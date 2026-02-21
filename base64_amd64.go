//go:build goexperiment.simd

package b64simd

import "simd/archsimd"

// SIMD constants, initialized per-tier in init().
var (
	// Encode: mulhi/mullo sextet extraction (shared by AVX2 and VBMI tiers).
	encMulhiM archsimd.Uint16x16
	encMulhiC archsimd.Uint16x16
	encMulloM archsimd.Uint16x16
	encMulloC archsimd.Uint16x16

	// Decode: Muła/Nojiri nibble-LUT validation + roll-table + packing (shared).
	nibbleMask archsimd.Uint8x32
	decLutHi   archsimd.Uint8x32
	decLutLo   archsimd.Uint8x32
	decLutRoll archsimd.Uint8x32
	dec2F      archsimd.Uint8x32
	decShift4  archsimd.Uint32x8
	decMaddub  archsimd.Int8x32
	decMaddwd  archsimd.Int16x16

	// Tier 1 encode: -4 offset reshuffle + 4-op ASCII mapping.
	encAVX2Reshuf archsimd.Int8x32
	enc51         archsimd.Uint8x32
	enc25         archsimd.Int8x32
	encLut        archsimd.Uint8x32

	// Tier 1 decode: VPSHUFB within lanes + VPERMD cross-lane.
	decAVX2Compact archsimd.Int8x32
	decAVX2Permute archsimd.Uint32x8

	// Tier 2 encode: VPERMB reshuffle + VPERMI2B 64-entry LUT.
	encVBMIReshuf archsimd.Uint8x32
	encVBMILutLo  archsimd.Uint8x32
	encVBMILutHi  archsimd.Uint8x32

	// Tier 2 decode: single VPERMB replaces VPSHUFB+VPERMD.
	decVBMICompact archsimd.Uint8x32
)

// Tier 1: AVX2 (works under Rosetta emulation).
// Tier 2: AVX-512 + VBMI (server-grade: EPYC Zen 4+, Ice Lake+).
func init() {
	if !archsimd.X86.AVX2() {
		return
	}
	initAVX2()
	bulkEncode = encodeAVX2
	bulkDecode = decodeAVX2
	StdEncoding.useSIMD = true
	RawStdEncoding.useSIMD = true

	if !archsimd.X86.AVX512() || !archsimd.X86.AVX512VBMI() {
		return
	}
	initVBMI()
	bulkEncode = encodeVBMI
	bulkDecode = decodeVBMI
}

// --- Helpers ---

// dup duplicates a 16-byte pattern into both lanes of a 32-byte array.
func dup(b [16]byte) [32]byte {
	var r [32]byte
	copy(r[:16], b[:])
	copy(r[16:], b[:])
	return r
}

// u16pair fills a 256-bit vector with alternating uint16 values.
func u16pair(even, odd uint16) archsimd.Uint16x16 {
	var a [16]uint16
	for i := 0; i < 16; i += 2 {
		a[i] = even
		a[i+1] = odd
	}
	return archsimd.LoadUint16x16(&a)
}

// --- Init ---

func initAVX2() {
	encMulhiM = u16pair(0xFC00, 0x0FC0)
	encMulhiC = u16pair(0x0040, 0x0400)
	encMulloM = u16pair(0x03F0, 0x003F)
	encMulloC = u16pair(0x0010, 0x0100)

	// -4 offset reshuffle: lane 0 indices +4 (loaded 4 bytes early), lane 1 normal.
	encAVX2Reshuf = archsimd.LoadUint8x32(&[32]byte{
		5, 4, 6, 5, 8, 7, 9, 8, 11, 10, 12, 11, 14, 13, 15, 14, // lane 0 (+4)
		1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10, // lane 1
	}).AsInt8x32()

	enc51 = archsimd.BroadcastUint8x32(51)
	enc25 = archsimd.BroadcastInt8x32(25)
	el := dup([16]byte{
		65,                                               // A-Z
		71,                                               // a-z
		252, 252, 252, 252, 252, 252, 252, 252, 252, 252, // 0-9
		237, // +
		240, // /
		0, 0,
	})
	encLut = archsimd.LoadUint8x32(&el)

	nibbleMask = archsimd.BroadcastUint8x32(0x0F)
	decShift4 = archsimd.BroadcastUint32x8(4)
	dec2F = archsimd.BroadcastUint8x32(0x2F)

	hi := dup([16]byte{
		0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x08,
		0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
	})
	decLutHi = archsimd.LoadUint8x32(&hi)
	lo := dup([16]byte{
		0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
		0x11, 0x11, 0x13, 0x1A, 0x1B, 0x1B, 0x1B, 0x1A,
	})
	decLutLo = archsimd.LoadUint8x32(&lo)
	roll := dup([16]byte{
		0, 16, 19, 4, 191, 191, 185, 185, // '/', '+', '0'-'9', 'A'-'Z', 'a'-'z'
		0, 0, 0, 0, 0, 0, 0, 0,
	})
	decLutRoll = archsimd.LoadUint8x32(&roll)

	mu := dup([16]byte{0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01,
		0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01})
	decMaddub = archsimd.LoadUint8x32(&mu).AsInt8x32()
	decMaddwd = archsimd.LoadInt16x16(&[16]int16{
		0x1000, 1, 0x1000, 1, 0x1000, 1, 0x1000, 1,
		0x1000, 1, 0x1000, 1, 0x1000, 1, 0x1000, 1,
	})

	cs := dup([16]byte{2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0x80, 0x80, 0x80, 0x80})
	decAVX2Compact = archsimd.LoadUint8x32(&cs).AsInt8x32()
	decAVX2Permute = archsimd.LoadUint32x8(&[8]uint32{0, 1, 2, 4, 5, 6, 7, 7})
}

func initVBMI() {
	encVBMIReshuf = archsimd.LoadUint8x32(&[32]byte{
		1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
		13, 12, 14, 13, 16, 15, 17, 16, 19, 18, 20, 19, 22, 21, 23, 22,
	})

	// 64-entry LUT split across two 32-byte vectors for VPERMI2B.
	var lo32, hi32 [32]byte
	for i := range 26 {
		lo32[i] = 65 // A-Z
	}
	for i := 26; i < 32; i++ {
		lo32[i] = 71 // a-z (26-31)
	}
	for i := range 20 {
		hi32[i] = 71 // a-z (32-51)
	}
	for i := 20; i < 30; i++ {
		hi32[i] = 252 // 0-9 (52-61)
	}
	hi32[30] = 237 // +
	hi32[31] = 240 // /
	encVBMILutLo = archsimd.LoadUint8x32(&lo32)
	encVBMILutHi = archsimd.LoadUint8x32(&hi32)

	// Byte compaction: extract 3 of every 4 bytes, cross-lane.
	decVBMICompact = archsimd.LoadUint8x32(&[32]byte{
		2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12,
		18, 17, 16, 22, 21, 20, 26, 25, 24, 30, 29, 28,
	})
}

// --- Tier 1: AVX2 ---

// encodeAVX2 encodes 24-byte blocks using the -4 offset load trick.
func encodeAVX2(dst, src []byte) (di, si int) {
	srcEnd := len(src) - 32
	dstEnd := len(dst) - 32
	if 6 > srcEnd || 8 > dstEnd {
		return 0, 0
	}

	// Scalar preamble: 2 triplets to establish si >= 4 for the offset load.
	for i := 0; i < 2; i++ {
		v := uint(src[3*i])<<16 | uint(src[3*i+1])<<8 | uint(src[3*i+2])
		dst[4*i+0] = encodeStd[v>>18&0x3F]
		dst[4*i+1] = encodeStd[v>>12&0x3F]
		dst[4*i+2] = encodeStd[v>>6&0x3F]
		dst[4*i+3] = encodeStd[v&0x3F]
	}
	si = 6
	di = 8

	// Local copies keep constants in registers (measured +37% without).
	reshuf := encAVX2Reshuf
	mulhiM := encMulhiM
	mulhiC := encMulhiC
	mulloM := encMulloM
	mulloC := encMulloC
	c51 := enc51
	c25 := enc25
	lut := encLut

	for si <= srcEnd && di <= dstEnd {
		raw := archsimd.LoadUint8x32Slice(src[si-4 : si+28])
		reshuffled := raw.PermuteOrZeroGrouped(reshuf)
		asU16 := reshuffled.AsUint16x16()
		hi := asU16.And(mulhiM).MulHigh(mulhiC)
		lo := asU16.And(mulloM).Mul(mulloC)
		sextets := hi.Or(lo).AsUint8x32()
		reduced := sextets.SubSaturated(c51)
		above25 := sextets.AsInt8x32().Greater(c25)
		idx := reduced.Sub(above25.ToInt8x32().AsUint8x32())
		delta := lut.PermuteOrZeroGrouped(idx.AsInt8x32())
		result := sextets.Add(delta)
		d := dst[di:]
		result.StoreSlice(d[:32])
		si += 24
		di += 32
	}
	return di, si
}

// decodeAVX2 decodes 32-byte blocks. Stops on invalid input.
func decodeAVX2(dst, src []byte) (di, si int) {
	// Local copies keep constants in registers (measured +32% without).
	nmask := nibbleMask
	lutHi := decLutHi
	lutLo := decLutLo
	lutRoll := decLutRoll
	c2F := dec2F
	shift4 := decShift4
	maddub := decMaddub
	maddwd := decMaddwd
	compact := decAVX2Compact
	perm := decAVX2Permute

	srcEnd := len(src) - 32
	dstEnd := len(dst) - 32
	for si <= srcEnd && di <= dstEnd {
		in := archsimd.LoadUint8x32Slice(src[si : si+32])

		hiNib := in.AsUint32x8().ShiftRight(shift4).AsUint8x32().And(nmask)
		loNib := in.And(nmask)
		if !lutHi.PermuteOrZeroGrouped(hiNib.AsInt8x32()).And(
			lutLo.PermuteOrZeroGrouped(loNib.AsInt8x32())).IsZero() {
			break
		}

		eq2F := in.Equal(c2F).ToInt8x32().AsUint8x32()
		roll := lutRoll.PermuteOrZeroGrouped(hiNib.Add(eq2F).AsInt8x32())
		translated := in.Add(roll)

		merged := translated.DotProductPairsSaturated(maddub)
		packed := merged.DotProductPairs(maddwd)
		compacted := packed.AsUint8x32().PermuteOrZeroGrouped(compact)
		result := compacted.AsUint32x8().Permute(perm).AsUint8x32()

		d := dst[di:]
		result.StoreSlice(d[:32])
		si += 32
		di += 24
	}
	return di, si
}

// --- Tier 2: VBMI ---

// encodeVBMI encodes 24-byte blocks. VPERMB crosses lanes; VPERMI2B maps sextets→ASCII.
func encodeVBMI(dst, src []byte) (di, si int) {
	srcEnd := len(src) - 32
	dstEnd := len(dst) - 32
	if srcEnd < 0 || dstEnd < 0 {
		return 0, 0
	}

	reshuf := encVBMIReshuf
	mulhiM := encMulhiM
	mulhiC := encMulhiC
	mulloM := encMulloM
	mulloC := encMulloC
	lutLo := encVBMILutLo
	lutHi := encVBMILutHi

	for si <= srcEnd && di <= dstEnd {
		raw := archsimd.LoadUint8x32Slice(src[si : si+32])
		reshuffled := raw.Permute(reshuf)
		asU16 := reshuffled.AsUint16x16()
		hi := asU16.And(mulhiM).MulHigh(mulhiC)
		lo := asU16.And(mulloM).Mul(mulloC)
		sextets := hi.Or(lo).AsUint8x32()
		delta := lutLo.ConcatPermute(lutHi, sextets)
		result := sextets.Add(delta)
		d := dst[di:]
		result.StoreSlice(d[:32])
		si += 24
		di += 32
	}
	return di, si
}

// decodeVBMI decodes 32-byte blocks. VPERMB compaction replaces VPSHUFB+VPERMD.
func decodeVBMI(dst, src []byte) (di, si int) {
	nmask := nibbleMask
	lutHi := decLutHi
	lutLo := decLutLo
	lutRoll := decLutRoll
	c2F := dec2F
	shift4 := decShift4
	maddub := decMaddub
	maddwd := decMaddwd
	compact := decVBMICompact

	srcEnd := len(src) - 32
	dstEnd := len(dst) - 32
	for si <= srcEnd && di <= dstEnd {
		in := archsimd.LoadUint8x32Slice(src[si : si+32])

		hiNib := in.AsUint32x8().ShiftRight(shift4).AsUint8x32().And(nmask)
		loNib := in.And(nmask)
		if !lutHi.PermuteOrZeroGrouped(hiNib.AsInt8x32()).And(
			lutLo.PermuteOrZeroGrouped(loNib.AsInt8x32())).IsZero() {
			break
		}

		eq2F := in.Equal(c2F).ToInt8x32().AsUint8x32()
		roll := lutRoll.PermuteOrZeroGrouped(hiNib.Add(eq2F).AsInt8x32())
		translated := in.Add(roll)

		merged := translated.DotProductPairsSaturated(maddub)
		packed := merged.DotProductPairs(maddwd)
		result := packed.AsUint8x32().Permute(compact)

		d := dst[di:]
		result.StoreSlice(d[:32])
		si += 32
		di += 24
	}
	return di, si
}
