//go:build goexperiment.simd

package simdenc

import "simd/archsimd"

const (
	encodeStd = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	encodeURL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)

// Per-alphabet encode constants.
type encodeAlpha struct {
	alphabet      string
	sextetToAscii archsimd.Uint8x32 // AVX2: range-based sextet→ASCII offset
	asciiTableLo  archsimd.Uint8x32 // VBMI: 64-entry table, low 32 entries
	asciiTableHi  archsimd.Uint8x32 // VBMI: 64-entry table, high 32 entries
	asciiTable512 archsimd.Uint8x64 // AVX-512: 64-entry table in one vector
}

// Per-alphabet decode constants.
type decodeAlpha struct {
	validHi, validLo archsimd.Uint8x32 // Muła/Nojiri validation tables
	rollTable        archsimd.Uint8x32 // ASCII-to-sextet offset by high nibble
	special          archsimd.Uint8x32 // broadcast of special char ('/' or '_')
	shift            archsimd.Uint8x32 // broadcast of special char shift
	decTable         *[256]byte         // scalar tail lookup
}

var (
	encAlphas [2]encodeAlpha
	decAlphas [2]decodeAlpha
	decTables [2][256]byte
	hasAVX512 bool
)

// Shared encode constants (same for both alphabets).
var (
	encSextetMaskHi  archsimd.Uint16x16
	encSextetShiftHi archsimd.Uint16x16
	encSextetMaskLo  archsimd.Uint16x16
	encSextetShiftLo archsimd.Uint16x16
)

// AVX2 encode constants.
var (
	encOffsetShuffle   archsimd.Int8x32  // byte grouping with -4 offset trick
	encLastLowerSextet archsimd.Uint8x32 // broadcast 51
	encLastUpperSextet archsimd.Int8x32  // broadcast 25
)

// VBMI encode constants.
var encCrossLaneShuffle archsimd.Uint8x32

// AVX-512 encode constants.
var (
	encShuffle512   archsimd.Uint8x64
	encMaskHi512    archsimd.Uint16x32
	encShiftHi512   archsimd.Uint16x32
	encMaskLo512    archsimd.Uint16x32
	encShiftLo512   archsimd.Uint16x32
)

// Shared decode constants.
var (
	nibbleMask     archsimd.Uint8x32
	nibbleShift    archsimd.Uint32x8
	combinePairs   archsimd.Int8x32  // [64, 1, ...] adjacent 6-bit pairs → 12-bit
	combineQuads   archsimd.Int16x16 // [4096, 1, ...] adjacent 12-bit pairs → 24-bit
	extractShuffle archsimd.Int8x32  // AVX2: intra-lane byte extract
	extractPermute archsimd.Uint32x8 // AVX2: cross-lane dword pack
	extractVBMI    archsimd.Uint8x32 // VBMI: single cross-lane byte extract+pack
)

// --- Helpers ---

func alt16(even, odd uint16) archsimd.Uint16x16 {
	lane := archsimd.BroadcastUint16x8(even).InterleaveLo(archsimd.BroadcastUint16x8(odd))
	var z archsimd.Uint16x16
	return z.SetLo(lane).SetHi(lane)
}

func dupBytes(v archsimd.Uint8x16) archsimd.Uint8x32 {
	var z archsimd.Uint8x32
	return z.SetLo(v).SetHi(v)
}

// --- Init ---

func init() {
	if !archsimd.X86.AVX2() {
		return
	}
	simdEncode = doEncode
	simdDecode = doDecode

	// Shared encode: sextet extraction masks.
	encSextetMaskHi = alt16(0xFC00, 0x0FC0)
	encSextetShiftHi = alt16(0x0040, 0x0400)
	encSextetMaskLo = alt16(0x03F0, 0x003F)
	encSextetShiftLo = alt16(0x0010, 0x0100)

	// AVX2 encode.
	encOffsetShuffle = archsimd.LoadUint8x32(&[32]byte{
		5, 4, 6, 5, 8, 7, 9, 8, 11, 10, 12, 11, 14, 13, 15, 14,
		1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
	}).AsInt8x32()
	encLastLowerSextet = archsimd.BroadcastUint8x32(51)
	encLastUpperSextet = archsimd.BroadcastInt8x32(25)

	// VBMI encode.
	encCrossLaneShuffle = archsimd.LoadUint8x32(&[32]byte{
		1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
		13, 12, 14, 13, 16, 15, 17, 16, 19, 18, 20, 19, 22, 21, 23, 22,
	})

	// Shared decode.
	nibbleMask = archsimd.BroadcastUint8x32(0x0F)
	nibbleShift = archsimd.BroadcastUint32x8(4)
	combinePairs = dupBytes(archsimd.LoadUint8x16(&[16]byte{
		0x40, 1, 0x40, 1, 0x40, 1, 0x40, 1, 0x40, 1, 0x40, 1, 0x40, 1, 0x40, 1,
	})).AsInt8x32()
	maddLane := archsimd.BroadcastInt16x8(0x1000).InterleaveLo(archsimd.BroadcastInt16x8(1))
	var maddZ archsimd.Int16x16
	combineQuads = maddZ.SetLo(maddLane).SetHi(maddLane)

	// AVX2 decode: two-step extract + pack.
	extractShuffle = dupBytes(archsimd.LoadUint8x16(&[16]byte{
		2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0x80, 0x80, 0x80, 0x80,
	})).AsInt8x32()
	extractPermute = archsimd.LoadUint32x8(&[8]uint32{0, 1, 2, 4, 5, 6, 7, 7})

	// Per-alphabet constants.
	initAlphabet(alphabetStd, encodeStd, 0x2F, 237, 240, 0xFF,
		[16]byte{0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x08, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10},
		[16]byte{0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x13, 0x1A, 0x1B, 0x1B, 0x1B, 0x1A},
		[16]byte{0, 16, 19, 4, 191, 191, 185, 185, 0, 0, 0, 0, 0, 0, 0, 0})
	initAlphabet(alphabetURL, encodeURL, 0x5F, 239, 32, 0x03,
		[16]byte{0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x08, 0x20, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01},
		[16]byte{0x0B, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x07, 0x37, 0x37, 0x35, 0x37, 0x27},
		[16]byte{0, 0, 17, 4, 191, 191, 185, 185, 224, 0, 0, 0, 0, 0, 0, 0})

	if !archsimd.X86.AVX512() || !archsimd.X86.AVX512VBMI() {
		return
	}
	hasAVX512 = true

	// VBMI decode: single cross-lane extract+pack.
	extractVBMI = archsimd.LoadUint8x32(&[32]byte{
		2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12,
		18, 17, 16, 22, 21, 20, 26, 25, 24, 30, 29, 28,
	})

	// AVX-512 encode: byte grouping.
	var grouping [64]byte
	pat := [16]byte{1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10}
	for lane := range 4 {
		for i := range 16 {
			grouping[lane*16+i] = pat[i] + byte(lane*12)
		}
	}
	encShuffle512 = archsimd.LoadUint8x64(&grouping)

	// AVX-512 encode: widened sextet extraction masks.
	var z32 archsimd.Uint16x32
	encMaskHi512 = z32.SetLo(encSextetMaskHi).SetHi(encSextetMaskHi)
	encShiftHi512 = z32.SetLo(encSextetShiftHi).SetHi(encSextetShiftHi)
	encMaskLo512 = z32.SetLo(encSextetMaskLo).SetHi(encSextetMaskLo)
	encShiftLo512 = z32.SetLo(encSextetShiftLo).SetHi(encSextetShiftLo)
}

func initAlphabet(idx uint8, alphabet string, specialChar, enc62, enc63, specialShift byte,
	valHi, valLo, roll [16]byte) {
	a := &encAlphas[idx]
	d := &decAlphas[idx]

	// Encode tables.
	a.alphabet = alphabet
	a.sextetToAscii = dupBytes(archsimd.LoadUint8x16(&[16]byte{
		65, 71, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, enc62, enc63,
	}))
	var lo32, hi32 [32]byte
	for i := range 26 {
		lo32[i] = 65
	}
	for i := 26; i < 32; i++ {
		lo32[i] = 71
	}
	for i := range 20 {
		hi32[i] = 71
	}
	for i := 20; i < 30; i++ {
		hi32[i] = 252
	}
	hi32[30], hi32[31] = enc62, enc63
	a.asciiTableLo = archsimd.LoadUint8x32(&lo32)
	a.asciiTableHi = archsimd.LoadUint8x32(&hi32)

	// AVX-512 table.
	var t512 [64]byte
	for i := range 26 {
		t512[i] = 65
	}
	for i := 26; i < 52; i++ {
		t512[i] = 71
	}
	for i := 52; i < 62; i++ {
		t512[i] = 252
	}
	t512[62], t512[63] = enc62, enc63
	a.asciiTable512 = archsimd.LoadUint8x64(&t512)

	// Decode tables.
	for i := range decTables[idx] {
		decTables[idx][i] = 0xFF
	}
	for i := range 64 {
		decTables[idx][alphabet[i]] = byte(i)
	}
	d.decTable = &decTables[idx]
	d.special = archsimd.BroadcastUint8x32(specialChar)
	d.shift = archsimd.BroadcastUint8x32(specialShift)
	d.validHi = dupBytes(archsimd.LoadUint8x16(&valHi))
	d.validLo = dupBytes(archsimd.LoadUint8x16(&valLo))
	d.rollTable = dupBytes(archsimd.LoadUint8x16(&roll))
}

// --- Encode ---

func doEncode(alphabet uint8, dst, src []byte) {
	a := &encAlphas[alphabet]
	n := len(src)
	di, si := 0, 0
	alpha := a.alphabet

	// Stage 1: AVX-512 (48 bytes/iter).
	if hasAVX512 && n >= 64 {
		shuffle := encShuffle512
		maskHi := encMaskHi512
		shiftHi := encShiftHi512
		maskLo := encMaskLo512
		shiftLo := encShiftLo512
		asciiTable := a.asciiTable512

		srcEnd, dstEnd := n-64, len(dst)-64
		for si <= srcEnd && di <= dstEnd {
			grouped := archsimd.LoadUint8x64Slice(src[si : si+64]).Permute(shuffle)
			w := grouped.AsUint16x32()
			sextets := w.And(maskHi).MulHigh(shiftHi).Or(w.And(maskLo).Mul(shiftLo)).AsUint8x64()
			result := sextets.Add(asciiTable.Permute(sextets))
			result.StoreSlice(dst[di : di+64])
			si += 48
			di += 64
		}
	}

	// Stage 2: VBMI 256-bit (24 bytes/iter, cross-lane).
	if hasAVX512 && n-si >= 32 {
		shuffle := encCrossLaneShuffle
		maskHi := encSextetMaskHi
		shiftHi := encSextetShiftHi
		maskLo := encSextetMaskLo
		shiftLo := encSextetShiftLo
		tableLo := a.asciiTableLo
		tableHi := a.asciiTableHi

		srcEnd, dstEnd := len(src)-32, len(dst)-32
		for si <= srcEnd && di <= dstEnd {
			grouped := archsimd.LoadUint8x32Slice(src[si : si+32]).Permute(shuffle)
			w := grouped.AsUint16x16()
			sextets := w.And(maskHi).MulHigh(shiftHi).Or(w.And(maskLo).Mul(shiftLo)).AsUint8x32()
			asciiOffset := tableLo.ConcatPermute(tableHi, sextets)
			result := sextets.Add(asciiOffset)
			result.StoreSlice(dst[di : di+32])
			si += 24
			di += 32
		}
	}

	// Stage 2 (AVX2 fallback): offset-load trick (24 bytes/iter, lane-local).
	if !hasAVX512 && n >= 34 {
		// Scalar preamble: encode 2 triplets so si >= 4 for the offset load.
		for i := 0; i < 2; i++ {
			v := uint(src[3*i])<<16 | uint(src[3*i+1])<<8 | uint(src[3*i+2])
			dst[4*i], dst[4*i+1], dst[4*i+2], dst[4*i+3] = alpha[v>>18&0x3F], alpha[v>>12&0x3F], alpha[v>>6&0x3F], alpha[v&0x3F]
		}
		si = 6
		di = 8

		shuffle := encOffsetShuffle
		maskHi := encSextetMaskHi
		shiftHi := encSextetShiftHi
		maskLo := encSextetMaskLo
		shiftLo := encSextetShiftLo
		lastLower := encLastLowerSextet
		lastUpper := encLastUpperSextet
		sextetToAscii := a.sextetToAscii

		srcEnd, dstEnd := len(src)-32, len(dst)-32
		for si <= srcEnd && di <= dstEnd {
			grouped := archsimd.LoadUint8x32Slice(src[si-4 : si+28]).PermuteOrZeroGrouped(shuffle)
			w := grouped.AsUint16x16()
			sextets := w.And(maskHi).MulHigh(shiftHi).Or(w.And(maskLo).Mul(shiftLo)).AsUint8x32()
			saturated := sextets.SubSaturated(lastLower)
			pastUpper := sextets.AsInt8x32().Greater(lastUpper).ToInt8x32().AsUint8x32()
			rangeIdx := saturated.Sub(pastUpper)
			asciiOffset := sextetToAscii.PermuteOrZeroGrouped(rangeIdx.AsInt8x32())
			result := sextets.Add(asciiOffset)
			result.StoreSlice(dst[di : di+32])
			si += 24
			di += 32
		}
	}

	// Stage 3: scalar tail.
	for si+2 < n {
		v := uint(src[si])<<16 | uint(src[si+1])<<8 | uint(src[si+2])
		dst[di], dst[di+1], dst[di+2], dst[di+3] = alpha[v>>18&0x3F], alpha[v>>12&0x3F], alpha[v>>6&0x3F], alpha[v&0x3F]
		si += 3
		di += 4
	}
	switch n - si {
	case 2:
		v := uint(src[si])<<16 | uint(src[si+1])<<8
		dst[di], dst[di+1], dst[di+2] = alpha[v>>18&0x3F], alpha[v>>12&0x3F], alpha[v>>6&0x3F]
	case 1:
		v := uint(src[si]) << 16
		dst[di], dst[di+1] = alpha[v>>18&0x3F], alpha[v>>12&0x3F]
	}
}

// --- Decode ---

func doDecode(alphabet uint8, dst, src []byte) (int, int) {
	a := &decAlphas[alphabet]
	di, si := 0, 0

	if hasAVX512 {
		nibMask := nibbleMask
		nibShift := nibbleShift
		validHi := a.validHi
		validLo := a.validLo
		rollTable := a.rollTable
		specialChar := a.special
		specialShift := a.shift
		pairs := combinePairs
		quads := combineQuads
		extract := extractVBMI

		srcEnd, dstEnd := len(src)-32, len(dst)-32
		for si <= srcEnd && di <= dstEnd {
			encoded := archsimd.LoadUint8x32Slice(src[si : si+32])
			hiNib := encoded.AsUint32x8().ShiftRight(nibShift).AsUint8x32().And(nibMask)
			loNib := encoded.And(nibMask)
			if !validHi.PermuteOrZeroGrouped(hiNib.AsInt8x32()).And(
				validLo.PermuteOrZeroGrouped(loNib.AsInt8x32())).IsZero() {
				break
			}
			isSpecial := encoded.Equal(specialChar).ToInt8x32().AsUint8x32().And(specialShift)
			roll := rollTable.PermuteOrZeroGrouped(hiNib.Add(isSpecial).AsInt8x32())
			sextets := encoded.Add(roll)
			twelveBit := sextets.DotProductPairsSaturated(pairs)
			twentyFourBit := twelveBit.DotProductPairs(quads)
			result := twentyFourBit.AsUint8x32().Permute(extract)
			result.StoreSlice(dst[di : di+32])
			si += 32
			di += 24
		}
	} else {
		nibMask := nibbleMask
		nibShift := nibbleShift
		validHi := a.validHi
		validLo := a.validLo
		rollTable := a.rollTable
		specialChar := a.special
		specialShift := a.shift
		pairs := combinePairs
		quads := combineQuads
		extractShuf := extractShuffle
		extractPerm := extractPermute

		srcEnd, dstEnd := len(src)-32, len(dst)-32
		for si <= srcEnd && di <= dstEnd {
			encoded := archsimd.LoadUint8x32Slice(src[si : si+32])
			hiNib := encoded.AsUint32x8().ShiftRight(nibShift).AsUint8x32().And(nibMask)
			loNib := encoded.And(nibMask)
			if !validHi.PermuteOrZeroGrouped(hiNib.AsInt8x32()).And(
				validLo.PermuteOrZeroGrouped(loNib.AsInt8x32())).IsZero() {
				break
			}
			isSpecial := encoded.Equal(specialChar).ToInt8x32().AsUint8x32().And(specialShift)
			roll := rollTable.PermuteOrZeroGrouped(hiNib.Add(isSpecial).AsInt8x32())
			sextets := encoded.Add(roll)
			twelveBit := sextets.DotProductPairsSaturated(pairs)
			twentyFourBit := twelveBit.DotProductPairs(quads)
			result := twentyFourBit.AsUint8x32().PermuteOrZeroGrouped(extractShuf).AsUint32x8().Permute(extractPerm).AsUint8x32()
			result.StoreSlice(dst[di : di+32])
			si += 32
			di += 24
		}
	}

	// Scalar tail.
	table := a.decTable
	for si+3 < len(src) {
		va, vb, vc, vd := table[src[si]], table[src[si+1]], table[src[si+2]], table[src[si+3]]
		if (va|vb|vc|vd)&0x80 != 0 {
			break
		}
		dst[di], dst[di+1], dst[di+2] = va<<2|vb>>4, vb<<4|vc>>2, vc<<6|vd
		si += 4
		di += 3
	}
	if rem := len(src) - si; rem >= 2 {
		va, vb := table[src[si]], table[src[si+1]]
		if (va|vb)&0x80 != 0 {
			return di, si
		}
		dst[di] = va<<2 | vb>>4
		di++
		si += 2
		if rem >= 3 {
			vc := table[src[si]]
			if vc&0x80 != 0 {
				return di, si
			}
			dst[di] = vb<<4 | vc>>2
			di++
			si++
		}
	}
	return di, si
}
