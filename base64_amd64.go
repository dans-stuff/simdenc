//go:build goexperiment.simd

package simdenc

import (
	"os"
	"simd/archsimd"
)

// Per-alphabet encode constants.
type encodeAlpha struct {
	alphabet      string
	sextetToAscii archsimd.Uint8x32 // AVX2: range-based sextet→ASCII offset
	asciiTable512 archsimd.Uint8x64 // AVX-512: 64-entry table in one vector
}

// Per-alphabet decode constants.
type decodeAlpha struct {
	validHi, validLo archsimd.Uint8x32 // Muła/Nojiri validation tables
	rollTable        archsimd.Uint8x32 // ASCII-to-sextet offset by high nibble
	special          archsimd.Uint8x32 // broadcast of special char ('/' or '_')
	shift            archsimd.Uint8x32 // broadcast of special char shift
}

var (
	encAlphas [2]encodeAlpha
	decAlphas [2]decodeAlpha
	hasAVX512 bool
)

// Shared encode constants (same for both alphabets).
var (
	encSextetMaskHi  archsimd.Uint16x16
	encSextetShiftHi archsimd.Uint16x16
	encSextetMaskLo  archsimd.Uint16x16
	encSextetShiftLo archsimd.Uint16x16
)

// SSE encode constants.
var (
	encSSEShuffle      archsimd.Int8x16  // byte grouping (no offset trick)
	encSSELowerSextet  archsimd.Uint8x16 // broadcast 51
	encSSEUpperSextet  archsimd.Int8x16  // broadcast 25
	encSSEMaskHi       archsimd.Uint16x8
	encSSEShiftHi      archsimd.Uint16x8
	encSSEMaskLo       archsimd.Uint16x8
	encSSEShiftLo      archsimd.Uint16x8
)

// Per-alphabet SSE encode constant.
type encodeAlphaSSE struct {
	sextetToAscii archsimd.Uint8x16
}

var encAlphasSSE [2]encodeAlphaSSE

// AVX2 encode constants.
var (
	encOffsetShuffle   archsimd.Int8x32  // byte grouping with -4 offset trick
	encLastLowerSextet archsimd.Uint8x32 // broadcast 51
	encLastUpperSextet archsimd.Int8x32  // broadcast 25
)

// AVX-512 encode constants.
var (
	encShuffle512   archsimd.Uint8x64
	encMaskHi512    archsimd.Uint16x32
	encShiftHi512   archsimd.Uint16x32
	encMaskLo512    archsimd.Uint16x32
	encShiftLo512   archsimd.Uint16x32
)

// SSE decode constants (128-bit).
var (
	decNibbleMask16   archsimd.Uint8x16
	decNibbleShift16  archsimd.Uint32x4
	decCombinePairs16 archsimd.Int8x16
	decCombineQuads16 archsimd.Int16x8
	decExtract16 archsimd.Int8x16 // intra-register byte extract+pack
)

// AVX2/VBMI decode constants (256-bit).
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

	// SSE encode constants.
	encSSEShuffle = archsimd.LoadUint8x16(&[16]byte{
		1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
	}).AsInt8x16()
	encSSELowerSextet = archsimd.BroadcastUint8x16(51)
	encSSEUpperSextet = archsimd.BroadcastInt8x16(25)
	alt8 := func(even, odd uint16) archsimd.Uint16x8 {
		return archsimd.BroadcastUint16x8(even).InterleaveLo(archsimd.BroadcastUint16x8(odd))
	}
	encSSEMaskHi = alt8(0xFC00, 0x0FC0)
	encSSEShiftHi = alt8(0x0040, 0x0400)
	encSSEMaskLo = alt8(0x03F0, 0x003F)
	encSSEShiftLo = alt8(0x0010, 0x0100)

	// Shared encode: sextet extraction masks (AVX2 width).
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

	// SSE decode (128-bit).
	decNibbleMask16 = archsimd.BroadcastUint8x16(0x0F)
	decNibbleShift16 = archsimd.BroadcastUint32x4(4)
	pairsLane := archsimd.LoadUint8x16(&[16]byte{
		0x40, 1, 0x40, 1, 0x40, 1, 0x40, 1, 0x40, 1, 0x40, 1, 0x40, 1, 0x40, 1,
	})
	decCombinePairs16 = pairsLane.AsInt8x16()
	decCombineQuads16 = archsimd.BroadcastInt16x8(0x1000).InterleaveLo(archsimd.BroadcastInt16x8(1))
	decExtract16 = archsimd.LoadUint8x16(&[16]byte{
		2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0x80, 0x80, 0x80, 0x80,
	}).AsInt8x16()

	// AVX2 decode (256-bit).
	nibbleMask = archsimd.BroadcastUint8x32(0x0F)
	nibbleShift = archsimd.BroadcastUint32x8(4)
	combinePairs = dupBytes(pairsLane).AsInt8x32()
	maddLane := archsimd.BroadcastInt16x8(0x1000).InterleaveLo(archsimd.BroadcastInt16x8(1))
	var maddZ archsimd.Int16x16
	combineQuads = maddZ.SetLo(maddLane).SetHi(maddLane)

	// AVX2 decode: two-step extract + pack.
	extractShuffle = dupBytes(archsimd.LoadUint8x16(&[16]byte{
		2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0x80, 0x80, 0x80, 0x80,
	})).AsInt8x32()
	extractPermute = archsimd.LoadUint32x8(&[8]uint32{0, 1, 2, 4, 5, 6, 7, 7})

	// Per-alphabet constants.
	initAlphabet(alphabetStd, encodeStdAlpha, 0x2F, 237, 240, 0xFF,
		[16]byte{0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x08, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10},
		[16]byte{0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x13, 0x1A, 0x1B, 0x1B, 0x1B, 0x1A},
		[16]byte{0, 16, 19, 4, 191, 191, 185, 185, 0, 0, 0, 0, 0, 0, 0, 0})
	initAlphabet(alphabetURL, encodeURLAlpha, 0x5F, 239, 32, 0x03,
		[16]byte{0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x08, 0x20, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01},
		[16]byte{0x0B, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x07, 0x37, 0x37, 0x35, 0x37, 0x27},
		[16]byte{0, 0, 17, 4, 191, 191, 185, 185, 224, 0, 0, 0, 0, 0, 0, 0})

	if !archsimd.X86.AVX512() || !archsimd.X86.AVX512VBMI() || os.Getenv("SIMDENC_NO_AVX512") != "" {
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
	sseLut := archsimd.LoadUint8x16(&[16]byte{
		65, 71, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, enc62, enc63,
	})
	encAlphasSSE[idx].sextetToAscii = sseLut
	a.sextetToAscii = dupBytes(sseLut)
	// AVX-512 table (only if 512-bit instructions available).
	if archsimd.X86.AVX512() {
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
	}

	// Decode SIMD constants.
	d.special = archsimd.BroadcastUint8x32(specialChar)
	d.shift = archsimd.BroadcastUint8x32(specialShift)
	d.validHi = dupBytes(archsimd.LoadUint8x16(&valHi))
	d.validLo = dupBytes(archsimd.LoadUint8x16(&valLo))
	d.rollTable = dupBytes(archsimd.LoadUint8x16(&roll))
}

// --- Encode ---

func encodeSSE(alphabet uint8, dst, src []byte, di, si int) (int, int) {
	shuffle := encSSEShuffle
	maskHi := encSSEMaskHi
	shiftHi := encSSEShiftHi
	maskLo := encSSEMaskLo
	shiftLo := encSSEShiftLo
	lowerSextet := encSSELowerSextet
	upperSextet := encSSEUpperSextet
	sextetToAscii := encAlphasSSE[alphabet].sextetToAscii

	srcEnd, dstEnd := len(src)-16, len(dst)-16
	for si <= srcEnd && di <= dstEnd {
		grouped := archsimd.LoadUint8x16Slice(src[si : si+16]).PermuteOrZero(shuffle)
		w := grouped.AsUint16x8()
		sextets := w.And(maskHi).MulHigh(shiftHi).Or(w.And(maskLo).Mul(shiftLo)).AsUint8x16()
		saturated := sextets.SubSaturated(lowerSextet)
		pastUpper := sextets.AsInt8x16().Greater(upperSextet).ToInt8x16().AsUint8x16()
		rangeIdx := saturated.Sub(pastUpper)
		asciiOffset := sextetToAscii.PermuteOrZero(rangeIdx.AsInt8x16())
		result := sextets.Add(asciiOffset)
		d := dst[di:]
		result.StoreSlice(d[:16])
		si += 12
		di += 16
	}
	return di, si
}

// encodeSSEAVX2 fuses SSE bookends + AVX2 middle into a single function call.
// One hoist for both SSE and AVX2 register sets. Does both SSE iterations first
// (preamble + tail), then the AVX2 loop fills the middle.
// Returns (di, si) past the last byte fully encoded by SIMD.
func encodeSSEAVX2(alphabet uint8, a *encodeAlpha, dst, src []byte) (int, int) {
	n := len(src)

	// SSE constants.
	sseShuffle := encSSEShuffle
	sseMaskHi := encSSEMaskHi
	sseShiftHi := encSSEShiftHi
	sseMaskLo := encSSEMaskLo
	sseShiftLo := encSSEShiftLo
	sseLowerSextet := encSSELowerSextet
	sseUpperSextet := encSSEUpperSextet
	sseSextetToAscii := encAlphasSSE[alphabet].sextetToAscii

	// AVX2 constants.
	avxOffsetShuffle := encOffsetShuffle
	avxSextetMaskHi := encSextetMaskHi
	avxSextetShiftHi := encSextetShiftHi
	avxSextetMaskLo := encSextetMaskLo
	avxSextetShiftLo := encSextetShiftLo
	avxLastLowerSextet := encLastLowerSextet
	avxLastUpperSextet := encLastUpperSextet
	avxSextetToAscii := a.sextetToAscii

	// --- SSE preamble: encode src[0:12] → dst[0:16] ---

	grouped := archsimd.LoadUint8x16Slice(src[0:16]).PermuteOrZero(sseShuffle)
	w := grouped.AsUint16x8()
	sextets := w.And(sseMaskHi).MulHigh(sseShiftHi).Or(w.And(sseMaskLo).Mul(sseShiftLo)).AsUint8x16()
	saturated := sextets.SubSaturated(sseLowerSextet)
	pastUpper := sextets.AsInt8x16().Greater(sseUpperSextet).ToInt8x16().AsUint8x16()
	rangeIdx := saturated.Sub(pastUpper)
	asciiOffset := sseSextetToAscii.PermuteOrZero(rangeIdx.AsInt8x16())
	result := sextets.Add(asciiOffset)
	result.StoreSlice(dst[0:16])

	// --- AVX2 middle: loop from si=12 (after SSE preamble) ---
	// AVX2 needs si >= 4 for the -4 offset trick; si=12 satisfies this.
	di, si := 16, 12
	srcEnd, dstEnd := n-28, len(dst)-32
	for si <= srcEnd && di <= dstEnd {
		grouped256 := archsimd.LoadUint8x32Slice(src[si-4 : si+28]).PermuteOrZeroGrouped(avxOffsetShuffle)
		w256 := grouped256.AsUint16x16()
		sextets256 := w256.And(avxSextetMaskHi).MulHigh(avxSextetShiftHi).Or(w256.And(avxSextetMaskLo).Mul(avxSextetShiftLo)).AsUint8x32()
		saturated256 := sextets256.SubSaturated(avxLastLowerSextet)
		pastUpper256 := sextets256.AsInt8x32().Greater(avxLastUpperSextet).ToInt8x32().AsUint8x32()
		rangeIdx256 := saturated256.Sub(pastUpper256)
		asciiOffset256 := avxSextetToAscii.PermuteOrZeroGrouped(rangeIdx256.AsInt8x32())
		result256 := sextets256.Add(asciiOffset256)
		d := dst[di:]
		result256.StoreSlice(d[:32])
		si += 24
		di += 32
	}

	// --- SSE cleanup: encode remaining bytes after AVX2 loop ---
	srcEnd16, dstEnd16 := n-16, len(dst)-16
	for si <= srcEnd16 && di <= dstEnd16 {
		grouped = archsimd.LoadUint8x16Slice(src[si : si+16]).PermuteOrZero(sseShuffle)
		w = grouped.AsUint16x8()
		sextets = w.And(sseMaskHi).MulHigh(sseShiftHi).Or(w.And(sseMaskLo).Mul(sseShiftLo)).AsUint8x16()
		saturated = sextets.SubSaturated(sseLowerSextet)
		pastUpper = sextets.AsInt8x16().Greater(sseUpperSextet).ToInt8x16().AsUint8x16()
		rangeIdx = saturated.Sub(pastUpper)
		asciiOffset = sseSextetToAscii.PermuteOrZero(rangeIdx.AsInt8x16())
		result = sextets.Add(asciiOffset)
		d := dst[di:]
		result.StoreSlice(d[:16])
		si += 12
		di += 16
	}

	return di, si
}

// encode512 runs the 512-bit encode loop only. Returns (di, si) indicating
// how far it got. Caller handles remaining bytes via SSE or scalar.
func encode512(a *encodeAlpha, dst, src []byte) (di, si int) {
	shuffle512 := encShuffle512
	maskHi512 := encMaskHi512
	shiftHi512 := encShiftHi512
	maskLo512 := encMaskLo512
	shiftLo512 := encShiftLo512
	asciiTable512 := a.asciiTable512

	srcEnd, dstEnd := len(src)-64, len(dst)-64
	for si <= srcEnd && di <= dstEnd {
		grouped := archsimd.LoadUint8x64Slice(src[si : si+64]).Permute(shuffle512)
		w := grouped.AsUint16x32()
		sextets := w.And(maskHi512).MulHigh(shiftHi512).Or(w.And(maskLo512).Mul(shiftLo512)).AsUint8x64()
		result := sextets.Add(asciiTable512.Permute(sextets))
		d := dst[di:]
		result.StoreSlice(d[:64])
		si += 48
		di += 64
	}
	return di, si
}

// doEncode does the complete encode: SIMD bulk + scalar remainder.
// The caller only needs to handle padding.
func doEncode(alphabet uint8, dst, src []byte) {
	n := len(src)
	alpha := encAlphabets[alphabet]

	// SIMD bulk.
	di, si := 0, 0
	if n >= 28 {
		if n < 120 {
			di, si = encodeSSE(alphabet, dst, src, 0, 0)
		} else if hasAVX512 && n >= 256 {
			a := &encAlphas[alphabet]
			di, si = encode512(a, dst, src)
			di, si = encodeSSE(alphabet, dst, src, di, si)
		} else {
			a := &encAlphas[alphabet]
			di, si = encodeSSEAVX2(alphabet, a, dst, src)
		}
	}

	// Scalar remainder: full triplets.
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

func decodeSSE(a *decodeAlpha, dst, src []byte, di, si int) (int, int) {
	nibbleMask := decNibbleMask16
	nibbleShift := decNibbleShift16
	validHi := a.validHi.GetLo()
	validLo := a.validLo.GetLo()
	rollTable := a.rollTable.GetLo()
	special := a.special.GetLo()
	shift := a.shift.GetLo()
	combinePairs := decCombinePairs16
	combineQuads := decCombineQuads16
	extract := decExtract16

	srcEnd, dstEnd := len(src)-16, len(dst)-16
	for si <= srcEnd && di <= dstEnd {
		encoded := archsimd.LoadUint8x16Slice(src[si : si+16])
		hiNib := encoded.AsUint32x4().ShiftRight(nibbleShift).AsUint8x16().And(nibbleMask)
		loNib := encoded.And(nibbleMask)
		if !validHi.PermuteOrZero(hiNib.AsInt8x16()).And(
			validLo.PermuteOrZero(loNib.AsInt8x16())).IsZero() {
			break
		}
		isSpecial := encoded.Equal(special).ToInt8x16().AsUint8x16().And(shift)
		roll := rollTable.PermuteOrZero(hiNib.Add(isSpecial).AsInt8x16())
		sextets := encoded.Add(roll)
		twelveBit := sextets.DotProductPairsSaturated(combinePairs)
		twentyFourBit := twelveBit.DotProductPairs(combineQuads)
		result := twentyFourBit.AsUint8x16().PermuteOrZero(extract)
		d := dst[di:]
		result.StoreSlice(d[:16])
		si += 16
		di += 12
	}
	return di, si
}

// decodeSSEAVX2 fuses SSE bookends + AVX2 middle into a single function call.
// One hoist for both SSE and AVX2 register sets.
func decodeSSEAVX2(a *decodeAlpha, dst, src []byte) (int, int) {
	n := len(src)

	// SSE constants.
	sseNibbleMask := decNibbleMask16
	sseNibbleShift := decNibbleShift16
	sseValidHi := a.validHi.GetLo()
	sseValidLo := a.validLo.GetLo()
	sseRollTable := a.rollTable.GetLo()
	sseSpecial := a.special.GetLo()
	sseShift := a.shift.GetLo()
	sseCombinePairs := decCombinePairs16
	sseCombineQuads := decCombineQuads16
	sseExtract := decExtract16

	// AVX2 constants.
	avxNibbleMask := nibbleMask
	avxNibbleShift := nibbleShift
	avxValidHi := a.validHi
	avxValidLo := a.validLo
	avxRollTable := a.rollTable
	avxSpecial := a.special
	avxShift := a.shift
	avxCombinePairs := combinePairs
	avxCombineQuads := combineQuads
	avxExtractShuffle := extractShuffle
	avxExtractPermute := extractPermute

	// --- SSE preamble: decode src[0:16] → dst[0:12] (store 16, 4 garbage) ---

	encoded := archsimd.LoadUint8x16Slice(src[0:16])
	hiNib := encoded.AsUint32x4().ShiftRight(sseNibbleShift).AsUint8x16().And(sseNibbleMask)
	loNib := encoded.And(sseNibbleMask)
	if !sseValidHi.PermuteOrZero(hiNib.AsInt8x16()).And(
		sseValidLo.PermuteOrZero(loNib.AsInt8x16())).IsZero() {
		return 0, 0
	}
	isSpecial := encoded.Equal(sseSpecial).ToInt8x16().AsUint8x16().And(sseShift)
	roll := sseRollTable.PermuteOrZero(hiNib.Add(isSpecial).AsInt8x16())
	sextets := encoded.Add(roll)
	twelveBit := sextets.DotProductPairsSaturated(sseCombinePairs)
	twentyFourBit := twelveBit.DotProductPairs(sseCombineQuads)
	result := twentyFourBit.AsUint8x16().PermuteOrZero(sseExtract)
	result.StoreSlice(dst[0:16])

	// --- AVX2 middle: loop from si=16 (after SSE preamble) ---
	di, si := 12, 16
	srcEnd, dstEnd := n-32, len(dst)-32
	for si <= srcEnd && di <= dstEnd {
		encoded256 := archsimd.LoadUint8x32Slice(src[si : si+32])
		hiNib256 := encoded256.AsUint32x8().ShiftRight(avxNibbleShift).AsUint8x32().And(avxNibbleMask)
		loNib256 := encoded256.And(avxNibbleMask)
		if !avxValidHi.PermuteOrZeroGrouped(hiNib256.AsInt8x32()).And(
			avxValidLo.PermuteOrZeroGrouped(loNib256.AsInt8x32())).IsZero() {
			break
		}
		isSpecial256 := encoded256.Equal(avxSpecial).ToInt8x32().AsUint8x32().And(avxShift)
		roll256 := avxRollTable.PermuteOrZeroGrouped(hiNib256.Add(isSpecial256).AsInt8x32())
		sextets256 := encoded256.Add(roll256)
		twelveBit256 := sextets256.DotProductPairsSaturated(avxCombinePairs)
		twentyFourBit256 := twelveBit256.DotProductPairs(avxCombineQuads)
		result256 := twentyFourBit256.AsUint8x32().PermuteOrZeroGrouped(avxExtractShuffle).AsUint32x8().Permute(avxExtractPermute).AsUint8x32()
		d := dst[di:]
		result256.StoreSlice(d[:32])
		si += 32
		di += 24
	}

	// --- SSE cleanup: decode remaining bytes after AVX2 loop ---
	// AVX2's 32-byte stores write 8 garbage bytes past the 24 valid bytes,
	// so we can't use a pre-positioned tail like encode does. Instead, loop
	// SSE from where AVX2 left off.
	srcEnd16, dstEnd16 := len(src)-16, len(dst)-16
	for si <= srcEnd16 && di <= dstEnd16 {
		encoded = archsimd.LoadUint8x16Slice(src[si : si+16])
		hiNib = encoded.AsUint32x4().ShiftRight(sseNibbleShift).AsUint8x16().And(sseNibbleMask)
		loNib = encoded.And(sseNibbleMask)
		if !sseValidHi.PermuteOrZero(hiNib.AsInt8x16()).And(
			sseValidLo.PermuteOrZero(loNib.AsInt8x16())).IsZero() {
			break
		}
		isSpecial = encoded.Equal(sseSpecial).ToInt8x16().AsUint8x16().And(sseShift)
		roll = sseRollTable.PermuteOrZero(hiNib.Add(isSpecial).AsInt8x16())
		sextets = encoded.Add(roll)
		twelveBit = sextets.DotProductPairsSaturated(sseCombinePairs)
		twentyFourBit = twelveBit.DotProductPairs(sseCombineQuads)
		result = twentyFourBit.AsUint8x16().PermuteOrZero(sseExtract)
		d := dst[di:]
		result.StoreSlice(d[:16])
		si += 16
		di += 12
	}

	return di, si
}

func decodeVBMI(a *decodeAlpha, dst, src []byte) (int, int) {
	nibbleMask := nibbleMask
	nibbleShift := nibbleShift
	validHi := a.validHi
	validLo := a.validLo
	rollTable := a.rollTable
	special := a.special
	shift := a.shift
	combinePairs := combinePairs
	combineQuads := combineQuads
	extractVBMI := extractVBMI

	di, si := 0, 0
	srcEnd, dstEnd := len(src)-32, len(dst)-32
	for si <= srcEnd && di <= dstEnd {
		encoded := archsimd.LoadUint8x32Slice(src[si : si+32])
		hiNib := encoded.AsUint32x8().ShiftRight(nibbleShift).AsUint8x32().And(nibbleMask)
		loNib := encoded.And(nibbleMask)
		if !validHi.PermuteOrZeroGrouped(hiNib.AsInt8x32()).And(
			validLo.PermuteOrZeroGrouped(loNib.AsInt8x32())).IsZero() {
			break
		}
		isSpecial := encoded.Equal(special).ToInt8x32().AsUint8x32().And(shift)
		roll := rollTable.PermuteOrZeroGrouped(hiNib.Add(isSpecial).AsInt8x32())
		sextets := encoded.Add(roll)
		twelveBit := sextets.DotProductPairsSaturated(combinePairs)
		twentyFourBit := twelveBit.DotProductPairs(combineQuads)
		result := twentyFourBit.AsUint8x32().Permute(extractVBMI)
		d := dst[di:]
		result.StoreSlice(d[:32])
		si += 32
		di += 24
	}
	return di, si
}

// doDecode does the complete decode: SIMD bulk + scalar remainder.
// Returns (decoded bytes, source bytes consumed). If si < len(src),
// an invalid character was encountered and the caller should delegate
// to stdlib for error reporting.
func doDecode(alphabet uint8, dst, src []byte) (int, int) {
	n := len(src)
	a := &decAlphas[alphabet]

	// SIMD bulk.
	di, si := 0, 0
	if n >= 64 {
		if hasAVX512 {
			di, si = decodeVBMI(a, dst, src)
		} else {
			di, si = decodeSSEAVX2(a, dst, src)
		}
	} else if n >= 16 {
		di, si = decodeSSE(a, dst, src, 0, 0)
	}

	// Scalar remainder: full quads only. Partial blocks (2-3 trailing chars)
	// are left for the stdlib fallback in Decode(), which handles newline
	// stripping and error reporting correctly.
	table := &decTables[alphabet]
	for si+3 < n {
		va, vb, vc, vd := table[src[si]], table[src[si+1]], table[src[si+2]], table[src[si+3]]
		if (va|vb|vc|vd)&0x80 != 0 {
			break
		}
		dst[di], dst[di+1], dst[di+2] = va<<2|vb>>4, vb<<4|vc>>2, vc<<6|vd
		si += 4
		di += 3
	}
	return di, si
}
