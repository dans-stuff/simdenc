//go:build goexperiment.simd

package simdenc

import (
	"os"
	"simd/archsimd"
)

// Per-alphabet encode constants.
type encodeAlpha struct {
	sextetToAscii archsimd.Uint8x32 // AVX2: range-based sextet→ASCII offset
	asciiTable512 archsimd.Uint8x64 // AVX-512: 64-entry table in one vector
}

// Per-alphabet decode constants.
type decodeAlpha struct {
	validHi, validLo archsimd.Uint8x32 // Muła/Nojiri validation tables
	rollTable        archsimd.Uint8x32 // ASCII-to-sextet offset by high nibble
	special          archsimd.Uint8x32 // broadcast of special char ('/' or '_')
	shift            archsimd.Uint8x32 // broadcast of special char shift
	// 512-bit VPERMI2B decode LUT (two halves of 128-entry table).
	// Valid base64 chars → 6-bit value; invalid → 0x80.
	lutLo archsimd.Uint8x64 // entries 0-63
	lutHi archsimd.Uint8x64 // entries 64-127
}

// SIMD constants are package-level vars initialized via Load + As casts.
// Load and As have no ISA requirement, so these are safe on any CPU.
//
// Repeating patterns are expressed as named uint64 constants, then loaded
// at the target width (128/256/512-bit). This gives each magic number a
// human-readable name while keeping declarations compact.
//
// IMPORTANT: Constants must be declared at package level, not inside
// function bodies. Package-level vars compile to a single VMOVDQU load
// from the .data section per constant. Inline locals require building
// each value on the stack (8 MOVQ immediates + 8 MOVQ stores + 1
// VMOVDQU load per 512-bit value), causing ~6% regression at 512-bit
// width. Moving the .As*() cast into the function (while keeping the
// Load at package level) is also slower (~5%). See Experiment 44.

// Shared uint64 patterns (little-endian byte order) reused across widths.
const (
	// Sextet extraction masks/multipliers for the Muła 3→4 byte expansion.
	// Each 32-bit word isolates high or low bits of adjacent bytes, then a
	// multiply-shift moves them into the correct 6-bit sextet positions.
	maskHi  = uint64(0x0FC0FC000FC0FC00) // AND mask: keep bits [11:6] of bytes 1,2
	shiftHi = uint64(0x0400004004000040) // mulhi16 shift: move masked bits into place
	maskLo  = uint64(0x003F03F0003F03F0) // AND mask: keep bits [5:0] of bytes 0,1
	shiftLo = uint64(0x0100001001000010) // mullo16 shift: move masked bits into place

	// Decode recombination: reverse the sextet split, packing four 6-bit
	// values back into three 8-bit bytes via multiply-add (PMADDUBSW/PMADDWD).
	combPairs = uint64(0x0140014001400140) // PMADDUBSW: adjacent sextets → 12-bit pairs
	combQuads = uint64(0x0001100000011000) // PMADDWD: adjacent pairs → 24-bit triplets

	// Decode nibble validation: high/low nibble masks for the Muła error check.
	nibble   = uint64(0x0F0F0F0F0F0F0F0F) // AND mask to isolate low nibble
	nibShift = uint64(0x0000000400000004) // right-shift by 4 to get high nibble

	// Decode extract shuffle: after recombination each 32-bit word holds 3
	// result bytes in positions [2:0] with byte [3] as garbage. This shuffle
	// packs the 3 good bytes from each of 4 words into 12 contiguous bytes,
	// with the remaining 4 positions zeroed (0x80).
	extractLo = uint64(0x090A040506000102) // bytes 0-7: pick good bytes from words 0-2
	extractHi = uint64(0x808080800C0D0E08) // bytes 8-15: pick good bytes from word 3 + zero-fill
)

// Broadcast fills: single byte repeated across all 8 positions of a uint64.
const (
	fill2F = uint64(0x2F2F2F2F2F2F2F2F) // '/' (standard special char)
	fill5F = uint64(0x5F5F5F5F5F5F5F5F) // '_' (URL special char)
	fillFF = uint64(0xFFFFFFFFFFFFFFFF) // standard special shift
	fill03 = uint64(0x0303030303030303) // URL special shift
	fill33 = uint64(0x3333333333333333) // lower sextet mask (0x3F >> 2)
	fill19 = uint64(0x1919191919191919) // upper sextet bound (25)
	fill80 = uint64(0x8080808080808080) // high-bit mask
)

// Per-alphabet encode constants. sextetToAscii LUTs are 16-byte range-based
// offset tables tiled across both 128-bit lanes of a 256-bit vector.
var encAlphas = [2]encodeAlpha{
	{sextetToAscii: archsimd.LoadUint64x4(&[4]uint64{0xFCFCFCFCFCFC4741, 0x0000F0EDFCFCFCFC, 0xFCFCFCFCFCFC4741, 0x0000F0EDFCFCFCFC}).AsUint8x32()},
	{sextetToAscii: archsimd.LoadUint64x4(&[4]uint64{0xFCFCFCFCFCFC4741, 0x000020EFFCFCFCFC, 0xFCFCFCFCFCFC4741, 0x000020EFFCFCFCFC}).AsUint8x32()},
}

// Per-alphabet SSE encode constants (128-bit sextetToAscii LUTs).
var encAlphasSSE = [2]archsimd.Uint8x16{
	archsimd.LoadUint64x2(&[2]uint64{0xFCFCFCFCFCFC4741, 0x0000F0EDFCFCFCFC}).AsUint8x16(),
	archsimd.LoadUint64x2(&[2]uint64{0xFCFCFCFCFCFC4741, 0x000020EFFCFCFCFC}).AsUint8x16(),
}

// Per-alphabet decode constants. Validation and roll tables are 16-byte
// Muła/Nojiri LUTs tiled to both lanes of 256-bit vectors.
var decAlphas = [2]decodeAlpha{
	{ // Standard: special='/', shift=0xFF
		special:   archsimd.LoadUint64x4(&[4]uint64{fill2F, fill2F, fill2F, fill2F}).AsUint8x32(),
		shift:     archsimd.LoadUint64x4(&[4]uint64{fillFF, fillFF, fillFF, fillFF}).AsUint8x32(),
		validHi:   archsimd.LoadUint64x4(&[4]uint64{0x0804080402011010, 0x1010101010101010, 0x0804080402011010, 0x1010101010101010}).AsUint8x32(),
		validLo:   archsimd.LoadUint64x4(&[4]uint64{0x1111111111111115, 0x1A1B1B1B1A131111, 0x1111111111111115, 0x1A1B1B1B1A131111}).AsUint8x32(),
		rollTable: archsimd.LoadUint64x4(&[4]uint64{0xB9B9BFBF04131000, 0x0000000000000000, 0xB9B9BFBF04131000, 0x0000000000000000}).AsUint8x32(),
	},
	{ // URL: special='_', shift=0x03
		special:   archsimd.LoadUint64x4(&[4]uint64{fill5F, fill5F, fill5F, fill5F}).AsUint8x32(),
		shift:     archsimd.LoadUint64x4(&[4]uint64{fill03, fill03, fill03, fill03}).AsUint8x32(),
		validHi:   archsimd.LoadUint64x4(&[4]uint64{0x2008100804020101, 0x0101010101010101, 0x2008100804020101, 0x0101010101010101}).AsUint8x32(),
		validLo:   archsimd.LoadUint64x4(&[4]uint64{0x030303030303030B, 0x2737353737070303, 0x030303030303030B, 0x2737353737070303}).AsUint8x32(),
		rollTable: archsimd.LoadUint64x4(&[4]uint64{0xB9B9BFBF04110000, 0x00000000000000E0, 0xB9B9BFBF04110000, 0x00000000000000E0}).AsUint8x32(),
	},
}

var hasAVX2 = archsimd.X86.AVX2() && os.Getenv("SIMDENC_NO_AVX2") == ""
var hasAVX512 = hasAVX2 && archsimd.X86.AVX512() && archsimd.X86.AVX512VBMI() && os.Getenv("SIMDENC_NO_AVX512") == ""

// init sets the SIMD dispatch functions and builds per-alphabet AVX-512 LUTs.
func init() {
	if !hasAVX2 {
		return
	}
	simdEncode = doEncode
	simdDecode = doDecode

	if !hasAVX512 {
		return
	}

	// Per-alphabet AVX-512 tables (require loops to construct).
	for idx, alpha := range encAlphabets {
		// Encode: 64-entry ASCII offset table.
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
		t512[62], t512[63] = byte(alpha[62])-62, byte(alpha[63])-63
		encAlphas[idx].asciiTable512 = archsimd.LoadUint8x64(&t512)

		// Decode: VPERMI2B LUT (128-entry table split across two Uint8x64).
		var lo, hi [64]byte
		for i := range 64 {
			lo[i] = 0x80
			hi[i] = 0x80
		}
		for i, c := range alpha {
			if c < 64 {
				lo[c] = byte(i)
			} else {
				hi[c-64] = byte(i)
			}
		}
		decAlphas[idx].lutLo = archsimd.LoadUint8x64(&lo)
		decAlphas[idx].lutHi = archsimd.LoadUint8x64(&hi)
	}
}

// --- Encode ---
//
// Pipeline: encode512 or encodeAVX2 → encodeSSE → scalar tail.
// Each function starts at position 0 in its slices and returns source bytes
// consumed. The caller derives di from the fixed 3:4 ratio (di = si*4/3).

// encSSEShuffle rearranges 12 input bytes into position for 3→4 expansion.
// Each group of 3 source bytes (a triplet) maps to 4 output positions:
// src[1,0,2,1, 4,3,5,4, 7,6,8,7, 10,9,11,10] — each triplet's bytes are
// duplicated so the mask/shift steps can extract all four 6-bit sextets.
var encSSEShuffle = archsimd.LoadUint64x2(&[2]uint64{0x0405030401020001, 0x0A0B090A07080607}).AsInt8x16()
var encSSELowerSextet = archsimd.LoadUint64x2(&[2]uint64{fill33, fill33}).AsUint8x16()
var encSSEUpperSextet = archsimd.LoadUint64x2(&[2]uint64{fill19, fill19}).AsInt8x16()
var encSSEMaskHi = archsimd.LoadUint64x2(&[2]uint64{maskHi, maskHi}).AsUint16x8()
var encSSEShiftHi = archsimd.LoadUint64x2(&[2]uint64{shiftHi, shiftHi}).AsUint16x8()
var encSSEMaskLo = archsimd.LoadUint64x2(&[2]uint64{maskLo, maskLo}).AsUint16x8()
var encSSEShiftLo = archsimd.LoadUint64x2(&[2]uint64{shiftLo, shiftLo}).AsUint16x8()

// encodeSSE runs the 128-bit SSE encode loop (12 src → 16 dst per iteration).
func encodeSSE(alphabet uint8, dst, src []byte) int {
	shuffle := encSSEShuffle
	maskHi := encSSEMaskHi
	shiftHi := encSSEShiftHi
	maskLo := encSSEMaskLo
	shiftLo := encSSEShiftLo
	lowerSextet := encSSELowerSextet
	upperSextet := encSSEUpperSextet
	sextetToAscii := encAlphasSSE[alphabet]

	si, di := 0, 0
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
		result.StoreSlice(dst[di : di+16])
		si += 12
		di += 16
	}
	return si
}

// encAVX2Shuffle is the AVX2 version of encSSEShuffle. Because AVX2 loads
// 32 bytes starting from src[si-4], the byte indices are offset: lane 0
// (bytes 4-15 of the load) uses indices starting at 4, lane 1 (bytes 0-11
// of the second group) uses the same pattern as SSE.
var encAVX2Shuffle = archsimd.LoadUint64x4(&[4]uint64{
	0x0809070805060405, 0x0E0F0D0E0B0C0A0B,
	0x0405030401020001, 0x0A0B090A07080607,
}).AsInt8x32()
var encSextetMaskHi = archsimd.LoadUint64x4(&[4]uint64{maskHi, maskHi, maskHi, maskHi}).AsUint16x16()
var encSextetShiftHi = archsimd.LoadUint64x4(&[4]uint64{shiftHi, shiftHi, shiftHi, shiftHi}).AsUint16x16()
var encSextetMaskLo = archsimd.LoadUint64x4(&[4]uint64{maskLo, maskLo, maskLo, maskLo}).AsUint16x16()
var encSextetShiftLo = archsimd.LoadUint64x4(&[4]uint64{shiftLo, shiftLo, shiftLo, shiftLo}).AsUint16x16()
var encAVX2LowerSextet = archsimd.LoadUint64x4(&[4]uint64{fill33, fill33, fill33, fill33}).AsUint8x32()
var encAVX2UpperSextet = archsimd.LoadUint64x4(&[4]uint64{fill19, fill19, fill19, fill19}).AsInt8x32()

// encodeAVX2 runs the 256-bit AVX2 encode loop (24 src → 32 dst per iteration).
// Starts with a fixed scalar preamble of 2 triplets (6 src → 8 dst) to satisfy
// the -4 offset trick (the AVX2 load reads src[si-4:si+28]).
func encodeAVX2(alphabet uint8, a *encodeAlpha, dst, src []byte) int {
	// Fixed scalar preamble: 2 triplets → si=6, di=8.
	alpha := encAlphabets[alphabet]
	v0 := uint(src[0])<<16 | uint(src[1])<<8 | uint(src[2])
	dst[0], dst[1], dst[2], dst[3] = alpha[v0>>18&0x3F], alpha[v0>>12&0x3F], alpha[v0>>6&0x3F], alpha[v0&0x3F]
	v1 := uint(src[3])<<16 | uint(src[4])<<8 | uint(src[5])
	dst[4], dst[5], dst[6], dst[7] = alpha[v1>>18&0x3F], alpha[v1>>12&0x3F], alpha[v1>>6&0x3F], alpha[v1&0x3F]

	offsetShuffle := encAVX2Shuffle
	sextetMaskHi := encSextetMaskHi
	sextetShiftHi := encSextetShiftHi
	sextetMaskLo := encSextetMaskLo
	sextetShiftLo := encSextetShiftLo
	lastLowerSextet := encAVX2LowerSextet
	lastUpperSextet := encAVX2UpperSextet
	sextetToAscii := a.sextetToAscii

	base, di := 2, 8 // base = si(6) - 4 = 2
	srcEnd, dstEnd := len(src)-32, len(dst)-32
	for base <= srcEnd && di <= dstEnd {
		grouped := archsimd.LoadUint8x32Slice(src[base : base+32]).PermuteOrZeroGrouped(offsetShuffle)
		w := grouped.AsUint16x16()
		sextets := w.And(sextetMaskHi).MulHigh(sextetShiftHi).Or(w.And(sextetMaskLo).Mul(sextetShiftLo)).AsUint8x32()
		saturated := sextets.SubSaturated(lastLowerSextet)
		pastUpper := sextets.AsInt8x32().Greater(lastUpperSextet).ToInt8x32().AsUint8x32()
		rangeIdx := saturated.Sub(pastUpper)
		asciiOffset := sextetToAscii.PermuteOrZeroGrouped(rangeIdx.AsInt8x32())
		result := sextets.Add(asciiOffset)
		result.StoreSlice(dst[di : di+32])
		base += 24
		di += 32
	}
	return base + 4
}

// encShuffle512 is the 512-bit version of encSSEShuffle. It processes 48 src
// bytes into 64 output bytes. Each 128-bit lane uses the same 3→4 pattern
// but with byte indices offset by 12 per lane (12 src bytes per lane).
var encShuffle512 = archsimd.LoadUint64x8(&[8]uint64{
	0x0405030401020001, 0x0A0B090A07080607,
	0x10110F100D0E0C0D, 0x1617151613141213,
	0x1C1D1B1C191A1819, 0x222321221F201E1F,
	0x2829272825262425, 0x2E2F2D2E2B2C2A2B,
}).AsUint8x64()
var encMaskHi512 = archsimd.LoadUint64x8(&[8]uint64{maskHi, maskHi, maskHi, maskHi, maskHi, maskHi, maskHi, maskHi}).AsUint16x32()
var encShiftHi512 = archsimd.LoadUint64x8(&[8]uint64{shiftHi, shiftHi, shiftHi, shiftHi, shiftHi, shiftHi, shiftHi, shiftHi}).AsUint16x32()
var encMaskLo512 = archsimd.LoadUint64x8(&[8]uint64{maskLo, maskLo, maskLo, maskLo, maskLo, maskLo, maskLo, maskLo}).AsUint16x32()
var encShiftLo512 = archsimd.LoadUint64x8(&[8]uint64{shiftLo, shiftLo, shiftLo, shiftLo, shiftLo, shiftLo, shiftLo, shiftLo}).AsUint16x32()

// encode512 runs the 512-bit encode loop (48 src → 64 dst per iteration).
func encode512(a *encodeAlpha, dst, src []byte) int {
	shuffle512 := encShuffle512
	maskHi512 := encMaskHi512
	shiftHi512 := encShiftHi512
	maskLo512 := encMaskLo512
	shiftLo512 := encShiftLo512
	asciiTable512 := a.asciiTable512

	si, di := 0, 0
	srcEnd, dstEnd := len(src)-64, len(dst)-64
	for si <= srcEnd && di <= dstEnd {
		grouped := archsimd.LoadUint8x64Slice(src[si : si+64]).Permute(shuffle512)
		w := grouped.AsUint16x32()
		sextets := w.And(maskHi512).MulHigh(shiftHi512).Or(w.And(maskLo512).Mul(shiftLo512)).AsUint8x64()
		result := sextets.Add(asciiTable512.Permute(sextets))
		result.StoreSlice(dst[di : di+64])
		si += 48
		di += 64
	}
	return si
}

// doEncode runs the SIMD waterfall and returns source bytes consumed.
// Caller guarantees len(src) >= 16 and handles any remainder via stdlib.
func doEncode(alphabet uint8, dst, src []byte) int {
	n := len(src)
	si := 0

	if hasAVX512 && n >= 64 {
		si = encode512(&encAlphas[alphabet], dst, src)
	} else if n >= 36 {
		si = encodeAVX2(alphabet, &encAlphas[alphabet], dst, src)
	}
	di := si * 4 / 3
	if si+16 <= n {
		si += encodeSSE(alphabet, dst[di:], src[si:])
	}

	return si
}

// --- Decode ---
//
// Pipeline: decode512 or decodeAVX2 → decodeSSE → scalar tail.
// Same pattern as encode: each function starts at 0 and returns source bytes
// consumed. The caller derives di from the fixed 4:3 ratio (di = si * 3 / 4).

var decNibbleMask16 = archsimd.LoadUint64x2(&[2]uint64{nibble, nibble}).AsUint8x16()
var decNibbleShift16 = archsimd.LoadUint64x2(&[2]uint64{nibShift, nibShift}).AsUint32x4()
var decCombinePairs16 = archsimd.LoadUint64x2(&[2]uint64{combPairs, combPairs}).AsInt8x16()
var decCombineQuads16 = archsimd.LoadUint64x2(&[2]uint64{combQuads, combQuads}).AsInt16x8()
var decExtract16 = archsimd.LoadUint64x2(&[2]uint64{extractLo, extractHi}).AsInt8x16()

// decodeSSE runs the 128-bit SSE decode loop. Processes 16 src → 12 dst
// bytes per iteration. Returns source bytes consumed.
func decodeSSE(a *decodeAlpha, dst, src []byte) int {
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

	si, di := 0, 0
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
		result.StoreSlice(dst[di : di+16])
		si += 16
		di += 12
	}
	return si
}

var decNibbleMask32 = archsimd.LoadUint64x4(&[4]uint64{nibble, nibble, nibble, nibble}).AsUint8x32()
var decNibbleShift32 = archsimd.LoadUint64x4(&[4]uint64{nibShift, nibShift, nibShift, nibShift}).AsUint32x8()
var decCombinePairs32 = archsimd.LoadUint64x4(&[4]uint64{combPairs, combPairs, combPairs, combPairs}).AsInt8x32()
var decCombineQuads32 = archsimd.LoadUint64x4(&[4]uint64{combQuads, combQuads, combQuads, combQuads}).AsInt16x16()
var decExtractShuffle = archsimd.LoadUint64x4(&[4]uint64{extractLo, extractHi, extractLo, extractHi}).AsInt8x32()

// decExtractPermute compacts the output of the extract shuffle. After VPSHUFB,
// each 128-bit lane has 12 good bytes in positions [11:0] and 4 garbage bytes
// in [15:12]. This VPERMD moves the 6 good dwords (3 per lane) into the low
// 24 bytes: dwords {0,1,2, 4,5,6} → positions {0,1,2,3,4,5}, with dword 7
// duplicated in positions {6,7} (overwritten by the next iteration).
var decExtractPermute = archsimd.LoadUint64x4(&[4]uint64{
	0x0000000100000000, 0x0000000400000002,
	0x0000000600000005, 0x0000000700000007,
}).AsUint32x8()

// decodeAVX2 runs the 256-bit AVX2 decode loop. Processes 32 src → 24 dst
// bytes per iteration. Returns source bytes consumed.
func decodeAVX2(a *decodeAlpha, dst, src []byte) int {
	avxNibbleMask := decNibbleMask32
	avxNibbleShift := decNibbleShift32
	avxValidHi := a.validHi
	avxValidLo := a.validLo
	avxRollTable := a.rollTable
	avxSpecial := a.special
	avxShift := a.shift
	avxCombinePairs := decCombinePairs32
	avxCombineQuads := decCombineQuads32
	avxExtractShuffle := decExtractShuffle
	avxExtractPermute := decExtractPermute

	si, di := 0, 0
	srcEnd, dstEnd := len(src)-32, len(dst)-32
	for si <= srcEnd && di <= dstEnd {
		encoded := archsimd.LoadUint8x32Slice(src[si : si+32])
		hiNib := encoded.AsUint32x8().ShiftRight(avxNibbleShift).AsUint8x32().And(avxNibbleMask)
		loNib := encoded.And(avxNibbleMask)
		if !avxValidHi.PermuteOrZeroGrouped(hiNib.AsInt8x32()).And(
			avxValidLo.PermuteOrZeroGrouped(loNib.AsInt8x32())).IsZero() {
			break
		}
		isSpecial := encoded.Equal(avxSpecial).ToInt8x32().AsUint8x32().And(avxShift)
		roll := avxRollTable.PermuteOrZeroGrouped(hiNib.Add(isSpecial).AsInt8x32())
		sextets := encoded.Add(roll)
		twelveBit := sextets.DotProductPairsSaturated(avxCombinePairs)
		twentyFourBit := twelveBit.DotProductPairs(avxCombineQuads)
		result := twentyFourBit.AsUint8x32().PermuteOrZeroGrouped(avxExtractShuffle).AsUint32x8().Permute(avxExtractPermute).AsUint8x32()
		result.StoreSlice(dst[di : di+32])
		si += 32
		di += 24
	}
	return si
}

var decHighBitMask512 = archsimd.LoadUint64x8(&[8]uint64{fill80, fill80, fill80, fill80, fill80, fill80, fill80, fill80}).AsUint8x64()
var decCombinePairs512 = archsimd.LoadUint64x8(&[8]uint64{combPairs, combPairs, combPairs, combPairs, combPairs, combPairs, combPairs, combPairs}).AsInt8x64()
var decCombineQuads512 = archsimd.LoadUint64x8(&[8]uint64{combQuads, combQuads, combQuads, combQuads, combQuads, combQuads, combQuads, combQuads}).AsInt16x32()

// decExtract512 is the 512-bit version of the extract shuffle. It packs the
// 3 good bytes from each 32-bit word across all 4 lanes (48 bytes total from
// 64) into contiguous positions. The last 16 bytes are zeroed (unused).
var decExtract512 = archsimd.LoadUint64x8(&[8]uint64{
	0x090A040506000102, 0x161011120C0D0E08,
	0x1C1D1E18191A1415, 0x292A242526202122,
	0x363031322C2D2E28, 0x3C3D3E38393A3435,
	0, 0,
}).AsUint8x64()

// decode512 runs the 512-bit VPERMI2B decode loop. Processes 64 src → 48 dst
// bytes per iteration. Uses ConcatPermute (VPERMI2B) for combined validation +
// translation in one instruction. Returns source bytes consumed.
func decode512(a *decodeAlpha, dst, src []byte) int {
	lutLo := a.lutLo
	lutHi := a.lutHi
	highBitMask := decHighBitMask512
	combinePairs := decCombinePairs512
	combineQuads := decCombineQuads512
	extract := decExtract512
	var zero archsimd.Uint8x64

	si, di := 0, 0
	srcEnd, dstEnd := len(src)-64, len(dst)-64
	for si <= srcEnd && di <= dstEnd {
		encoded := archsimd.LoadUint8x64Slice(src[si : si+64])

		// VPERMI2B: translate ASCII → 6-bit values. Invalid chars → 0x80.
		sextets := lutLo.ConcatPermute(lutHi, encoded)

		// Error detection: any non-ASCII input byte has bit 7 set (caught by OR),
		// any invalid ASCII char produced 0x80 from the LUT (also caught by OR).
		errors := encoded.Or(sextets).And(highBitMask)
		if errors.Equal(zero).ToBits() != ^uint64(0) {
			break
		}

		// Pack 6-bit values into bytes: VPMADDUBSW + VPMADDWD.
		twelveBit := sextets.DotProductPairsSaturated(combinePairs)
		twentyFourBit := twelveBit.DotProductPairs(combineQuads)

		// Compact: extract 48 useful bytes from 64 (skip every 4th byte).
		result := twentyFourBit.AsUint8x64().Permute(extract)
		result.StoreSlice(dst[di : di+64])
		si += 64
		di += 48
	}
	return si
}

// doDecode runs the SIMD waterfall and returns (decoded bytes, source bytes
// consumed). Caller guarantees len(src) >= 16 and handles any remainder via stdlib.
func doDecode(alphabet uint8, dst, src []byte) (int, int) {
	n := len(src)
	a := &decAlphas[alphabet]
	si := 0

	if hasAVX512 && n >= 64 {
		si = decode512(a, dst, src)
	} else if n >= 64 {
		si = decodeAVX2(a, dst, src)
	}
	di := si * 3 / 4
	if si+16 <= n {
		si += decodeSSE(a, dst[di:], src[si:])
	}

	return si * 3 / 4, si
}
