//go:build goexperiment.simd

package simdenc

import "simd/archsimd"

// --- Helpers ---

// dup duplicates a 16-byte pattern into both lanes of a 32-byte array.
func dup(b [16]byte) [32]byte {
	var r [32]byte
	copy(r[:16], b[:])
	copy(r[16:], b[:])
	return r
}

// alternateU16x16 fills a 256-bit vector with alternating uint16 values.
func alternateU16x16(even, odd uint16) archsimd.Uint16x16 {
	var a [16]uint16
	for i := 0; i < 16; i += 2 {
		a[i] = even
		a[i+1] = odd
	}
	return archsimd.LoadUint16x16(&a)
}

// alternateU16x32 fills a 512-bit vector with alternating uint16 values.
func alternateU16x32(even, odd uint16) archsimd.Uint16x32 {
	var a [32]uint16
	for i := 0; i < 32; i += 2 {
		a[i] = even
		a[i+1] = odd
	}
	return archsimd.LoadUint16x32(&a)
}

// --- Init ---

// Tier 1: AVX2 (works under Rosetta emulation).
// Tier 2: AVX-512 + VBMI (server-grade: EPYC Zen 4+, Ice Lake+).
func init() {
	if !archsimd.X86.AVX2() {
		return
	}
	initShared()
	initAVX2()
	bulkEncode = encodeAVX2
	bulkDecode = decodeAVX2
	StdEncoding.useSIMD = true
	RawStdEncoding.useSIMD = true

	if !archsimd.X86.AVX512() || !archsimd.X86.AVX512VBMI() {
		return
	}
	initVBMI()
	init512()
	bulkEncode = func(dst, src []byte) (di, si int) {
		di, si = encode512(dst, src)
		di2, si2 := encodeVBMI(dst[di:], src[si:])
		return di + di2, si + si2
	}
	bulkDecode = decodeVBMI
}

// ========================================================================
// Shared constants (used by both AVX2 and VBMI tiers)
// ========================================================================

var (
	// Encode: sextet extraction.
	// Base64 splits every 3 bytes into four 6-bit values ("sextets").
	// We do this by treating the bytes as 16-bit words, masking out the bits
	// we want, and multiplying to shift them into the right position.
	// "Hi" extracts sextets from the high bits of each word pair,
	// "Lo" extracts sextets from the low bits.
	encSextetMaskHi  archsimd.Uint16x16
	encSextetShiftHi archsimd.Uint16x16
	encSextetMaskLo  archsimd.Uint16x16
	encSextetShiftLo archsimd.Uint16x16

	// Decode: character validation.
	// We check if every input byte is a valid base64 character by splitting
	// each byte into its high and low 4-bit nibbles, looking up both nibbles
	// in separate tables, and ANDing the results. If the AND is non-zero,
	// the character is invalid. This is the Muła/Nojiri validation algorithm.
	nibbleMask      archsimd.Uint8x32 // 0x0F: mask to extract a 4-bit nibble
	decValidationHi archsimd.Uint8x32 // validation flags indexed by high nibble
	decValidationLo archsimd.Uint8x32 // validation flags indexed by low nibble

	// Decode: ASCII-to-sextet translation.
	// Each valid base64 character needs an offset subtracted to get back to
	// its 6-bit value. For example, 'A' (65) maps to 0, so we add -65.
	// The offset depends on which character range the byte falls in.
	// The slash '/' is a special case that needs its own detection.
	decAsciiToSextetTable archsimd.Uint8x32
	decIsSlashChar        archsimd.Uint8x32 // broadcast 0x2F ('/')

	// Decode: combining sextets back into bytes.
	// Four 6-bit sextets [a,b,c,d] become three output bytes. We do this in
	// two steps using multiply-and-add:
	//   Step 1 (decCombinePairs):  [a,b] → a*64 + b  (two sextets → 12 bits)
	//   Step 2 (decCombineQuads):  [ab,cd] → ab*4096 + cd  (four sextets → 24 bits)
	decNibbleShift  archsimd.Uint32x8  // shift right by 4 to extract high nibble
	decCombinePairs archsimd.Int8x32   // [64, 1, 64, 1, ...] weights for pair combining
	decCombineQuads archsimd.Int16x16  // [4096, 1, 4096, 1, ...] weights for quad combining
)

func initShared() {
	encSextetMaskHi = alternateU16x16(0xFC00, 0x0FC0)
	encSextetShiftHi = alternateU16x16(0x0040, 0x0400)
	encSextetMaskLo = alternateU16x16(0x03F0, 0x003F)
	encSextetShiftLo = alternateU16x16(0x0010, 0x0100)

	nibbleMask = archsimd.BroadcastUint8x32(0x0F)
	decNibbleShift = archsimd.BroadcastUint32x8(4)
	decIsSlashChar = archsimd.BroadcastUint8x32(0x2F) // '/'

	hi := dup([16]byte{
		0x10, 0x10, 0x01, 0x02, 0x04, 0x08, 0x04, 0x08,
		0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
	})
	decValidationHi = archsimd.LoadUint8x32(&hi)
	lo := dup([16]byte{
		0x15, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
		0x11, 0x11, 0x13, 0x1A, 0x1B, 0x1B, 0x1B, 0x1A,
	})
	decValidationLo = archsimd.LoadUint8x32(&lo)
	roll := dup([16]byte{
		0, 16, 19, 4, 191, 191, 185, 185, // '/', '+', '0'-'9', 'A'-'Z', 'a'-'z'
		0, 0, 0, 0, 0, 0, 0, 0,
	})
	decAsciiToSextetTable = archsimd.LoadUint8x32(&roll)

	mu := dup([16]byte{0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01,
		0x40, 0x01, 0x40, 0x01, 0x40, 0x01, 0x40, 0x01})
	decCombinePairs = archsimd.LoadUint8x32(&mu).AsInt8x32()
	decCombineQuads = archsimd.LoadInt16x16(&[16]int16{
		0x1000, 1, 0x1000, 1, 0x1000, 1, 0x1000, 1,
		0x1000, 1, 0x1000, 1, 0x1000, 1, 0x1000, 1,
	})
}

// ========================================================================
// Tier 1: AVX2
// ========================================================================

var (
	// Encode: rearranges raw input bytes into 3-byte encoding groups.
	// AVX2 can only shuffle within each 16-byte lane, so we use the -4 offset
	// load trick: load starting 4 bytes early, then the shuffle indices in
	// lane 0 are offset by +4 to compensate.
	encOffsetByteGrouping archsimd.Int8x32

	// Encode: sextet-to-ASCII mapping.
	// Each 6-bit sextet (0-63) maps to an ASCII character. We figure out which
	// character range each sextet belongs to (A-Z, a-z, 0-9, +, /), then look
	// up the offset to add. encLastLowerSextet (51) and encLastUpperSextet (25)
	// are the boundary values used to determine the range.
	encLastLowerSextet archsimd.Uint8x32 // broadcast 51: sextet index of 'z'
	encLastUpperSextet archsimd.Int8x32  // broadcast 25: sextet index of 'Z'
	encSextetToAscii   archsimd.Uint8x32 // per-range offset to convert sextet → ASCII char

	// Decode: extracts 3 decoded bytes from every 4-byte group, within each lane.
	decExtractWithinLanes archsimd.Int8x32
	// Decode: packs the extracted bytes from both lanes into contiguous output.
	decPackAcrossLanes archsimd.Uint32x8
)

func initAVX2() {
	encOffsetByteGrouping = archsimd.LoadUint8x32(&[32]byte{
		5, 4, 6, 5, 8, 7, 9, 8, 11, 10, 12, 11, 14, 13, 15, 14, // lane 0 (+4 offset)
		1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10, // lane 1
	}).AsInt8x32()

	encLastLowerSextet = archsimd.BroadcastUint8x32(51)
	encLastUpperSextet = archsimd.BroadcastInt8x32(25)
	el := dup([16]byte{
		65,                                               // A-Z: add 65 ('A')
		71,                                               // a-z: add 71 ('a' - 26)
		252, 252, 252, 252, 252, 252, 252, 252, 252, 252, // 0-9: add 252 ('0' - 52, wraps unsigned)
		237, // +
		240, // /
		0, 0,
	})
	encSextetToAscii = archsimd.LoadUint8x32(&el)

	cs := dup([16]byte{2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, 0x80, 0x80, 0x80, 0x80})
	decExtractWithinLanes = archsimd.LoadUint8x32(&cs).AsInt8x32()
	decPackAcrossLanes = archsimd.LoadUint32x8(&[8]uint32{0, 1, 2, 4, 5, 6, 7, 7})
}

// encodeAVX2 encodes 24 source bytes into 32 base64 characters per iteration.
// Uses the -4 offset load trick because AVX2 can only shuffle within 16-byte
// lanes, so we load 4 bytes early and adjust the shuffle indices to compensate.
func encodeAVX2(dst, src []byte) (di, si int) {
	srcEnd := len(src) - 32
	dstEnd := len(dst) - 32
	if 6 > srcEnd || 8 > dstEnd {
		return 0, 0
	}

	// Scalar preamble: encode 2 triplets so si >= 4 for the offset load.
	for i := 0; i < 2; i++ {
		v := uint(src[3*i])<<16 | uint(src[3*i+1])<<8 | uint(src[3*i+2])
		dst[4*i+0] = encodeStd[v>>18&0x3F]
		dst[4*i+1] = encodeStd[v>>12&0x3F]
		dst[4*i+2] = encodeStd[v>>6&0x3F]
		dst[4*i+3] = encodeStd[v&0x3F]
	}
	si = 6
	di = 8

	// Local copies keep constants in registers (+37% without — see RESEARCH.md).
	byteGrouping := encOffsetByteGrouping
	sextetMaskHi := encSextetMaskHi
	sextetShiftHi := encSextetShiftHi
	sextetMaskLo := encSextetMaskLo
	sextetShiftLo := encSextetShiftLo
	lastLower := encLastLowerSextet
	lastUpper := encLastUpperSextet
	sextetToAscii := encSextetToAscii

	for si <= srcEnd && di <= dstEnd {
		// Load 32 bytes (starting 4 early for the offset trick).
		inputBytes := archsimd.LoadUint8x32Slice(src[si-4 : si+28])

		// Rearrange bytes into 3-byte encoding groups.
		groupedBytes := inputBytes.PermuteOrZeroGrouped(byteGrouping)

		// Extract four 6-bit sextets from each 3-byte group.
		words := groupedBytes.AsUint16x16()
		hiSextets := words.And(sextetMaskHi).MulHigh(sextetShiftHi)
		loSextets := words.And(sextetMaskLo).Mul(sextetShiftLo)
		sextets := hiSextets.Or(loSextets).AsUint8x32()

		// Map each sextet (0-63) to its ASCII character (A-Z, a-z, 0-9, +, /).
		// Figure out which range each sextet belongs to, then add the right offset.
		saturatedIndex := sextets.SubSaturated(lastLower)
		pastUppercase := sextets.AsInt8x32().Greater(lastUpper)
		rangeIndex := saturatedIndex.Sub(pastUppercase.ToInt8x32().AsUint8x32())
		asciiOffset := sextetToAscii.PermuteOrZeroGrouped(rangeIndex.AsInt8x32())
		asciiOutput := sextets.Add(asciiOffset)

		d := dst[di:]
		asciiOutput.StoreSlice(d[:32])
		si += 24
		di += 32
	}
	return di, si
}

// decodeAVX2 decodes 32 base64 characters into 24 bytes per iteration.
// Stops at the first invalid character and returns how far it got.
func decodeAVX2(dst, src []byte) (di, si int) {
	// Local copies keep constants in registers (+32% without — see RESEARCH.md).
	nibMask := nibbleMask
	validHi := decValidationHi
	validLo := decValidationLo
	asciiToSextetTable := decAsciiToSextetTable
	slashChar := decIsSlashChar
	nibShift := decNibbleShift
	combinePairs := decCombinePairs
	combineQuads := decCombineQuads
	extractBytes := decExtractWithinLanes
	packAcrossLanes := decPackAcrossLanes

	srcEnd := len(src) - 32
	dstEnd := len(dst) - 32
	for si <= srcEnd && di <= dstEnd {
		encoded := archsimd.LoadUint8x32Slice(src[si : si+32])

		// Validate: split each byte into nibbles, look up both in validation
		// tables, AND the results. Non-zero means an invalid character.
		hiNibble := encoded.AsUint32x8().ShiftRight(nibShift).AsUint8x32().And(nibMask)
		loNibble := encoded.And(nibMask)
		if !validHi.PermuteOrZeroGrouped(hiNibble.AsInt8x32()).And(
			validLo.PermuteOrZeroGrouped(loNibble.AsInt8x32())).IsZero() {
			break
		}

		// Translate ASCII characters back to 6-bit sextet values.
		isSlash := encoded.Equal(slashChar).ToInt8x32().AsUint8x32()
		asciiToSextet := asciiToSextetTable.PermuteOrZeroGrouped(hiNibble.Add(isSlash).AsInt8x32())
		sextets := encoded.Add(asciiToSextet)

		// Combine four 6-bit sextets into three bytes:
		//   [a,b] → a*64+b = 12 bits,  then [ab,cd] → ab*4096+cd = 24 bits
		twelveBitValues := sextets.DotProductPairsSaturated(combinePairs)
		twentyFourBitValues := twelveBitValues.DotProductPairs(combineQuads)

		// Extract the 3 decoded bytes from each 4-byte group and pack contiguously.
		decodedBytes := twentyFourBitValues.AsUint8x32().PermuteOrZeroGrouped(extractBytes).AsUint32x8().Permute(packAcrossLanes).AsUint8x32()

		d := dst[di:]
		decodedBytes.StoreSlice(d[:32])
		si += 32
		di += 24
	}
	return di, si
}

// ========================================================================
// Tier 2: VBMI (256-bit with cross-lane byte permutes)
// ========================================================================

var (
	// Encode: rearranges bytes into 3-byte encoding groups.
	// Unlike AVX2, VBMI's cross-lane VPERMB can reach any byte in the full
	// 32-byte vector, so no -4 offset trick or scalar preamble is needed.
	encByteGrouping archsimd.Uint8x32

	// Encode: sextet-to-ASCII lookup table, split across two 32-byte halves.
	// VPERMI2B (ConcatPermute) can index into 64 entries across two vectors,
	// which replaces the 4-operation range-detection logic from AVX2 with a
	// single table lookup.
	encAsciiTableLo archsimd.Uint8x32 // entries 0-31 (A-Z + start of a-z)
	encAsciiTableHi archsimd.Uint8x32 // entries 32-63 (rest of a-z, 0-9, +, /)

	// Decode: extracts 3 decoded bytes from every 4-byte group, cross-lane.
	// Single VPERMB replaces the two-step extract+pack that AVX2 needs.
	decExtractAndPack archsimd.Uint8x32
)

func initVBMI() {
	encByteGrouping = archsimd.LoadUint8x32(&[32]byte{
		1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10,
		13, 12, 14, 13, 16, 15, 17, 16, 19, 18, 20, 19, 22, 21, 23, 22,
	})

	var lo32, hi32 [32]byte
	for i := range 26 {
		lo32[i] = 65 // sextets 0-25 → A-Z
	}
	for i := 26; i < 32; i++ {
		lo32[i] = 71 // sextets 26-31 → a-f
	}
	for i := range 20 {
		hi32[i] = 71 // sextets 32-51 → g-z
	}
	for i := 20; i < 30; i++ {
		hi32[i] = 252 // sextets 52-61 → 0-9
	}
	hi32[30] = 237 // sextet 62 → +
	hi32[31] = 240 // sextet 63 → /
	encAsciiTableLo = archsimd.LoadUint8x32(&lo32)
	encAsciiTableHi = archsimd.LoadUint8x32(&hi32)

	decExtractAndPack = archsimd.LoadUint8x32(&[32]byte{
		2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12,
		18, 17, 16, 22, 21, 20, 26, 25, 24, 30, 29, 28,
	})
}

// encodeVBMI encodes 24 source bytes into 32 base64 characters per iteration.
// Faster than AVX2 at small sizes because cross-lane VPERMB eliminates the
// offset load trick and its scalar preamble.
func encodeVBMI(dst, src []byte) (di, si int) {
	srcEnd := len(src) - 32
	dstEnd := len(dst) - 32
	if srcEnd < 0 || dstEnd < 0 {
		return 0, 0
	}

	byteGrouping := encByteGrouping
	sextetMaskHi := encSextetMaskHi
	sextetShiftHi := encSextetShiftHi
	sextetMaskLo := encSextetMaskLo
	sextetShiftLo := encSextetShiftLo
	asciiTableLo := encAsciiTableLo
	asciiTableHi := encAsciiTableHi

	for si <= srcEnd && di <= dstEnd {
		inputBytes := archsimd.LoadUint8x32Slice(src[si : si+32])
		groupedBytes := inputBytes.Permute(byteGrouping)

		words := groupedBytes.AsUint16x16()
		hiSextets := words.And(sextetMaskHi).MulHigh(sextetShiftHi)
		loSextets := words.And(sextetMaskLo).Mul(sextetShiftLo)
		sextets := hiSextets.Or(loSextets).AsUint8x32()

		// Single-instruction sextet→ASCII via 64-entry lookup table.
		asciiOffset := asciiTableLo.ConcatPermute(asciiTableHi, sextets)
		asciiOutput := sextets.Add(asciiOffset)

		d := dst[di:]
		asciiOutput.StoreSlice(d[:32])
		si += 24
		di += 32
	}
	return di, si
}

// decodeVBMI decodes 32 base64 characters into 24 bytes per iteration.
// Faster than AVX2 because cross-lane VPERMB extracts and packs the decoded
// bytes in a single step instead of two.
func decodeVBMI(dst, src []byte) (di, si int) {
	nibMask := nibbleMask
	validHi := decValidationHi
	validLo := decValidationLo
	asciiToSextetTable := decAsciiToSextetTable
	slashChar := decIsSlashChar
	nibShift := decNibbleShift
	combinePairs := decCombinePairs
	combineQuads := decCombineQuads
	extractBytes := decExtractAndPack

	srcEnd := len(src) - 32
	dstEnd := len(dst) - 32
	for si <= srcEnd && di <= dstEnd {
		encoded := archsimd.LoadUint8x32Slice(src[si : si+32])

		// Validate all 32 characters at once.
		hiNibble := encoded.AsUint32x8().ShiftRight(nibShift).AsUint8x32().And(nibMask)
		loNibble := encoded.And(nibMask)
		if !validHi.PermuteOrZeroGrouped(hiNibble.AsInt8x32()).And(
			validLo.PermuteOrZeroGrouped(loNibble.AsInt8x32())).IsZero() {
			break
		}

		// Translate ASCII → sextets.
		isSlash := encoded.Equal(slashChar).ToInt8x32().AsUint8x32()
		asciiToSextet := asciiToSextetTable.PermuteOrZeroGrouped(hiNibble.Add(isSlash).AsInt8x32())
		sextets := encoded.Add(asciiToSextet)

		// Combine sextets → bytes, then extract and pack in one step.
		twelveBitValues := sextets.DotProductPairsSaturated(combinePairs)
		twentyFourBitValues := twelveBitValues.DotProductPairs(combineQuads)
		decodedBytes := twentyFourBitValues.AsUint8x32().Permute(extractBytes)

		d := dst[di:]
		decodedBytes.StoreSlice(d[:32])
		si += 32
		di += 24
	}
	return di, si
}

// ========================================================================
// Tier 2: AVX-512 (512-bit encode only; decode stays at 256-bit because
// the serial dependency chain regresses when widened — see RESEARCH.md)
// ========================================================================

var (
	// Encode: same concepts as 256-bit, but processing 48 bytes → 64 characters.
	encByteGrouping512 archsimd.Uint8x64
	encSextetMaskHi512  archsimd.Uint16x32
	encSextetShiftHi512 archsimd.Uint16x32
	encSextetMaskLo512  archsimd.Uint16x32
	encSextetShiftLo512 archsimd.Uint16x32

	// Encode: 64-entry sextet→ASCII table. At 512 bits, VPERMB can index all
	// 64 entries from a single vector, so no split table needed.
	encAsciiTable512 archsimd.Uint8x64
)

func init512() {
	var grouping [64]byte
	pat := [16]byte{1, 0, 2, 1, 4, 3, 5, 4, 7, 6, 8, 7, 10, 9, 11, 10}
	for lane := range 4 {
		for i := range 16 {
			grouping[lane*16+i] = pat[i] + byte(lane*12)
		}
	}
	encByteGrouping512 = archsimd.LoadUint8x64(&grouping)

	encSextetMaskHi512 = alternateU16x32(0xFC00, 0x0FC0)
	encSextetShiftHi512 = alternateU16x32(0x0040, 0x0400)
	encSextetMaskLo512 = alternateU16x32(0x03F0, 0x003F)
	encSextetShiftLo512 = alternateU16x32(0x0010, 0x0100)

	var asciiTable [64]byte
	for i := range 26 {
		asciiTable[i] = 65 // sextets 0-25 → A-Z
	}
	for i := 26; i < 52; i++ {
		asciiTable[i] = 71 // sextets 26-51 → a-z
	}
	for i := 52; i < 62; i++ {
		asciiTable[i] = 252 // sextets 52-61 → 0-9
	}
	asciiTable[62] = 237 // sextet 62 → +
	asciiTable[63] = 240 // sextet 63 → /
	encAsciiTable512 = archsimd.LoadUint8x64(&asciiTable)
}

// encode512 encodes 48 source bytes into 64 base64 characters per iteration.
// Uses 512-bit vectors for maximum throughput on large inputs. The wider VPERMB
// can hold all 64 ASCII table entries in a single vector.
func encode512(dst, src []byte) (di, si int) {
	srcEnd := len(src) - 64
	dstEnd := len(dst) - 64
	if srcEnd < 0 || dstEnd < 0 {
		return 0, 0
	}

	byteGrouping := encByteGrouping512
	sextetMaskHi := encSextetMaskHi512
	sextetShiftHi := encSextetShiftHi512
	sextetMaskLo := encSextetMaskLo512
	sextetShiftLo := encSextetShiftLo512
	asciiTable := encAsciiTable512

	for si <= srcEnd && di <= dstEnd {
		inputBytes := archsimd.LoadUint8x64Slice(src[si : si+64])
		groupedBytes := inputBytes.Permute(byteGrouping)

		words := groupedBytes.AsUint16x32()
		hiSextets := words.And(sextetMaskHi).MulHigh(sextetShiftHi)
		loSextets := words.And(sextetMaskLo).Mul(sextetShiftLo)
		sextets := hiSextets.Or(loSextets).AsUint8x64()

		asciiOffset := asciiTable.Permute(sextets)
		asciiOutput := sextets.Add(asciiOffset)

		d := dst[di:]
		asciiOutput.StoreSlice(d[:64])
		si += 48
		di += 64
	}
	return di, si
}
