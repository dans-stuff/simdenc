//go:build goexperiment.simd

package simdenc

import "simd/archsimd"

const hexDigits = "0123456789abcdef"

// AVX-512 constants, loaded in init (they need loops to build).
var (
	hexTable512 archsimd.Uint8x64 // hexDigits repeated 4x; lookups only use the low 4 bits of an index
	hexDup0     archsimd.Uint8x64 // lanes 2i and 2i+1 = source byte i, for i < 32
	hexDup1     archsimd.Uint8x64 // same for source byte i+32
	hexLutLo512 archsimd.Uint8x64 // ASCII 0-63  -> nibble, 0x80 if not a hex digit
	hexLutHi512 archsimd.Uint8x64 // ASCII 64-127 -> nibble, 0x80 if not a hex digit
	hexPack512  archsimd.Uint8x64 // even bytes of two vectors, 64 bytes in all
	hexMult512  archsimd.Int8x64  // {16, 1} pairs: high nibble * 16 + low nibble
)

// hexTable256 is hexDigits in each 128-bit lane, for VPSHUFB.
var hexTable256 = [32]byte([]byte(hexDigits + hexDigits))

func init() {
	if !hasAVX512 {
		return
	}
	var table, dup0, dup1, pack [64]byte
	var mult [64]int8
	for i := range 64 {
		table[i] = hexDigits[i&15]
		mult[i] = 1
		if i%2 == 0 {
			mult[i] = 16
		}
	}
	for i := range 32 {
		dup0[2*i], dup0[2*i+1] = byte(i), byte(i)
		dup1[2*i], dup1[2*i+1] = byte(32+i), byte(32+i)
		pack[i], pack[32+i] = byte(2*i), byte(64+2*i)
	}
	lutLo, lutHi := decodeLUT(hexDigits, "0123456789ABCDEF")
	hexTable512, hexDup0, hexDup1 = archsimd.LoadUint8x64(&table), archsimd.LoadUint8x64(&dup0), archsimd.LoadUint8x64(&dup1)
	hexLutLo512, hexLutHi512, hexPack512 = archsimd.LoadUint8x64(&lutLo), archsimd.LoadUint8x64(&lutHi), archsimd.LoadUint8x64(&pack)
	hexMult512 = archsimd.LoadInt8x64(&mult)
}

// hexEncodeBlocks encodes as many whole blocks as it can, widest vectors
// first, and returns the source bytes consumed.
func hexEncodeBlocks(dst, src []byte) int {
	n := 0
	if hasAVX512 && len(src) >= 64 {
		n += hexEncode512(dst, src)
	}
	if hasAVX2 && len(src)-n >= 16 {
		n += hexEncode256(dst[n*2:], src[n:])
	}
	return n
}

// hexDecodeBlocks decodes as many whole blocks as it can, widest vectors
// first, stopping before the first block that holds a non-hex character. It
// returns the characters consumed.
func hexDecodeBlocks(dst, src []byte) int {
	n := 0
	if hasAVX512 && len(src) >= 128 {
		n += hexDecode512(dst, src)
	}
	if hasAVX2 && len(src)-n >= 32 {
		n += hexDecode256(dst[n/2:], src[n:])
	}
	return n
}

// hexEncode512 encodes 64 source bytes into 128 digits per iteration. A
// byte-duplicating VPERMB puts source byte i in output lanes 2i and 2i+1.
// Shifting each 16-bit word right by 4 leaves the high nibble in the low 4
// bits of its low byte; merging that into the even lanes gives high nibbles
// in even lanes and low nibbles in odd lanes. A final VPERMB maps nibbles to
// ASCII, ignoring the stray upper bits because the table repeats every 16.
func hexEncode512(dst, src []byte) int {
	table, dup0, dup1 := hexTable512, hexDup0, hexDup1
	even := archsimd.Mask8x64FromBits(0x5555555555555555)

	si := 0
	for si+64 <= len(src) && si*2+128 <= len(dst) {
		v := archsimd.LoadUint8x64Slice(src[si : si+64])
		d0, d1 := v.Permute(dup0), v.Permute(dup1)
		n0 := d0.AsUint16x32().ShiftAllRight(4).AsUint8x64().Merge(d0, even)
		n1 := d1.AsUint16x32().ShiftAllRight(4).AsUint8x64().Merge(d1, even)
		table.Permute(n0).StoreSlice(dst[si*2 : si*2+64])
		table.Permute(n1).StoreSlice(dst[si*2+64 : si*2+128])
		si += 64
	}
	return si
}

// hexDecode512 decodes 128 digits into 64 bytes per iteration. One VPERMI2B
// validates and translates (0x80 marks a non-hex character); the OR of every
// input and translated byte has its sign bit set exactly when a character was
// invalid or non-ASCII. VPMADDUBSW with {16, 1} joins nibble pairs into words
// and one more VPERMI2B packs the low byte of each word.
func hexDecode512(dst, src []byte) int {
	lutLo, lutHi := hexLutLo512, hexLutHi512
	mult, pack := hexMult512, hexPack512
	var zero archsimd.Int8x64

	si := 0
	for si+128 <= len(src) && si/2+64 <= len(dst) {
		c0 := archsimd.LoadUint8x64Slice(src[si : si+64])
		c1 := archsimd.LoadUint8x64Slice(src[si+64 : si+128])
		n0, n1 := lutLo.ConcatPermute(lutHi, c0), lutLo.ConcatPermute(lutHi, c1)
		if c0.Or(n0).Or(c1.Or(n1)).AsInt8x64().Less(zero).ToBits() != 0 {
			break
		}
		b0 := n0.DotProductPairsSaturated(mult).AsUint8x64()
		b1 := n1.DotProductPairsSaturated(mult).AsUint8x64()
		b0.ConcatPermute(b1, pack).StoreSlice(dst[si/2 : si/2+64])
		si += 128
	}

	return si
}

// hexEncode256 encodes 16 source bytes into 32 digits per iteration. Each
// byte is widened to a 16-bit word w; (w>>4 | w<<8) & 0x0f0f puts the high
// nibble in the low byte and the low nibble in the high byte, which is the
// output order, and VPSHUFB maps nibbles to ASCII.
func hexEncode256(dst, src []byte) int {
	table, mask := archsimd.LoadUint8x32(&hexTable256), archsimd.BroadcastUint8x32(0x0f)

	si := 0
	for si+16 <= len(src) && si*2+32 <= len(dst) {
		w := archsimd.LoadUint8x16Slice(src[si : si+16]).ExtendToUint16()
		nibbles := w.ShiftAllRight(4).Or(w.ShiftAllLeft(8)).AsUint8x32().And(mask)
		table.PermuteOrZeroGrouped(nibbles.AsInt8x32()).StoreSlice(dst[si*2 : si*2+32])
		si += 16
	}
	return si
}

// hexDecode256 decodes 32 digits into 16 bytes per iteration. A byte is a
// digit if c-'0' <= 9 and a letter if (c|0x20)-'a' <= 5 (both unsigned),
// which accepts exactly [0-9a-fA-F]. VPMADDUBSW with {16, 1} joins nibble
// pairs, VPSHUFB picks the low byte of each word and VPERMD joins the lanes.
func hexDecode256(dst, src []byte) int {
	zero, nine, five, ten := archsimd.BroadcastUint8x32('0'), archsimd.BroadcastUint8x32(9), archsimd.BroadcastUint8x32(5), archsimd.BroadcastUint8x32(10)
	letterA, caseBit := archsimd.BroadcastUint8x32('a'), archsimd.BroadcastUint8x32(0x20)
	mult := archsimd.BroadcastUint16x16(0x0110).AsInt8x32() // bytes 16, 1
	evens := archsimd.LoadUint64x4(&[4]uint64{0x0E0C0A0806040200, 0x8080808080808080, 0x0E0C0A0806040200, 0x8080808080808080}).AsInt8x32()
	lanes := archsimd.LoadUint32x8(&[8]uint32{0, 1, 4, 5, 0, 0, 0, 0})

	si := 0
	for si+32 <= len(src) && si/2+16 <= len(dst) {
		c := archsimd.LoadUint8x32Slice(src[si : si+32])
		digit := c.Sub(zero)
		letter := c.Or(caseBit).Sub(letterA)
		isDigit, isLetter := digit.Min(nine).Equal(digit), letter.Min(five).Equal(letter)
		if isDigit.Or(isLetter).ToBits() != ^uint32(0) {
			break
		}
		nibbles := digit.Merge(letter.Add(ten), isDigit)
		packed := nibbles.DotProductPairsSaturated(mult).AsUint8x32().PermuteOrZeroGrouped(evens)
		packed.AsUint32x8().Permute(lanes).AsUint8x32().GetLo().StoreSlice(dst[si/2 : si/2+16])
		si += 32
	}
	return si
}
