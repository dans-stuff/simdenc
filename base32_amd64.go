//go:build goexperiment.simd

package simdenc

import (
	"math/bits"
	"simd/archsimd"
)

var base32Alphabets = [2]string{
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", // base32AlphabetStd
	"0123456789ABCDEFGHIJKLMNOPQRSTUV", // base32AlphabetHex
}

// AVX-512 constants, loaded in init. Encode and decode work on 8-character
// groups: 5 bytes <-> 8 five-bit values.
var (
	// Encode.
	base32Gather0 archsimd.Uint8x64    // 16-bit window per output char, for source groups 0-3
	base32Gather1 archsimd.Uint8x64    // same for source groups 4-7
	base32Shifts  archsimd.Uint16x32   // right shift that brings each char's 5 bits to the bottom
	base32Compact archsimd.Uint8x64    // low bytes of the 64 shifted words
	base32Table   [2]archsimd.Uint8x64 // alphabet repeated twice: lookups only use the low 5 bits

	// Decode.
	base32LutLo   [2]archsimd.Uint8x64 // ASCII 0-63  -> 5-bit value, 0x80 if invalid
	base32LutHi   [2]archsimd.Uint8x64 // ASCII 64-127 -> 5-bit value, 0x80 if invalid
	base32Mult1   archsimd.Int8x64     // {32, 1}: two 5-bit values -> 10 bits
	base32Mult2   archsimd.Int16x32    // {1024, 1}: two 10-bit values -> 20 bits
	base32Extract archsimd.Uint8x64    // the 5 bytes of each 40-bit lane, in order
)

func init() {
	if !hasAVX512 {
		return
	}
	var gather0, gather1, compact, extract [64]byte
	var shifts [32]uint16
	for w := range 32 {
		group, char := w/8, w%8
		first := 5 * char / 8             // byte of the group holding the char's top bit
		shifts[w] = uint16(11 - 5*char%8) // 16-bit window [first, first+1] shifted so the char is in bits 0-4
		for g, out := range []*[64]byte{&gather0, &gather1} {
			// Little-endian word: low byte = window's second byte, high byte = first.
			base := 5*group + 20*g + first
			out[2*w], out[2*w+1] = byte(base+1), byte(base)
		}
	}
	for p := range 64 {
		compact[p] = byte(2 * p)
		if p >= 32 {
			compact[p] = byte(64 + 2*(p-32))
		}
	}
	for g := range 8 {
		for t := range 5 {
			extract[5*g+t] = byte(8*g + 4 - t) // 40-bit value in a little-endian qword: top byte is byte 4
		}
	}
	var mult1 [64]int8
	var mult2 [32]int16
	for i := range 64 {
		mult1[i] = 1
		if i%2 == 0 {
			mult1[i] = 32
		}
	}
	for i := range 32 {
		mult2[i] = 1
		if i%2 == 0 {
			mult2[i] = 1024
		}
	}
	base32Gather0, base32Gather1 = archsimd.LoadUint8x64(&gather0), archsimd.LoadUint8x64(&gather1)
	base32Shifts = archsimd.LoadUint16x32(&shifts)
	base32Compact, base32Extract = archsimd.LoadUint8x64(&compact), archsimd.LoadUint8x64(&extract)
	base32Mult1, base32Mult2 = archsimd.LoadInt8x64(&mult1), archsimd.LoadInt16x32(&mult2)

	for a, alpha := range base32Alphabets {
		var table [64]byte
		for i := range table {
			table[i] = alpha[i%32]
		}
		lutLo, lutHi := decodeLUT(alpha)
		base32Table[a] = archsimd.LoadUint8x64(&table)
		base32LutLo[a], base32LutHi[a] = archsimd.LoadUint8x64(&lutLo), archsimd.LoadUint8x64(&lutHi)
	}
}

// base32EncodeBlocks encodes as many whole 5-byte groups as it can and returns
// the source bytes consumed.
func base32EncodeBlocks(alphabet uint8, dst, src []byte) int {
	if !hasAVX512 || len(src) < 40 {
		return 0
	}
	return base32Encode512(alphabet, dst, src)
}

// base32DecodeBlocks decodes as many whole 8-character groups as it can,
// stopping before the first block with an invalid character, and returns the
// source characters consumed.
func base32DecodeBlocks(alphabet uint8, dst, src []byte) int {
	if !hasAVX512 || len(src) < 64 {
		return 0
	}
	return base32Decode512(alphabet, dst, src)
}

// base32Encode512 encodes 40 source bytes into 64 characters per iteration.
// A VPERMB per half builds, for every output char, the 16-bit window of
// source bytes that holds its 5 bits; a variable shift moves the 5 bits to
// the bottom of each word (bytes past the end of src only reach unused
// window bits). One VPERMI2B keeps the low byte of all 64 words
// and a last VPERMB maps values to ASCII, ignoring the stray upper bits
// because the table repeats every 32 entries.
func base32Encode512(alphabet uint8, dst, src []byte) int {
	gather0, gather1, shifts, compact := base32Gather0, base32Gather1, base32Shifts, base32Compact
	table := base32Table[alphabet]

	si, di := 0, 0
	for si+64 <= len(src) && di+64 <= len(dst) {
		v := archsimd.LoadUint8x64Slice(src[si : si+64])
		w0 := v.Permute(gather0).AsUint16x32().ShiftRight(shifts).AsUint8x64()
		w1 := v.Permute(gather1).AsUint16x32().ShiftRight(shifts).AsUint8x64()
		table.Permute(w0.ConcatPermute(w1, compact)).StoreSlice(dst[di : di+64])
		si += 40
		di += 64
	}
	// Tail: the whole groups left, at most 8, through a partial load and store.
	if n := min((len(src)-si)/5, 8); n > 0 && di+8*n <= len(dst) {
		v := archsimd.LoadUint8x64SlicePart(src[si:])
		w0 := v.Permute(gather0).AsUint16x32().ShiftRight(shifts).AsUint8x64()
		w1 := v.Permute(gather1).AsUint16x32().ShiftRight(shifts).AsUint8x64()
		table.Permute(w0.ConcatPermute(w1, compact)).StoreSlicePart(dst[di : di+8*n])
		si += 5 * n
	}
	return si
}

// base32Decode512 decodes 64 characters into 40 bytes per iteration. One
// VPERMI2B validates and translates (0x80 marks an invalid character).
// VPMADDUBSW and VPMADDWD join the values into 20-bit halves; shifting and
// OR-ing the two halves of each 64-bit lane makes the 40-bit group, and a
// VPERMB picks its 5 bytes, of which the masked store keeps the first 40.
func base32Decode512(alphabet uint8, dst, src []byte) int {
	lutLo, lutHi := base32LutLo[alphabet], base32LutHi[alphabet]
	mult1, mult2, extract := base32Mult1, base32Mult2, base32Extract
	var zero archsimd.Int8x64

	si, di := 0, 0
	for si+64 <= len(src) && di+40 <= len(dst) {
		c := archsimd.LoadUint8x64Slice(src[si : si+64])
		v := lutLo.ConcatPermute(lutHi, c)
		if c.Or(v).AsInt8x64().Less(zero).ToBits() != 0 {
			break
		}
		halves := v.DotProductPairsSaturated(mult1).DotProductPairs(mult2).AsUint64x8()
		groups := halves.ShiftAllLeft(20).Or(halves.ShiftAllRight(32))
		groups.AsUint8x64().Permute(extract).StoreSlicePart(dst[di : di+40])
		si += 64
		di += 40
	}
	// Tail: the whole groups left, at most 8, up to the first invalid character.
	if n := min((len(src)-si)/8, 8); n > 0 {
		c := archsimd.LoadUint8x64SlicePart(src[si:])
		v := lutLo.ConcatPermute(lutHi, c)
		bad := c.Or(v).AsInt8x64().Less(zero).ToBits()
		if n = min(n, bits.TrailingZeros64(bad)/8); n > 0 && di+5*n <= len(dst) {
			halves := v.DotProductPairsSaturated(mult1).DotProductPairs(mult2).AsUint64x8()
			groups := halves.ShiftAllLeft(20).Or(halves.ShiftAllRight(32))
			groups.AsUint8x64().Permute(extract).StoreSlicePart(dst[di : di+5*n])
			si += 8 * n
		}
	}
	return si
}
