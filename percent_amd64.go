//go:build goexperiment.simd

package simdenc

import (
	"math/bits"
	"simd/archsimd"
)

// AVX-512 VBMI2 constants, loaded in init. VBMI2 provides Compress, which
// both kernels use to delete the bytes that do not appear in the output.
var (
	percentLutLo [2]archsimd.Uint8x64 // class of ASCII 0-63 per mode
	percentLutHi [2]archsimd.Uint8x64 // class of ASCII 64-127 per mode
	percentPlus  [2]archsimd.Uint8x64 // what '+' unescapes to, per mode
	percentTable archsimd.Uint8x64    // upper-case hex digits repeated 4x: lookups only use the low 4 bits
	percentRep3  archsimd.Uint8x64    // lanes 3i..3i+2 = source byte i, for i < 21
	percentIdx1  archsimd.Uint8x64    // lane i = i+1 (clamped)
	percentIdx2  archsimd.Uint8x64    // lane i = i+2 (clamped)
)

const (
	percentLane0 = 0x1249249249249249 // bits 0, 3, 6, ... 60: first lane of each 3-lane group
	percentLane1 = 0x2492492492492492 // bits 1, 4, 7, ... 61: second lane
	percentLanes = 1<<63 - 1          // lanes 0-62: the 21 groups
)

func init() {
	if !hasVBMI2 {
		return
	}
	var table, rep3, idx1, idx2 [64]byte
	for i := range 64 {
		table[i] = upperHex[i&15]
		rep3[i] = byte(i / 3 % 21)
		idx1[i], idx2[i] = byte(min(i+1, 63)), byte(min(i+2, 63))
	}
	percentTable, percentRep3 = archsimd.LoadUint8x64(&table), archsimd.LoadUint8x64(&rep3)
	percentIdx1, percentIdx2 = archsimd.LoadUint8x64(&idx1), archsimd.LoadUint8x64(&idx2)
	for mode := range 2 {
		var lo, hi [64]byte
		for c := range 64 {
			lo[c], hi[c] = percentClass[mode][c], percentClass[mode][64+c]
		}
		percentLutLo[mode], percentLutHi[mode] = archsimd.LoadUint8x64(&lo), archsimd.LoadUint8x64(&hi)
	}
	percentPlus[percentQuery] = archsimd.BroadcastUint8x64(' ')
	percentPlus[percentPath] = archsimd.BroadcastUint8x64('+')
}

// percentCountBlocks counts, in whole 64-byte blocks, the bytes that escape
// to %XX and the spaces that become '+', and returns the bytes scanned.
func percentCountBlocks(mode int, src []byte) (escapes, spaces, si int) {
	if !hasVBMI2 {
		return 0, 0, 0
	}
	lutLo, lutHi := percentLutLo[mode], percentLutHi[mode]
	isEscape, isSpace := archsimd.BroadcastUint8x64(classEscape), archsimd.BroadcastUint8x64(classSpace)
	var zero archsimd.Int8x64

	for ; si+64 <= len(src); si += 64 {
		v := archsimd.LoadUint8x64Slice(src[si : si+64])
		class := lutLo.ConcatPermute(lutHi, v)
		// A byte of 0x80 or more is always escaped; the table only knows ASCII.
		high := v.AsInt8x64().Less(zero).ToBits()
		escapes += bits.OnesCount64(class.Equal(isEscape).ToBits() | high)
		spaces += bits.OnesCount64(class.Equal(isSpace).ToBits() &^ high)
	}
	return escapes, spaces, si
}

// percentEscapeBlocks escapes whole 64-byte blocks of src into dst and returns
// the bytes consumed and written. A block with nothing to escape is copied
// (spaces become '+' in query mode). Otherwise 21 bytes are expanded: each
// source byte is repeated in three lanes, the lanes become '%' and two hex
// digits for a byte that escapes, and Compress drops the two spare lanes of
// every byte that does not.
func percentEscapeBlocks(mode int, dst, src []byte) (si, di int) {
	if !hasVBMI2 {
		return 0, 0
	}
	lutLo, lutHi, table, rep3 := percentLutLo[mode], percentLutHi[mode], percentTable, percentRep3
	isEscape, isSpace := archsimd.BroadcastUint8x64(classEscape), archsimd.BroadcastUint8x64(classSpace)
	percent, plus := archsimd.BroadcastUint8x64('%'), archsimd.BroadcastUint8x64('+')
	lane0, lane1 := archsimd.Mask8x64FromBits(percentLane0), archsimd.Mask8x64FromBits(percentLane1)
	var zero archsimd.Int8x64

	for si+64 <= len(src) && di+64 <= len(dst) {
		v := archsimd.LoadUint8x64Slice(src[si : si+64])
		class := lutLo.ConcatPermute(lutHi, v)
		if class.Equal(isEscape).Or(v.AsInt8x64().Less(zero)).ToBits() == 0 {
			plus.Merge(v, class.Equal(isSpace)).StoreSlice(dst[di : di+64])
			si += 64
			di += 64
			continue
		}

		t := v.Permute(rep3)
		class = lutLo.ConcatPermute(lutHi, t)
		escaped := class.Equal(isEscape).Or(t.AsInt8x64().Less(zero))
		nibbles := t.AsUint16x32().ShiftAllRight(4).AsUint8x64().Merge(t, lane1) // lane 1: high nibble, lane 2: low
		digits := percent.Merge(table.Permute(nibbles), lane0)                   // lane 0: '%'
		out := digits.Merge(plus.Merge(t, class.Equal(isSpace)), escaped)        // escaped bytes take all 3 lanes
		keep := escaped.Or(lane0).ToBits() & percentLanes
		out.Compress(archsimd.Mask8x64FromBits(keep)).StoreSlicePart(dst[di : di+bits.OnesCount64(keep)])
		si += 21
		di += bits.OnesCount64(keep)
	}
	return si, di
}

// percentUnescapeBlocks decodes src into dst in 64-byte blocks, up to the first
// block holding a malformed %XX, and returns the bytes consumed and written.
// For every '%' the two lanes after it are looked up as hex digits and joined
// into the byte, which replaces the '%' lane; Compress then drops the digit
// lanes. A '%' in the last four lanes starts the next block instead, so no
// sequence is ever split.
func percentUnescapeBlocks(mode int, dst, src []byte) (si, di int) {
	if !hasVBMI2 {
		return 0, 0
	}
	lutLo, lutHi := hexLutLo512, hexLutHi512
	idx1, idx2, plusTo := percentIdx1, percentIdx2, percentPlus[mode]
	percent, plus := archsimd.BroadcastUint8x64('%'), archsimd.BroadcastUint8x64('+')
	highNibble, highBit := archsimd.BroadcastUint8x64(0xF0), archsimd.BroadcastUint8x64(0x80)
	var zero archsimd.Int8x64

	for si+64 <= len(src) && di+64 <= len(dst) {
		v := archsimd.LoadUint8x64Slice(src[si : si+64])
		isPercent := v.Equal(percent)
		percents := isPercent.ToBits()
		text := plusTo.Merge(v, v.Equal(plus))
		if percents == 0 {
			text.StoreSlice(dst[di : di+64])
			si += 64
			di += 64
			continue
		}

		n := 64
		if late := percents >> 60; late != 0 {
			n = 60 + bits.TrailingZeros64(late)
		}
		nibbles := lutLo.ConcatPermute(lutHi, v).Or(v.And(highBit)) // 0x80 marks a non-hex or non-ASCII byte
		high, low := nibbles.Permute(idx1), nibbles.Permute(idx2)
		used := uint64(1)<<n - 1
		if high.Or(low).AsInt8x64().Less(zero).ToBits()&percents&used != 0 {
			break
		}
		joined := high.AsUint16x32().ShiftAllLeft(4).AsUint8x64().And(highNibble).Or(low) // the 16-bit shift lets a neighbour's bits into the low nibble
		keep := used &^ (percents<<1 | percents<<2)
		joined.Merge(text, isPercent).Compress(archsimd.Mask8x64FromBits(keep)).StoreSlicePart(dst[di : di+bits.OnesCount64(keep)])
		si += n
		di += bits.OnesCount64(keep)
	}
	return si, di
}
