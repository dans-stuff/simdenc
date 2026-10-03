package simdenc

import "encoding/hex"

// HexEncodedLen returns the length of an encoding of n source bytes.
func HexEncodedLen(n int) int { return n * 2 }

// HexDecodedLen returns the length of a decoding of x source bytes.
func HexDecodedLen(x int) int { return x / 2 }

// HexEncode encodes src into HexEncodedLen(len(src)) bytes of dst as
// lowercase hexadecimal and returns that length. It matches hex.Encode.
func HexEncode(dst, src []byte) int {
	n := hexEncodeBlocks(dst, src)
	hex.Encode(dst[n*2:], src[n:])
	return len(src) * 2
}

// HexDecode decodes src into HexDecodedLen(len(src)) bytes of dst and
// returns the number of bytes written. It matches hex.Decode, including the
// returned count and error (hex.InvalidByteError or hex.ErrLength) on bad
// input.
func HexDecode(dst, src []byte) (int, error) {
	n := hexDecodeBlocks(dst, src)
	m, err := hex.Decode(dst[n/2:], src[n:])
	return n/2 + m, err
}

// HexEncodeToString returns the lowercase hexadecimal encoding of src.
func HexEncodeToString(src []byte) string {
	dst := make([]byte, HexEncodedLen(len(src)))
	HexEncode(dst, src)
	return string(dst)
}

// HexDecodeString returns the bytes represented by the hexadecimal string s.
// On error it returns the bytes decoded before the error, like
// hex.DecodeString.
func HexDecodeString(s string) ([]byte, error) {
	dst := make([]byte, HexDecodedLen(len(s)))
	n, err := HexDecode(dst, []byte(s))
	return dst[:n], err
}

// AppendHexEncode appends the lowercase hexadecimal encoding of src to dst.
func AppendHexEncode(dst, src []byte) []byte {
	n := HexEncodedLen(len(src))
	dst = grow(dst, n)
	HexEncode(dst[len(dst)-n:], src)
	return dst
}

// AppendHexDecode appends the bytes represented by the hexadecimal src to
// dst. On error it appends the bytes decoded before the error.
func AppendHexDecode(dst, src []byte) ([]byte, error) {
	n := HexDecodedLen(len(src))
	dst = grow(dst, n)
	nn, err := HexDecode(dst[len(dst)-n:], src)
	return dst[:len(dst)-n+nn], err
}
