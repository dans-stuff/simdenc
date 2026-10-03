package simdenc

import "encoding/base32"

const (
	base32AlphabetStd uint8 = 0
	base32AlphabetHex uint8 = 1
)

// Base32Encoding defines a base32 encoding/decoding scheme.
type Base32Encoding struct {
	base     *base32.Encoding
	alphabet uint8 // base32AlphabetStd or base32AlphabetHex
}

// Pre-built encodings matching encoding/base32.
var (
	Base32StdEncoding = &Base32Encoding{base: base32.StdEncoding, alphabet: base32AlphabetStd}
	Base32HexEncoding = &Base32Encoding{base: base32.HexEncoding, alphabet: base32AlphabetHex}
)

// WithPadding returns a new Base32Encoding identical to enc but with the given
// padding character, or NoPadding to disable padding.
func (enc Base32Encoding) WithPadding(padding rune) *Base32Encoding {
	enc.base = enc.base.WithPadding(padding)
	return &enc
}

func (enc *Base32Encoding) EncodedLen(n int) int { return enc.base.EncodedLen(n) }

func (enc *Base32Encoding) DecodedLen(n int) int { return enc.base.DecodedLen(n) }

// Encode encodes src into dst like base32.Encoding.Encode. SIMD handles whole
// 5-byte groups; the standard library finishes the rest, including padding.
func (enc *Base32Encoding) Encode(dst, src []byte) {
	si := base32EncodeBlocks(enc.alphabet, dst, src)
	enc.base.Encode(dst[si/5*8:], src[si:])
}

func (enc *Base32Encoding) EncodeToString(src []byte) string {
	buf := make([]byte, enc.EncodedLen(len(src)))
	enc.Encode(buf, src)
	return string(buf)
}

func (enc *Base32Encoding) AppendEncode(dst, src []byte) []byte {
	n := enc.EncodedLen(len(src))
	dst = grow(dst, n)
	enc.Encode(dst[len(dst)-n:], src)
	return dst
}

// Decode decodes src into dst like base32.Encoding.Decode. SIMD handles whole
// 8-character groups up to the first invalid character; the standard library
// decodes the rest, so padding rules and error offsets are its own.
func (enc *Base32Encoding) Decode(dst, src []byte) (int, error) {
	si := base32DecodeBlocks(enc.alphabet, dst, src)
	n, err := enc.base.Decode(dst[si/8*5:], src[si:])
	if corrupt, ok := err.(base32.CorruptInputError); ok {
		err = corrupt + base32.CorruptInputError(si)
	}
	return si/8*5 + n, err
}

func (enc *Base32Encoding) DecodeString(s string) ([]byte, error) {
	dst := make([]byte, enc.DecodedLen(len(s)))
	n, err := enc.Decode(dst, []byte(s))
	return dst[:n], err
}

func (enc *Base32Encoding) AppendDecode(dst, src []byte) ([]byte, error) {
	n := enc.DecodedLen(len(src))
	dst = grow(dst, n)
	nn, err := enc.Decode(dst[len(dst)-n:], src)
	return dst[:len(dst)-n+nn], err
}
