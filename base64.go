package simdenc

import "encoding/base64"

const (
	StdPadding rune = '='
	NoPadding  rune = -1

	alphabetStd uint8 = 0
	alphabetURL uint8 = 1
)

// Encoding defines a base64 encoding/decoding scheme.
type Encoding struct {
	base     *base64.Encoding
	padChar  rune
	alphabet uint8 // alphabetStd or alphabetURL; used by SIMD dispatch
}

// Pre-built encodings matching encoding/base64.
var (
	StdEncoding    = &Encoding{base: base64.StdEncoding, padChar: StdPadding, alphabet: alphabetStd}
	URLEncoding    = &Encoding{base: base64.URLEncoding, padChar: StdPadding, alphabet: alphabetURL}
	RawStdEncoding = &Encoding{base: base64.RawStdEncoding, padChar: NoPadding, alphabet: alphabetStd}
	RawURLEncoding = &Encoding{base: base64.RawURLEncoding, padChar: NoPadding, alphabet: alphabetURL}
)

// WithPadding returns a new Encoding identical to enc but with the given
// padding character, or NoPadding to disable padding.
func (enc Encoding) WithPadding(padding rune) *Encoding {
	enc.padChar = padding
	enc.base = enc.base.WithPadding(padding)
	return &enc
}

func (enc *Encoding) EncodedLen(n int) int {
	if enc.padChar == NoPadding {
		return (n*4 + 2) / 3
	}
	return (n + 2) / 3 * 4
}

func (enc *Encoding) DecodedLen(n int) int {
	if enc.padChar == NoPadding {
		return n * 3 / 4
	}
	return n / 4 * 3
}

// Encode encodes src into dst like base64.Encoding.Encode. SIMD handles whole
// 3-byte groups; the standard library finishes the rest, including padding.
func (enc *Encoding) Encode(dst, src []byte) {
	si := encodeBlocks(enc.alphabet, dst, src)
	enc.base.Encode(dst[si/3*4:], src[si:])
}

func (enc *Encoding) EncodeToString(src []byte) string {
	buf := make([]byte, enc.EncodedLen(len(src)))
	enc.Encode(buf, src)
	return string(buf)
}

func (enc *Encoding) AppendEncode(dst, src []byte) []byte {
	n := enc.EncodedLen(len(src))
	dst = grow(dst, n)
	enc.Encode(dst[len(dst)-n:], src)
	return dst
}

// Decode decodes src into dst like base64.Encoding.Decode. SIMD handles whole
// 4-character groups up to the first invalid character; the standard library
// decodes the rest, so padding rules and error offsets are its own.
func (enc *Encoding) Decode(dst, src []byte) (int, error) {
	si := decodeBlocks(enc.alphabet, dst, src)
	n, err := enc.base.Decode(dst[si/4*3:], src[si:])
	if corrupt, ok := err.(base64.CorruptInputError); ok {
		err = corrupt + base64.CorruptInputError(si)
	}
	return si/4*3 + n, err
}

func (enc *Encoding) DecodeString(s string) ([]byte, error) {
	dst := make([]byte, enc.DecodedLen(len(s)))
	n, err := enc.Decode(dst, []byte(s))
	return dst[:n], err
}

func (enc *Encoding) AppendDecode(dst, src []byte) ([]byte, error) {
	n := enc.DecodedLen(len(src))
	dst = grow(dst, n)
	nn, err := enc.Decode(dst[len(dst)-n:], src)
	return dst[:len(dst)-n+nn], err
}

func grow(s []byte, n int) []byte {
	if cap(s)-len(s) >= n {
		return s[:len(s)+n]
	}
	buf := make([]byte, len(s)+n)
	copy(buf, s)
	return buf
}
