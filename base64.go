// Package simdenc implements base64 encoding as specified by RFC 4648,
// with optional SIMD acceleration on amd64.
//
// The API mirrors encoding/base64. On platforms without SIMD support,
// all operations delegate to encoding/base64.
package simdenc

import "encoding/base64"

const (
	StdPadding rune = '='
	NoPadding  rune = -1

	alphabetStd uint8 = 0
	alphabetURL uint8 = 1

	encodeStdAlpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	encodeURLAlpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)

var encAlphabets = [2]string{encodeStdAlpha, encodeURLAlpha}

// Decode lookup tables: ASCII byte → 6-bit value, 0xFF for invalid.
var decTables [2][256]byte

func init() {
	for a := range 2 {
		for i := range decTables[a] {
			decTables[a][i] = 0xFF
		}
		for i, c := range encAlphabets[a] {
			decTables[a][c] = byte(i)
		}
	}
}

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

// rawStdlib provides no-padding stdlib encodings for decoding error paths.
var rawStdlib = [2]*base64.Encoding{base64.RawStdEncoding, base64.RawURLEncoding}

// SIMD dispatch: set by platform-specific init when SIMD is available.
// When nil, Encode/Decode delegate entirely to encoding/base64.
var (
	simdEncode func(alphabet uint8, dst, src []byte)
	simdDecode func(alphabet uint8, dst, src []byte) (int, int)
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

func (enc *Encoding) Encode(dst, src []byte) {
	if simdEncode == nil {
		enc.base.Encode(dst, src)
		return
	}
	simdEncode(enc.alphabet, dst, src)
	if enc.padChar != NoPadding {
		n := enc.EncodedLen(len(src))
		raw := (len(src)*4 + 2) / 3
		for i := raw; i < n; i++ {
			dst[i] = byte(enc.padChar)
		}
	}
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

func (enc *Encoding) Decode(dst, src []byte) (int, error) {
	if simdDecode == nil {
		return enc.base.Decode(dst, src)
	}
	if enc.padChar != NoPadding {
		for len(src) > 0 && src[len(src)-1] == byte(enc.padChar) {
			src = src[:len(src)-1]
		}
	}
	di, si := simdDecode(enc.alphabet, dst, src)
	if si == len(src) {
		return di, nil
	}
	n, err := rawStdlib[enc.alphabet].Decode(dst[di:], src[si:])
	return di + n, err
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
