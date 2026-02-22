// Package simdenc implements base64 encoding as specified by RFC 4648,
// with optional SIMD acceleration on amd64.
//
// The API mirrors encoding/base64. On platforms without SIMD support,
// all operations delegate to encoding/base64.
package simdenc

import (
	"encoding/base64"
	"strconv"
)

const (
	// StdPadding is the standard base64 padding character ('=').
	StdPadding rune = '='
	// NoPadding disables padding in the encoding.
	NoPadding rune = -1

	encodeStd = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	encodeURL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)

// Encoding defines a base64 encoding/decoding scheme.
type Encoding struct {
	base    *base64.Encoding // stdlib fallback (always works)
	encode  [64]byte         // for SIMD scalar tail
	decode  [256]byte        // for SIMD scalar tail
	padChar rune
	useSIMD bool // true when std alphabet AND SIMD detected at init
}

// Pre-built encodings matching encoding/base64.
var (
	StdEncoding    = newEncoding(base64.StdEncoding, encodeStd, StdPadding)
	URLEncoding    = newEncoding(base64.URLEncoding, encodeURL, StdPadding)
	RawStdEncoding = newEncoding(base64.RawStdEncoding, encodeStd, NoPadding)
	RawURLEncoding = newEncoding(base64.RawURLEncoding, encodeURL, NoPadding)
)

func newEncoding(std *base64.Encoding, alphabet string, pad rune) *Encoding {
	enc := &Encoding{base: std, padChar: pad}
	copy(enc.encode[:], alphabet)
	for i := range enc.decode {
		enc.decode[i] = 0xFF
	}
	for i, c := range []byte(alphabet) {
		enc.decode[c] = byte(i)
	}
	return enc
}

// WithPadding returns a new Encoding identical to enc but with the given
// padding character, or NoPadding to disable padding.
func (enc Encoding) WithPadding(padding rune) *Encoding {
	enc.padChar = padding
	enc.base = enc.base.WithPadding(padding)
	return &enc
}

// EncodedLen returns the length in bytes of the base64 encoding of n source bytes.
func (enc *Encoding) EncodedLen(n int) int { return enc.base.EncodedLen(n) }

// DecodedLen returns the maximum length in bytes of the decoded data given n encoded bytes.
func (enc *Encoding) DecodedLen(n int) int { return enc.base.DecodedLen(n) }

// Encode encodes src into EncodedLen(len(src)) bytes of dst.
func (enc *Encoding) Encode(dst, src []byte) {
	if !enc.useSIMD {
		enc.base.Encode(dst, src)
		return
	}
	di, si := bulkEncode(dst, src)
	enc.encodeScalar(dst[di:], src[si:])
	if enc.padChar != NoPadding {
		n := enc.EncodedLen(len(src))
		raw := (len(src)*4 + 2) / 3
		for i := raw; i < n; i++ {
			dst[i] = byte(enc.padChar)
		}
	}
}

// EncodeToString returns the base64 encoding of src as a string.
func (enc *Encoding) EncodeToString(src []byte) string {
	if !enc.useSIMD {
		return enc.base.EncodeToString(src)
	}
	buf := make([]byte, enc.EncodedLen(len(src)))
	enc.Encode(buf, src)
	return string(buf)
}

// AppendEncode appends the base64 encoding of src to dst.
func (enc *Encoding) AppendEncode(dst, src []byte) []byte {
	if !enc.useSIMD {
		return enc.base.AppendEncode(dst, src)
	}
	n := enc.EncodedLen(len(src))
	dst = grow(dst, n)
	enc.Encode(dst[len(dst)-n:], src)
	return dst
}

// Decode decodes src into at most DecodedLen(len(src)) bytes of dst.
func (enc *Encoding) Decode(dst, src []byte) (int, error) {
	if !enc.useSIMD {
		return enc.base.Decode(dst, src)
	}
	if enc.padChar != NoPadding {
		for len(src) > 0 && src[len(src)-1] == byte(enc.padChar) {
			src = src[:len(src)-1]
		}
	}
	di, si := bulkDecode(dst, src)
	n, err := enc.decodeScalar(dst[di:], src[si:], si)
	return di + n, err
}

// DecodeString returns the bytes represented by the base64 string s.
func (enc *Encoding) DecodeString(s string) ([]byte, error) {
	if !enc.useSIMD {
		return enc.base.DecodeString(s)
	}
	dst := make([]byte, enc.DecodedLen(len(s)))
	n, err := enc.Decode(dst, []byte(s))
	return dst[:n], err
}

// AppendDecode appends the base64 decoding of src to dst.
func (enc *Encoding) AppendDecode(dst, src []byte) ([]byte, error) {
	if !enc.useSIMD {
		return enc.base.AppendDecode(dst, src)
	}
	n := enc.DecodedLen(len(src))
	dst = grow(dst, n)
	nn, err := enc.Decode(dst[len(dst)-n:], src)
	return dst[:len(dst)-n+nn], err
}

// --- Scalar encode/decode (used as SIMD tail handlers) ---

func (enc *Encoding) encodeScalar(dst, src []byte) {
	e := &enc.encode
	di, si := 0, 0
	n := (len(src) / 3) * 3
	for si < n {
		v := uint(src[si])<<16 | uint(src[si+1])<<8 | uint(src[si+2])
		dst[di+0] = e[v>>18&0x3F]
		dst[di+1] = e[v>>12&0x3F]
		dst[di+2] = e[v>>6&0x3F]
		dst[di+3] = e[v&0x3F]
		si += 3
		di += 4
	}
	switch len(src) - si {
	case 1:
		v := uint(src[si])
		dst[di+0] = e[v>>2]
		dst[di+1] = e[(v&0x3)<<4]
	case 2:
		v := uint(src[si])<<8 | uint(src[si+1])
		dst[di+0] = e[v>>10&0x3F]
		dst[di+1] = e[v>>4&0x3F]
		dst[di+2] = e[(v&0xF)<<2]
	}
}

func (enc *Encoding) decodeScalar(dst, src []byte, srcOffset int) (int, error) {
	d := &enc.decode
	di, si := 0, 0
	n := (len(src) / 4) * 4
	for si < n {
		a, b, c, dd := d[src[si]], d[src[si+1]], d[src[si+2]], d[src[si+3]]
		if a|b|c|dd == 0xFF {
			for j := 0; j < 4; j++ {
				if d[src[si+j]] == 0xFF {
					return di, CorruptInputError(srcOffset + si + j)
				}
			}
		}
		dst[di+0] = a<<2 | b>>4
		dst[di+1] = b<<4 | c>>2
		dst[di+2] = c<<6 | dd
		si += 4
		di += 3
	}
	switch len(src) - si {
	case 2:
		a, b := d[src[si]], d[src[si+1]]
		if a == 0xFF {
			return di, CorruptInputError(srcOffset + si)
		}
		if b == 0xFF {
			return di, CorruptInputError(srcOffset + si + 1)
		}
		dst[di] = a<<2 | b>>4
		di++
	case 3:
		a, b, c := d[src[si]], d[src[si+1]], d[src[si+2]]
		if a == 0xFF {
			return di, CorruptInputError(srcOffset + si)
		}
		if b == 0xFF {
			return di, CorruptInputError(srcOffset + si + 1)
		}
		if c == 0xFF {
			return di, CorruptInputError(srcOffset + si + 2)
		}
		dst[di+0] = a<<2 | b>>4
		dst[di+1] = b<<4 | c>>2
		di += 2
	case 1:
		return di, CorruptInputError(srcOffset + si)
	}
	return di, nil
}

// --- Errors ---

// CorruptInputError is returned by Decode when the input contains invalid base64 data.
// The integer value indicates the byte offset of the first invalid character.
type CorruptInputError int64

func (e CorruptInputError) Error() string {
	return "illegal base64 data at input byte " + strconv.FormatInt(int64(e), 10)
}

// --- Helpers ---

// bulkEncode and bulkDecode are set by arch-specific init (e.g. base64_amd64.go).
// They are nil on platforms without SIMD support.
var (
	bulkEncode func(dst, src []byte) (int, int)
	bulkDecode func(dst, src []byte) (int, int)
)

func grow(s []byte, n int) []byte {
	if cap(s)-len(s) >= n {
		return s[:len(s)+n]
	}
	buf := make([]byte, len(s)+n)
	copy(buf, s)
	return buf
}
