package simdenc

import (
	"net/url"
	"strings"
	"unsafe"
)

// Percent-encoding with the behaviour of net/url's QueryEscape, PathEscape,
// QueryUnescape and PathUnescape.

const (
	percentQuery = 0 // query component: ' ' <-> '+'
	percentPath  = 1 // path segment: '+' stays '+'

	// Byte classes in percentClass.
	classKeep   = 0 // written as is
	classEscape = 1 // written as %XX
	classSpace  = 2 // written as '+' (query mode only)
)

// percentClass[mode][c] says what escaping does to byte c. It is derived from
// net/url itself so the two can never disagree.
var percentClass = func() (class [2][256]uint8) {
	for c := range 256 {
		s := string([]byte{byte(c)})
		switch q := url.QueryEscape(s); {
		case q == "+":
			class[percentQuery][c] = classSpace
		case q != s:
			class[percentQuery][c] = classEscape
		}
		if url.PathEscape(s) != s {
			class[percentPath][c] = classEscape
		}
	}
	return class
}()

// QueryEscape escapes s so it can be placed inside a URL query, like url.QueryEscape.
func QueryEscape(s string) string { return escape(s, percentQuery) }

// PathEscape escapes s so it can be placed inside a URL path segment, like url.PathEscape.
func PathEscape(s string) string { return escape(s, percentPath) }

// QueryUnescape reverses QueryEscape, like url.QueryUnescape.
func QueryUnescape(s string) (string, error) { return unescape(s, percentQuery) }

// PathUnescape reverses PathEscape, like url.PathUnescape.
func PathUnescape(s string) (string, error) { return unescape(s, percentPath) }

const upperHex = "0123456789ABCDEF"

// escape counts what needs escaping, so that it can return s itself when
// nothing does and otherwise allocate the exact result, then fills the result.
func escape(s string, mode int) string {
	src := bytesOf(s)
	escapes, spaces, si := percentCountBlocks(mode, src)
	for _, c := range src[si:] {
		switch percentClass[mode][c] {
		case classEscape:
			escapes++
		case classSpace:
			spaces++
		}
	}
	if escapes == 0 && spaces == 0 {
		return s
	}

	dst := make([]byte, len(s)+2*escapes)
	si, di := percentEscapeBlocks(mode, dst, src)
	for _, c := range src[si:] {
		switch percentClass[mode][c] {
		case classEscape:
			dst[di], dst[di+1], dst[di+2] = '%', upperHex[c>>4], upperHex[c&15]
			di += 3
		case classSpace:
			dst[di] = '+'
			di++
		default:
			dst[di] = c
			di++
		}
	}
	return stringOf(dst)
}

// unescape decodes whole %XX sequences with SIMD up to the first invalid one;
// net/url decodes the rest, so it also reports every error.
func unescape(s string, mode int) (string, error) {
	if !strings.Contains(s, "%") && (mode == percentPath || !strings.Contains(s, "+")) {
		return s, nil
	}
	dst := make([]byte, len(s))
	si, di := percentUnescapeBlocks(mode, dst, bytesOf(s))

	unescapeRest := url.PathUnescape
	if mode == percentQuery {
		unescapeRest = url.QueryUnescape
	}
	rest, err := unescapeRest(s[si:])
	if err != nil {
		return "", err
	}
	di += copy(dst[di:], rest)
	return stringOf(dst[:di]), nil
}

// bytesOf views s as a read-only []byte without copying. Callers never write to it.
func bytesOf(s string) []byte { return unsafe.Slice(unsafe.StringData(s), len(s)) }

// stringOf turns b into a string without copying. Callers never use b afterwards.
func stringOf(b []byte) string { return unsafe.String(unsafe.SliceData(b), len(b)) }
