package simdenc

import (
	"math/rand"
	"net/url"
	"strings"
	"testing"
)

// percentFuncs pairs each function with the net/url function it must match.
var percentEscapers = []struct {
	name      string
	got, want func(string) string
}{
	{"QueryEscape", QueryEscape, url.QueryEscape},
	{"PathEscape", PathEscape, url.PathEscape},
}

var percentUnescapers = []struct {
	name      string
	got, want func(string) (string, error)
}{
	{"QueryUnescape", QueryUnescape, url.QueryUnescape},
	{"PathUnescape", PathUnescape, url.PathUnescape},
}

// randText returns n bytes: mostly characters that stay as they are, and with
// probability special any byte at all, including spaces, '%' and '+'.
func randText(r *rand.Rand, n int, special float64) string {
	const plain = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~"
	b := make([]byte, n)
	for i := range b {
		if r.Float64() < special {
			b[i] = byte(r.Intn(256))
		} else {
			b[i] = plain[r.Intn(len(plain))]
		}
	}
	return string(b)
}

func checkEscape(t *testing.T, s string) {
	t.Helper()
	for _, e := range percentEscapers {
		if got, want := e.got(s), e.want(s); got != want {
			t.Fatalf("%s(%q) = %q, want %q", e.name, s, got, want)
		}
	}
}

func checkUnescape(t *testing.T, s string) {
	t.Helper()
	for _, u := range percentUnescapers {
		got, gerr := u.got(s)
		want, werr := u.want(s)
		if got != want || gerr != werr {
			t.Fatalf("%s(%q) = (%q, %v), want (%q, %v)", u.name, s, got, gerr, want, werr)
		}
	}
}

func TestPercentEscape(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(1))
		for n := range 300 {
			for _, special := range []float64{0, 0.02, 0.1, 0.5, 1} {
				checkEscape(t, randText(r, n, special))
			}
		}
		for _, n := range []int{1000, 4097, 65543} {
			for _, special := range []float64{0, 0.001, 0.1, 1} {
				checkEscape(t, randText(r, n, special))
			}
		}
		checkEscape(t, strings.Repeat(" ", 200))
		checkEscape(t, strings.Repeat("\xa0", 200)) // 0xA0 has the low bits of ' ' but must escape
	})
}

// Every byte value at every position of an otherwise clean string.
func TestPercentEscapeByteEverywhere(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		base := []byte(strings.Repeat("abcdefghij", 15))
		for pos := range base {
			for b := range 256 {
				s := append([]byte(nil), base...)
				s[pos] = byte(b)
				checkEscape(t, string(s))
			}
		}
	})
}

func TestPercentUnescapeValid(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(2))
		for n := range 300 {
			for _, special := range []float64{0, 0.02, 0.1, 0.5, 1} {
				s := randText(r, n, special)
				checkUnescape(t, url.QueryEscape(s))
				checkUnescape(t, url.PathEscape(s))
				checkUnescape(t, strings.ToLower(url.QueryEscape(s)))
				checkUnescape(t, s) // arbitrary text: '%' and '+' appear unescaped, so often malformed
			}
		}
		for _, n := range []int{1000, 4097, 65543} {
			s := randText(r, n, 0.1)
			checkUnescape(t, url.QueryEscape(s))
			checkUnescape(t, url.PathEscape(s))
		}
	})
}

// A %XX sequence starting at every offset around the 64-byte block boundary,
// followed by every length of tail, and truncated sequences at the very end.
func TestPercentUnescapeBoundaries(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		for offset := range 140 {
			for _, tail := range []string{"", "x", "xy", "%41", "%4", "%", "+", "%zz", "%4g"} {
				checkUnescape(t, strings.Repeat("a", offset)+"%41"+tail)
				checkUnescape(t, strings.Repeat("a", offset)+"%41%42%43"+tail)
				checkUnescape(t, strings.Repeat("a", offset)+tail)
			}
		}
	})
}

// Every two characters after a '%' placed where a 64-byte block ends.
func TestPercentUnescapePairs(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		prefix := strings.Repeat("a", 59)
		for i := range 1 << 16 {
			s := prefix + "%" + string([]byte{byte(i), byte(i >> 8)}) + strings.Repeat("b", 70)
			checkUnescape(t, s)
		}
	})
}

// Every byte value at every position of a string made of escapes.
func TestPercentUnescapeByteEverywhere(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		base := []byte(strings.Repeat("a%41+%7e", 18))
		for pos := range base {
			for b := range 256 {
				s := append([]byte(nil), base...)
				s[pos] = byte(b)
				checkUnescape(t, string(s))
			}
		}
	})
}

func FuzzPercentEscape(f *testing.F) {
	f.Add("")
	f.Add("a b&c=d/é\x00\xff")
	f.Add(strings.Repeat("a b%", 40))
	f.Fuzz(func(t *testing.T, s string) {
		eachTier(t, func(t *testing.T) { checkEscape(t, s) })
	})
}

func FuzzPercentUnescape(f *testing.F) {
	f.Add("")
	f.Add("a%20b+c%zz")
	f.Add(strings.Repeat("a%41+", 40))
	f.Fuzz(func(t *testing.T, s string) {
		eachTier(t, func(t *testing.T) { checkUnescape(t, s) })
	})
}
