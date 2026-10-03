package simdenc

import (
	"encoding/base32"
	"math/rand"
	"testing"
)

// base32Pairs pairs each simdenc encoding with the standard library encoding
// it must match exactly.
var base32Pairs = []struct {
	name string
	got  *Base32Encoding
	want *base32.Encoding
}{
	{"Std", Base32StdEncoding, base32.StdEncoding},
	{"Hex", Base32HexEncoding, base32.HexEncoding},
	{"StdNoPad", Base32StdEncoding.WithPadding(NoPadding), base32.StdEncoding.WithPadding(base32.NoPadding)},
	{"HexNoPad", Base32HexEncoding.WithPadding(NoPadding), base32.HexEncoding.WithPadding(base32.NoPadding)},
	{"Std/pad#", Base32StdEncoding.WithPadding('#'), base32.StdEncoding.WithPadding('#')},
}

var base32Cases = func() (cases []codecCase) {
	for _, p := range base32Pairs {
		cases = append(cases, codecCase{p.name, p.got, p.want})
	}
	return cases
}()

func TestBase32MatchesStdlib(t *testing.T) {
	checkCodecs(t, base32Cases, 110)
}

// The convenience methods around Encode and Decode, which checkCodecs does not reach.
func TestBase32Helpers(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(1))
		for _, p := range base32Pairs {
			for n := range 200 {
				src := rndBytes(r, n)
				if p.got.EncodedLen(n) != p.want.EncodedLen(n) || p.got.DecodedLen(n) != p.want.DecodedLen(n) {
					t.Fatalf("%s: length helpers differ at %d", p.name, n)
				}
				text := p.want.EncodeToString(src)
				if got := p.got.EncodeToString(src); got != text {
					t.Fatalf("%s: EncodeToString(%d bytes) = %q, want %q", p.name, n, got, text)
				}
				if got := p.got.AppendEncode([]byte("x"), src); string(got) != "x"+text {
					t.Fatalf("%s: AppendEncode(%d bytes) = %q", p.name, n, got)
				}
				for _, in := range []string{text, corrupt(r, text)} {
					want, werr := p.want.DecodeString(in)
					got, gerr := p.got.DecodeString(in)
					if string(got) != string(want) || gerr != werr {
						t.Fatalf("%s: DecodeString(%q) = %q, %v; want %q, %v", p.name, in, got, gerr, want, werr)
					}
					wantAppended, werr := p.want.AppendDecode([]byte("x"), []byte(in))
					gotAppended, gerr := p.got.AppendDecode([]byte("x"), []byte(in))
					if string(gotAppended) != string(wantAppended) || gerr != werr {
						t.Fatalf("%s: AppendDecode(%q) = %q, %v; want %q, %v", p.name, in, gotAppended, gerr, wantAppended, werr)
					}
				}
			}
		}
	})
}

// corrupt returns s with one byte replaced by '!', or s if it is empty.
func corrupt(r *rand.Rand, s string) string {
	if s == "" {
		return s
	}
	b := []byte(s)
	b[r.Intn(len(b))] = '!'
	return string(b)
}

func FuzzBase32DecodeMatchesStdlib(f *testing.F) {
	fuzzCodecs(f, base32Cases)
}
