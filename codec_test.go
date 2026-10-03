package simdenc

import (
	"bytes"
	"math/rand"
	"testing"
)

// codec is the method set shared by encoding/base64, encoding/base32 and
// simdenc's encodings, so one set of checks covers all of them.
type codec interface {
	EncodedLen(n int) int
	Encode(dst, src []byte)
	DecodedLen(n int) int
	Decode(dst, src []byte) (int, error)
}

// codecCase pairs a simdenc encoding with the standard library encoding it
// must match exactly.
type codecCase struct {
	name      string
	got, want codec
}

// sameDecode fails unless both codecs agree on the count, the error and the n
// decoded bytes. Bytes past n are unspecified.
func (c codecCase) sameDecode(t *testing.T, src []byte) {
	t.Helper()
	want := make([]byte, c.want.DecodedLen(len(src)))
	got := make([]byte, len(want))
	wn, werr := c.want.Decode(want, src)
	gn, gerr := c.got.Decode(got, src)
	if wn != gn || werr != gerr || !bytes.Equal(want[:wn], got[:gn]) {
		t.Fatalf("%s decode(%q): got n=%d err=%v, want n=%d err=%v", c.name, src, gn, gerr, wn, werr)
	}
}

func (c codecCase) encode(src []byte) []byte {
	dst := make([]byte, c.want.EncodedLen(len(src)))
	c.want.Encode(dst, src)
	return dst
}

// malformedTails are endings that exercise padding rules and line breaks.
var malformedTails = []string{"", "=", "==", "===", "====", "======", "========", "A=", "AA=", "QQ=", "QQ====",
	"QUJDRA=", "QUJDRA==", "QQ=A", "\n", "\r\n", "QUJD\nRA==", "QQ==\n", "QQ==QQ=="}

// checkCodecs runs every differential check on every case, under every CPU
// tier: encode at many lengths, every prefix of a valid encoding, every byte
// value at every position of an encoding of n bytes, and malformed endings
// after prefixes of several lengths.
func checkCodecs(t *testing.T, cases []codecCase, n int) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(1))
		for _, c := range cases {
			sizes := []int{4095, 4096, 4097, 65543}
			for size := range 300 {
				sizes = append(sizes, size)
			}
			for _, size := range sizes {
				src := rndBytes(r, size)
				got := make([]byte, len(c.encode(src)))
				c.got.Encode(got, src)
				if !bytes.Equal(got, c.encode(src)) {
					t.Fatalf("%s encode mismatch at n=%d", c.name, size)
				}
			}
			for size := range 200 {
				full := c.encode(rndBytes(r, size))
				for end := range len(full) + 1 {
					c.sameDecode(t, full[:end])
				}
			}
			base := c.encode(rndBytes(r, n))
			for pos := range base {
				for b := range 256 {
					src := bytes.Clone(base)
					src[pos] = byte(b)
					c.sameDecode(t, src)
				}
			}
			for _, tail := range malformedTails {
				for _, prefix := range []int{0, 12, 100} {
					c.sameDecode(t, append(c.encode(rndBytes(r, prefix)), tail...))
				}
			}
		}
	})
}

// fuzzCodecs fuzzes Decode on arbitrary input against the standard library.
func fuzzCodecs(f *testing.F, cases []codecCase) {
	f.Add([]byte("QQ="))
	f.Add([]byte("QUJDRA=="))
	f.Add(bytes.Repeat([]byte("QUJD"), 40))
	f.Fuzz(func(t *testing.T, src []byte) {
		eachTier(t, func(t *testing.T) {
			for _, c := range cases {
				c.sameDecode(t, src)
			}
		})
	})
}

func rndBytes(r *rand.Rand, n int) []byte {
	b := make([]byte, n)
	r.Read(b)
	return b
}
