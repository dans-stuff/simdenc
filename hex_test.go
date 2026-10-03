package simdenc

import (
	"bytes"
	"encoding/hex"
	"errors"
	"math/rand"
	"testing"
)

// sameDecode reports whether HexDecode agrees with hex.Decode on src in
// count, error and the entire destination buffer.
func sameHexDecode(t *testing.T, src []byte) {
	t.Helper()
	want := make([]byte, len(src)/2)
	got := make([]byte, len(src)/2)
	wn, werr := hex.Decode(want, src)
	gn, gerr := HexDecode(got, src)
	if wn != gn || !sameHexErr(werr, gerr) || !bytes.Equal(want, got) {
		t.Fatalf("decode(%q): got n=%d err=%v, want n=%d err=%v (dst equal: %v)",
			src, gn, gerr, wn, werr, bytes.Equal(want, got))
	}
}

func sameHexErr(a, b error) bool {
	if a == nil || b == nil {
		return a == b
	}
	var ia, ib hex.InvalidByteError
	if errors.As(a, &ia) {
		return errors.As(b, &ib) && ia == ib
	}
	return errors.Is(a, b)
}

func TestHexEncode(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(1))
		sizes := []int{4095, 4096, 4097, 65543}
		for n := range 300 {
			sizes = append(sizes, n)
		}
		for _, n := range sizes {
			src := rndBytes(r, n)
			want := make([]byte, n*2)
			got := make([]byte, n*2)
			hex.Encode(want, src)
			if HexEncode(got, src) != n*2 || !bytes.Equal(want, got) {
				t.Fatalf("encode mismatch at n=%d", n)
			}
			if string(AppendHexEncode([]byte("x"), src)) != "x"+string(want) {
				t.Fatalf("AppendHexEncode mismatch at n=%d", n)
			}
		}
	})
}

func TestHexDecodeValid(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(2))
		for n := range 300 {
			src := []byte(hex.EncodeToString(rndBytes(r, n)))
			sameHexDecode(t, src)
			sameHexDecode(t, bytes.ToUpper(src))
			sameHexDecode(t, src[:len(src)/2]) // odd or even prefix of valid text
		}
		sameHexDecode(t, []byte(hex.EncodeToString(rndBytes(r, 65537))))
	})
}

// Every byte value in every position relative to the 32/64/128-char blocks.
func TestHexDecodeBadByteEverywhere(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(3))
		base := []byte(hex.EncodeToString(rndBytes(r, 150)))
		for pos := range base {
			for b := range 256 {
				src := bytes.Clone(base)
				src[pos] = byte(b)
				sameHexDecode(t, src)
			}
		}
	})
}

func TestHexDecodeLengths(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(4))
		bad := []byte("gG/:@`\x00\x80\xff ")
		for n := range 330 {
			valid := []byte(hex.EncodeToString(rndBytes(r, n/2+1)))[:n]
			sameHexDecode(t, valid)
			for _, b := range bad {
				for _, pos := range []int{0, n / 2, n - 1} {
					if pos >= 0 && pos < n {
						src := bytes.Clone(valid)
						src[pos] = b
						sameHexDecode(t, src)
					}
				}
			}
		}
	})
}

func TestHexDecodePairs(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		for i := range 1 << 16 {
			sameHexDecode(t, []byte{byte(i), byte(i >> 8)})
		}
	})
}

func TestHexStringsAndAppend(t *testing.T) {
	eachTier(t, func(t *testing.T) {
		r := rand.New(rand.NewSource(5))
		src := rndBytes(r, 1000)
		s := HexEncodeToString(src)
		if s != hex.EncodeToString(src) {
			t.Fatal("HexEncodeToString mismatch")
		}
		got, err := HexDecodeString(s)
		if err != nil || !bytes.Equal(got, src) {
			t.Fatalf("HexDecodeString roundtrip failed: %v", err)
		}
		got, err = AppendHexDecode([]byte("pre"), []byte(s))
		if err != nil || string(got[:3]) != "pre" || !bytes.Equal(got[3:], src) {
			t.Fatalf("AppendHexDecode failed: %v", err)
		}
		wantB, wantErr := hex.DecodeString(s[:400] + "zz" + s[402:])
		gotB, gotErr := HexDecodeString(s[:400] + "zz" + s[402:])
		if !bytes.Equal(wantB, gotB) || !sameHexErr(wantErr, gotErr) {
			t.Fatalf("HexDecodeString error path: got %d bytes, %v; want %d bytes, %v", len(gotB), gotErr, len(wantB), wantErr)
		}
		ab, aerr := AppendHexDecode([]byte("pre"), []byte(s[:400]+"zz"+s[402:]))
		wantA, _ := hex.AppendDecode([]byte("pre"), []byte(s[:400]+"zz"+s[402:]))
		if aerr == nil || !bytes.Equal(ab, wantA) {
			t.Fatalf("AppendHexDecode error path mismatch: %v", aerr)
		}
		if HexEncodedLen(7) != 14 || HexDecodedLen(15) != 7 {
			t.Fatal("length helpers")
		}
	})
}

func FuzzHexDecode(f *testing.F) {
	f.Add([]byte(""))
	f.Add([]byte("00ff"))
	f.Add(bytes.Repeat([]byte("0123456789abcdefABCDEF"), 20))
	f.Add(append(bytes.Repeat([]byte("ab"), 100), 'g'))
	f.Fuzz(func(t *testing.T, src []byte) {
		eachTier(t, func(t *testing.T) { sameHexDecode(t, src) })
	})
}

func FuzzHexEncode(f *testing.F) {
	f.Add([]byte(""))
	f.Add(bytes.Repeat([]byte{0, 0xff, 0x5a}, 50))
	f.Fuzz(func(t *testing.T, src []byte) {
		eachTier(t, func(t *testing.T) {
			if got := HexEncodeToString(src); got != hex.EncodeToString(src) {
				t.Fatalf("encode mismatch for %x", src)
			}
		})
	})
}
