//go:build goexperiment.simd

package simdenc

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"testing"

	cristalhq "github.com/cristalhq/base64"
	emmansun "github.com/emmansun/base64"
)

// --- Correctness ---

// encodingPair pairs a simdenc encoding with its stdlib equivalent.
type encodingPair struct {
	name string
	enc  *Encoding
	std  *base64.Encoding
}

var allEncodings = []encodingPair{
	{"RawStd", RawStdEncoding, base64.RawStdEncoding},
	{"Std", StdEncoding, base64.StdEncoding},
	{"RawURL", RawURLEncoding, base64.RawURLEncoding},
	{"URL", URLEncoding, base64.URLEncoding},
}

func TestRoundtrip(t *testing.T) {
	sizes := []int{0, 1, 2, 3, 4, 12, 13, 15, 16, 24, 36, 48, 100, 1000, 4096}
	for _, e := range allEncodings {
		for _, n := range sizes {
			t.Run(fmt.Sprintf("%s/n=%d", e.name, n), func(t *testing.T) {
				src := randbytes(t, n)
				encoded := e.enc.EncodeToString(src)
				want := e.std.EncodeToString(src)
				if encoded != want {
					t.Fatalf("encode mismatch:\n  got:  %q\n  want: %q", encoded, want)
				}
				decoded, err := e.enc.DecodeString(encoded)
				if err != nil {
					t.Fatalf("decode: %v", err)
				}
				if !bytes.Equal(decoded, src) {
					t.Fatal("roundtrip mismatch")
				}
			})
		}
	}
}

func TestDecodeInvalid(t *testing.T) {
	_, err := RawStdEncoding.DecodeString("!!!")
	if err == nil {
		t.Fatal("expected error for invalid input")
	}
}

func TestEncodedLen(t *testing.T) {
	for n := 0; n < 100; n++ {
		got := RawStdEncoding.EncodedLen(n)
		want := base64.RawStdEncoding.EncodedLen(n)
		if got != want {
			t.Errorf("RawStdEncoding.EncodedLen(%d) = %d, want %d", n, got, want)
		}
		got = StdEncoding.EncodedLen(n)
		want = base64.StdEncoding.EncodedLen(n)
		if got != want {
			t.Errorf("StdEncoding.EncodedLen(%d) = %d, want %d", n, got, want)
		}
	}
}

func TestDecodedLen(t *testing.T) {
	for n := 0; n < 100; n++ {
		got := RawStdEncoding.DecodedLen(n)
		want := base64.RawStdEncoding.DecodedLen(n)
		if got != want {
			t.Errorf("RawStdEncoding.DecodedLen(%d) = %d, want %d", n, got, want)
		}
		got = StdEncoding.DecodedLen(n)
		want = base64.StdEncoding.DecodedLen(n)
		if got != want {
			t.Errorf("StdEncoding.DecodedLen(%d) = %d, want %d", n, got, want)
		}
	}
}

func TestAppendEncode(t *testing.T) {
	src := randbytes(t, 100)
	prefix := []byte("prefix:")
	got := RawStdEncoding.AppendEncode(prefix, src)
	want := base64.RawStdEncoding.EncodeToString(src)
	if string(got) != "prefix:"+want {
		t.Fatalf("AppendEncode mismatch")
	}
}

func TestWithPadding(t *testing.T) {
	src := randbytes(t, 100)
	enc := StdEncoding.WithPadding('#')
	std := base64.StdEncoding.WithPadding('#')
	encoded := enc.EncodeToString(src)
	want := std.EncodeToString(src)
	if encoded != want {
		t.Fatalf("WithPadding encode mismatch:\n  got:  %q\n  want: %q", encoded, want)
	}
	decoded, err := enc.DecodeString(encoded)
	if err != nil {
		t.Fatalf("WithPadding decode: %v", err)
	}
	if !bytes.Equal(decoded, src) {
		t.Fatal("WithPadding roundtrip mismatch")
	}
}

func TestAppendDecode(t *testing.T) {
	src := randbytes(t, 100)
	encoded := base64.RawStdEncoding.EncodeToString(src)
	prefix := []byte("prefix:")
	got, err := RawStdEncoding.AppendDecode(prefix, []byte(encoded))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got[:7], prefix) {
		t.Fatalf("prefix lost")
	}
	if !bytes.Equal(got[7:], src) {
		t.Fatalf("AppendDecode mismatch")
	}
}

func randbytes(t *testing.T, n int) []byte {
	t.Helper()
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		t.Fatal(err)
	}
	return b
}

// --- Fuzz ---

func FuzzEncode(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0})
	f.Add([]byte{0, 1, 2})
	f.Add(randfuzz(64))
	f.Add(randfuzz(1000))

	f.Fuzz(func(t *testing.T, data []byte) {
		for _, e := range allEncodings {
			got := e.enc.EncodeToString(data)
			want := e.std.EncodeToString(data)
			if got != want {
				t.Fatalf("%s mismatch for len %d:\n  got:  %q\n  want: %q", e.name, len(data), got, want)
			}
		}
	})
}

func FuzzDecode(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{0})
	f.Add([]byte{0, 1, 2})
	f.Add(randfuzz(64))
	f.Add(randfuzz(1000))

	f.Fuzz(func(t *testing.T, data []byte) {
		for _, e := range allEncodings {
			encoded := e.std.EncodeToString(data)
			got, err := e.enc.DecodeString(encoded)
			if err != nil {
				t.Fatalf("%s decode of valid input failed: %v", e.name, err)
			}
			if !bytes.Equal(got, data) {
				t.Fatalf("%s decode mismatch for len %d", e.name, len(data))
			}
		}

		// Feed raw random bytes as base64 input; simdenc must not panic.
		_, stdErr := base64.RawStdEncoding.DecodeString(string(data))
		_, simdErr := RawStdEncoding.DecodeString(string(data))
		if stdErr != nil && simdErr == nil {
			t.Fatalf("stdlib returned error but simdenc did not for raw input %q", data)
		}
	})
}

func randfuzz(n int) []byte {
	b := make([]byte, n)
	rand.Read(b)
	return b
}

// --- Benchmarks ---
//
// Four implementations compared at each size:
//   simd      — production path (best available SIMD tier, or stdlib fallback)
//   emmansun  — github.com/emmansun/base64 (AVX2, hand-tuned asm)
//   cristalhq — github.com/cristalhq/base64 (pure Go, Turbo-Base64 port)
//   stdlib    — encoding/base64

func BenchmarkEncode(b *testing.B) {
	em := emmansun.StdEncoding.WithPadding(emmansun.NoPadding)
	cr := cristalhq.RawStdEncoding
	for _, size := range []int{3, 12, 24, 48, 64, 100, 128, 256, 1000, 10000, 65536, 131072, 1000000, 2000000, 4000000, 8000000, 16000000} {
		raw := make([]byte, size)
		rand.Read(raw)
		n := RawStdEncoding.EncodedLen(size)

		b.Run(fmt.Sprintf("simd/%d", size), func(b *testing.B) {
			dst := make([]byte, n)
			b.SetBytes(int64(size))
			for b.Loop() {
				RawStdEncoding.Encode(dst, raw)
			}
		})
		b.Run(fmt.Sprintf("emmansun/%d", size), func(b *testing.B) {
			dst := make([]byte, n)
			b.SetBytes(int64(size))
			for b.Loop() {
				em.Encode(dst, raw)
			}
		})
		b.Run(fmt.Sprintf("cristalhq/%d", size), func(b *testing.B) {
			dst := make([]byte, n)
			b.SetBytes(int64(size))
			for b.Loop() {
				cr.Encode(dst, raw)
			}
		})
		b.Run(fmt.Sprintf("stdlib/%d", size), func(b *testing.B) {
			dst := make([]byte, n)
			b.SetBytes(int64(size))
			for b.Loop() {
				base64.RawStdEncoding.Encode(dst, raw)
			}
		})
	}
}

func BenchmarkDecode(b *testing.B) {
	em := emmansun.StdEncoding.WithPadding(emmansun.NoPadding)
	cr := cristalhq.RawStdEncoding
	for _, size := range []int{64, 100, 128, 256, 1000, 10000, 65536, 131072, 1000000, 2000000, 4000000, 8000000, 16000000} {
		raw := make([]byte, size)
		rand.Read(raw)
		enc := []byte(base64.RawStdEncoding.EncodeToString(raw))
		dn := RawStdEncoding.DecodedLen(len(enc))

		b.Run(fmt.Sprintf("simd/%d", size), func(b *testing.B) {
			dst := make([]byte, dn)
			b.SetBytes(int64(len(enc)))
			for b.Loop() {
				RawStdEncoding.Decode(dst, enc)
			}
		})
		b.Run(fmt.Sprintf("emmansun/%d", size), func(b *testing.B) {
			dst := make([]byte, dn)
			b.SetBytes(int64(len(enc)))
			for b.Loop() {
				em.Decode(dst, enc)
			}
		})
		b.Run(fmt.Sprintf("cristalhq/%d", size), func(b *testing.B) {
			dst := make([]byte, dn)
			b.SetBytes(int64(len(enc)))
			for b.Loop() {
				cr.Decode(dst, enc)
			}
		})
		b.Run(fmt.Sprintf("stdlib/%d", size), func(b *testing.B) {
			dst := make([]byte, dn)
			b.SetBytes(int64(len(enc)))
			for b.Loop() {
				base64.RawStdEncoding.Decode(dst, enc)
			}
		})
	}
}
