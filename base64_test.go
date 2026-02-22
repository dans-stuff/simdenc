//go:build goexperiment.simd

package simdenc

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"testing"

	emmansun "github.com/emmansun/base64"
)

// --- Correctness ---

func TestRoundtrip(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 12, 13, 15, 16, 24, 36, 48, 100, 1000, 4096} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			src := randbytes(t, n)
			encoded := RawStdEncoding.EncodeToString(src)
			want := base64.RawStdEncoding.EncodeToString(src)
			if encoded != want {
				t.Fatalf("encode mismatch:\n  got:  %q\n  want: %q", encoded, want)
			}
			decoded, err := RawStdEncoding.DecodeString(encoded)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if !bytes.Equal(decoded, src) {
				t.Fatal("roundtrip mismatch")
			}
		})
	}
}

func TestPaddedRoundtrip(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 12, 24, 100} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			src := randbytes(t, n)
			encoded := StdEncoding.EncodeToString(src)
			want := base64.StdEncoding.EncodeToString(src)
			if encoded != want {
				t.Fatalf("padded encode mismatch:\n  got:  %q\n  want: %q", encoded, want)
			}
			decoded, err := StdEncoding.DecodeString(encoded)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if !bytes.Equal(decoded, src) {
				t.Fatal("padded roundtrip mismatch")
			}
		})
	}
}

func TestDecodeInvalid(t *testing.T) {
	_, err := RawStdEncoding.DecodeString("!!!")
	if err == nil {
		t.Fatal("expected error for invalid input")
	}
}

func TestDecodeAgainstStdlib(t *testing.T) {
	for _, n := range []int{16, 32, 64, 128, 256, 1024} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			raw := randbytes(t, n)
			encoded := base64.RawStdEncoding.EncodeToString(raw)
			decoded, err := RawStdEncoding.DecodeString(encoded)
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if !bytes.Equal(decoded, raw) {
				t.Fatalf("mismatch at size %d", n)
			}
		})
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
		// Raw (no padding)
		gotRaw := RawStdEncoding.EncodeToString(data)
		wantRaw := base64.RawStdEncoding.EncodeToString(data)
		if gotRaw != wantRaw {
			t.Fatalf("RawStdEncoding mismatch for len %d:\n  got:  %q\n  want: %q", len(data), gotRaw, wantRaw)
		}

		// Padded
		gotPad := StdEncoding.EncodeToString(data)
		wantPad := base64.StdEncoding.EncodeToString(data)
		if gotPad != wantPad {
			t.Fatalf("StdEncoding mismatch for len %d:\n  got:  %q\n  want: %q", len(data), gotPad, wantPad)
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
		// 1. Encode with stdlib, decode with both, compare.
		encodedRaw := base64.RawStdEncoding.EncodeToString(data)
		gotRaw, err := RawStdEncoding.DecodeString(encodedRaw)
		if err != nil {
			t.Fatalf("RawStdEncoding decode of valid input failed: %v", err)
		}
		if !bytes.Equal(gotRaw, data) {
			t.Fatalf("RawStdEncoding decode mismatch for len %d", len(data))
		}

		encodedPad := base64.StdEncoding.EncodeToString(data)
		gotPad, err := StdEncoding.DecodeString(encodedPad)
		if err != nil {
			t.Fatalf("StdEncoding decode of valid input failed: %v", err)
		}
		if !bytes.Equal(gotPad, data) {
			t.Fatalf("StdEncoding decode mismatch for len %d", len(data))
		}

		// 2. Feed raw random bytes as base64 input; simdenc must not panic.
		//    If stdlib returns an error, simdenc should too.
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
// Three implementations compared at each size:
//   simd     — production path (best available SIMD tier, or stdlib fallback)
//   emmansun — github.com/emmansun/base64 (AVX2, hand-tuned asm)
//   stdlib   — encoding/base64

func BenchmarkEncode(b *testing.B) {
	em := emmansun.StdEncoding.WithPadding(emmansun.NoPadding)
	for _, size := range []int{100, 1000, 10000, 65536} {
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
	for _, size := range []int{100, 1000, 10000, 65536} {
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
		b.Run(fmt.Sprintf("stdlib/%d", size), func(b *testing.B) {
			dst := make([]byte, dn)
			b.SetBytes(int64(len(enc)))
			for b.Loop() {
				base64.RawStdEncoding.Decode(dst, enc)
			}
		})
	}
}
