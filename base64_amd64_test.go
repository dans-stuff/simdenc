//go:build goexperiment.simd

package simdenc

import (
	"bytes"
	"testing"
)

func TestDecode512(t *testing.T) {
	if !hasAVX512 {
		t.Skip("no AVX-512 VBMI")
	}
	// Verify decode512 produces correct output for a deterministic input.
	src := make([]byte, 100)
	for i := range src {
		src[i] = byte(i)
	}
	encoded := []byte(RawStdEncoding.EncodeToString(src))
	dst := make([]byte, RawStdEncoding.DecodedLen(len(encoded)))
	a := &decAlphas[alphabetStd]
	si := decode512(a, dst, encoded)
	if si != 64 {
		t.Fatalf("decode512: si=%d, want si=64", si)
	}
	di := si * 3 / 4 // 48
	if !bytes.Equal(dst[:di], src[:di]) {
		t.Fatalf("decode512 output mismatch:\n  got:  %v\n  want: %v", dst[:di], src[:di])
	}
}
