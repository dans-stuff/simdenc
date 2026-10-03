//go:build goexperiment.simd

package simdenc

import "testing"

// tier is one CPU tier to run a test under.
type tier struct {
	name string
	set  func() (restore func())
}

// lowerTo lowers the CPU flags and returns a function that restores them.
func lowerTo(avx2, avx512 bool) func() {
	a2, a512, v2 := hasAVX2, hasAVX512, hasVBMI2
	hasAVX2, hasAVX512, hasVBMI2 = avx2 && hasAVX2, avx512 && hasAVX512, avx512 && hasVBMI2
	return func() { hasAVX2, hasAVX512, hasVBMI2 = a2, a512, v2 }
}

// tiers lists every tier this CPU can run: AVX-512, AVX2 without AVX-512, and
// no SIMD at all (the reference code alone).
func tiers(t testing.TB) []tier {
	t.Helper()
	var out []tier
	if hasAVX512 {
		out = append(out, tier{"avx512", func() func() { return lowerTo(true, true) }})
	}
	if hasAVX2 {
		out = append(out, tier{"avx2", func() func() { return lowerTo(true, false) }})
	}
	return append(out, tier{"reference", func() func() { return lowerTo(false, false) }})
}
