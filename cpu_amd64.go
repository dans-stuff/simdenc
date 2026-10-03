//go:build goexperiment.simd

package simdenc

import "simd/archsimd"

// CPU tiers shared by every codec. Tests lower these to exercise narrower tiers.
var (
	hasAVX2   = archsimd.X86.AVX2()
	hasAVX512 = hasAVX2 && archsimd.X86.AVX512() && archsimd.X86.AVX512VBMI()
	hasVBMI2  = hasAVX512 && archsimd.X86.AVX512VBMI2()
)
