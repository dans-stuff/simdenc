// Package simdenc implements text encodings with SIMD acceleration written
// in pure Go using simd/archsimd (build with GOEXPERIMENT=simd on amd64).
//
// The base64 API mirrors encoding/base64. On platforms or CPUs without a
// supported SIMD tier, all operations delegate to the standard library or
// to a scalar reference implementation with identical results.
package simdenc
