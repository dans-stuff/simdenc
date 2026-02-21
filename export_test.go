//go:build goexperiment.simd

package b64simd

// EncodeAVX2Direct calls encodeAVX2 directly.
func EncodeAVX2Direct(dst, src []byte) (int, int) {
	return encodeAVX2(dst, src)
}
