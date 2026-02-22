//go:build goexperiment.simd && amd64

package simdenc

// EncodeAVX2Direct calls encodeAVX2 directly.
func EncodeAVX2Direct(dst, src []byte) (int, int) {
	return encodeAVX2(dst, src)
}
