//go:build !(amd64 && goexperiment.simd)

package simdenc

// Without SIMD no blocks are consumed and the reference code does all the work.

func encodeBlocks(alphabet uint8, dst, src []byte) int       { return 0 }
func decodeBlocks(alphabet uint8, dst, src []byte) int       { return 0 }
func hexEncodeBlocks(dst, src []byte) int                    { return 0 }
func hexDecodeBlocks(dst, src []byte) int                    { return 0 }
func base32EncodeBlocks(alphabet uint8, dst, src []byte) int { return 0 }
func base32DecodeBlocks(alphabet uint8, dst, src []byte) int { return 0 }
