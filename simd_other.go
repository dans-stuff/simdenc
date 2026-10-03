//go:build !(amd64 && goexperiment.simd)

package simdenc

// Without SIMD no blocks are consumed and the reference code does all the work.

func encodeBlocks(alphabet uint8, dst, src []byte) int { return 0 }
func decodeBlocks(alphabet uint8, dst, src []byte) int { return 0 }
