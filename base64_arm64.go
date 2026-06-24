//go:build arm64

package simdenc

//go:noescape
func encodeBlocksNEON(dst, src *byte, nblocks uint64, alpha *byte)

//go:noescape
func decodeBlocksNEON(dst, src *byte, nblocks uint64, tableA, tableB *byte) uint64

var (
	neonEncAlpha [2][64]byte
	neonDecA     [2][64]byte
	neonDecB     [2][64]byte
)

func init() {
	for a := range 2 {
		copy(neonEncAlpha[a][:], encAlphabets[a])
		for i := range neonDecA[a] {
			neonDecA[a][i] = 0xFF
		}
		for i := range neonDecB[a] {
			neonDecB[a][i] = 0xFF
		}
		for i := 0; i < 64; i++ {
			ch := encAlphabets[a][i]
			if ch < 64 {
				neonDecA[a][ch] = byte(i)
			} else {
				neonDecB[a][ch-64] = byte(i)
			}
		}
	}

	simdEncode = neonEncode
	simdDecode = neonDecode
}

func neonEncode(alphabet uint8, dst, src []byte) int {
	nblocks := len(src) / 48
	if nblocks == 0 {
		return 0
	}
	encodeBlocksNEON(&dst[0], &src[0], uint64(nblocks), &neonEncAlpha[alphabet][0])
	return nblocks * 48
}

func neonDecode(alphabet uint8, dst, src []byte) (int, int) {
	n := len(src)
	const blkIn = 64
	bulk := 0
	if n > blkIn {
		bulk = ((n - blkIn) / blkIn) * blkIn
	}
	if bulk == 0 {
		return 0, 0
	}
	nblocks := uint64(bulk / blkIn)
	st := decodeBlocksNEON(&dst[0], &src[0], nblocks,
		&neonDecA[alphabet][0], &neonDecB[alphabet][0])
	if st != 0 {
		return 0, 0
	}
	di := bulk / 4 * 3
	return di, bulk
}
