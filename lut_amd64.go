//go:build goexperiment.simd

package simdenc

// decodeLUT builds the 128-entry table that VPERMI2B uses to validate and
// translate ASCII in one step. In each spelling, character i has value i;
// every other byte maps to 0x80, so OR-ing the result with the input marks
// anything that is not a digit or is not ASCII. It returns the entries for
// ASCII 0-63 and 64-127.
func decodeLUT(spellings ...string) (lo, hi [64]byte) {
	for i := range 64 {
		lo[i], hi[i] = 0x80, 0x80
	}
	for _, s := range spellings {
		for value := range len(s) {
			if c := s[value]; c < 64 {
				lo[c] = byte(value)
			} else {
				hi[c-64] = byte(value)
			}
		}
	}
	return lo, hi
}
