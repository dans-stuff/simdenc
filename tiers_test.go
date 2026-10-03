package simdenc

import "testing"

// eachTier runs f once per CPU tier available (see tiers).
func eachTier(t *testing.T, f func(t *testing.T)) {
	for _, tr := range tiers(t) {
		t.Run(tr.name, func(t *testing.T) {
			defer tr.set()()
			f(t)
		})
	}
}
