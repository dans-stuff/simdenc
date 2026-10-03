//go:build !(amd64 && goexperiment.simd)

package simdenc

import "testing"

type tier struct {
	name string
	set  func() (restore func())
}

// tiers has only the reference path where no SIMD tier exists.
func tiers(t testing.TB) []tier {
	return []tier{{"reference", func() func() { return func() {} }}}
}
