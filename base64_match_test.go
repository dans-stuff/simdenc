package simdenc

import (
	"encoding/base64"
	"testing"
)

// base64Cases pairs each simdenc encoding, including custom padding, with the
// standard library encoding it must match exactly.
var base64Cases = []codecCase{
	{"Std", StdEncoding, base64.StdEncoding},
	{"URL", URLEncoding, base64.URLEncoding},
	{"RawStd", RawStdEncoding, base64.RawStdEncoding},
	{"RawURL", RawURLEncoding, base64.RawURLEncoding},
	{"Std/pad#", StdEncoding.WithPadding('#'), base64.StdEncoding.WithPadding('#')},
}

func TestBase64MatchesStdlib(t *testing.T) {
	checkCodecs(t, base64Cases, 110)
}

func FuzzBase64MatchesStdlib(f *testing.F) {
	fuzzCodecs(f, base64Cases)
}
