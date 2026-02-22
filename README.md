# simdenc [![Go Reference](https://pkg.go.dev/badge/github.com/dans-stuff/simdenc.svg)](https://pkg.go.dev/github.com/dans-stuff/simdenc) [![Go Report Card](https://goreportcard.com/badge/github.com/dans-stuff/simdenc)](https://goreportcard.com/report/github.com/dans-stuff/simdenc)

SIMD-accelerated encoding for Go. Currently supports **base64**.

- 🚀 **Up to 25x faster** base64 encoding than stdlib
- ⚡ **35 GB/s encode, 27 GB/s decode** on AMD EPYC Zen 4
- 🔧 **Pure Go** via `simd/archsimd` - no assembly files, no CGo
- 🔌 **Drop-in API** - same interface as `encoding/base64`

```go
import "github.com/dans-stuff/simdenc"

encoded := simdenc.StdEncoding.EncodeToString(data)
decoded, err := simdenc.StdEncoding.DecodeString(encoded)
```

## Performance

Measured on AMD EPYC Zen 4.

`StdEncoding.Encode` hits **35 GB/s** on large inputs. That's 25x faster than `encoding/base64` (1.4 GB/s) and 50% faster than the best hand-tuned AVX2 assembly library I could find (23 GB/s). At 1 KB it's still 22 GB/s, about 16x stdlib.

`StdEncoding.Decode` hits **27 GB/s**. 17x faster than stdlib (1.6 GB/s), and slightly ahead of the best assembly library (25 GB/s).

## How it works

CPU features are detected at init. The best available path is selected automatically:

- **No SIMD**: delegates to `encoding/base64` (ARM, older x86, etc.)
- **AVX2**: 256-bit encode/decode covering any x86 CPU from ~2013 onward, including under Rosetta 2
- **AVX-512 + VBMI**: 512-bit encode with adaptive 256-bit tail, VPERMB-accelerated decode. Targets AMD EPYC Zen 4+, Intel Ice Lake+

On tier 2 hardware, a single encode call chains all three: 512-bit bulk (48 bytes/iteration), 256-bit VBMI for the remainder (24 bytes/iteration), then scalar for the last few bytes.

## SIMD takeaways

Every optimization was A/B tested on EPYC with measured deltas. Some things we learned:

- **VPERMB is the star instruction.** A single VPERMB replaces the entire 4-instruction sextet-to-ASCII mapping with a 64-entry LUT. It also eliminates the cross-lane shuffle workarounds that AVX2 requires. This one instruction is responsible for most of the AVX-512 speedup.
- **512-bit encode works, 512-bit decode doesn't.** Encode got +55% from going 512-bit wide. Decode actually regressed 12% because the serial dependency chain (validate, translate, pack, compact) gets double-pumped on Zen 4 and the wider vectors don't help.
- **Adaptive width chaining matters.** Running 512-bit bulk followed by 256-bit VBMI cleanup gave +27% at 1 KB versus either width alone. Small inputs benefit more from fewer setup instructions; large inputs benefit from raw width.
- **Local copies of globals gave +37%.** The Go compiler can't prove that globals aren't modified by function calls in a loop, so it reloads them every iteration. Copying to a local variable before the loop lets the compiler keep them in registers.
- **Unsafe pointer loads measured 0%.** We tried bypassing `LoadSlice` with unsafe pointers. No measurable difference. The compiler already optimizes slice loads well when bounds are explicit.

See [RESEARCH.md](RESEARCH.md) for the full experiment log.

## The catch

`simd/archsimd` is still behind `GOEXPERIMENT=simd` and everything here depends on an unreleased version of Go (1.26rc1). This is an experiment. Treat it accordingly.

```bash
GOEXPERIMENT=simd go test -v ./...
GOEXPERIMENT=simd go test -bench Benchmark -benchtime 1s -count 3 -run XXX ./...
```

## API

Same as `encoding/base64`: `StdEncoding`, `RawStdEncoding`, `URLEncoding`, `RawURLEncoding`, plus `Encode`, `Decode`, `EncodeToString`, `DecodeString`, `AppendEncode`, `AppendDecode`, `EncodedLen`, `DecodedLen`, `WithPadding`.

## License

MIT
