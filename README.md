# simdenc

SIMD-accelerated encoding for Go. Currently supports **base64**.

**Up to 25x faster base64 encoding than stdlib. Faster than hand-written assembly. Pure Go.**

Go 1.26 ships with [`simd/archsimd`](https://pkg.go.dev/simd/archsimd), a package that lets you write SIMD code directly in Go. No assembly files, no CGo. I wanted to see how far that could go on a real workload, so I built a base64 encoder/decoder with it.

## Performance

Measured on AMD EPYC Zen 4.

`StdEncoding.Encode` hits **35 GB/s** on large inputs. That's 25x faster than `encoding/base64` (1.4 GB/s) and 50% faster than the best hand-tuned AVX2 assembly library I could find (23 GB/s). At 1 KB it's still 22 GB/s, about 16x stdlib.

`StdEncoding.Decode` hits **27 GB/s**. 17x faster than stdlib (1.6 GB/s), and slightly ahead of the best assembly library (25 GB/s).

## Usage

```go
import "github.com/dans-stuff/simdenc"

encoded := simdenc.StdEncoding.EncodeToString(data)
decoded, err := simdenc.StdEncoding.DecodeString(encoded)
```

Same API as `encoding/base64`: `StdEncoding`, `RawStdEncoding`, `URLEncoding`, `RawURLEncoding`, plus `Encode`, `Decode`, `EncodeToString`, `DecodeString`, `AppendEncode`, `AppendDecode`, `EncodedLen`, `DecodedLen`, `WithPadding`.

## How it works

CPU features are detected at init. The best available path is selected automatically:

- **No SIMD**: delegates to `encoding/base64` (ARM, older x86, etc.)
- **AVX2**: 256-bit encode/decode covering any x86 CPU from ~2013 onward, including under Rosetta 2
- **AVX-512 + VBMI**: 512-bit encode with adaptive 256-bit tail, VPERMB-accelerated decode. Targets AMD EPYC Zen 4+, Intel Ice Lake+

On tier 2 hardware, a single encode call chains all three: 512-bit bulk (48 bytes/iteration), 256-bit VBMI for the remainder (24 bytes/iteration), then scalar for the last few bytes.

Every optimization was A/B tested on EPYC with measured deltas. See [RESEARCH.md](RESEARCH.md) for the full experiment log.

## The catch

`simd/archsimd` is still behind `GOEXPERIMENT=simd` and everything here depends on an unreleased version of Go (1.26rc1). This is an experiment. Treat it accordingly.

```bash
GOEXPERIMENT=simd go test -v ./...
GOEXPERIMENT=simd go test -bench Benchmark -benchtime 1s -count 3 -run XXX ./...
```

## Why this exists

I wanted to understand what Go's new SIMD intrinsics are capable of on a well-studied problem. Base64 has decades of SIMD research behind it, which makes it a good benchmark for the tooling. The fact that pure Go can beat hand-written assembly here is a pretty exciting signal for where `archsimd` is heading.

## License

MIT
