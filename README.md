# simdenc [![Go Reference](https://pkg.go.dev/badge/github.com/dans-stuff/simdenc.svg)](https://pkg.go.dev/github.com/dans-stuff/simdenc) [![Go Report Card](https://goreportcard.com/badge/github.com/dans-stuff/simdenc)](https://goreportcard.com/report/github.com/dans-stuff/simdenc)

SIMD-accelerated encoding for Go. Currently supports **base64**.

- 🚀 **Up to 25x faster** base64 encoding than stdlib, **35 GB/s** on AMD EPYC Zen 4
- 🔧 **Pure Go** via `simd/archsimd` - no assembly files, no CGo, drop-in `encoding/base64` API

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

On the fastest hardware, a single encode call chains all three: 512-bit bulk (48 bytes/iteration), 256-bit cleanup (24 bytes/iteration), then scalar for the last few bytes.

## Things I learned

I built this to see what Go 1.26's new SIMD intrinsics can actually do. Base64 is a well-studied problem with decades of SIMD research, so it's a good test. Every optimization was A/B tested on EPYC. Here's what stood out:

**One instruction changed everything.** There's an AVX-512 instruction called VPERMB that can rearrange any of 64 bytes into any order in a single step. Base64 encoding needs to convert 6-bit values into ASCII characters, which normally takes 4 separate operations. With VPERMB you just build a 64-entry lookup table and do it in one shot. That single change is where most of the AVX-512 speedup comes from.

**Wider isn't always better.** Going from 256-bit to 512-bit vectors made encoding 55% faster, but decoding actually got 12% *slower*. Decoding has a longer chain of steps that depend on each other (validate, translate, pack, compact), and on Zen 4 each 512-bit step takes twice as long as its 256-bit equivalent. More width, same serial bottleneck.

**Mixing vector widths in a single call was a big win.** For encoding, we run the 512-bit path on the bulk of the data, then switch to 256-bit for the leftover 24-47 bytes instead of falling back to slow byte-at-a-time scalar code. That gave +27% at 1 KB versus using either width alone.

**Go's compiler has a quirk with global variables in loops.** Copying a global into a local variable before a hot loop gave +37%. The compiler can't prove that a global isn't modified by other goroutines or function calls mid-loop, so it reloads it from memory every single iteration. A local copy lets it stay in a register.

**`unsafe.Pointer` loads gave exactly 0% improvement.** We tried bypassing the safe `LoadSlice` API with raw pointer arithmetic. No difference at all. The compiler already eliminates bounds checks when you give it explicit slice bounds like `src[i:i+32]`.

**Rosetta 2 is weird.** Byte-identical x86 machine code in the same binary can run at completely different speeds under Rosetta, depending on where the function lands in memory. We confirmed this by copying a competitor's assembly function verbatim into our binary: the original ran at 6.3 GB/s, our copy ran at 2.2 GB/s, same instructions. Not actionable, just interesting.

See [RESEARCH.md](RESEARCH.md) for the full experiment log with all the numbers.

## The catch

`simd/archsimd` is still behind `GOEXPERIMENT=simd` and everything here depends on an unreleased version of Go (1.26rc1). This is an experiment. Treat it accordingly.

```bash
GOEXPERIMENT=simd go test -v ./...
GOEXPERIMENT=simd go test -bench Benchmark -benchtime 1s -count 3 -run XXX ./...
```

## API

Same as `encoding/base64`: `StdEncoding`, `RawStdEncoding`, `URLEncoding`, `RawURLEncoding`, plus `Encode`, `Decode`, `EncodeToString`, `DecodeString`, `AppendEncode`, `AppendDecode`, `EncodedLen`, `DecodedLen`, `WithPadding`.

## License

[MIT](LICENSE)
