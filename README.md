# simdenc [![Go Reference](https://pkg.go.dev/badge/github.com/dans-stuff/simdenc.svg)](https://pkg.go.dev/github.com/dans-stuff/simdenc) [![Go Report Card](https://goreportcard.com/badge/github.com/dans-stuff/simdenc)](https://goreportcard.com/report/github.com/dans-stuff/simdenc)

SIMD-accelerated encoding for Go. Currently supports **base64**.

- **Up to 25x faster** base64 encoding than stdlib, **35 GB/s** on AMD EPYC Zen 4
- **Pure Go** via `simd/archsimd` — no assembly files, no CGo, drop-in `encoding/base64` API
- **All encodings**: `StdEncoding`, `RawStdEncoding`, `URLEncoding`, `RawURLEncoding` — all SIMD-accelerated

```go
import "github.com/dans-stuff/simdenc"

encoded := simdenc.StdEncoding.EncodeToString(data)
decoded, err := simdenc.StdEncoding.DecodeString(encoded)
```

## Performance

Measured on AMD EPYC 9R14 (Zen 4), single core. Compared against every fast Go base64 library and two C/C++ libraries.

**Encode (GB/s):**

| Size | simdenc | emmansun | cristalhq | stdlib | aklomp (C) | simdutf (C++) |
|------|---------|----------|-----------|--------|------------|---------------|
| 100 B | 2.6 | 6.1 | 2.3 | 1.3 | 1.9 | 3.5 |
| 1 KB | 11.2 | 17.8 | 1.3 | 1.4 | 17.0 | 20.3 |
| 10 KB | 17.8 | 23.6 | 1.3 | 1.4 | 46.6 | 26.9 |
| 64 KB | 17.9 | 22.0 | 1.3 | 1.2 | 39.1 | 20.6 |

**Decode (GB/s):**

| Size | simdenc | emmansun | cristalhq | stdlib | aklomp (C) | simdutf (C++) |
|------|---------|----------|-----------|--------|------------|---------------|
| 100 B | 2.5 | 3.5 | 1.4 | 1.4 | 3.7 | 1.8 |
| 1 KB | 10.0 | 16.3 | 1.7 | 1.5 | 16.6 | 12.9 |
| 10 KB | 12.6 | 24.4 | 1.6 | 1.5 | 22.9 | 21.5 |
| 64 KB | 11.4 | 24.1 | 1.7 | 1.6 | 22.0 | 21.4 |

At 1 KB+, simdenc is significantly faster than stdlib and cristalhq for both encode and decode. The one library that beats us at large encode is [aklomp/base64](https://github.com/aklomp/base64) (C), which uses VPMULTISHIFTQB — an AVX-512 instruction not yet exposed by Go's `archsimd` package.

See [RESEARCH.md](RESEARCH.md) for the full experiment log.

## How it works

CPU features are detected at init. The best available path is selected automatically:

- **No SIMD**: delegates to `encoding/base64` (ARM, older x86, etc.)
- **AVX2**: 256-bit encode/decode covering any x86 CPU from ~2013 onward, including under Rosetta 2
- **AVX-512 + VBMI**: 512-bit encode with adaptive 256-bit tail, VPERMB-accelerated decode. Targets AMD EPYC Zen 4+, Intel Ice Lake+

On the fastest hardware, a single encode call chains two dedicated functions: 512-bit bulk (48 bytes/iteration), then 256-bit cleanup (24 bytes/iteration), then scalar for the last few bytes. Each function hoists only its own constants into local variables, keeping register pressure low.

## Things I learned

I built this to see what Go 1.26's new SIMD intrinsics can actually do. Base64 is a well-studied problem with decades of SIMD research, so it's a good test. Every optimization was A/B tested on EPYC. Here's what stood out:

**One instruction changed everything.** There's an AVX-512 instruction called VPERMB that can rearrange any of 64 bytes into any order in a single step. Base64 encoding needs to convert 6-bit values into ASCII characters, which normally takes 4 separate operations. With VPERMB you just build a 64-entry lookup table and do it in one shot. That single change is where most of the AVX-512 speedup comes from.

**Wider isn't always better.** Going from 256-bit to 512-bit vectors made encoding 55% faster, but decoding actually got 12% *slower*. Decoding has a longer chain of steps that depend on each other (validate, translate, pack, compact), and on Zen 4 each 512-bit step takes twice as long as its 256-bit equivalent. More width, same serial bottleneck.

**Mixing vector widths in a single call was a big win.** For encoding, we run the 512-bit path on the bulk of the data, then switch to 256-bit for the leftover 24-47 bytes instead of falling back to slow byte-at-a-time scalar code. The 256-bit stage reuses 4 bytes of overlap from the previous stage's output, eliminating the need for a separate scalar preamble. That gave +27% at 1 KB versus using either width alone.

**Go has no Loop Invariant Code Motion (LICM).** Global variables are reloaded from memory on every single loop iteration — even when there are no stores or function calls in the loop body. This isn't about aliasing or goroutines; Go's compiler simply doesn't implement this optimization ([tracked in golang/go#63670](https://github.com/golang/go/issues/63670)). The workaround is copying each global into a local variable before the loop (`v := globalVec`). The compiler treats locals as register-eligible, so they stay in registers across iterations. This gave +37% for encode and +32% for decode.

**Closures kill SIMD intrinsic inlining.** We tried a closure approach where a factory function captures all constants and returns the inner decode/encode function. The captured values were correctly hoisted (loaded once, not reloaded per iteration). But Go's compiler refuses to inline SIMD intrinsics inside closure bodies — `LoadUint8x32Slice` and `StoreSlice` become real `CALL` instructions instead of inline VMOVDQU. This caused a 7-8x slowdown, capping throughput at ~6 GB/s regardless of input size. This happens even when the closure is called directly from a local variable (not through a function pointer), so it's a fundamental compiler limitation, not a devirtualization issue.

**By-value struct arguments work as implicit hoisting — but only for small structs.** Passing a struct of constants by value copies them onto the callee's stack, which acts as a hoist. This is equivalent to individual `v := global` copies but more self-documenting. However, when the struct exceeds ~200 bytes (7+ YMM vectors), the copy overhead and register pressure cause regressions. A monolithic function that combines all tiers (512-bit + 256-bit + scalar) with all constants in scope regressed 52% at 1 KB compared to separate per-tier functions.

**Keeping per-tier functions small was critical.** Each SIMD tier (AVX-512, VBMI 256-bit, AVX2) is its own function that hoists only its own ~6-7 constants. Merging them into a single function hurt performance at all sizes above 100 bytes, because the compiler must manage register allocation for all code paths simultaneously. Separate functions let each hot loop use nearly all available registers for its own constants.

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
