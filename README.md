# simdenc [![Go Reference](https://pkg.go.dev/badge/github.com/dans-stuff/simdenc.svg)](https://pkg.go.dev/github.com/dans-stuff/simdenc) [![Go Report Card](https://goreportcard.com/badge/github.com/dans-stuff/simdenc)](https://goreportcard.com/report/github.com/dans-stuff/simdenc)

An experiment in SIMD-accelerated base64 encoding using Go 1.26's new `simd/archsimd` intrinsics. Pure Go, no assembly files, no CGo.

```go
import "github.com/dans-stuff/simdenc"

encoded := simdenc.StdEncoding.EncodeToString(data)
decoded, err := simdenc.StdEncoding.DecodeString(encoded)
```

Drop-in replacement for `encoding/base64`. Falls back to stdlib on platforms without SIMD support.

## Performance

63 GB/s encode, 31 GB/s decode. Measured on AMD EPYC 9B45 (Zen 5), single core.

**Encode (GB/s):**

| Size | simdenc | emmansun | cristalhq | stdlib |
|------|---------|----------|-----------|--------|
| 64 B | 6.4 | 7.4 | 3.7 | 1.9 |
| 256 B | 21.5 | 18.4 | 4.0 | 1.8 |
| 1 KB | 45.4 | 26.5 | 4.0 | 1.9 |
| 10 KB | 61.0 | 27.9 | 4.1 | 1.9 |
| 64 KB | 63.6 | 28.7 | 4.1 | 1.9 |
| 1 MB | 48.1 | 28.5 | 4.1 | 1.9 |

**Decode (GB/s):**

| Size | simdenc | emmansun | cristalhq | stdlib |
|------|---------|----------|-----------|--------|
| 64 B | 5.7 | 4.3 | 3.7 | 2.4 |
| 256 B | 15.7 | 13.3 | 4.0 | 2.8 |
| 1 KB | 25.7 | 24.1 | 4.0 | 2.8 |
| 10 KB | 30.5 | 30.7 | 4.0 | 2.9 |
| 64 KB | 31.2 | 32.4 | 4.0 | 2.9 |
| 1 MB | 30.9 | 32.0 | 4.0 | 2.9 |

The other libraries are there to contextualize where compiler-generated SIMD sits. emmansun uses hand-written AVX2 assembly. cristalhq uses Go with some SIMD tricks. The encode result is the interesting part: 33x stdlib from pure Go compiler intrinsics, about 2x the hand-written assembly at 64 KB. Decode converges with emmansun around 31 GB/s; they're ~4% faster at large sizes due to bounds checks and instruction scheduling that the Go compiler can't quite match.

See [RESEARCH.md](RESEARCH.md) for the full experiment log with 37 A/B tests.

## How it works

CPU features are detected at init. The best available path is selected automatically:

- **No SIMD**: delegates to `encoding/base64` (ARM, older x86, etc.)
- **SSE (128-bit)**: encode and decode for small inputs (< ~120 bytes)
- **AVX2 (256-bit)**: fused with SSE bookends for medium inputs. SSE handles the first and last 12/16 bytes, AVX2 fills the middle.
- **AVX-512 + VBMI**: standalone 512-bit encode loop with SSE cleanup, VPERMB-accelerated 256-bit decode

Each tier is its own function that hoists only its own constants. A tiny dispatch function selects the tier based on input size and CPU features. This matters because the Go compiler allocates registers per-function, so merging tiers into one function causes spills and 50%+ regressions.

## What I learned

I built this to see what Go 1.26's SIMD intrinsics can actually do. Base64 is a well-studied problem with decades of SIMD research, so it's a good test case. Every optimization was A/B tested on EPYC Zen 4 and Zen 5.

### VPERMB is the big win

There's an AVX-512 instruction called VPERMB that rearranges any of 64 bytes into any order in a single step. Base64 encoding converts 6-bit values to ASCII characters, which normally takes 4 operations. VPERMB turns it into a 64-entry lookup table and does it in one shot. That's where most of the AVX-512 encode speedup comes from.

### 512-bit is faster for encode, slower for decode

Going from 256-bit to 512-bit vectors made encoding 55% faster on Zen 4 and over 2x faster on Zen 5, but decoding got 12% slower. Decode has a longer dependency chain (validate, translate, pack, compact) that doesn't benefit from wider execution. Encode's pipeline is shorter and more parallel.

### Fused bookends

For encode and decode, we process the first and last few bytes with SSE (128-bit), then fill the middle with AVX2 or AVX-512. The SSE preamble and tail overlap slightly with the wider loop's output. The overlap is harmless (same data written twice) and eliminates the need for a separate scalar preamble or tail. This gave +27% at 1 KB for encode versus using either width alone.

### Go has no LICM

Global variables are reloaded from memory on every loop iteration, even when there are no stores or function calls in the loop body. Go's compiler simply doesn't implement Loop Invariant Code Motion ([golang/go#63670](https://github.com/golang/go/issues/63670)). The workaround is copying each global into a local before the loop (`v := globalVec`). Locals are register-eligible, so they stay in registers across iterations. This gave +37% for encode and +32% for decode.

### Closures kill SIMD inlining

We tried a closure approach where a factory function captures all constants and returns the inner function. The captured values were hoisted correctly (loaded once, not reloaded per iteration). But Go's compiler won't inline SIMD intrinsics inside closure bodies. `LoadUint8x32Slice` and `StoreSlice` become real `CALL` instructions instead of inline VMOVDQU. 7-8x slowdown, capping throughput at ~6 GB/s regardless of input size. This happens even when the closure is called directly from a local variable, so it's a compiler limitation, not a devirtualization issue.

### Small functions or nothing

Each SIMD tier is its own function with ~6-7 hoisted constants. Merging them into one function hurts at all sizes above 100 bytes because the compiler manages register allocation for all code paths simultaneously. Even adding a 3-line size check to a dispatch function can degrade the SIMD callees by 50%. Keep dispatch functions tiny: no constant hoisting, no loops, just a size check and a tail call.

### unsafe.Pointer loads: 0% improvement

We tried bypassing the safe `LoadSlice` API with raw pointer arithmetic. No difference. The compiler already eliminates bounds checks when you give it explicit slice bounds like `src[i:i+32]`.

### Rosetta 2 weirdness

Byte-identical x86 machine code in the same binary can run at completely different speeds under Rosetta 2, depending on where the function lands in memory. We copied a competitor's assembly function verbatim into our binary: the original ran at 6.3 GB/s, our copy ran at 2.2 GB/s, same instructions. Not actionable, just interesting.

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

[MIT](LICENSE)
