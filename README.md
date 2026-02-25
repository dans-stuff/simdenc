# simdenc [![Go Reference](https://pkg.go.dev/badge/github.com/dans-stuff/simdenc.svg)](https://pkg.go.dev/github.com/dans-stuff/simdenc) [![Go Report Card](https://goreportcard.com/badge/github.com/dans-stuff/simdenc)](https://goreportcard.com/report/github.com/dans-stuff/simdenc)

An experiment in SIMD-accelerated base64 encoding using Go 1.26's new `simd/archsimd` intrinsics. Pure Go, no assembly files, no CGo.

```go
import "github.com/dans-stuff/simdenc"

encoded := simdenc.StdEncoding.EncodeToString(data)
decoded, err := simdenc.StdEncoding.DecodeString(encoded)
```

Drop-in replacement for `encoding/base64`. Falls back to stdlib on platforms without SIMD support.

## Performance

65 GB/s encode, 80 GB/s decode. Measured on AMD EPYC 9B45 (Zen 5), single core.

**Encode (GB/s):**

| Size | simdenc | emmansun | cristalhq | stdlib |
|------|---------|----------|-----------|--------|
| 64 B | 6.4 | 7.4 | 3.7 | 1.9 |
| 256 B | 21.6 | 18.6 | 4.0 | 1.9 |
| 1 KB | 41.3 | 26.6 | 4.1 | 1.9 |
| 10 KB | 61.3 | 28.0 | 4.1 | 1.9 |
| 64 KB | 64.6 | 28.8 | 4.1 | 1.9 |
| 1 MB | 49.0 | 28.7 | 4.1 | 1.9 |

**Decode (GB/s):**

| Size | simdenc | emmansun | cristalhq | stdlib |
|------|---------|----------|-----------|--------|
| 64 B | 5.4 | 4.3 | 3.7 | 2.5 |
| 256 B | 18.4 | 13.5 | 4.0 | 2.8 |
| 1 KB | 42.5 | 24.3 | 4.0 | 2.8 |
| 10 KB | 68.5 | 30.8 | 4.0 | 2.9 |
| 64 KB | 80.3 | 32.4 | 4.0 | 2.9 |
| 1 MB | 63.4 | 32.5 | 4.0 | 2.9 |

The other libraries are there to contextualize where compiler-generated SIMD sits. emmansun uses hand-written AVX2 assembly. cristalhq uses Go with some SIMD tricks. Encode is 33x stdlib and 2.2x the hand-written assembly at 64 KB. Decode is the bigger story: 80 GB/s from pure Go compiler intrinsics, 2.5x emmansun's hand-written assembly. That's thanks to VPERMI2B (an AVX-512 VBMI instruction) which replaces the 6-instruction nibble-LUT validation+translation pipeline with a single combined lookup.

See [RESEARCH.md](RESEARCH.md) for the full experiment log with 44 A/B tests.

## How it works

CPU features are detected at init. The best available path is selected automatically:

- **No SIMD**: delegates to `encoding/base64` (ARM, older x86, etc.)
- **SSE (128-bit)**: encode and decode, used as cleanup after wider tiers
- **AVX2 (256-bit)**: 256-bit encode and decode for medium inputs, with SSE cleanup for remaining bytes
- **AVX-512 + VBMI**: 512-bit encode via VPERMB LUT, 512-bit decode via VPERMI2B combined validation+translation, SSE cleanup for tail bytes

Each tier is its own function that hoists only its own constants. A tiny dispatch function selects the tier based on input size and CPU features. This matters because the Go compiler allocates registers per-function, so merging tiers into one function causes spills and 50%+ regressions.

## What I learned

I built this to see what Go 1.26's SIMD intrinsics can actually do. Base64 is a well-studied problem with decades of SIMD research, so it's a good test case. Every optimization was A/B tested on EPYC Zen 4 and Zen 5.

### VPERMB and VPERMI2B are the big wins

VPERMB rearranges any of 64 bytes into any order in a single instruction. Base64 encoding converts 6-bit values to ASCII characters, which normally takes 4 operations. VPERMB turns it into a 64-entry lookup table and does it in one shot. That's where most of the AVX-512 encode speedup comes from.

VPERMI2B (exposed as `ConcatPermute` in archsimd) does the same thing but across a 128-entry table formed by concatenating two 64-byte registers. For decode, this replaces the entire nibble-LUT validation+translation pipeline (6 instructions: shift, mask, two LUT lookups, compare, add) with a single instruction that validates and translates simultaneously. Invalid characters produce 0x80, which is caught by a simple bit check. This is the single biggest optimization in the project, roughly doubling decode throughput.

### 512-bit helps both encode and decode

Going from 256-bit to 512-bit vectors made encoding 55% faster on Zen 4 (and over 2x on Zen 5). Decode initially got 12% *slower* with 512-bit vectors when using the same nibble-LUT algorithm. But switching to the VPERMI2B-based algorithm at 512-bit width changed the picture: the shorter pipeline (1 instruction for validate+translate instead of 6) more than compensates for Zen 4's 512-bit double-pumping. Decode is now 2x the 256-bit path at 10 KB+.

### Waterfall dispatch

Each SIMD tier is a standalone function. The dispatcher tries AVX-512 first, then AVX2, then SSE, each picking up where the previous left off. Early experiments used a "fused bookend" approach where SSE wrapped AVX2 (processing the first and last bytes with SSE, filling the middle with AVX2). Benchmarking showed the unfused waterfall is 12-18% faster at all sizes because each standalone function gets cleaner register allocation. SSE now only runs as cleanup after the wider tier finishes.

### Go has no LICM

Global variables are reloaded from memory on every loop iteration, even when there are no stores or function calls in the loop body. Go's compiler simply doesn't implement Loop Invariant Code Motion ([golang/go#15808](https://github.com/golang/go/issues/15808)). The workaround is copying each global into a local before the loop (`v := globalVec`). Locals are register-eligible, so they stay in registers across iterations. This gave +37% for encode and +32% for decode.

### Closures kill SIMD inlining

We tried a closure approach where a factory function captures all constants and returns the inner function. The captured values were hoisted correctly (loaded once, not reloaded per iteration). But Go's compiler won't inline SIMD intrinsics inside closure bodies. `LoadUint8x32Slice` and `StoreSlice` become real `CALL` instructions instead of inline VMOVDQU. 7-8x slowdown, capping throughput at ~6 GB/s regardless of input size. This happens even when the closure is called directly from a local variable, so it's a compiler limitation, not a devirtualization issue.

### Small functions or nothing

Each SIMD tier is its own function with ~6-7 hoisted constants. Merging them into one function hurts at all sizes above 100 bytes because the compiler manages register allocation for all code paths simultaneously. Even adding a 3-line size check to a dispatch function can degrade the SIMD callees by 50%. Keep dispatch functions tiny: no constant hoisting, no loops, just a size check and a tail call.

### Broadcast512 doesn't do what you'd expect

`Uint8x16.Broadcast512()` broadcasts *element zero* (a single byte via VPBROADCASTB), not the entire 128-bit lane. To replicate a 128-bit pattern across all lanes of a 512-bit register, you need `SetLo`/`SetHi` chains. The Go team plans to recognize this pattern and optimize it to VBROADCASTI32X4 in a future release.

### VPMULTISHIFTQB would make encode even faster

We prototyped a VPMULTISHIFTQB-based encode path in hand-written Go assembly. It reduces the encode inner loop from 7 SIMD instructions to 3 (VPERMB + VPMULTISHIFTQB + VPERMB), giving +37-125% encode throughput depending on size. But archsimd doesn't expose VPMULTISHIFTQB, so this can't ship as pure Go. It's the single biggest missing intrinsic for this workload.

### Alignment doesn't matter

archsimd uses VMOVDQU/VMOVDQU64 (unaligned loads) for all vector operations. At 256-bit width there's zero measurable penalty for misaligned data. At 512-bit width there's ~8% penalty for misaligned addresses, but Go's allocator naturally aligns heap memory. Not worth worrying about in practice.

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
