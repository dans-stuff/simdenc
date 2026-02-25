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

## What I learned about `simd/archsimd`

I built this to test Go 1.26's SIMD intrinsics, using base64 as the workload. Every finding below comes from 44 A/B tests on AMD EPYC Zen 4 and Zen 5. The archsimd-specific lessons apply to any SIMD Go code; the base64-specific findings are in the next section.

### How to declare SIMD constants

Express byte patterns as named `uint64` constants, then declare vectors as package-level `var` with `Load` + `.As*()` in one expression. Shadow into locals at the top of each function body (LICM workaround — see below).

```go
const maskHi = uint64(0x0FC0FC000FC0FC00) // AND mask: keep bits [11:6]

var encMaskHi512 = archsimd.LoadUint64x8(&[8]uint64{
    maskHi, maskHi, maskHi, maskHi, maskHi, maskHi, maskHi, maskHi,
}).AsUint16x32()

func encode512(dst, src []byte) {
    mask := encMaskHi512 // shadow into local — stays in register
    // ...
}
```

Why this matters: declaring constants inline inside the function body forces the compiler to build each 512-bit value on the stack (8 `MOVQ` immediates + 8 `MOVQ` stores + 1 `VMOVDQU64` — 17 instructions per constant), causing a ~6% regression vs a single `VMOVDQU64` load from `.data`. Moving just the `.As*()` cast into the function is also ~5% slower. (Experiment 44.)

### Go has no LICM

Global variables are reloaded from memory on every loop iteration, even when there are no stores or function calls in the loop body. Go's compiler doesn't implement Loop Invariant Code Motion ([golang/go#15808](https://github.com/golang/go/issues/15808)). The workaround is copying each global into a local before the loop (`v := globalVec`). Locals are register-eligible, so they stay in registers across iterations. This gave +37% for encode and +32% for decode.

### One SIMD tier per function

Register allocation is per-function. Merging SIMD tiers (e.g. SSE + AVX2) into one function causes spills and 50%+ regressions at all sizes above 100 bytes. Even adding a 3-line size check to a dispatch function degrades the SIMD callees. Keep dispatch functions tiny: a size check and a tail call, nothing else.

### Closures kill SIMD inlining

Go's compiler won't inline SIMD intrinsics inside closure bodies. `LoadUint8x32Slice` and `StoreSlice` become real `CALL` instructions instead of inline VMOVDQU. 7-8x slowdown, capping throughput at ~6 GB/s regardless of input size. This happens even when the closure is called directly from a local variable — it's a compiler limitation, not a devirtualization issue.

### `Broadcast512` broadcasts element zero

`Uint8x16.Broadcast512()` broadcasts *element zero* (a single byte via VPBROADCASTB), not the entire 128-bit lane. To replicate a 128-bit pattern across all lanes of a 512-bit register, you need `SetLo`/`SetHi` chains.

### Alignment doesn't matter

archsimd uses VMOVDQU/VMOVDQU64 (unaligned loads) everywhere. Go's allocator naturally aligns heap memory. Not worth worrying about in practice.

### What's not in archsimd yet

- **VPMULTISHIFTQB:** not exposed (would 2x our encode — see below)
- **VPTERNLOGD:** unexported `tern()` method, compiler auto-fusion doesn't trigger (0 VPTERNLOGD emissions in our entire build)
- **IsZero at 512-bit:** missing (workaround: `Equal(zero).ToBits()`)
- **Non-temporal stores, prefetch:** not exposed

## Base64-specific findings

VPERMB (`Permute` in archsimd) is a 64-entry byte lookup in a single instruction. For encode, it replaces the 4-operation range-based sextet-to-ASCII translation. VPERMI2B (`ConcatPermute`) does the same across a 128-entry table formed by concatenating two 64-byte registers. For decode, it replaces the entire nibble-LUT validation+translation pipeline (6 instructions: shift, mask, two LUT lookups, compare, add) with one instruction that validates and translates simultaneously. Invalid characters produce 0x80, caught by a simple bit check. VPERMI2B is the single biggest optimization in the project, roughly doubling decode throughput.

512-bit vectors help both encode and decode once you switch to the right algorithm. Encode was 55% faster on Zen 4 (over 2x on Zen 5). Decode initially got 12% *slower* at 512-bit with the same nibble-LUT algorithm, but VPERMI2B's shorter pipeline (1 instruction vs 6) more than compensates for Zen 4's 512-bit double-pumping. We also prototyped a VPMULTISHIFTQB-based encode path in hand-written Go assembly: a 3-instruction hot loop (VPERMB + VPMULTISHIFTQB + VPERMB) that runs +86-115% faster at compute-bound sizes. It can't ship without the intrinsic, but validates what's possible when archsimd catches up.

Byte-identical x86 machine code in the same binary can run at completely different speeds under Rosetta 2, depending on where the function lands in memory — same instructions, 3x performance gap. Not actionable, just interesting.

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
