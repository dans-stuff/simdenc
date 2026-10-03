# simdenc [![Go Reference](https://pkg.go.dev/badge/github.com/dans-stuff/simdenc.svg)](https://pkg.go.dev/github.com/dans-stuff/simdenc) [![Go Report Card](https://goreportcard.com/badge/github.com/dans-stuff/simdenc)](https://goreportcard.com/report/github.com/dans-stuff/simdenc)

SIMD-accelerated text encodings written with Go 1.26's `simd/archsimd` intrinsics. Pure Go: no assembly files, no cgo.

| Encoding | API | Same results as |
|---|---|---|
| base64 | `StdEncoding`, `URLEncoding`, `RawStdEncoding`, `RawURLEncoding` | `encoding/base64` |
| base32 | `Base32StdEncoding`, `Base32HexEncoding` | `encoding/base32` |
| hex | `HexEncode`, `HexDecode`, `HexEncodeToString`, ... | `encoding/hex` |
| percent-encoding | `QueryEscape`, `PathEscape`, `QueryUnescape`, `PathUnescape` | `net/url` |

```go
import "github.com/dans-stuff/simdenc"

encoded := simdenc.StdEncoding.EncodeToString(data)
decoded, err := simdenc.StdEncoding.DecodeString(encoded)

escaped := simdenc.QueryEscape("a b&c=d")
```

The base64 API is a drop-in replacement for `encoding/base64`, and the base32 and hex ones mirror their standard library packages. Everything is checked against the library in the last column: same output and, for bad input, the same error and the same offset. Where CPUs or platforms lack the SIMD tier, the standard library (or a short scalar loop) does the work.

## At a glance

simdenc on 64 KB inputs, one AMD EPYC 9B45 (Zen 5) core, against the fastest other implementation I found and against the Go standard library. "Fastest other" is `emmansun/base64` for base64 and `go-simd` for hex and base32 (both AVX2 assembly); for percent-encoding I found no SIMD library, so the standard library is the only comparison. "Mixed" text is about 10% bytes that need escaping, "clean" has none.

| function | simdenc | vs fastest other | vs standard library |
|---|---|---|---|
| base64 encode | 67.6 GB/s | 2.2x (emmansun) | 34x (2.00 GB/s) |
| base64 decode | 83.7 GB/s | 2.4x (emmansun) | 27x (3.11 GB/s) |
| hex encode | 58.9 GB/s | 1.0x (gosimd) | 33x (1.77 GB/s) |
| hex decode | 100.1 GB/s | 4.8x (gosimd) | 33x (3.04 GB/s) |
| base32 encode | 53.0 GB/s | 3.9x (gosimd) | 29x (1.85 GB/s) |
| base32 decode | 48.8 GB/s | 5.5x (gosimd) | 72x (0.68 GB/s) |
| QueryEscape, mixed text | 3.8 GB/s | none found | 6x (0.61 GB/s) |
| QueryEscape, clean text | 55.4 GB/s | none found | 18x (3.01 GB/s) |
| QueryUnescape, mixed text | 9.2 GB/s | none found | 23x (0.40 GB/s) |
| QueryUnescape, one escape at the end | 15.3 GB/s | none found | 30x (0.51 GB/s) |

Throughput counts source bytes for encode and encoded characters for decode. For percent-encoding, "mixed" is about 10% bytes that need escaping, "clean" has none and "one escape at the end" is clean text with a single escape as its last byte, which is the case where the unescape kernel runs over mostly clean blocks. Fully clean input to `QueryUnescape` is faster still (see the per-size table), but that path is the standard library's own vectorised `strings.Contains` finding nothing to decode, not our kernel. The per-size tables below cover 64 bytes to 1 MB (hex from 16 bytes), and the notes under them say where simdenc is not ahead (short inputs especially).

## Performance

All numbers are from one AMD EPYC 9B45 (Zen 5) core, `GOEXPERIMENT=simd`, Go 1.26.4. They are medians of 5 runs of `go test -bench` with the spread (max minus min, as a percentage of the median) in brackets. One machine, one session: indicative, not a guarantee for other CPUs. The benchmark harness that produced these tables is not part of this repository.

The competitors are emmansun and cristalhq (base64; the benchmarks in `base64_test.go` compare against them), `go-simd` (hex and base32: AVX2 assembly), and `net/url`.

### base64

**Base64Encode** (GB/s, median of 5 runs; ± is max-min spread)

| size | cristalhq | emmansun | simdenc | stdlib |
|---|---|---|---|---|
| 64 | 3.78 (14%) | 7.56 (12%) | 5.92 (4%) | 1.93 (9%) |
| 256 | 4.28 (8%) | 19.0 (19%) | 18.8 (6%) | 2.01 (8%) |
| 1024 | 4.25 (6%) | 26.7 (6%) | 44.0 (12%) | 2.01 (3%) |
| 10240 | 4.22 (6%) | 29.6 (5%) | 61.5 (9%) | 2.05 (4%) |
| 65536 | 4.33 (4%) | 31.3 (2%) | 67.6 (2%) | 2.00 (4%) |
| 1048576 | 4.23 (3%) | 29.7 (2%) | 48.5 (3%) | 2.03 (1%) |

**Base64Decode** (GB/s, median of 5 runs; ± is max-min spread)

| size | cristalhq | emmansun | simdenc | stdlib |
|---|---|---|---|---|
| 64 | 3.74 (12%) | 4.74 (8%) | 5.90 (7%) | 2.33 (23%) |
| 256 | 4.01 (6%) | 14.0 (6%) | 18.6 (5%) | 2.95 (4%) |
| 1024 | 4.17 (4%) | 25.9 (4%) | 46.8 (7%) | 2.85 (6%) |
| 10240 | 4.21 (6%) | 33.5 (3%) | 77.7 (6%) | 3.10 (7%) |
| 65536 | 4.21 (5%) | 34.5 (4%) | 83.7 (8%) | 3.11 (11%) |
| 1048576 | 4.21 (5%) | 34.1 (2%) | 63.1 (5%) | 3.06 (4%) |

### hex

**HexEncode** (GB/s, median of 5 runs; ± is max-min spread)

| size | gosimd | simdenc | stdlib |
|---|---|---|---|
| 16 | 3.82 (3%) | 3.20 (6%) | 1.63 (4%) |
| 32 | 7.37 (5%) | 5.16 (7%) | 1.70 (5%) |
| 64 | 13.8 (5%) | 12.4 (5%) | 1.69 (6%) |
| 256 | 34.3 (8%) | 30.4 (17%) | 1.64 (12%) |
| 1024 | 53.8 (6%) | 48.2 (5%) | 1.70 (4%) |
| 10240 | 54.5 (9%) | 56.5 (5%) | 1.73 (4%) |
| 65536 | 56.9 (2%) | 58.9 (11%) | 1.77 (3%) |
| 1048576 | 37.3 (10%) | 38.5 (3%) | 1.72 (9%) |

**HexDecode** (GB/s, median of 5 runs; ± is max-min spread)

| size | gosimd | simdenc | stdlib |
|---|---|---|---|
| 16 | 4.56 (6%) | 2.41 (2%) | 2.84 (2%) |
| 32 | 10.3 (9%) | 4.65 (1%) | 2.78 (18%) |
| 64 | 14.2 (2%) | 19.6 (7%) | 2.89 (3%) |
| 256 | 18.8 (3%) | 50.7 (5%) | 2.88 (4%) |
| 1024 | 20.1 (3%) | 80.1 (6%) | 3.03 (4%) |
| 10240 | 20.2 (4%) | 93.7 (5%) | 3.05 (6%) |
| 65536 | 21.0 (3%) | 100.1 (8%) | 3.04 (2%) |
| 1048576 | 20.5 (6%) | 75.5 (2%) | 2.92 (4%) |

### base32

**Base32Encode** (GB/s, median of 5 runs; ± is max-min spread)

| size | gosimd | simdenc | stdlib |
|---|---|---|---|
| 64 | 4.05 (6%) | 5.23 (3%) | 1.77 (3%) |
| 256 | 8.12 (4%) | 11.8 (5%) | 1.80 (6%) |
| 1024 | 11.8 (4%) | 36.3 (39%) | 1.90 (10%) |
| 10240 | 13.6 (4%) | 51.0 (6%) | 1.87 (4%) |
| 65536 | 13.5 (3%) | 53.0 (12%) | 1.85 (6%) |
| 1048576 | 13.6 (3%) | 43.1 (3%) | 1.86 (4%) |

**Base32Decode** (GB/s, median of 5 runs; ± is max-min spread)

| size | gosimd | simdenc | stdlib |
|---|---|---|---|
| 64 | 4.10 (8%) | 4.17 (7%) | 0.60 (8%) |
| 256 | 7.33 (2%) | 15.0 (16%) | 0.65 (21%) |
| 1024 | 8.36 (3%) | 32.4 (9%) | 0.70 (5%) |
| 10240 | 8.96 (9%) | 43.7 (4%) | 0.68 (6%) |
| 65536 | 8.87 (4%) | 48.8 (7%) | 0.68 (5%) |
| 1048576 | 9.08 (2%) | 54.5 (5%) | 0.72 (3%) |

### percent-encoding

"mixed" text has about 10% of its bytes needing escapes; "clean" has none, where `net/url` and simdenc both return the input unchanged (simdenc without allocating).

**QueryEscape** (GB/s, median of 5 runs; ± is max-min spread)

| size | simdenc | stdlib |
|---|---|---|
| clean/64 | 11.6 (3%) | 2.77 (24%) |
| clean/256 | 27.3 (3%) | 2.49 (9%) |
| clean/1024 | 43.9 (11%) | 3.08 (6%) |
| clean/10240 | 53.5 (11%) | 3.21 (8%) |
| clean/65536 | 55.4 (5%) | 3.01 (6%) |
| clean/1048576 | 55.3 (6%) | 3.15 (4%) |
| mixed/64 | 1.00 (10%) | 0.63 (7%) |
| mixed/256 | 2.08 (10%) | 0.77 (12%) |
| mixed/1024 | 2.81 (5%) | 0.78 (6%) |
| mixed/10240 | 3.56 (5%) | 0.72 (10%) |
| mixed/65536 | 3.78 (7%) | 0.61 (7%) |
| mixed/1048576 | 3.92 (3%) | 0.37 (4%) |
| sparse/64 | 2.32 (10%) | 0.93 (6%) |
| sparse/256 | 5.83 (6%) | 0.89 (3%) |
| sparse/1024 | 8.93 (6%) | 1.12 (5%) |
| sparse/10240 | 13.0 (6%) | 1.21 (8%) |
| sparse/65536 | 13.3 (12%) | 1.21 (10%) |
| sparse/1048576 | 13.0 (6%) | 1.35 (7%) |

**QueryUnescape** (GB/s, median of 5 runs; ± is max-min spread)

| size | simdenc | stdlib |
|---|---|---|
| clean/64 | 9.26 (2%) | 1.06 (7%) |
| clean/256 | 26.3 (24%) | 1.36 (4%) |
| clean/1024 | 48.0 (6%) | 1.38 (4%) |
| clean/10240 | 57.7 (2%) | 1.38 (7%) |
| clean/65536 | 60.9 (4%) | 1.41 (8%) |
| clean/1048576 | 57.4 (3%) | 1.34 (8%) |
| mixed/64 | 2.06 (6%) | 0.47 (11%) |
| mixed/256 | 1.79 (5%) | 0.58 (4%) |
| mixed/1024 | 6.02 (3%) | 0.48 (9%) |
| mixed/10240 | 8.46 (7%) | 0.43 (13%) |
| mixed/65536 | 9.21 (6%) | 0.40 (5%) |
| mixed/1048576 | 7.79 (7%) | 0.37 (3%) |
| sparse/64 | 2.38 (8%) | 0.41 (8%) |
| sparse/256 | 5.67 (7%) | 0.47 (8%) |
| sparse/1024 | 9.11 (1%) | 0.52 (5%) |
| sparse/10240 | 14.3 (6%) | 0.54 (11%) |
| sparse/65536 | 15.3 (5%) | 0.51 (7%) |
| sparse/1048576 | 13.0 (14%) | 0.52 (6%) |

### Reading the numbers

- **base64**: decode is ahead of emmansun's AVX2 assembly at every size, 1.2x at 64 B, 1.3x at 256 B and 1.8-2.4x from 1 KB. Encode is behind it at 64 B (0.78x), level at 256 B (0.99x) and 1.6-2.2x ahead from 1 KB.
- **hex**: decode is 2.7x `go-simd` at 256 B and 3.7-4.8x from 1 KB. Encode is **level** with it: 0.9x from 64 B to 1 KB and 1.0x from 10 KB. Below 64 B simdenc leaves hex to the standard library and trails `go-simd`: decode 0.53x at 16 characters and 0.45x at 32, encode 0.84x at 16 bytes and 0.7x at 32. An earlier run of mine had simdenc 1.1-1.7x ahead on hex encode only because `go-simd` read 35 GB/s that session against 57 here; within a session the spread is a few percent, between sessions on this shared VM it was 30%. Ablation runs point at store throughput as the encode limit.
- **base32**: decode is level with `go-simd` at 64 B and 2.0x ahead at 256 B and 3.9-6.0x from 1 KB; encode is 1.3-3.9x ahead at every size.
- **percent-encoding**: escape is 1.6x `net/url` at 64 B of mixed text and 10.6x at 1 MB (6x at 64 KB); with one escape at the end 2.5x to 11x; on clean text 4.2x to 18x. Unescape is 3.1x at 256 B of mixed text and 23x at 64 KB; with one escape at the end 5.8x to 30x; on clean text 8.7x to 43x (the `strings.Contains` path described above).
- Very short inputs pay for the extra call before the reference code: in an interleaved before/after run, 12-byte base64 encode went from 8.2 to 11.2 ns when the cascade replaced the old direct path.

## How it works

Every codec has the same shape. CPU features are detected at init. A kernel per vector width consumes as many whole blocks as it can, widest first: 512-bit, then 256-bit, then 128-bit. A reference implementation (the standard library, or a short scalar loop) finishes what is left, and it is the only code that reports errors, so errors are always the reference's own. A kernel never reports an error: at the first block that holds invalid input it stops and hands over.

| | 512-bit | 256-bit | 128-bit | needs |
|---|---|---|---|---|
| base64 | yes | yes | yes | AVX-512 VBMI for 512; AVX2 for 256 and 128 |
| hex | yes | yes | | AVX-512 VBMI for 512; AVX2 for 256 |
| base32 | yes | | | AVX-512 VBMI |
| percent-encoding | yes | | | AVX-512 VBMI2 (for `VPCOMPRESSB`) |

Where a codec has no 256-bit tier, CPUs without the 512-bit one use the reference code. AVX2 has no byte permute and no 16-bit variable shift, so base32 and percent-encoding would each need a different algorithm there, not just narrower vectors. Hex has no 128-bit tier because it would save at most 8 bytes of scalar work. (I also tried replacing hex's narrower tiers with one masked partial-vector step after the 512-bit loop: 10-16% faster for decode at 10-64 KB, 10-27% slower at 64-256 B, the common size.)

Each tier is its own function that hoists only its own constants. This matters because the Go compiler allocates registers per function, so merging tiers into one function causes spills and 50%+ regressions.

Tests run every case on every tier the machine has (`eachTier` lowers the CPU flags), against the reference library: encode at many lengths, every prefix of a valid encoding, every byte value at every position, malformed endings and padding, and Go fuzz targets. For base32 a deliberate bug put into a kernel was confirmed to fail the tests; for hex and percent-encoding the tests caught real bugs while the kernels were being written. The base64 kernels predate this and were not mutation-tested.

## What I learned about `simd/archsimd`

I built this to test Go 1.26's SIMD intrinsics, starting with base64 as the workload. The first findings come from 44 A/B tests on AMD EPYC Zen 4 and Zen 5. The archsimd-specific lessons apply to any SIMD Go code; the base64-specific findings follow.

### How to declare SIMD constants

Express byte patterns as named `uint64` constants, then declare vectors as package-level `var` with `Load` + `.As*()` in one expression. Shadow into locals at the top of each function body (LICM workaround, see below).

```go
const maskHi = uint64(0x0FC0FC000FC0FC00) // AND mask: keep bits [11:6]

var encMaskHi512 = archsimd.LoadUint64x8(&[8]uint64{
    maskHi, maskHi, maskHi, maskHi, maskHi, maskHi, maskHi, maskHi,
}).AsUint16x32()

func encode512(dst, src []byte) {
    mask := encMaskHi512 // shadow into local: stays in register
    // ...
}
```

Why this matters: declaring constants inline inside the function body forces the compiler to build each 512-bit value on the stack (8 `MOVQ` immediates + 8 `MOVQ` stores + 1 `VMOVDQU64`, 17 instructions per constant), causing a ~6% regression vs a single `VMOVDQU64` load from `.data`. Moving just the `.As*()` cast into the function is also ~5% slower. (Experiment 44.) 512-bit loads fault on CPUs without AVX-512, so the newer codecs load their 512-bit constants in `init`, behind the CPU check.

### Go has no LICM

Global variables are reloaded from memory on every loop iteration, even when there are no stores or function calls in the loop body. Go's compiler doesn't implement Loop Invariant Code Motion ([golang/go#15808](https://github.com/golang/go/issues/15808)). The workaround is copying each global into a local before the loop (`v := globalVec`). Locals are register-eligible, so they stay in registers across iterations. This gave +37% for encode and +32% for decode.

### One SIMD tier per function

Register allocation is per-function. Merging SIMD tiers (e.g. SSE + AVX2) into one function causes spills and 50%+ regressions at all sizes above 100 bytes. Even adding a 3-line size check to a dispatch function degrades the SIMD callees. Keep dispatch functions tiny.

### Closures kill SIMD inlining

Go's compiler won't inline SIMD intrinsics inside closure bodies. `LoadUint8x32Slice` and `StoreSlice` become real `CALL` instructions instead of inline VMOVDQU. 7-8x slowdown, capping throughput at ~6 GB/s regardless of input size. This happens even when the closure is called directly from a local variable. It's a compiler limitation, not a devirtualization issue.

### Other things worth knowing

- **`Merge(y, mask)` keeps the receiver where the mask is true.** `x.Merge(y, m)` is `m ? x : y`, which reads backwards from a blend. Three of my first drafts had it inverted.
- **A 16-bit shift leaks into the neighbouring byte.** Shifting a byte vector as 16-bit words moves bits across byte lanes; either the table you index afterwards ignores those bits (a lookup table repeated to fill a 64-entry `Permute`) or you mask them.
- **`Permute` and `ConcatPermute` index with only the low 6 or 7 bits**, so a 128-entry lookup does not see a byte's top bit; check it separately when non-ASCII input must be rejected.
- **Partial loads and stores are cheap enough for tails** (`LoadUint8x64SlicePart`, `StoreSlicePart`) but cost about 20% when used on every iteration of a hot loop; use the full-width forms in the loop and the partial ones for the last block.
- **`Broadcast512` broadcasts element zero.** `Uint8x16.Broadcast512()` broadcasts one byte, not the 128-bit lane.
- **Alignment doesn't matter.** archsimd uses unaligned loads everywhere, and Go's allocator aligns heap memory.

### What's not in archsimd yet

- **VPMULTISHIFTQB:** not exposed (would speed up base64 encode; base32 works around it with a gather and variable shifts)
- **VPTERNLOGD:** unexported `tern()` method, compiler auto-fusion doesn't trigger
- **IsZero at 512-bit:** missing (workaround: `Equal(zero).ToBits()`)
- **Non-temporal stores, prefetch:** not exposed

## Base64-specific findings

VPERMB (`Permute` in archsimd) is a 64-entry byte lookup in a single instruction. For encode, it replaces the 4-operation range-based sextet-to-ASCII translation. VPERMI2B (`ConcatPermute`) does the same across a 128-entry table formed by concatenating two 64-byte registers. For decode, it replaces the entire nibble-LUT validation+translation pipeline (6 instructions: shift, mask, two LUT lookups, compare, add) with one instruction that validates and translates simultaneously. Invalid characters produce 0x80, caught by a simple bit check. VPERMI2B is the single biggest optimization in the project, roughly doubling decode throughput, and it is the core of every other decoder here.

512-bit vectors help both encode and decode once you switch to the right algorithm. Encode was 55% faster on Zen 4 (over 2x on Zen 5). Decode initially got 12% *slower* at 512-bit with the same nibble-LUT algorithm, but VPERMI2B's shorter pipeline (1 instruction vs 6) more than compensates for Zen 4's 512-bit double-pumping. We also prototyped a VPMULTISHIFTQB-based encode path in hand-written Go assembly: a 3-instruction hot loop (VPERMB + VPMULTISHIFTQB + VPERMB) that runs +86-115% faster at compute-bound sizes. It can't ship without the intrinsic, but validates what's possible when archsimd catches up.

Byte-identical x86 machine code in the same binary can run at completely different speeds under Rosetta 2, depending on where the function lands in memory: same instructions, 3x performance gap. Not actionable, just interesting.

See [RESEARCH.md](RESEARCH.md) for the full experiment log (it describes the code as it was then, including tiers and a NEON backend that no longer exist).

## The catch

`simd/archsimd` is still behind `GOEXPERIMENT=simd`, and everything here depends on it (Go 1.26). This is an experiment. Treat it accordingly. Without the experiment, or off amd64, the standard library does all the work.

```bash
GOEXPERIMENT=simd go test ./...
GOEXPERIMENT=simd go test -run XXX -fuzz FuzzHexDecode -fuzztime 10s .   # likewise FuzzBase32DecodeMatchesStdlib, FuzzPercentUnescape, ...
```

## Details

- **base64**: `Encoding` has the methods of `base64.Encoding`: `Encode`, `Decode`, `EncodeToString`, `DecodeString`, `AppendEncode`, `AppendDecode`, `EncodedLen`, `DecodedLen`, `WithPadding`.
- **base32**: `Base32Encoding` has the same methods; `Base32StdEncoding` and `Base32HexEncoding` are the two alphabets.
- **hex**: `HexEncode`, `HexDecode`, `HexEncodeToString`, `HexDecodeString`, `AppendHexEncode`, `AppendHexDecode`, `HexEncodedLen`, `HexDecodedLen` (lower-case output, either case accepted).
- **percent-encoding**: `QueryEscape` and `PathEscape` (`' '` becomes `+` only in the query form), `QueryUnescape` and `PathUnescape`. No allocation when nothing needs escaping or unescaping.

## License

[MIT](LICENSE)
