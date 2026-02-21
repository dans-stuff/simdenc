# AVX-512 Base64 Research Notes

## Optimization Catalog (AMD EPYC Zen 4)

Every non-obvious pattern in the codebase, with its measured A/B impact:

| # | Optimization | Delta | Verdict |
|---|---|---|---|
| 1 | Local copies of globals in loops | +37% enc, +32% dec | KEEP — compiler can't hoist global loads |
| 2 | 512-bit encode (VPERMB LUT, 48B/iter) | +28% at 1K, +55% at 64K vs AVX2 | KEEP |
| 3 | VPERMB cross-lane reshuffle (no preamble) | +62% at 100, +27% at 1K vs AVX2 | KEEP |
| 4 | ConcatPermute (VPERMI2B) LUT vs 4-op | +10% at 1K, +19% at 64K | KEEP |
| 5 | VPERMB decode compaction vs VPSHUFB+VPERMD | +5% at 1K, +9% at 64K | KEEP |
| 6 | Adaptive chaining (encode512 + encodeVBMI) | +27% at 1K vs either alone | KEEP |
| 7 | `d := dst[di:]` reslice before StoreSlice | +3% at 10K/64K | KEEP — zero complexity cost |
| 8 | -4 offset load trick (AVX2 encode) | Required (no VPERMB without VBMI) | KEEP |
| 9 | Unsafe pointer loads | 0% | REMOVED — no benefit |
| 10 | `for range n` counted loops | -27% decode at 64K | REMOVED — regression |

## Server: AMD EPYC Zen 4

Supported extensions: AVX-512F, DQ, CD, BW, VL, IFMA, VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, BF16.

Zen 4 executes 512-bit ops by double-pumping through 256-bit execution ports.
Same throughput as two AVX2 ops, but saves instruction count and loop overhead.

## Key archsimd 512-bit Operations

| Operation | Type | Instruction | Notes |
|-----------|------|-------------|-------|
| `Uint8x64.Permute(indices)` | VPERMB | VBMI | Full 64-byte cross-lane byte permute |
| `Uint8x64.PermuteOrZeroGrouped(indices)` | VPSHUFB | AVX512BW | Per-lane shuffle, high bit zeroes |
| `Uint8x64.DotProductPairsSaturated(Int8x64)` | VPMADDUBSW | AVX512BW | u8*i8 pairs -> i16 |
| `Int16x32.DotProductPairs(Int16x32)` | VPMADDWD | AVX512BW | i16 pairs -> i32 |
| `Uint16x32.MulHigh/Mul` | VPMULHUW/VPMULLW | AVX512BW | 16-bit multiply |
| `Uint8x64.Equal(y).ToBits()` | VPCMPB+KMOV | AVX512BW | Compare -> k-register -> uint64 |
| `Uint8x64.Compress(mask)` | VPCOMPRESSB | VBMI2 | Pack selected bytes |
| `Uint16x32.ShiftLeftConcat/ShiftRightConcat` | VPSHLDVW/VPSHRDVW | VBMI2 | Concatenate-and-shift |
| `Uint32x16.ShiftAllRight(n)` | VPSRLD | AVX512F | Broadcast shift |
| `Uint8x32.Permute(indices)` | VPERMB | VBMI+VL | 32-byte cross-lane permute (256-bit) |

**Missing from archsimd:**
- No `IsZero()` on 512-bit types -- use `Equal(zero).ToBits() == ^uint64(0)`
- No `LoadUint8x64Slice` with < 64 bytes (must ensure 64 readable bytes)

## Type Conversions (512-bit)

All As* conversions exist: `Uint8x64.AsInt8x64()`, `Uint8x64.AsUint16x32()`, `Uint8x64.AsUint32x16()`,
`Int8x64.AsUint8x64()`, `Uint16x32.AsUint8x64()`, `Uint32x16.AsUint8x64()`, etc.
`Mask8x64.ToInt8x64()` converts mask to vector (0xFF/-1 per set element, 0 per unset).

## Experiment 1: Full 512-bit Encode (Tasks 1+2+3) -- WIN

**Changes from AVX2:**
1. Process 48 src -> 64 dst per iteration (vs 24->32)
2. VPERMB for cross-lane reshuffle -- eliminates -4 offset trick and scalar preamble
3. 64-entry LUT via VPERMB for sextet->ASCII -- replaces 4 ops (SubSaturated+Greater+subtract+PermuteOrZeroGrouped) with 1

**Results (AMD EPYC, GB/s):**
| Size | AVX2 | AVX-512 | Speedup |
|------|------|---------|---------|
| 1K | 13.8 | 17.6 | 1.28x |
| 10K | 22.1 | 35.7 | 1.61x |
| 64K | 22.8 | 35.3 | 1.55x |

Verdict: **Large win.** VPERMB 64-entry LUT is the key enabler -- reduces the sextet->ASCII
mapping from 4 instructions to 1. Cross-lane VPERMB reshuffle also eliminates the -4 offset
load trick and its 24-byte scalar preamble.

## Experiment 2: Full 512-bit Decode (Tasks 3+4+5) -- REGRESSION

**Changes from AVX2:**
1. Process 64 src -> 48 dst per iteration (vs 32->24)
2. Mask register validation (Equal+ToBits) instead of And().IsZero()
3. VPERMB for byte compaction -- replaces VPSHUFB+VPERMD (2 ops -> 1)
4. Same nibble-LUT validation + VPMADDUBSW/VPMADDWD packing, just wider

**Results (AMD EPYC, GB/s):**
| Size | AVX2 | AVX-512 | Speedup |
|------|------|---------|---------|
| 1K | 19.0 | 14.5 | 0.76x |
| 10K | 24.1 | 21.9 | 0.91x |
| 64K | 25.2 | 22.2 | 0.88x |

Verdict: **Regression.** Decode has more serial dependencies in the pipeline
(validate -> translate -> VPMADDUBSW -> VPMADDWD -> compact). Each step is
double-pumped on Zen 4. The instruction count savings from VPERMB compaction
don't compensate for the increased latency.

## Experiment 3: VPCOMPRESSB for Decode (Task 6) -- NOT VIABLE

VPCOMPRESSB removes bytes by mask but doesn't do bit-level packing. The 4-sextet->3-byte
transformation requires arithmetic (shifting and combining 6-bit fields within bytes).
VPCOMPRESSB can only help after bytes are already packed, at which point VPERMB already works.

## Experiment 4: VPSHLDVW/VPSHRDVW for Encode (Task 7) -- NO IMPROVEMENT

VPSHLDVW is a funnel shift: `result = (x << shift) | (y >> (16-shift))` per 16-bit element.
For encode sextet extraction, this doesn't reduce ops vs AND+MulHigh/AND+Mul.
Both approaches need 4 operations to extract 4 sextets from packed bytes.
The mulhi/mullo trick is already optimal.

## Optimal Configuration for Zen 4

**Encode: AVX-512** (35 GB/s, 55% faster than AVX2)
**Decode: AVX2** (25 GB/s, AVX-512 decode regresses 12%)

This is wired up in `init()` as `bulkEncode = encode512; bulkDecode = decodeSIMD`.

## Experiment 5: AVX2+VBMI Hybrid Encode (Task 11) -- WIN at small sizes

**Change:** Replace PermuteOrZeroGrouped (VPSHUFB, per-lane) with Permute (VPERMB, cross-lane)
for the encode reshuffle. Eliminates the -4 offset load trick and 24-byte scalar preamble.
Same sextet extraction (mulhi/mullo) and ASCII mapping (SubSaturated+Greater+VPSHUFB).

**Results (AMD EPYC, GB/s):**
| Size | AVX2 | Hybrid | AVX-512 | Speedup (vs AVX2) |
|------|------|--------|---------|-------------------|
| 100 | 2.2 | 3.6 | 2.2 | 1.62x |
| 1K | 13.9 | 17.7 | 18.0 | 1.27x |
| 10K | 21.7 | 23.7 | 34.7 | 1.09x |
| 64K | 24.2 | 22.6 | 34.0 | 0.93x |

Verdict: **Win at small sizes** (eliminating the preamble helps a lot at 100-1K).
At large sizes, AVX-512's 2x width + VPERMB LUT still dominates.

## Experiment 6: AVX2+VBMI Hybrid Decode (Task 12) -- WIN everywhere

**Change:** Replace VPSHUFB+VPERMD (2-op compaction) with single VPERMB (cross-lane byte
permute). Same validation, translation, and VPMADDUBSW+VPMADDWD packing.

**Results (AMD EPYC, GB/s):**
| Size | AVX2 | Hybrid | AVX-512 | Speedup (vs AVX2) |
|------|------|--------|---------|-------------------|
| 100 | 4.0 | 3.9 | 2.6 | ~1.0x |
| 1K | 19.7 | 20.7 | 14.3 | 1.05x |
| 10K | 25.6 | 25.2 | 21.8 | ~1.0x |
| 64K | 24.7 | 27.0 | 22.1 | 1.09x |

Verdict: **Consistent small win.** Saving 1 instruction (VPERMD) per iteration adds up.
Best decode throughput at 64K: **27 GB/s**, beating both pure AVX2 and emmansun.

## Experiment 7: ConcatPermute (VPERMI2B) 64-entry LUT at 256-bit -- WIN

**Change:** Replace the 4-op ASCII mapping in hybrid encode:
```
SubSaturated + Greater + Sub + PermuteOrZeroGrouped  (4 ops)
```
with a single ConcatPermute (VPERMI2B) from a split 64-entry LUT:
```
lutLo.ConcatPermute(lutHi, sextets)  (1 op)
```
ConcatPermute selects from two 32-byte vectors (= 64 entries), same trick as
AVX-512's VPERMB LUT but at 256-bit width. Requires AVX-512 VBMI + VL.

**Results (AMD EPYC, GB/s):**
| Size | Hybrid | HybridLUT | Improvement |
|------|--------|-----------|-------------|
| 100 | 3.6 | 3.4 | ~same |
| 1K | 17.3 | **19.1** | +10% |
| 10K | 22.7 | **25.4** | +12% |
| 64K | 22.4 | **26.6** | +19% |

Verdict: **Significant win at all sizes ≥1K.** VPERMI2B removes 3 ops from the inner
loop (SubSaturated, Greater, Sub). HybridLUT now matches emmansun at 1K and beats
it at 10K+. At 100 bytes the per-iteration savings are too small relative to setup cost.

## Experiment 8: Adaptive Encode (AVX-512 bulk + hybridLUT tail) -- WIN

**Change:** Chain encode512 → encodeHybridLUT so the 24-47 byte remainder after AVX-512
is processed by hybridLUT (24 bytes/iter) instead of falling to scalar (3 bytes/iter).
Size-adaptive dispatch: hybridLUT for <256 bytes, AVX-512+hybridLUT for ≥256.

**Results (AMD EPYC, GB/s):**
| Size | Hybrid | AVX-512 | Adaptive | Best |
|------|--------|---------|----------|------|
| 100 | 3.5 | 2.2 | 3.3 | Hybrid |
| 1K | 17.8 | 17.7 | **22.5** | Adaptive (+27%) |
| 10K | 22.8 | 35.2 | 34.1 | AVX-512 (~same) |
| 64K | 23.9 | 35.0 | **36.0** | Adaptive (+3%) |

Verdict: **Win.** The hybrid tail reclaims 24-47 bytes that would otherwise go scalar.
At 1K this is a 27% improvement over either path alone. At 64K the tail benefit
is smaller but still measurable.

## Experiment 9: Chained Decode (hybrid + SWAR tail) -- REGRESSION

**Change:** Chain decodeHybrid → swarDecodeBulk so the 8-31 byte remainder is
processed by SWAR (8 bytes/iter) instead of falling to scalar.

**Results (AMD EPYC, GB/s):**
| Size | Hybrid | Chained |
|------|--------|---------|
| 100 | 4.1 | 1.8 |
| 1K | 20.0 | 15.1 |
| 10K | 25.5 | 24.7 |
| 64K | 26.2 | 26.4 |

Verdict: **Regression.** SWAR decode has significant per-call overhead (function call +
loop setup) that dominates for small tails. At 64K the 8-byte tail is too small
to benefit. Not worth the complexity.

## Experiment 10: Exhaustive AVX-512 Operation Audit

Audited ALL archsimd operations for potential base64 benefit:

### VPDPBUSD (DotProductQuadruple) -- NOT VIABLE
Could theoretically replace VPMADDUBSW+VPMADDWD (2→1) in decode packing.
For 4 sextets [a,b,c,d], we need: a×2^18 + b×2^12 + c×2^6 + d.
VPDPBUSD multiplies i8×u8 groups of 4 → i32. The multiplier 2^12=4096
**doesn't fit in a byte** (max signed i8 = 127). Not viable.

### GF2P8AFFINEQB (GaloisFieldAffineTransform) -- NOT VIABLE
Applies an 8×8 GF(2) bit-matrix per byte. Two problems:
1. Sextet extraction needs bits from **multiple** input bytes (e.g., s1 = (b0&3)<<4 | b1>>4).
   A single matrix operates on one byte at a time.
2. ASCII mapping requires modular addition, not XOR. GF(2) operations are XOR-based.

### VPERMI2B (ConcatPermute) -- NO CLEAR BENEFIT
Permutes from a combined 64-byte pool of two source vectors. Could theoretically
help by reshuffling from two chunks simultaneously, but our current approach already
loads one chunk and reshuffles within it. No clear instruction count reduction.

### VPEXPANDB (Expand) -- NO CLEAR BENEFIT
Inverse of Compress: scatters packed bytes to mask positions. Doesn't help with
base64's core operations (validation, translation, or bit packing).

### VPALIGNR (ConcatShiftBytesRightGrouped) -- NO CLEAR BENEFIT
Byte alignment within 128-bit groups. Our decode already uses optimal
VPMADDUBSW+VPMADDWD+VPERMB pipeline.

**Conclusion:** No remaining AVX-512 operations can improve the core encode/decode loops.
The current adaptive encode and hybrid decode are optimal for Zen 4.

## Optimal Configuration

**Encode: Adaptive** (AVX-512 bulk + hybridLUT tail, 35 GB/s at 64K, 22.5 GB/s at 1K)
- Size-adaptive: hybridLUT for <256 bytes, AVX-512+hybridLUT for ≥256
- AVX-512: VPERMB 64-entry LUT (4 ops → 1) + 2x width for bulk
- HybridLUT tail: ConcatPermute (VPERMI2B) for 64-entry LUT at 256-bit width
- Reclaims 24-47 byte remainder that would otherwise go scalar

**Decode: Hybrid** (AVX2-width + VPERMB compaction, 27 GB/s at 64K)
- AVX2 width avoids Zen 4 double-pumping penalty on serial chain
- VPERMB cross-lane compaction saves 1 instruction vs pure AVX2

## Full Benchmark Summary (AMD EPYC Zen 4, GB/s)

### Encode
| Size | SWAR | AVX2 | Hybrid | HybridLUT | AVX-512 | Adaptive | emmansun |
|------|------|------|--------|-----------|---------|----------|----------|
| 100 | 0.53 | 2.2 | 3.6 | 3.4 | 2.2 | 3.3 | 6.1 |
| 1K | 0.52 | 13.5 | 17.3 | 19.1 | 17.8 | **22.5** | 19.2 |
| 10K | 0.53 | 22.4 | 22.7 | 25.4 | 35.0 | **35.3** | 24.3 |
| 64K | 0.53 | 24.0 | 22.4 | 26.6 | 35.0 | **34.7** | 24.6 |

### Decode
| Size | SWAR | AVX2 | Hybrid | AVX-512 | emmansun | stdlib |
|------|------|------|--------|---------|----------|--------|
| 100 | 0.56 | 3.9 | **4.1** | 2.5 | 4.1 | 1.5 |
| 1K | 0.55 | 19.5 | **20.8** | 14.5 | 17.4 | 1.6 |
| 10K | 0.56 | 24.9 | **25.6** | 21.9 | 24.9 | 1.6 |
| 64K | 0.56 | 25.7 | **26.0** | 22.1 | 24.8 | 1.6 |

### vs emmansun (hand-tuned AVX2 asm competitor)
- Encode 64K: **35 GB/s vs 25 GB/s** (1.40x faster)
- Encode 1K: **22.5 GB/s vs 19 GB/s** (1.18x faster)
- Decode 64K: **26 GB/s vs 24 GB/s** (1.09x faster)
- Both using pure Go + archsimd intrinsics (no hand-written asm)
- emmansun wins at 100 bytes (6.1 vs 3.3 GB/s) due to hand-tuned asm overhead

## Required CPU Features per Path

| Path | Required Features | Width |
|------|-------------------|-------|
| SWAR | None (portable) | 64-bit |
| AVX2 encode | AVX2 | 256-bit |
| AVX2 decode | AVX2 | 256-bit |
| Hybrid encode | AVX2 + VBMI (+ VL) | 256-bit |
| Hybrid decode | AVX2 + VBMI (+ VL) | 256-bit |
| AVX-512 encode | AVX-512BW + VBMI | 512-bit |
| AVX-512 decode | AVX-512BW + VBMI + VBMI2 | 512-bit |
| Adaptive encode | AVX-512BW + VBMI + VL | 512+256 |

## Micro-optimization A/B Tests (AMD EPYC Zen 4)

### Local copies of global constants in SIMD loops

**Test:** Compare accessing SIMD constants from global vars vs local copies
in the hot loop. Go's compiler can keep local variables in registers but
must reload globals on every iteration.

**Result:** +37% encode, +32% decode without local copies. The compiler
reloads all 8 constants from memory each iteration without local copies.
**Verdict: KEEP.**

### Unsafe pointer loads vs slice-based LoadUint8x32Slice

**Test:** Replace `archsimd.LoadUint8x32Slice(src[si-4 : si+28])` with
`unsafe.Pointer`-based loads to eliminate bounds checks.

**Result:** Zero measurable change on both Rosetta and EPYC. The compiler
already eliminates bounds checks when the slice capacity is proven sufficient.
**Verdict: REMOVE. No benefit, worse readability.**

## Rosetta 2 Performance Investigation

### Background

Under Apple's Rosetta 2 (x86_64 → ARM64 AOT translation), our SIMD-intrinsic
encode runs at ~2.3 GB/s while emmansun's hand-written assembly runs at ~5.9 GB/s.
On native AMD EPYC, both achieve ~19 GB/s (97% parity). This is a Rosetta-specific
2.5x performance gap.

### Hypotheses Tested and Ruled Out

| # | Hypothesis | Test | Result |
|---|-----------|------|--------|
| 1 | Bounds checks in hot loop | Unsafe pointer loads/stores | Zero change (2.3 GB/s) |
| 2 | Index arithmetic overhead | Pointer-based iteration | Zero change |
| 3 | Dual loop condition overhead | Single remaining counter | Zero change (even with 18-instruction loop) |
| 4 | Loop structure | `for range N` counted loop | Zero change |
| 5 | Mixed scalar+SIMD in function | Split SIMD into //go:noinline helper | Zero change |
| 6 | Indirect function call via bulkEncode | Direct call via export_test.go | Zero change |
| 7 | Benchmark interference/warmup | Separate benchmark runs | Same numbers |
| 8 | Missing VZEROUPPER | Added vzeroupper() call at function exit | Zero change |
| 9 | ABI0 vs ABIInternal calling convention | Hand-written .s assembly (ABI0) | Zero change (2.2 GB/s) |
| 10 | Go runtime stack checks/GC preemption | NOSPLIT .s function, no stack checks | Zero change |
| 11 | Function prologue/frame pointer | Minimal .s prologue | Zero change |
| 12 | 16-byte alignment of hot loop | NOP-padded to align VMOVDQU to mod16=0 | Zero change |
| 13 | Loop pattern (DECQ/JNE vs CMPQ/JB+JMP) | Matched emmansun's exact loop structure | Zero change |
| 14 | VMOVDQU vs VBROADCASTI128 constants | Used VBROADCASTI128 from RODATA | Zero change |
| 15 | Go compiler vs hand-written assembly | Same algorithm in .s file | Zero change |
| 16 | Go-specific overhead | Same algorithm in C (clang -O2) | Also ~2.0 GB/s |

### Key Finding: Byte-identical code, 2.8x performance difference

We copied emmansun's entire `encodeAsm` function verbatim into our package
under a different symbol name (`copyEncodeAsm`). Same dead code paths (SSE, AVX),
same AVX2 hot loop, same RODATA constants, same function structure.

**Machine code comparison** (hot loop body, 17 instructions):

Both functions produce **byte-for-byte identical** machine code for all 17
instructions. The ONLY difference is a 4-byte RIP-relative displacement in one
VPSHUFB instruction (pointing to different RODATA symbols at addresses 64 bytes
apart in the same section).

| Metric | emmansun original | our copy |
|--------|------------------|----------|
| Function address | 0x11acd00 | 0x11b40a0 |
| Hot loop VMOVDQU addr | 0x11aced0 (mod16=0) | 0x11b425b (mod16=11) |
| RODATA address | 0x11d7a40 | 0x11d7a00 |
| Performance (1K) | **6.3 GB/s** | **2.2 GB/s** |
| Performance (10K) | **6.4 GB/s** | **2.3 GB/s** |
| Performance (64K) | **6.4 GB/s** | **2.3 GB/s** |
| Ratio | 1.0x | **2.8x slower** |

This was also confirmed with a standalone C program (clang -O2 -mavx2): the same
algorithm also runs at ~2.0 GB/s under Rosetta, confirming it's not Go-specific.

Calling emmansun's original function via `go:linkname` from our benchmark
achieves the full 6.3 GB/s, ruling out calling-context effects.

### Conclusion

The performance difference is an artifact of Rosetta 2's AOT translation.
Two copies of the same function in the same binary produce different-quality
ARM64 translations based on the function's position in the code section.

This is NOT actionable from user code — there is no way to control function
placement in Go's linker. On native x86_64 hardware (AMD EPYC), our code
achieves **97% parity** with emmansun at 1K and **107%** at 10K.

### Native AMD EPYC AVX2-only Results

| Size | Our AVX2 | emmansun | Parity |
|------|----------|----------|--------|
| 1K | 18.5 GB/s | 19.0 GB/s | **97%** |
| 10K | 23.5 GB/s | 22.0 GB/s | **107%** |
| 64K | 24.0 GB/s | 24.6 GB/s | **98%** |

### init() dispatch logic
```
if AVX2 + AVX-512BW + VBMI + VL:
    bulkEncode = encodeSizeAdaptive  // AVX-512 bulk + hybrid tail
    bulkDecode = decodeHybrid        // AVX2 width + VPERMB compaction
elif AVX2 + VBMI + VL:
    bulkEncode = encodeHybrid
    bulkDecode = decodeHybrid
elif AVX2:
    bulkEncode = encodeSIMD
    bulkDecode = decodeSIMD
else:
    (falls through to SWAR default)
```
