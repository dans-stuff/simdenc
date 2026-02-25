# AVX-512 Base64 Research Notes

## Optimization Catalog (AMD EPYC Zen 4)

Every non-obvious pattern in the codebase, with its measured A/B impact:

| # | Optimization | Delta | Verdict |
|---|---|---|---|
| 1 | By-value struct args (replaces local copies) | +37% enc, +32% dec | KEEP — Go lacks LICM (see #18-21) |
| 2 | 512-bit encode (VPERMB LUT, 48B/iter) | +28% at 1K, +55% at 64K vs AVX2 | KEEP |
| 3 | VPERMB cross-lane reshuffle (no preamble) | +62% at 100, +27% at 1K vs AVX2 | KEEP |
| 4 | ConcatPermute (VPERMI2B) LUT vs 4-op | +10% at 1K, +19% at 64K | KEEP |
| 5 | VPERMB decode compaction vs VPSHUFB+VPERMD | +5% at 1K, +9% at 64K | KEEP |
| 6 | Preamble elimination (offset sharing) | +73% at 200, +30% at 2K | KEEP — key insight |
| 7 | `d := dst[di:]` reslice before StoreSlice | +3% at 10K/64K | KEEP — zero complexity cost |
| 8 | -4 offset load trick (AVX2 encode) | Required (no VPERMB without VBMI) | KEEP |
| 9 | Unsafe pointer loads | 0% | REMOVED — no benefit |
| 10 | `for range n` counted loops | -27% decode at 64K | REMOVED — regression |
| 11 | srcEnd bounds fix (`-28` not `-32`) | +93% enc at 100B | KEEP — was losing an iteration |
| 12 | Branchless decode special-char handling | 0% (cleaner code) | KEEP — removes branch, same speed |
| 13 | Package-level function vars (no stub file) | 0% | REMOVED — replaced by direct dispatch (see #15) |
| 14 | Branchless decode error accumulation | -3% to -10% | REMOVED — loop-carried dependency |
| 15 | Direct dispatch (no function variable) | +9% enc at 100B | KEEP — eliminates indirect call overhead |
| 16 | Monolithic encode (all stages inlined) | +12% enc at 100B, +17% at 128B | KEEP — no sub-function calls in hot path |
| 17 | Inline scalar decode tail | +33% dec at 100B, +43% at 128B | KEEP — eliminates stdlib tail call (~10ns) |
| 18 | Go lacks LICM (root cause investigation) | N/A | FINDING — all globals reload every iteration |
| 19 | Struct fields behind pointer | -22% to -27% dec | REMOVED — same LICM problem as globals |
| 20 | Closures capturing constants | -69% to -87% | REMOVED — SIMD intrinsics not inlined in closures |
| 21 | By-value struct arguments | 0% (same as hoisting) | KEEP — replaces manual hoisting, cleaner code |
| 22 | SIMD constant construction in init() | 0% | KEEP — cleaner init, Broadcast256 pitfall documented |
| 44 | SIMD constant declaration strategies | -5% to -6% for alternatives | FINDING — package-level var + As cast is optimal |

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

## Optimal Configuration for Zen 4 (early — superseded by Experiment 11)

**Encode: AVX-512** (35 GB/s, 55% faster than AVX2)
**Decode: AVX2** (25 GB/s, AVX-512 decode regresses 12%)

Note: This was the early finding. Experiment 11 (preamble elimination) simplified
the encode pipeline further — see the updated Optimal Configuration section below.

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

## Experiment 11: Preamble Elimination via Offset Sharing -- KEY INSIGHT

**Discovery:** The performance gap between AVX2 encode (with its 6-byte scalar preamble)
and VBMI encode (which uses cross-lane VPERMB to skip the preamble) was NOT about
instruction quality. It was entirely about **iteration count**.

The AVX2 -4 offset trick requires 4 readable bytes before the first real byte. The old
`encodeAVX2` used a 6-byte scalar preamble (producing 8 dst bytes) to provide this overlap.
But at certain sizes, the preamble cost an entire SIMD iteration — at 200 bytes, AVX2 got
7 iterations while VBMI got 8.

**Key insight:** encode512 always leaves si ≥ 48 after any iteration. Passing `src[si-4:]`
to an AVX2 function that starts at si=4 (no preamble) gives it the 4-byte overlap for free.
When encode512 can't run, a 6-byte stdlib encode provides the same overlap.

**encodeAVX2NoPreamble:** Same as encodeAVX2 but starts at si=4 with no scalar preamble.
The caller guarantees 4 readable bytes exist before the data.

**Iteration count proof (VBMI vs AVX2NoPreamble after encode512):**

Every input size produces identical SIMD iteration counts. The preamble was the
*entire* source of the performance difference.

**Results (AMD EPYC, GB/s, encode512 + AVX2NoPreamble + stdlib):**
| Size | Old (512+VBMI) | New (512+AVX2np) | Delta |
|------|----------------|------------------|-------|
| 200 | 5.5 | 9.5 | +73% |
| 2000 | 24.0 | 31.3 | +30% |
| 5000 | 30.6 | 35.2 | +15% |
| 10000 | 35.2 | 35.3 | ~same |

Verdict: **Major simplification.** Eliminated encodeVBMI entirely. The "preamble elimination"
philosophy — maximize bytes on the SIMD path by removing setup overhead — matters more than
using fancier instructions.

## Experiment 12: Simplified Dispatch (encodeSIMDFull) -- SIMPLIFICATION

**Change:** Replaced function-pointer dispatch (`bulkEncode`) and multiple encode functions
(encodeAVX2, encodeVBMI, encodeHybridLUT, encodeSizeAdaptive) with a single `encodeSIMDFull`
function containing a clean 3-stage pipeline:

```
Stage 1: encode512        (48 bytes/iter, needs ≥64 readable bytes, AVX-512 only)
Stage 2: encodeAVX2NoPreamble (24 bytes/iter, needs 4-byte overlap from prior stage)
Stage 3: stdlib           (remaining bytes)
```

When encode512 can't run (no AVX-512 or input too small), a 6-byte stdlib preamble
provides the overlap that AVX2 needs.

Also added `len(src) < 34` fast exit to `Encode()` to avoid SIMD overhead for small inputs.
At 48 bytes this recovered function-pointer overhead: 1,624 MB/s vs 1,109 MB/s before.

Replaced stdlib calls for preamble/tail with inline `encodeScalarTail()` — avoids method
dispatch and loop setup overhead of `base64.RawStdEncoding.Encode()`. This improved
small-size encode by 7-18% (e.g., 128 bytes: 5.85 → 6.92 GB/s, beating emmansun).

**Dead code removed:** encodeVBMI, encodeAVX2 (with preamble), encodeScalar, SWAR encode,
encodeHybridLUT, encodeSizeAdaptive, all VBMI-encode-only constants.

## Experiment 13: srcEnd Bounds Fix in encodeAVX2NoPreamble -- BUG FIX

**Discovery:** `encodeAVX2NoPreamble` used `srcEnd = len(src) - 32`, but the load is
`src[si-4 : si+28]`, meaning the upper bound is `si+28`, not `si+32`. The off-by-4
caused the loop to exit one iteration early at certain sizes, dropping to the scalar
tail prematurely.

**Fix:** `srcEnd = len(src) - 28`.

**Results (AMD EPYC, GB/s):**
| Size | Before | After | Delta |
|------|--------|-------|-------|
| 100 | 2.43 | 4.70 | **+93%** |
| 1K | 18.9 | 18.9 | ~same |
| 10K | 32.3 | 32.3 | ~same |
| 64K | 34.5 | 34.5 | ~same |

Verdict: **Critical fix at small sizes.** At 100 bytes, the old code got 2 SIMD iterations
instead of 3, forcing 24 extra bytes through the scalar tail. At large sizes the lost
iteration is a rounding error.

## Experiment 14: Branchless Decode Special-Char Handling -- SIMPLIFICATION

**Discovery:** The decode loop had a branch `if isURLAlphabet { isSpecial = isSpecial.And(shift) }`
to handle the different special characters ('/' vs '_'). Since the AND with 0xFF is a
no-op for standard encoding, we can make it unconditional.

**Change:** Store `decSpecialShift` per-alphabet (0xFF for std, 0x03 for URL). The AND
is always executed — it's a no-op for std and the actual shift for URL.

**Result:** Zero measurable performance change. Removes a branch from the hot path
and an `isURL bool` field from the alphabet struct. **Verdict: KEEP for simplicity.**

## Experiment 15: Package-Level Function Variables -- ARCHITECTURE

**Discovery:** The original design used a `base64_other.go` stub file with `//go:build !amd64`
to provide no-op implementations on non-SIMD platforms. This is unnecessary.

**Change:** `base64.go` declares `var simdEncode func(...)` and `var simdDecode func(...)`,
initially nil. The amd64 init() sets them to real implementations. `Encode()`/`Decode()`
check for nil and fall back to `encoding/base64`.

**Result:** Eliminated the stub file entirely. Zero performance change. The nil check
is a single comparison that the branch predictor handles trivially. **3 files total**
(base64.go, base64_amd64.go, base64_test.go), down from 4.

Also eliminated in the same pass:
- `rawBase` field on Encoding → replaced with `rawStdlib[enc.alphabet]` package-level lookup
- SIMD branches in helper methods (EncodeToString, DecodeString, etc.) → they just call Encode/Decode
- Inlined EncodedLen/DecodedLen as pure math (no delegation to stdlib)

## Experiment 16: Branchless Decode Error Accumulation -- REGRESSION

**Change:** Replace `if !IsZero() { break }` with `errors = errors.Or(validation)`,
checking once after the loop. If any error found, return (0, 0) to let stdlib re-decode
with exact error reporting. Goal: remove a branch from the hot path.

**Results (AMD EPYC, GB/s):**
| Size | Branching | Branchless | Delta |
|------|-----------|------------|-------|
| 100 | 3.3 | 3.2 | -3% |
| 1K | 17.3 | 16.8 | -3% |
| 10K | 24.7 | 23.1 | -6% |
| 64K | 24.8 | 22.4 | -10% |

Verdict: **Regression.** The `errors = errors.Or(...)` creates a **loop-carried dependency**:
every iteration must read `errors`, OR the new result, and write back. This serializes part
of the execution pipeline. The original branch (`IsZero() → break`) has no loop-carried
dependency — the branch predictor handles the always-taken path at near-zero cost.

### How much does validation cost?

Removing validation entirely (no VPSHUFB lookups, no VPTEST) gives:
| Size | With validation | No validation | Delta |
|------|----------------|---------------|-------|
| 1K | 17.3 GB/s | 20.0 GB/s | +16% |
| 10K | 24.7 GB/s | 29.7 GB/s | +20% |
| 64K | 24.8 GB/s | 30.4 GB/s | +23% |

So validation costs ~20% of decode throughput. However:
- **Cannot be replaced with a range check.** Checking `sextets ≤ 63` misses 57 invalid bytes
  that map to valid-looking sextets through the roll table (0x00-0x0F, 0x10-0x1F, etc.).
- **Cannot be deferred.** Branchless accumulation adds a loop-carried dep and is slower.
- **The 4-instruction validation (2 VPSHUFB + VPAND + VPTEST) is already minimal**
  for the Muła/Nojiri scheme. The nibble lookups cannot be reduced further.

## Optimal Configuration (Final)

**Encode: 4-tier dispatch** (SSE → fused SSE+AVX2 → encode512+SSE → scalar)
- encodeSSE: 12 bytes/iter, 128-bit, for inputs < 120 bytes
- encodeSSEAVX2: SSE bookends + AVX2 middle, for 120-255 bytes (or 120+ without AVX-512)
- encode512 + encodeSSE cleanup: 48 bytes/iter 512-bit bulk + SSE remainder, for 256+ bytes
- Inline scalar tail for < 28 bytes and final partial triplets
- Peak Zen 4: 36.7 GB/s. Peak Zen 5: 63.6 GB/s

**Decode: 3-tier dispatch** (SSE → fused SSE+AVX2 / VBMI → scalar)
- decodeSSE: 16 bytes/iter, 128-bit, for inputs 16-44 bytes
- decodeVBMI: 32 bytes/iter, VPERMB 256-bit, for 45+ bytes (AVX-512 machines)
- decodeSSEAVX2: SSE bookends + AVX2 middle, for 45+ bytes (no AVX-512)
- Inline scalar tail for < 16 bytes and final partial quads
- Peak Zen 4: 24.4 GB/s. Peak Zen 5: 31.2 GB/s

## Full Benchmark Summary (AMD EPYC Zen 4, GB/s)

### Encode (current: SSE + fused SSE+AVX2 + encode512+SSE + scalar)
| Size | simdenc | emmansun | stdlib | vs emmansun |
|------|---------|----------|--------|-------------|
| 64 | 4.2 | 4.3 | 1.3 | 0.98x |
| 256 | **14.5** | 10.7 | 1.4 | **1.36x** |
| 1K | **28.0** | 19.1 | 1.4 | **1.47x** |
| 10K | **36.0** | 23.3 | 1.4 | **1.55x** |
| 64K | **36.7** | 23.4 | 1.4 | **1.57x** |
| 1M | **34.9** | 23.9 | 1.5 | **1.46x** |

### Decode (current: SSE + VBMI/fused SSE+AVX2 + scalar)
| Size | simdenc | emmansun | stdlib | vs emmansun |
|------|---------|----------|--------|-------------|
| 64 | **2.0** | 1.3 | 0.7 | **1.54x** |
| 256 | **5.7** | 4.3 | 0.8 | **1.33x** |
| 1K | **10.0** | 8.4 | 1.6 | **1.19x** |
| 10K | 24.1 | 24.8 | 1.5 | 0.97x |
| 64K | 24.4 | 25.0 | 1.5 | 0.98x |
| 1M | **24.1** | 23.9 | 1.5 | **1.01x** |

### vs emmansun (hand-tuned AVX2 asm competitor)
**Zen 4:**
- Encode 64K: **36.7 GB/s vs 23.4 GB/s** (1.57x faster)
- Decode 64K: 24.4 GB/s vs 25.0 GB/s (0.98x, within noise)

**Zen 5:**
- Encode 64K: **63.6 GB/s vs 28.7 GB/s** (2.22x faster)
- Decode 64K: 31.2 GB/s vs 32.4 GB/s (0.96x, hand-written asm advantage)

All using pure Go + archsimd intrinsics (no hand-written assembly).

## Required CPU Features per Path

| Path | Required Features | Width | Role |
|------|-------------------|-------|------|
| encodeSSE | SSSE3 (via AVX2 gate) | 128-bit | Small encode (< 120 B), cleanup after encode512 |
| encodeSSEAVX2 | AVX2 | 128+256-bit | Medium encode (120-255 B), fused bookend |
| encode512 | AVX-512BW + VBMI | 512-bit | Large encode (256+ B) |
| decodeSSE | SSE4.1 | 128-bit | Small decode (16-44 B) |
| decodeSSEAVX2 | AVX2 | 128+256-bit | Medium/large decode (no AVX-512), fused bookend |
| decodeVBMI | AVX2 + VBMI (VL) | 256-bit | Large decode (45+ B, AVX-512 machines) |
| stdlib fallback | None | scalar | Non-amd64 or no AVX2 |

## Micro-optimization A/B Tests (AMD EPYC Zen 4)

### By-value struct arguments (replaces local copies of globals)

**Test:** Compare accessing SIMD constants from global vars vs by-value struct args.
Go lacks LICM entirely (see experiment 18), so globals are reloaded every iteration.
By-value struct args are stack-local and stay in registers.

**Result:** +37% encode, +32% decode vs bare globals. By-value struct args perform
identically to manual `v := global` hoisting, but the function signature explicitly
declares its dependencies instead of relying on mysterious copy-before-loop patterns.
**Verdict: KEEP — adopted as standard pattern (see experiment 21).**

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

## Competitive Landscape (AMD EPYC Zen 4)

All numbers in GB/s, measured on the same machine with benchtime=2s, count=3.

### Encode
| Size | **simdenc** | emmansun (Go asm) | cristalhq (Go) | stdlib | aklomp (C) | simdutf (C++) |
|------|------------|-------------------|----------------|--------|------------|---------------|
| 100 | **4.3** | 5.8 | 2.3 | 1.3 | 1.9 | 3.5 |
| 1K | **19.3** | 18.1 | 1.3 | 1.4 | 17.0 | 20.3 |
| 10K | **33.4** | 22.8 | 1.3 | 1.4 | **46.6** | 26.9 |
| 64K | **32.3** | 21.9 | 1.3 | 1.3 | **39.1** | 20.6 |

### Decode
| Size | **simdenc** | emmansun (Go asm) | cristalhq (Go) | stdlib | aklomp (C) | simdutf (C++) |
|------|------------|-------------------|----------------|--------|------------|---------------|
| 100 | **3.0** | 3.8 | 1.5 | 1.3 | 3.7 | 1.8 |
| 1K | **15.5** | 15.3 | 1.6 | 1.5 | 16.6 | 12.9 |
| 10K | **23.1** | 22.6 | 1.6 | 1.5 | 22.9 | 21.5 |
| 64K | **23.7** | 24.3 | 1.7 | 1.6 | 22.0 | 21.4 |

### Libraries tested
- **simdenc** — this library; Go + archsimd intrinsics, AVX-512 + AVX2
- **emmansun/base64** — hand-tuned AVX2 Go assembly
- **cristalhq/base64** — pure Go, Turbo-Base64 algorithm port (no SIMD)
- **aklomp/base64** — C library with AVX-512 VBMI, compiled with GCC 13 -O3 -march=native
- **simdutf** — C++ library with "icelake" path, compiled with GCC 13 -O3 -march=native
- **encoding/base64** — Go stdlib

### Key findings
1. **aklomp beats us at large encode by 20-40%** — they use VPMULTISHIFTQB, an AVX-512 VBMI
   instruction that extracts 8 arbitrary bit fields from 64-bit elements in one step. This
   replaces our 4-instruction mulhi/mullo sextet extraction. archsimd does not expose this
   instruction (as of Go 1.26).
2. **We beat every Go library at ≥1K.** emmansun is the closest competitor.
3. **For decode, we match or beat aklomp (C) at all sizes.** Our decode pipeline is already
   optimal — the Muła/Nojiri validation + VPERMB compaction matches what C compilers produce.
4. **simdutf is surprisingly slow** — its "icelake" path underperforms both aklomp and simdenc.
5. **cristalhq is stdlib-speed** — the "3x faster" claims in their README are from older Go versions.

### Missing instruction: VPMULTISHIFTQB
aklomp's encode advantage comes from `_mm512_multishift_epi64_epi8` (VPMULTISHIFTQB),
which extracts 8 arbitrary 8-bit fields from each 64-bit lane. For base64 encode, this
replaces the entire mulhi/mullo sextet extraction pipeline (4 ops → 1 op). It requires
AVX-512 VBMI, which Zen 4 has. However, Go's archsimd package does not expose this
instruction. If it did, we could likely match aklomp's encode throughput.

### init() dispatch logic
```
// Package-level function vars: simdEncode, simdDecode (nil by default → stdlib fallback)
if AVX2:
    simdEncode = doEncode          // 4-tier: SSE → SSE+AVX2 → 512+SSE → scalar
    simdDecode = doDecode          // 3-tier: SSE → VBMI/SSE+AVX2 → scalar
    populate encAlphas, decAlphas  // per-alphabet SIMD constants (std + URL)
    populate shared constants      // nibbleMask, combinePairs, etc.
    if AVX512 + VBMI:
        hasAVX512 = true           // enables encode512 + VPERMB decode
        populate 512-bit encode constants
```

`doEncode` dispatches: n<28 → scalar only, n<120 → encodeSSE, n<256 or no AVX-512 → encodeSSEAVX2, else → encode512 + encodeSSE cleanup. All paths include inline scalar tail.

`doDecode` dispatches: n<16 → scalar only, n<45 → decodeSSE, n≥45+AVX-512 → decodeVBMI, else → decodeSSEAVX2. All paths include inline scalar tail.

No stub file needed. When `simdEncode`/`simdDecode` are nil (non-amd64 or no AVX2), `base64.go` delegates directly to `encoding/base64`.

## Experiment 18: Go Lacks LICM — Root Cause of Global Reloading

**Discovery:** The +37% gain from hoisting globals into locals (experiment 1) is NOT caused by
goroutine aliasing, pointer escape, or function calls. Go's compiler simply **does not implement
Loop Invariant Code Motion** (LICM). Even the most trivial case reloads:

```go
var globalInt int
func sum(n int) int {
    s := 0
    for i := range n { s += globalInt }  // reloads globalInt EVERY iteration
    return s
}
```

No stores, no function calls, no aliasing — the compiler still emits a memory load per iteration.
This is tracked in [golang/go#63670](https://github.com/golang/go/issues/63670). An experimental
LICM pass exists but is not production-ready.

**Implication:** ANY global variable read inside a hot loop must be hoisted manually. This applies
to all Go code, not just SIMD.

## Experiment 19: Struct Fields Behind Pointers — REGRESSION

**Test:** Pack all decode SIMD constants into a struct, pass as `*decodeConsts` pointer,
read fields directly in the loop (no manual hoisting).

**Results (AMD EPYC, GB/s, decode):**
| Size | Hoisted locals | Struct pointer | Delta |
|------|---------------|----------------|-------|
| 1K | 17.3 | 13.5 | -22% |
| 64K | 24.8 | 18.1 | -27% |

**Why:** Same LICM problem. Reading `p.nibMask` through a pointer reloads from memory every
iteration, identical to reading a global. The compiler can't prove the pointer target isn't
modified. The archsimd documentation explicitly warns: "It is not recommended to put [vector
types] in an aggregate type."

**Verdict: REMOVED.** Pointer-to-struct does not solve hoisting.

## Experiment 20: Closures — REGRESSION (7-8x slower)

**Hypothesis:** A factory function captures all constants in a closure, returns the inner
decode function. Captured locals are loaded once at function entry (confirmed by assembly).
Eliminates hoisting entirely.

**Results (AMD EPYC, GB/s):**
| Size | Direct function | Closure | Delta |
|------|----------------|---------|-------|
| 100 | 16.0 | 5.0 | -69% |
| 10K | 46.0 | 6.2 | -87% |
| 64K | 49.1 | 6.2 | -87% |

**Root cause:** Go's compiler **does not inline SIMD intrinsics inside closure bodies.**
`LoadUint8x32Slice` and `StoreSlice` become actual CALL instructions instead of inline
VMOVDQU. This happens even when:
- The closure is called directly from a local variable (compiler can see the target)
- The closure body is a normal function with a loop (not a trivial one-liner)

Assembly proof:
- `decodeDirect`: 2 CALLs (both `runtime.panicBounds`)
- `makeDecoder.func1`: 5 CALLs (2 `panicBounds` + `LoadUint8x32Slice` + `StoreSlice` + `morestack`)

Each non-inlined SIMD call has full calling convention overhead (spill/reload YMM registers).
At 2048 iterations for 64K input, that's ~4096 extra function calls.

**Verdict: NOT VIABLE.** Closures are fundamentally incompatible with SIMD intrinsic inlining.

## Experiment 21: By-Value Struct Arguments — WIN (replaces hoisting)

**Hypothesis:** Pass all constants as a by-value struct argument. The copy onto the callee's
stack acts as an implicit hoist. The callee reads struct fields as stack-local values.

**Results (AMD EPYC, GB/s):**
| Size | Manual hoisting | By-value struct arg | Struct built locally | Delta |
|------|----------------|--------------------|--------------------|-------|
| 100 | 16.3 | 17.0 | 16.6 | ~same |
| 10K | 45.6 | 46.5 | 45.2 | ~same |
| 64K | 47.3 | 48.9 | 48.0 | ~same |

Assembly confirms: all SIMD intrinsics fully inlined, only `panicBounds` CALLs remain.
Identical to manual hoisting.

**Also tested:** `c := globalStruct` (copy entire global struct into local). Same performance
as individual hoisting — the compiler treats all fields of a stack-local struct as locals.

**Architecture adopted:** Split constants into shared + per-alphabet structs:
- `encodeConsts` / `decodeConsts` — one global instance, shared across alphabets
- `encodeAlpha` / `decodeAlpha` — `[2]` array, indexed by alphabet

Dispatchers are trivial:
```go
func doEncode(alphabet uint8, dst, src []byte) {
    encode(encShared, encAlphas[alphabet], dst, src)
}
```

**Benefits over manual hoisting:**
1. No mysterious `v := globalVec` lines that look like dead code
2. Function signatures declare their dependencies explicitly
3. Shared constants are defined once, not duplicated per alphabet
4. `init()` populates struct fields directly — no intermediate variables needed

**Verdict: KEEP.** This is now the standard pattern for all SIMD functions.

## Experiment 22: SIMD Constant Construction in init()

**Change:** Replaced byte-array construction (`LoadUint8x32(&[32]byte{...})`) with SIMD
operations where cleaner: `BroadcastUint8x32`, `InterleaveLo`, `SetLo/SetHi`.

**Key discovery:** `archsimd.Broadcast256()` on element types (Uint8x16, Uint16x8) broadcasts
**element zero** (VPBROADCASTB/W), NOT the 128-bit lane (VBROADCASTI128). To duplicate a
128-bit lane into both halves of a 256-bit vector, use `SetLo(v).SetHi(v)`.

Helper functions:
```go
func dupBytes(v archsimd.Uint8x16) archsimd.Uint8x32 {
    var z archsimd.Uint8x32
    return z.SetLo(v).SetHi(v)
}

func alt16(even, odd uint16) archsimd.Uint16x16 {
    lane := BroadcastUint16x8(even).InterleaveLo(BroadcastUint16x8(odd))
    var z archsimd.Uint16x16
    return z.SetLo(lane).SetHi(lane)
}
```

**Verdict: KEEP.** Cleaner init code, no byte arrays for simple patterns.

## Experiment 23: Monolithic Function Body Size — KEY FINDING

**Hypothesis:** The 40-55% regression from by-value struct args (experiment 21 at scale) was caused
by copying large structs. Testing on branch `monolithic-experiment`.

**Experiments (AMD EPYC, GB/s):**

| Approach | Encode 64KB | Decode 64KB | Notes |
|---|---|---|---|
| Separate per-tier functions (baseline) | 35.1 | 24.6 | 5 functions, 482 lines |
| Fully monolithic, shared hoist at top | 19.1 (-46%) | 17.0 (-31%) | Individual `v := global` hoisting |
| Fully monolithic, branched hoist | 18.0 (-49%) | 20.8 (-15%) | Constants inside if/else branches |
| Hybrid (encode512 separate, rest branched) | 28.5 (-19%) | 20.8 (-15%) | encode512 must be separate |
| By-value struct (monolithic) | 15.9 (-55%) | — | Struct copy + large body |

**Root cause: NOT struct copy.** Even with individual `v := global` hoisting (no struct at all),
the fully monolithic function regresses 31-46%. The compiler generates worse code for larger
function bodies — more register spilling, worse scheduling. This is a fundamental Go compiler
limitation, distinct from LICM.

**Key findings:**
1. encode512 (512-bit) MUST be a separate function — combining 512-bit and 256-bit tiers in
   one body causes massive register pressure
2. Branched hoisting (constants inside if/else) recovers significant performance vs shared
   hoisting at function entry — the compiler treats each branch as a smaller scope
3. Even with branched hoisting, monolithic functions still regress 15-19% vs separate functions
4. The regression scales with function body size: more SIMD operations in one body = worse code

**Verdict:** Separate per-tier functions are the only way to get full performance. The hybrid
approach is a middle ground but still loses 15-19% at large sizes.

## Experiment 24: Selective Hoisting — Not All Variables Need It

**Hypothesis:** Variables used late in the loop's dependency chain may not need hoisting —
the load latency gets masked by preceding computation.

**Test:** Starting from a unified decode function (10 hoisted SIMD vectors, scalar tail in
separate function), systematically un-hoist variables and measure impact.

**Results (AMD EPYC, GB/s, decode):**

| Config | Hoisted | 1KB | 10KB | 64KB |
|---|---|---|---|---|
| All hoisted | 10 | 17.1 | 21.2 | 21.7 |
| No extractVBMI | 9 | 16.9 | 21.2 | 21.6 |
| No combineQuads | 9 | 17.2 | 21.9 | 21.1 |
| No quads + no extractVBMI | 8 | 16.4 | 20.5 | 20.1 |
| No special + shift + extractVBMI | 7 | 14.5 | 18.1 | 18.3 |

**Key findings:**
1. **Late-pipeline variables can stay as globals at zero cost.** `extractVBMI` (VPERMB at end)
   and `combineQuads` (VPMADDWD after VPMADDUBSW) each cost 0-2% when un-hoisted individually.
   Their load latency is hidden behind the chain of preceding SIMD operations.
2. **Mid-pipeline variables must be hoisted.** `a.special` and `a.shift` (VPCMPEQB + VPAND,
   used in the middle of the pipeline) cost 16% when un-hoisted. The load competes with
   active computation.
3. **Non-linear interaction.** Un-hoisting both late-pipeline vars together costs 7%, while
   each individually costs ~0%. The two loads compete for memory bandwidth or pipeline slots.
4. **Heuristic:** Hoist variables used before or during the longest dependency chain. Variables
   used only after the chain completes (last 1-2 operations before store) may not need hoisting.

### Assembly analysis (decode, 10 hoisted, VBMI path)

- Function size: 371 bytes, `locals=0x8` (zero SIMD stack spills)
- Register allocation: Y0-Y9 for constants, Y10-Y13 for temporaries (14/16 YMM used)
- VBMI branch: `TESTB DL, DL / JEQ / VPERMB / JMP` — 4 instructions
- AVX2 fallback: loads extractShuffle/extractPermute from memory (cold path, un-hoisted)
- Only CALLs: `runtime.panicBounds` (bounds check failures, never taken in valid input)

Despite zero spills and clean register allocation, this unified function still runs 12% slower
than the separate-function baseline. The regression source is unclear — possibly branch
misprediction effects or compiler scheduling differences around the if/else.

## Experiment 25: Scalar Tail Separation

**Test:** Move the scalar decode tail from inside the SIMD function to a separate dispatcher.

**Results (AMD EPYC, GB/s, decode):**
| Size | Inline tail | Separate tail | Delta |
|---|---|---|---|
| 1KB | 16.8 | 17.1 | +2% |
| 10KB | 20.0 | 21.2 | +6% |
| 64KB | 20.6 | 21.7 | +5% |

**Verdict:** Separating the scalar tail reduces function body size, improving register allocation
for the SIMD loop. The scalar tail adds ~30 lines of code that the compiler must account for
in its allocation decisions, even though it rarely executes.

## Experiment 26: Branch Inside SIMD Loop Costs 8-9% — KEY FINDING

**Hypothesis:** The 12% regression in unified decode (vs separate functions) comes from the
`if useVBMI { VPERMB } else { VPSHUFB+VPERMD }` branch inside the hot loop.

**Test:** Remove the branch entirely, hardcoding VPERMB compaction (VBMI-only).
Same 10 hoisted variables, same function structure, just no branch.

**Results (AMD EPYC, GB/s, decode):**
| Config | 1KB | 10KB | 64KB |
|---|---|---|---|
| Separate funcs (baseline) | 18.7 | — | 24.6 |
| Unified + branch | 17.1 | 21.2 | 21.7 |
| Branch-free | 18.5 | 23.2 | 23.6 |
| Branch-free + `d := dst[di:]` reslice | **19.3** | **24.3** | **24.3** |

**Analysis:**
- Removing the branch recovered 8-9% (21.7 → 23.6 at 64KB)
- Adding the reslice trick recovered another 3% (23.6 → 24.3)
- Combined: within 1% of the separate-function baseline

**Why does a perfectly-predicted branch cost 8-9%?**
The branch predictor handles the always-taken path at near-zero cost, but the branch
still disrupts the compiler's code generation:
1. **Dead code in the binary:** The AVX2 fallback path (2 memory loads + VPSHUFB + VPERMD)
   is generated even on VBMI machines. This occupies instruction cache uselessly.
2. **Phi node register pressure:** `var result` declared before the branch creates a phi
   node that the compiler must schedule around, potentially forcing a register spill.
3. **Scheduling disruption:** The TESTB+JEQ between VPMADDWD and VPERMB prevents the
   compiler from scheduling them back-to-back for optimal pipeline utilization.
4. **Function size:** 371 bytes with branch vs 326 bytes without = 12% larger instruction footprint.

**Assembly comparison:**
| Metric | With branch | Branch-free |
|---|---|---|
| Function size | 371 bytes | 326 bytes |
| YMM registers | 14 (Y0-Y13) | 14 (Y0-Y13) |
| Stack spills | 0 | 0 |
| Instructions in loop | ~25 + branch | ~22 |

**Verdict:** Even perfectly-predicted branches are expensive inside SIMD loops.
For the primary VBMI target, use a branch-free decode. For AVX2-only fallback,
dispatch to a separate function before entering the loop (not inside it).

**Implication for code organization:** Instead of a single function with an internal
branch, use two function bodies (VBMI and AVX2) selected by the dispatcher. This
matches the performance of separate functions while keeping the dispatch logic clean.

## Experiment 27: Why the Reslice Trick Works — Bounds Check Elimination

**Pattern:** `d := dst[di:]` followed by `result.StoreSlice(d[:32])` instead of
`result.StoreSlice(dst[di : di+32])`.

**Assembly diff (decodeVBMI):**

Without reslice (239 bytes):
```asm
LEA  32(R8), R9      ; compute di+32
...
CMP  R8, R9           ; check di < di+32 (bounds check for StoreSlice)
```

With reslice (230 bytes):
```asm
CMP  CX, R8           ; check di < len(dst) (simpler, uses pre-computed value)
```

**Root cause:** Without the reslice, `StoreSlice(dst[di : di+32])` requires the compiler
to verify `di+32 <= cap(dst)`. This needs a LEA to compute `di+32` and an extra CMP.
With the reslice, the compiler knows `d` starts at `dst[di:]` with capacity `cap(dst)-di`,
so `StoreSlice(d[:32])` just needs to check `32 <= cap(d)`, which is equivalent to
`di+32 <= cap(dst)` — but the compiler proves this from the loop condition
`di <= len(dst)-32`, eliminating both instructions.

**Impact:** 2 fewer instructions per loop iteration. Function size: 230 vs 239 bytes (-4%).
Performance: +3% at 64KB (24.3 vs 23.6 GB/s).

**Verdict: KEEP.** Apply `d := dst[di:]` before every StoreSlice in hot loops.
Also works for LoadSlice: `s := src[si:]; archsimd.LoadUint8x32Slice(s[:32])`.

## Summary of Hoisting Approaches Tested

| Approach | Intrinsics inlined? | Constants hoisted? | Performance | Verdict |
|---|---|---|---|---|
| Manual `v := global` | Yes | Yes | Baseline | Works, but ugly |
| By-value struct arg | Yes | Yes | Same | **Adopted** — clean, explicit |
| Local struct copy | Yes | Yes | Same | Works, but still hoisting |
| Individual func args | Yes | Yes | Same | Works, but too many params |
| Struct pointer | Yes | No (LICM) | -22 to -27% | Rejected |
| Closure capturing | No | Yes | -69 to -87% | Rejected — no intrinsic inlining |

## Experiment 28: Slice Consumption Pattern — REGRESSION

**Hypothesis**: Instead of tracking `si`/`di` indices, progressively consume slices: `s = s[32:]`, `d = d[24:]`. Loop body always works from index 0. This extends the reslice trick (experiment 27) to both loads and stores, potentially eliminating all index arithmetic.

**Applied to**: encode512, encodeAVX2, decodeVBMI, decodeAVX2.

**Results** (64KB, GB/s):

| Configuration | Encode | Decode |
|---|---|---|
| Baseline (index + reslice trick) | 35.1 | 24.3 |
| Full slice consumption (all funcs) | 34.5 | 22.2 |
| Encode slice + decode index | 33.0 | 23.6 |

**Root cause**: A slice is 3 words (ptr, len, cap) vs 1 word for an index. Each reslice (`s = s[32:]`) updates all 3 fields. Two slices (src + dst) = 6 values to keep live in the loop, vs 2 integers (si, di) with the index approach. In SIMD functions already using 10-11 hoisted YMM variables out of 16 available, this extra GP register pressure causes spilling.

**Key insight**: The reslice trick (`d := dst[di:]` before `StoreSlice(d[:32])`) is the optimal balance — it gets bounds check elimination for the store without adding persistent register pressure in the loop. The reslice is a temporary that the compiler can optimize away after the store, while slice consumption makes the resliced values persistent across iterations.

**Conclusion**: Slice consumption is an excellent general Go pattern (parsers, protocol decoders, streaming) but not suitable for SIMD hot loops at the register pressure cliff. Reverted to index-based loops with the reslice trick for stores only.

## Experiment 29: Independent Register Allocation via Separate Functions

**Question**: Does Go's compiler respect branch-scoped register allocation? If we have two code paths (VBMI vs AVX2) in one function, do they get independent register budgets?

**Finding**: The question is moot because Go does NOT inline the SIMD functions — they're too large. `doDecode` compiles to `CALL decodeVBMI` / `CALL decodeAVX2`. Each function gets fully independent register allocation by definition.

**Register usage** (from assembly):

| Function | YMM registers used | Count |
|---|---|---|
| decodeVBMI | Y0-Y13 | 14 of 16 |
| decodeAVX2 | Y0-Y14 | 15 of 16 |

Both functions independently saturate nearly all available YMM registers. decodeAVX2 uses one more (Y14) because it needs two extract constants (`extractShuffle` + `extractPermute`) vs VBMI's single `extractVBMI`.

**Implication**: The two-level architecture (dispatcher → separate SIMD functions) is optimal precisely because each function gets its own full register budget. If they were merged into one function body, even with branches, the compiler would likely allocate globally across both paths, reducing the effective budget for each. This confirms the findings from experiment 23 (monolithic function body size) and experiment 26 (branch cost).

## Experiment 30: Flush-Start AVX2 Encode (VPERMD + VPSHUFB) — MIXED

**Problem**: AVX2 encode uses a -4 offset load trick because VPSHUFB can only shuffle within 128-bit lanes. The group [bytes 15,16,17] spans the lane boundary. The -4 offset shifts everything so no group crosses the boundary, but requires a 6-byte scalar preamble on the first call.

**Approach**: Use VPERMD to rearrange dwords across lanes first, placing bytes 0-11 in lane 0 and bytes 12-23 in lane 1, then VPSHUFB for intra-lane byte grouping. This loads from src[si:si+32] with no offset trick, eliminating the preamble.

**Tradeoff**: One extra instruction per iteration (VPERMD), and srcEnd = len(src)-32 vs len(src)-28, meaning fewer SIMD iterations at the tail.

**Results** (GB/s):

| Size | Flush-start | Offset (-4) | Delta |
|---|---|---|---|
| 48 | 1.58 | 1.48 | +7% (no preamble) |
| 64 | 2.49 | 2.38 | +5% |
| 100 | 2.40 | 5.39 | -55% (lost 1 SIMD iteration) |
| 128 | 5.59 | 5.83 | -4% |
| 1000 | 17.6 | 19.9 | -12% |
| 65536 | 33.5 | 33.9 | -1% |

**Analysis**: The flush path wins at tiny sizes (48-64 bytes) where eliminating the preamble matters, but loses at medium-to-large sizes due to: (a) extra VPERMD instruction costing ~1-3% throughput, and (b) wider read window (32 vs 28 bytes) losing entire SIMD iterations at the tail. At n=100, this loses a full iteration — the offset path gets 2 AVX2 iterations vs flush getting only 1.

**Conclusion**: The -4 offset trick is worth keeping. The 6-byte scalar preamble (~10ns) is far cheaper than losing SIMD iterations. The offset trick's tighter read window (28 vs 32 bytes) extracts more SIMD iterations from the same data. Reverted.

## Experiment 31: Overlapping AVX2 Tail Iteration — WIN (+7-23%)

**Insight**: The AVX2 encode loop advances by 24-byte stride. When the stride doesn't align with the end of the data, up to 23 bytes are left for scalar processing (1-7 full triplets). But there's a valid SIMD position that the stride jumped over. Since base64 is deterministic, re-encoding overlapping bytes produces identical output — the overlap is harmless.

**Implementation**: After the forward AVX2 loop, compute the last valid 3-aligned load position. If it's before the current `si` but covers new bytes (`last+24 > si+6`), call encodeAVX2 again at that position. This is just a second call to the same function — no code duplication, no changes to encodeAVX2.

```go
if last := (n-28) - (n-28)%3; last >= 4 && last < si && last+24 > si+6 {
    di, si = encodeAVX2(a, dst, src, last/3*4, last)
}
```

**Key details**:
- `last+24 > si`: ensures the overlap covers NEW bytes past current si (not a pure re-encode)
- `+6` threshold: avoids firing when only 1-2 scalar triplets would be saved (the function call overhead with re-hoisting 8 variables costs more than 1-2 scalar iterations)
- The function call overhead (~10ns for hoisting + CALL) is amortized by replacing 3-7 scalar iterations (~5ns each)

**Results** (GB/s, median of 5 runs):

| Size | Overlap | Baseline | Delta |
|---|---|---|---|
| 64 | 2.86 | 2.38 | +18% |
| 100 | 5.50 | 5.39 | +2% |
| 128 | 5.75 | 5.83 | -1% (noise) |
| 256 | 10.9 | 8.2 | +23% |
| 1000 | 23.4 | 19.9 | +15% |
| 10000 | 34.8 | 33.5 | +4% |
| 65536 | 36.6 | 33.9 | +7% |

**Development journey**: Three implementations were tested:
1. **Second call from doEncode**: Clean but bloated doEncode by 192 bytes. Showed mixed results with some sizes regressing ~8% — initially suspected function size issue, but actually noise.
2. **Duplicated body inside encodeAVX2**: Grew encodeAVX2 from 791→1214 bytes (+53%). The larger function degraded register allocation, confirmed by assembly analysis.
3. **Single-body loop restructure**: Kept one copy of the loop body with a conditional backup. Grew encodeAVX2 by only 135 bytes. Mixed results.
4. **Final: second call (same as #1) with `+6` threshold**: The simplest approach — just call encodeAVX2 twice. The key realization: encodeAVX2 is NOT inlined into doEncode (verified via assembly: real CALL instructions). The 186-byte doEncode growth is all argument setup and panicBounds entries, not SIMD codegen. The original "regression" at large sizes was server noise.

**Architectural insight**: The overlap technique treats SIMD encode as a composable primitive. The dispatcher can call it with any valid (si, di) parameters, including overlapping regions. The function doesn't need to know whether its output overlaps with a previous call. This enables flexible dispatch strategies without modifying the SIMD inner loop.

## Experiment 32: SSE 128-bit Encode and Decode — WIN at small sizes

**Motivation:** Below ~120 bytes, the AVX2 loop gets very few iterations, and the function call + constant hoisting overhead dominates. A 128-bit SSE path processes 12→16 bytes (encode) or 16→12 bytes (decode) per iteration with fewer hoisted constants.

**Implementation:**

`encodeSSE(alphabet, dst, src, di, si)` — processes 12 input → 16 output per iteration. Uses the same sextet extraction (mulhi/mullo) and ASCII mapping (range-based SubSaturated+Greater+PermuteOrZero) as the AVX2 path, but at 128-bit width. Hoists ~8 SSE constants.

`decodeSSE(a, dst, src, di, si)` — processes 16 input → 12 output per iteration. Same nibble-LUT validation + VPMADDUBSW + VPMADDWD pipeline as VBMI decode, but at 128-bit width with VPSHUFB for the final byte extraction. Hoists ~8 SSE constants.

Both functions accept `(di, si)` parameters so they can be called as cleanup after wider loops.

**Results (AMD EPYC Zen 4, GB/s):**

| Size | encodeSSE | scalar | Delta |
|------|-----------|--------|-------|
| 36 | 2.5 | 1.3 | +92% |
| 60 | 5.1 | 1.3 | +292% |
| 96 | 9.2 | 1.4 | +557% |

| Size | decodeSSE | scalar | Delta |
|------|-----------|--------|-------|
| 36 | 2.8 | 1.5 | +87% |
| 60 | 5.0 | 1.5 | +233% |
| 96 | 7.8 | 1.6 | +388% |

**Verdict: WIN.** SSE is the optimal path for inputs below ~120 bytes where AVX2 doesn't get enough iterations to amortize its setup cost.

## Experiment 33: Fused SSE+AVX2 Bookend Pattern — WIN (+27% encode, +15% decode)

**Insight:** Applying the same bookend technique from encode (experiment 11) to both encode and decode at the SSE+AVX2 level. SSE processes the first and last few bytes, AVX2 fills the middle. The SSE preamble and tail overlap slightly with the AVX2 region — the overlap is harmless (deterministic, same data written twice).

**Implementation:**

`encodeSSEAVX2(alphabet, a, dst, src)` — SSE preamble (12→16 bytes), SSE tail (last valid 12→16 position), AVX2 loop from si=12 fills the middle. Returns the tail position so the caller knows how far SIMD reached.

`decodeSSEAVX2(a, dst, src)` — SSE preamble (16→12 bytes), SSE tail (last valid 16→12 position), AVX2 loop from si=16 fills the middle.

**Results (AMD EPYC Zen 4, GB/s):**

| Size | fused(sse+avx2) | avx2-only | SSE-only | Best standalone |
|------|-----------------|-----------|----------|----------------|
| 150 | 13.4 | 10.5 | 9.5 | fused (+27%) |
| 256 | 14.5 | 12.0 | 7.8 | fused (+21%) |
| 512 | 23.1 | 20.1 | 7.1 | fused (+15%) |

**Verdict: WIN.** The fused pattern gives the best throughput for 120-256 bytes, the awkward middle ground where SSE alone doesn't have enough width and AVX2 alone wastes iterations on preamble/tail.

## Experiment 34: Register Pressure from Constant Co-hoisting — KEY FINDING

**Problem:** Initially, the 512-bit encode path was a single fused function `encodeSSE512` that contained both the 512-bit bulk loop AND the SSE cleanup loop. This hoisted ~12 SIMD constants (6 for 512-bit, 6 for SSE) at function entry.

**Discovery:** A/B benchmarking showed a consistent 10-11% encode regression at large sizes (400KB+) compared to the old code that used a standalone `encode512`. The fused function's 12 hoisted constants exceeded the comfortable register budget, causing spills in the hot 512-bit loop body.

**Fix:** Split back to standalone `encode512` (6 constants) called by `doEncode`, followed by `encodeSSE` for cleanup. Each function independently hoists only its own constants.

**Results (AMD EPYC Zen 4, GB/s, encode):**

| Size | fused (12 constants) | split (6+6) | Delta |
|------|---------------------|-------------|-------|
| 500 | 22.1 | 25.2 | +14% |
| 2K | 30.5 | 31.4 | +3% |
| 400K | 32.8 | 36.4 | +11% |

**Assembly evidence:** `encode512` standalone: 230 bytes, `locals=0x8` (no YMM spills, 6 YMM constants + 2 temporaries). `encodeSSE512` fused: 425 bytes, 12 hoisted YMM vectors saturating all 16 registers, forcing spills in the 512-bit loop body.

**Verdict:** Confirms experiment 23 (monolithic function body size). The register pressure cliff is real: 6-7 SIMD constants per function is the sweet spot on amd64 (16 YMM registers - ~2 for temporaries - ~2 for bookkeeping = 10-12 usable). Beyond that, the compiler spills.

## Experiment 35: Inline Scalar Tails in doEncode/doDecode — WIN (+74% small encode)

**Problem:** The original architecture had SIMD in `base64_amd64.go` and scalar remainder in `base64.go`. The `Encode()` method called `simdEncode` (function pointer) for SIMD, then ran scalar cleanup. This meant every encode/decode paid function pointer overhead (~14ns) even when the SIMD path would handle everything.

**Change:** Moved the scalar remainder loop into `doEncode` and `doDecode` directly. The public `Encode()`/`Decode()` methods now just call `simdEncode`/`simdDecode` which handle the complete job (SIMD bulk + scalar tail). No more function pointer overhead for the hot path.

**Results (AMD EPYC Zen 4, A/B interleaved, median):**

| Size | Before | After | Delta |
|------|--------|-------|-------|
| 30 B encode | 1.2 GB/s | 2.1 GB/s | +74% |
| 60 B encode | 5.1 GB/s | 5.8 GB/s | +14% |
| 30 B decode | 1.5 GB/s | 3.3 GB/s | +124% |
| 60 B decode | 4.5 GB/s | 4.6 GB/s | +3% |
| 400K encode | 34.5 GB/s | 36.2 GB/s | +5% |

**Verdict: WIN.** The biggest gains are at small sizes where the function pointer overhead dominated. No regressions at any size.

## Experiment 36: Dispatch Threshold Tuning — OPTIMIZATION

**Method:** Ran `BenchmarkEncodeSchemes` and `BenchmarkDecodeSchemes` at all sizes from 12 to 1M bytes to find crossover points between tiers.

**Encode thresholds (Zen 4):**

| Range | Best path | Why |
|-------|-----------|-----|
| < 28 bytes | scalar only (doEncode returns immediately) | SIMD overhead exceeds benefit |
| 28–119 bytes | encodeSSE | AVX2 gets too few iterations at this size |
| 120–255 bytes | encodeSSEAVX2 (fused bookend) | Sweet spot for SSE+AVX2 fusion |
| 256+ bytes (AVX-512) | encode512 + encodeSSE cleanup | 512-bit bulk dominates |
| 120+ bytes (no AVX-512) | encodeSSEAVX2 | Best available without 512-bit |

**Decode thresholds (Zen 4):**

| Range | Best path | Why |
|-------|-----------|-----|
| < 16 bytes | scalar only | Not enough bytes for a single SSE iteration |
| 16–44 bytes | decodeSSE | Handles small inputs efficiently |
| 45+ bytes (AVX-512) | decodeVBMI | VPERMB compaction at 256-bit width |
| 45+ bytes (no AVX-512) | decodeSSEAVX2 | Fused bookend pattern |

**Key finding:** The SSE threshold for encode (120 bytes) is much higher than the naive estimate (~30 bytes for 2 iterations). This is because encodeSSEAVX2 gets its first AVX2 iteration at ~40 bytes input, but doesn't beat pure SSE until ~120 bytes due to the overhead of hoisting both SSE and AVX2 constants.

## Experiment 37: AMD EPYC 9B45 (Zen 5) Benchmarks — 2x FASTER ENCODE

**Server:** AMD EPYC 9B45 (Zen 5), 2 vCPUs. AVX-512 + VBMI supported. Unlike Zen 4 which double-pumps 512-bit ops, Zen 5 has wider execution units.

**Results (GB/s, benchtime=1s, count=3):**

### Encode
| Size | simdenc | emmansun | cristalhq | stdlib | vs emmansun |
|------|---------|----------|-----------|--------|-------------|
| 64 B | 6.4 | 7.4 | 3.7 | 1.9 | 0.86x |
| 256 B | 21.5 | 18.4 | 4.0 | 1.8 | **1.17x** |
| 1 KB | 45.4 | 26.5 | 4.0 | 1.9 | **1.71x** |
| 10 KB | 61.0 | 27.9 | 4.1 | 1.9 | **2.19x** |
| 64 KB | 63.6 | 28.7 | 4.1 | 1.9 | **2.22x** |
| 1 MB | 48.1 | 28.5 | 4.1 | 1.9 | **1.69x** |

### Decode
| Size | simdenc | emmansun | cristalhq | stdlib | vs emmansun |
|------|---------|----------|-----------|--------|-------------|
| 64 B | 5.7 | 4.3 | 3.7 | 2.4 | **1.33x** |
| 256 B | 15.7 | 13.3 | 4.0 | 2.8 | **1.18x** |
| 1 KB | 25.7 | 24.1 | 4.0 | 2.8 | **1.07x** |
| 10 KB | 30.5 | 30.7 | 4.0 | 2.9 | 0.99x |
| 64 KB | 31.2 | 32.4 | 4.0 | 2.9 | 0.96x |
| 1 MB | 30.9 | 32.0 | 4.0 | 2.9 | 0.97x |

### Zen 4 → Zen 5 comparison (simdenc only)
| Size | Zen 4 enc | Zen 5 enc | Zen 4 dec | Zen 5 dec |
|------|-----------|-----------|-----------|-----------|
| 1 KB | 28.0 | 45.4 (+62%) | 10.0 | 25.7 (+157%) |
| 10 KB | 36.0 | 61.0 (+69%) | 24.1 | 30.5 (+27%) |
| 64 KB | 36.7 | 63.6 (+73%) | 24.4 | 31.2 (+28%) |
| 1 MB | 34.9 | 48.1 (+38%) | 24.1 | 30.9 (+28%) |

**Key findings:**

1. **Encode scales dramatically on Zen 5.** Peak encode jumps from 36.7 to 63.6 GB/s (+73%). This strongly suggests Zen 5 executes 512-bit ops natively (single-pump) instead of double-pumping like Zen 4. Our `encode512` loop benefits directly because each 512-bit VPERMB, VPMULLW, VPMULHUW executes in one cycle instead of two.

2. **Decode improvement is more modest.** Peak decode: 24.4 → 31.2 GB/s (+28%). Decode uses 256-bit VBMI (not 512-bit), so it doesn't benefit from wider execution. The 28% improvement is from general Zen 5 IPC gains and higher memory bandwidth.

3. **emmansun encode barely improves on Zen 5** (23.4 → 28.7 GB/s, +23%). They use AVX2 (256-bit) only — their code doesn't benefit from 512-bit native execution. This is why our lead over them widens from 1.5x on Zen 4 to 2.2x on Zen 5.

4. **Decode gap with emmansun at large sizes.** emmansun's hand-written AVX2 assembly is ~4% faster at 64KB+ decode. Both use the same algorithm (nibble-LUT validation + VPMADDUBSW + VPMADDWD), but their assembly avoids bounds checks and has tighter instruction scheduling. Our Go compiler-generated code has one extra bounds check per loop iteration and 3 extra SIMD instructions (special char handling).

5. **1 MB encode drops from 63.6 to 48.1 GB/s.** This is L2/L3 cache pressure — 1 MB input + 1.33 MB output exceeds the per-core L2 cache. The 64 KB peak is entirely in L1/L2.


## Experiment 38: archsimd API Audit (Go 1.26rc1)

Definitive inventory of AVX-512 instructions relevant to base64 optimization in `simd/archsimd`.

### VPMULTISHIFTQB — NOT AVAILABLE

**Status:** Not exposed in archsimd at any level (no public method, no unexported method).

**Evidence:**
- `grep -ri "multishift\|MultishiftQB" $GOROOT/src/simd/archsimd/` → zero results
- Not present in `ops_amd64.go`, `ops_internal_amd64.go`, `extra_amd64.go`, `shuffles_amd64.go`, or any codegen YAML
- The Go assembler DOES know the mnemonic (`AVPMULTISHIFTQB` in `cmd/internal/obj/x86/aenum.go`), so it can be used from hand-written `.s` files

**Impact:** Cannot replace the 4-instruction mulhi/mullo sextet extraction in `encode512` with a single instruction via intrinsics. Workaround: hand-written Go assembly (see Experiment 41).

### VPTERNLOGD — UNEXPORTED (compiler auto-fusion only)

**Status:** Present as unexported `tern()` method on Int32/Uint32/Int64/Uint64 types at all widths (128/256/512). Located in `ops_internal_amd64.go` lines 596-692.

**Evidence:**
```go
// ops_internal_amd64.go
func (x Uint32x16) tern(table uint8, y Uint32x16, z Uint32x16) Uint32x16
// Asm: VPTERNLOGD, CPU Feature: AVX512
```

**Design intent** (from `_gen/simdgen/ops/BitwiseLogic/categories.yaml`):
> "We also have PTEST and VPTERNLOG, those should be hidden from the users and only appear in rewrite rules."

The compiler's SSA rewrite rules are supposed to auto-fuse `And`/`Or`/`Xor`/`AndNot` chains into VPTERNLOGD. Whether this actually happens needs verification (see Experiment 43).

**Impact:** Cannot call directly. If auto-fusion works, our existing boolean expressions may already benefit. If not, there's no workaround short of assembly.

### ConcatPermute (VPERMI2B) — FULLY AVAILABLE at all widths

**Status:** Public method on `Uint8x16`, `Uint8x32`, `Uint8x64` (and signed variants).

**Evidence:**
```go
// ops_amd64.go:1307
func (x Uint8x32) ConcatPermute(y Uint8x32, indices Uint8x32) Uint8x32
// Asm: VPERMI2B, CPU Feature: AVX512VBMI

// ops_amd64.go:1323
func (x Uint8x64) ConcatPermute(y Uint8x64, indices Uint8x64) Uint8x64
// Asm: VPERMI2B, CPU Feature: AVX512VBMI
```

**Index semantics:** "Only the needed bits to represent xy's index are used in indices' elements."
- At 256-bit: xy is 64 bytes, so indices use low 6 bits (0-63). Selects from 64 entries.
- At 512-bit: xy is 128 bytes, so indices use low 7 bits (0-127). Selects from 128 entries.

**Impact for decode (Experiment 39):** At 256-bit, ConcatPermute selects from 64 bytes — NOT enough for the paper's 128-entry decode LUT (needs to map all 128 ASCII values). At 512-bit, it selects from 128 bytes — exactly what the paper needs. This means the VPERMI2B decode optimization requires either:
1. A 512-bit ConcatPermute (requires 512-bit decode pipeline — ties to Experiment 40), or
2. A different encoding of the LUT that fits in 64 entries at 256-bit

**Option 2 analysis:** Base64 uses ASCII values in ranges 0x2B-0x7A. With 64 entries at 256-bit, we can only index 0-63. ASCII 'A' is 0x41 (65) — already out of range. So 256-bit ConcatPermute CANNOT directly implement the paper's decode LUT. We'd need to subtract a base offset first, which adds an instruction and still can't handle the full ASCII range in 64 entries.

**Conclusion:** The VPERMI2B decode translation requires 512-bit width. This couples Task 2 (VPERMI2B decode) with Task 3 (512-bit decode). They should be attempted together.

### IsZero — NOT AVAILABLE at 512-bit

**Status:** Available on all 128-bit and 256-bit integer vector types. NOT available on any 512-bit type.

**Evidence:**
```
// extra_amd64.go — all IsZero signatures (exhaustive):
func (x Uint8x16) IsZero() bool
func (x Uint8x32) IsZero() bool
func (x Int8x16) IsZero() bool
func (x Int8x32) IsZero() bool
// ... Int16x8, Int16x16, Int32x4, Int32x8, Int64x2, Int64x4
// ... Uint16x8, Uint16x16, Uint32x4, Uint32x8, Uint64x2, Uint64x4
// NO 512-bit variants (no Uint8x64, Int8x64, etc.)
```

**Workaround options:**
1. `vec.Equal(zero).ToBits() == 0` — `Mask8x64.ToBits()` returns `uint64`, confirmed available. Cost: VPCMPB (k-register) + KMOVQ + TEST. Likely 2-3 cycles vs 1 cycle for VPTEST.
2. Split to 256-bit halves: `vec.GetHi().Or(vec.GetLo()).IsZero()` — VEXTRACTI + VPOR + VPTEST. Also 2-3 cycles.
3. For decode validation specifically: check the high bit of all bytes. `vec.Greater(threshold)` produces a Mask8x64, then `.ToBits() != 0` detects any invalid byte.

### Non-temporal Stores and Prefetch — NOT AVAILABLE

**Evidence:** `grep -rn "NonTemporal\|StreamStore\|Prefetch\|NTStore" $GOROOT/src/simd/archsimd/` → zero results. No cache control instructions exposed. L1-chunked processing (Experiment 42) cannot use NT stores via archsimd.

### Additional 512-bit Operations Available

Full Uint8x64 method inventory (ops relevant to base64):
- `Add`, `Sub`, `SubSaturated`, `And`, `Or`, `Xor`, `AndNot` — all arithmetic/bitwise
- `Equal`, `Greater`, `Less` etc. → `Mask8x64` → `.ToBits() uint64`
- `Permute(Uint8x64)` — VPERMB, full cross-lane (already used in encode512)
- `PermuteOrZeroGrouped(Int8x64)` — VPSHUFB, per-128-bit-lane
- `ConcatPermute(Uint8x64, Uint8x64)` — VPERMI2B, 128-entry lookup
- `Compress(Mask8x64)` — VPCOMPRESSB
- `DotProductPairsSaturated(Int8x64)` → `Int16x32` — VPMADDUBSW at 512-bit
- `GetHi()`/`GetLo()` → Uint8x32, `SetHi()`/`SetLo()` for width conversion
- `Broadcast512()` on Uint8x16 to fill all 4 lanes


## Experiment 39: 512-bit VPERMI2B Decode (ConcatPermute)

**Hypothesis:** Replace the 6-instruction nibble-LUT validation+translation pipeline in decode with a single VPERMI2B (ConcatPermute) lookup from a 128-entry table. Process 64 encoded bytes → 48 decoded bytes per iteration at 512-bit width.

**Design:**
- Build a 128-entry LUT split across two `Uint8x64` vectors (`lutLo` for entries 0-63, `lutHi` for entries 64-127). Valid base64 ASCII chars map to their 6-bit sextet value; invalid chars map to 0x80.
- `lutLo.ConcatPermute(lutHi, encoded)` performs combined validation + translation in one instruction (VPERMI2B).
- Error detection: `(encoded | sextets) & 0x80` catches both non-ASCII input bytes (bit 7 set in `encoded`) and invalid chars (bit 7 set in `sextets` from LUT).
- Rest of pipeline unchanged: VPMADDUBSW + VPMADDWD + VPERMB for sextet packing and byte compaction.

**Bug found and fixed:** `Broadcast512()` on `Uint8x16` broadcasts only **element zero** (a single byte via VPBROADCASTB), not the whole 128-bit lane. The `decCombinePairs512` constant was initialized as `LoadUint8x16([0x40, 1, ...]).Broadcast512()`, which produced `[0x40, 0x40, 0x40, ...]` (all 64s) instead of `[0x40, 1, 0x40, 1, ...]`. Fix: build via `SetLo(pairs256).SetHi(pairs256)` from the existing 256-bit constant.

**Threshold:** decode512 fires at ≥192 encoded bytes (3+ iterations) to avoid overhead at small sizes where Zen 4 double-pumping hurts.

**Results (AMD EPYC Zen 4, double-pumps 512-bit):**

| Encoded Size | Main (VBMI 256) | New (512+VBMI) | Δ |
|---|---|---|---|
| 86 (raw 64) | 2.7 GB/s | 2.7 GB/s | — |
| 134 (raw 100) | 3.1 GB/s | 3.2 GB/s | — |
| 172 (raw 128) | 5.6 GB/s | 5.4 GB/s | — |
| 342 (raw 256) | 7.5 GB/s | 8.5 GB/s | +14% |
| 1334 (raw 1K) | 15.1 GB/s | 22.7 GB/s | **+50%** |
| 13334 (raw 10K) | 17.8 GB/s | 44.0 GB/s | **+147%** |
| 174764 (raw 131K) | 18.3 GB/s | 45.9 GB/s | **+151%** |
| 1333334 (raw 1M) | 19.2 GB/s | 39.9 GB/s | **+108%** |

**Key observations:**

1. **No regressions at any size.** The threshold at 192 encoded bytes ensures small inputs use VBMI-only.
2. **+50% at 1 KB, +150% at 64 KB+ on Zen 4** — even with double-pumping of 512-bit ops.
3. **On Zen 5 (single-pump 512-bit), expect even larger gains.** Zen 4 double-pumps 512-bit, so each VPERMI2B costs ~2 cycles instead of 1. Zen 5 should roughly halve the per-iteration cost.
4. **Instruction count reduction:** The old nibble-LUT pipeline uses 6 instructions (shift, 2× and, 2× pshufb, cmp) for validation+translation. VPERMI2B replaces all of them with 1 ConcatPermute + 1 Or + 1 And + 1 Equal+ToBits for validation — net savings of ~2 instructions per iteration.
5. **The 1M decode drops from peak** (45.9 GB/s at 131K → 39.9 GB/s at 1M) due to L2/L3 cache pressure, same pattern as encode. See Experiment 40 for chunking analysis.


## Experiment 40: L2-Chunked Encode for Large Inputs

**Hypothesis:** Processing large inputs in L2-friendly chunks (24K-192K) should sustain near-peak throughput at 1MB+ by keeping the working set in L2 cache.

**Method:** Wrapped `encode512` in a chunking loop that processes `chunkSrc` input bytes at a time (multiple of 48, encode512's stride). Tested chunk sizes: 24K, 48K, 96K, 192K. Compared against unchunked encode512 at 131K, 1M, 4M, 16M.

**Results (AMD EPYC Zen 4):**

| Size | Unchunked | 24K | 48K | 96K | 192K |
|---|---|---|---|---|---|
| 131K | 32.1 GB/s | 34.0 GB/s | 32.1 GB/s | 31.2 GB/s | 33.1 GB/s |
| 1M | 30.1 GB/s | 30.0 GB/s | 29.1 GB/s | 30.4 GB/s | 31.0 GB/s |
| 4M | 30.1 GB/s | 30.0 GB/s | 29.8 GB/s | 31.0 GB/s | 30.1 GB/s |
| 16M | 18.1 GB/s | 18.1 GB/s | 19.4 GB/s | 19.1 GB/s | 18.4 GB/s |

**Result: NEGLIGIBLE BENEFIT.** All measurements within noise at 1M and 4M. At 16M, 48K chunks show ~7% improvement (18.1→19.4 GB/s), but this is marginal and inconsistent.

**Analysis:** The theoretical prediction was correct: base64 encode is a single-pass streaming algorithm. It never re-reads data, so cache eviction doesn't cause misses. The hardware prefetcher handles linear sequential access patterns well. The throughput drop at large sizes is a **fundamental memory bandwidth limit** (L2→L3→DRAM latency), not a cache reuse problem.

**What would actually help:**
- **Non-temporal stores** (VMOVNTDQ): bypass L2 cache for dst writes, freeing cache capacity for src reads. ~20-30% improvement expected at DRAM-bound sizes.
- **Software prefetch** (PREFETCHT0): pre-load src data ahead of consumption.
- Neither is available in archsimd (confirmed in Experiment 38).

**Conclusion:** The 1M encode regression (~9% on Zen 4) and 16M regression (~44%) are inherent memory bandwidth limits. No pure-software chunking approach can address this without NT stores or prefetch. The code was NOT modified — chunking is not worth the complexity for negligible gain.


## Experiment 41: VPMULTISHIFTQB Encode via Assembly (3-Instruction Hot Loop)

**Hypothesis:** Replace the 5-instruction mulhi/mullo sextet extraction (AND + MULHIGH + AND + MUL + OR) with a single VPMULTISHIFTQB instruction, which extracts all 8 sextets from each 64-bit lane in one operation. Requires a Go assembly (.s) file since archsimd doesn't expose VPMULTISHIFTQB.

**Design:**
- New byte grouping shuffle: packs 6 source bytes per qword in big-endian order [s2,s1,s0,s5,s4,s3,x,x] so VPMULTISHIFTQB sees contiguous 6-bit fields within each 64-bit lane.
- Shift vector: [18, 12, 6, 0, 42, 36, 30, 24] per qword — extracts 4 sextets from each of 2 triples packed into one 64-bit lane.
- Direct ASCII LUT: maps sextet → ASCII directly (not offset), so the pipeline is just VPERMB → VPMULTISHIFTQB → VPERMB. No ADD needed since VPERMB only uses the low 6 bits of each index.
- Full loop in assembly to avoid per-iteration function call overhead.

**Implementation:** Added `base64_amd64.s` with `encode512ms` function using Go's ABI0 calling convention. Constants defined as GLOBL/DATA in the .s file. Go side: `//go:noescape` declaration, `directLUT [64]byte` field on `encodeAlpha`.

**Results (AMD EPYC Zen 4, double-pumps 512-bit):**

| Size | simd (mulhi/mullo) | multishift (VPMULTISHIFTQB) | Δ |
|---|---|---|---|
| 256 | 12.6 GB/s | 27.0 GB/s | **+115%** |
| 1K | 23.7 GB/s | 46.5 GB/s | **+96%** |
| 10K | 33.2 GB/s | 61.9 GB/s | **+86%** |
| 65K | 32.5 GB/s | 40.6 GB/s | **+25%** |
| 131K | 31.6 GB/s | 41.7 GB/s | **+32%** |
| 1M | 29.9 GB/s | 35.1 GB/s | **+17%** |
| 4M | 28.6 GB/s | 33.7 GB/s | **+18%** |
| 16M | 18.0 GB/s | 17.4 GB/s | ~0% (bandwidth-limited) |

**Key observations:**

1. **+86-115% at compute-bound sizes (256B-10K).** The 3-instruction loop vs 7-instruction loop translates almost directly to 2x throughput when not memory-bound.
2. **61.9 GB/s encode at 10K on Zen 4** — this is within striking distance of aklomp's C implementation (which uses the same VPMULTISHIFTQB approach). On Zen 5 (single-pump), expect even higher.
3. **+17-32% at L2/L3-bound sizes (65K-4M).** Even when memory bandwidth starts limiting, fewer instructions per iteration still helps — less time wasted on instruction decode/execution leaves more bandwidth for memory operations.
4. **Convergence at 16M.** Both paths hit the same DRAM bandwidth wall.
5. **Broadcast512 pitfall avoided.** The constants are defined as 64-byte DATA blocks in the .s file, avoiding the `Broadcast512()` element-zero trap discovered in Experiment 39.

**Assembly maintenance cost:** The `.s` file is 52 lines with 3 SIMD instructions in the hot loop. The assembly is simple enough that it's unlikely to need changes unless the function signature changes. If Go adds VPMULTISHIFTQB to archsimd in a future release, the .s file can be deleted and replaced with intrinsics.

**Not shipped.** The prototype validates the approach but introduces a hand-written assembly file, which defeats the project's goal of pure Go with compiler intrinsics. Parked until archsimd exposes VPMULTISHIFTQB.


## Experiment 43: VPTERNLOGD Auto-Fusion Verification

**Test:** Does Go 1.26rc1's compiler auto-fuse And/Or/Xor chains into VPTERNLOGD?

**Method:** Added `//go:noinline` function `ternCheck(a, b, c Uint8x32) = a.Or(b).Or(c)` and inspected assembly via `-gcflags='-S'`.

**Result: NO FUSION.**

```asm
// ternCheck: a.Or(b).Or(c)
VPOR Y0, Y1, Y1    // tmp = a | b
VPOR Y1, Y2, Y0    // result = tmp | c
RET
```

Two separate VPOR instructions. No VPTERNLOGD emitted anywhere in the entire simdenc build (0 occurrences across 37 total VPOR/VPAND/VPXOR instructions).

**Also checked:** Full `go build -gcflags='-S'` of simdenc: zero VPTERNLOGD in the entire binary. The rewrite rules documented in archsimd's codegen YAML ("should only appear in rewrite rules") are either:
1. Not yet implemented in Go 1.26rc1, or
2. Only trigger for patterns not present in our code

**Impact:** VPTERNLOGD is effectively unavailable — neither callable nor auto-generated. This doesn't affect our current code (no 3-operand boolean chains in hot loops), but rules out any optimization that depends on it.

### Summary Table

| Instruction | archsimd status | Width available | Usable for base64? |
|---|---|---|---|
| VPMULTISHIFTQB | **Not exposed** | N/A | Encode sextet extraction — assembly only |
| VPTERNLOGD | Unexported (`tern`) | 128/256/512 | Maybe auto-fused by compiler |
| VPERMI2B | **Public** (`ConcatPermute`) | 128/256/512 | Decode LUT needs 512-bit (128 entries) |
| VPTEST (IsZero) | Public | 128/256 only | 512-bit workaround: Equal+ToBits |
| VPCOMPRESSB | Public (`Compress`) | 512 | Not useful (byte-level, not bit-level) |
| NT stores | **Not exposed** | N/A | No cache control available |
| Prefetch | **Not exposed** | N/A | No cache control available |


## Experiment 44: SIMD Constant Declaration Strategies — KEY FINDING

**Question:** SIMD code requires many lookup tables, shuffle masks, and broadcast constants. What is the best way to declare them using Go's `simd/archsimd` intrinsics?

**Three approaches tested** (encode512, 65536B input, AMD EPYC Zen 5):

| Approach | Assembly per 512-bit constant | ns/op | MB/s | Delta |
|---|---|---|---|---|
| A: Package-level var with `.As*()` cast, shadowed into local | 1 `VMOVDQU64` from .data | ~1920 | ~34,100 | baseline |
| B: Package-level `LoadUint64x8` (no cast), `.As*()` in function | 1 `VMOVDQU64` + extra ref? | ~2020 | ~32,400 | **-5%** |
| C: Fully inline (declared inside function body) | 8 MOVQ imm + 8 MOVQ stack + 1 VMOVDQU64 | ~1960 | ~33,400 | **-6%** |

**Approach A (winner):**
```go
// Package level: Load + As in one expression
var encMaskHi512 = archsimd.LoadUint64x8(&[8]uint64{...}).AsUint16x32()

// Function body: shadow into local (hoisting for LICM workaround)
func encode512(...) {
    maskHi512 := encMaskHi512
    ...
}
```
Compiles to a single `VMOVDQU64 symbol(SB), ZMM` per constant. The entire 512-bit value lives in the `.data` section and loads directly into a ZMM register. Function total: 204 bytes, NOSPLIT, 8 bytes stack. 5 loads for 5 constants.

**Approach B (Load at package level, cast in function):**
```go
var encMaskHi512Raw = archsimd.LoadUint64x8(&[8]uint64{...})  // Uint64x8, no cast
func encode512(...) {
    maskHi512 := encMaskHi512Raw.AsUint16x32()  // cast here
    ...
}
```
~5% slower. The compiler apparently doesn't fully elide the type reference indirection.

**Approach C (fully inline):**
```go
func encode512(...) {
    maskHi512 := archsimd.LoadUint64x8(&[8]uint64{...}).AsUint16x32()
    ...
}
```
The compiler cannot materialize 512-bit immediates directly into ZMM registers. Each constant requires building on the stack: 8 `MOVQ` immediates into registers, 8 `MOVQ` stores to stack slots, then 1 `VMOVDQU64` from stack. That's 17 instructions per constant vs 1 instruction for approach A. Function total: 701 bytes, 328 bytes stack. ~6% regression.

**Recommendation for Go SIMD intrinsics users:**

1. Express repeating byte patterns as named `uint64` constants with comments explaining the pattern
2. Declare SIMD vectors as package-level `var` with `Load` + `.As*()` in one expression
3. Shadow into locals at the top of the function body (Go lacks LICM, see Experiment 18)
4. Place each function's constants immediately above the function for locality

This pattern gives optimal codegen (single load per constant), readable declarations (named patterns with comments), and clean function bodies (short shadow block).
