#include "textflag.h"

// ARM64 NEON base64 encode/decode.
//
// Reference implementation for when Go adds NEON support to archsimd.
// Uses VLD3/VST4 structure load/store for natural 3-byte↔4-sextet mapping,
// and TBL4 (4-register table lookup) for alphabet translation.
//
// Performance on M1 Pro: encode ~19 GB/s, decode ~12 GB/s (~3× stdlib).
// Bottlenecked by VLD3/VST4 throughput (encode) and TBL4 instruction count
// (decode). 128-bit NEON width is the fundamental limit vs x86 256/512-bit.

// func encodeBlocksNEON(dst, src *byte, nblocks uint64, alpha *byte)
// Per iter: 48 src bytes → 64 dst bytes.
TEXT ·encodeBlocksNEON(SB), NOSPLIT, $0-32
	MOVD	dst+0(FP), R1
	MOVD	src+8(FP), R0
	MOVD	nblocks+16(FP), R2
	MOVD	alpha+24(FP), R3
	VLD1	(R3), [V24.B16, V25.B16, V26.B16, V27.B16]
	VMOVI	$0x3F, V30.B16
loop_enc:
	VLD3.P	48(R0), [V0.B16, V1.B16, V2.B16]
	VUSHR	$2, V0.B16, V4.B16
	VUSHR	$4, V1.B16, V5.B16
	VSLI	$4, V0.B16, V5.B16
	VAND	V30.B16, V5.B16, V5.B16
	VUSHR	$6, V2.B16, V6.B16
	VSLI	$2, V1.B16, V6.B16
	VAND	V30.B16, V6.B16, V6.B16
	VAND	V30.B16, V2.B16, V7.B16
	VTBL	V4.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V4.B16
	VTBL	V5.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V5.B16
	VTBL	V6.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V6.B16
	VTBL	V7.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V7.B16
	VST4.P	[V4.B16, V5.B16, V6.B16, V7.B16], 64(R1)
	SUBS	$1, R2, R2
	BNE	loop_enc
	RET

// func decodeBlocksNEON(dst, src *byte, nblocks uint64, tableA, tableB *byte) uint64
// Per iter: 64 src bytes → 48 dst bytes. Returns 0 ok, nonzero on invalid.
//
// Uses two 64-byte direct-lookup tables (tableA for input bytes 0-63,
// tableB for 64-127). Each entry is the 6-bit sextet value, or 0xFF for
// invalid. Inputs ≥128 are detected via the sign bit.
TEXT ·decodeBlocksNEON(SB), NOSPLIT, $0-48
	MOVD	dst+0(FP), R1
	MOVD	src+8(FP), R0
	MOVD	nblocks+16(FP), R2
	MOVD	tableA+24(FP), R3
	VLD1	(R3), [V20.B16, V21.B16, V22.B16, V23.B16]
	MOVD	tableB+32(FP), R3
	VLD1	(R3), [V24.B16, V25.B16, V26.B16, V27.B16]
	VMOVI	$64, V16.B16
	VMOVI	$0x80, V19.B16
	MOVD	ZR, R4
	CBZ	R2, dec_done
loop_dec:
	VLD4.P	64(R0), [V0.B16, V1.B16, V2.B16, V3.B16]
	VSUB	V16.B16, V0.B16, V4.B16
	VSUB	V16.B16, V1.B16, V5.B16
	VSUB	V16.B16, V2.B16, V6.B16
	VSUB	V16.B16, V3.B16, V7.B16
	VTBL	V0.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V8.B16
	VTBL	V1.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V9.B16
	VTBL	V2.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V10.B16
	VTBL	V3.B16, [V20.B16, V21.B16, V22.B16, V23.B16], V11.B16
	VTBL	V4.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V12.B16
	VTBL	V5.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V13.B16
	VTBL	V6.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V14.B16
	VTBL	V7.B16, [V24.B16, V25.B16, V26.B16, V27.B16], V15.B16
	VCMTST	V19.B16, V0.B16, V4.B16
	VCMTST	V19.B16, V1.B16, V5.B16
	VCMTST	V19.B16, V2.B16, V6.B16
	VCMTST	V19.B16, V3.B16, V7.B16
	VORR	V12.B16, V8.B16,  V8.B16
	VORR	V13.B16, V9.B16,  V9.B16
	VORR	V14.B16, V10.B16, V10.B16
	VORR	V15.B16, V11.B16, V11.B16
	VORR	V4.B16,  V8.B16,  V8.B16
	VORR	V5.B16,  V9.B16,  V9.B16
	VORR	V6.B16,  V10.B16, V10.B16
	VORR	V7.B16,  V11.B16, V11.B16
	VORR	V8.B16,  V9.B16,  V17.B16
	VORR	V10.B16, V11.B16, V18.B16
	VORR	V17.B16, V18.B16, V17.B16
	VMOV	V17.D[0], R5
	VMOV	V17.D[1], R6
	ORR	R5, R6, R5
	ORR	R5, R4, R4
	VSHL	$2, V8.B16,  V0.B16
	VUSHR	$4, V9.B16,  V17.B16
	VORR	V17.B16, V0.B16, V0.B16
	VSHL	$4, V9.B16,  V1.B16
	VUSHR	$2, V10.B16, V17.B16
	VORR	V17.B16, V1.B16, V1.B16
	VSHL	$6, V10.B16, V2.B16
	VORR	V11.B16, V2.B16, V2.B16
	VST3.P	[V0.B16, V1.B16, V2.B16], 48(R1)
	SUBS	$1, R2, R2
	BNE	loop_dec
dec_done:
	AND	$0x8080808080808080, R4, R4
	CMP	$0, R4
	CSET	NE, R0
	MOVD	R0, ret+40(FP)
	RET
