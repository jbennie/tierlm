/* tlm_quant.h - quantised block formats and dot-product kernels.
 *
 * Patterns: P-40 GGUF block layouts (byte-exact), P-42 quantise activations
 * once, P-43 SIMD == scalar bit for bit on integer paths, P-44 offsets/mins
 * folded through activation group sums, P-45 sign trick, P-47 FP8 E4M3 with
 * 128x128 scales, P-48 MXFP4, P-50 keep weights packed.
 *
 * Byte layouts (little-endian; QK = values per block):
 *   Q8_0 : QK 32,  34 B : f16 d, int8 q[32]                    x = d*q
 *   Q4_K : QK 256, 144 B: f16 d, f16 dmin, u8 scales[12], u8 qs[128]
 *   Q5_K : QK 256, 176 B: f16 d, f16 dmin, u8 scales[12], u8 qh[32], u8 qs[128]
 *   Q6_K : QK 256, 210 B: u8 ql[128], u8 qh[64], int8 scales[16], f16 d
 *   Q8_K : QK 256, 292 B: f32 d, int8 q[256], int16 bsums[16]   (activations)
 *   MXFP4: QK 32,  17 B : u8 e8m0 scale, u8 qs[16] (elements 0..15 in low
 *          nibbles, 16..31 in high nibbles; E2M1 values)
 *   FP8E4M3_B128: raw e4m3 bytes, row-major, plus a separate f32 scale per
 *          128x128 block (scale tensor stored after the weights).
 * K-quant 6-bit scale/min packing and element ordering follow the public
 * GGML definitions (see fragments/quant_kernels.md F-quant-02..06).
 */
#ifndef TLM_QUANT_H
#define TLM_QUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TLM_DT_F32 = 0,
    TLM_DT_F16 = 1,
    TLM_DT_BF16 = 2,
    TLM_DT_Q8_0 = 8,
    TLM_DT_Q4_K = 12,
    TLM_DT_Q5_K = 13,
    TLM_DT_Q6_K = 14,
    TLM_DT_Q8_K = 15,
    TLM_DT_MXFP4 = 39,
    TLM_DT_FP8E4M3_B128 = 64
} tlm_dtype;

const char *tlm_dtype_name(tlm_dtype t);
int         tlm_dtype_from_name(const char *name, tlm_dtype *out);   /* "q6_k", "Q8_0", ... */
int         tlm_dtype_block(tlm_dtype t);       /* values per block (1 for float types) */
size_t      tlm_dtype_block_bytes(tlm_dtype t); /* bytes per block */
/* Bytes for a row of n values (n must be a multiple of the block size;
 * FP8E4M3_B128 returns n, scales are accounted separately). */
size_t      tlm_row_bytes(tlm_dtype t, int64_t n);

/* fp16 / bf16 helpers (exact, round-to-nearest-even on the way in) */
float    tlm_f16_to_f32(uint16_t h);
uint16_t tlm_f32_to_f16(float f);
float    tlm_bf16_to_f32(uint16_t h);
uint16_t tlm_f32_to_bf16(float f);
float    tlm_e4m3_to_f32(uint8_t v);     /* NaN codes 0x7F/0xFF -> NaN */
float    tlm_e8m0_to_f32(uint8_t e);     /* 2^(e-127); e==255 -> NaN; e==0 -> 2^-127 */

/* Quantise a row of n floats into dtype (reference quantisers, deterministic). */
int tlm_quantize_row(tlm_dtype t, const float *x, void *dst, int64_t n);
/* Dequantise a row (FP8E4M3_B128 not supported here: needs block scales). */
int tlm_dequantize_row(tlm_dtype t, const void *src, float *y, int64_t n);

/* Activation quantisation, done once per input vector and shared (P-42). */
typedef struct {
    tlm_dtype kind;      /* TLM_DT_Q8_K for K-quant weights, TLM_DT_Q8_0 for Q8_0 / MXFP4 weights */
    int64_t   n;
    void     *data;      /* blocks */
} tlm_qact;
size_t tlm_qact_bytes(tlm_dtype kind, int64_t n);
void   tlm_quantize_act(tlm_dtype kind, const float *x, int64_t n, void *dst);
/* Which activation kind a weight dtype consumes. F32/F16/BF16 weights use f32 activations. */
tlm_dtype tlm_act_kind_for(tlm_dtype weight);

/* Dot product of one weight row with a quantised (or f32) activation.
 * `act` points at the tlm_quantize_act output (or float* for float weights). */
float tlm_vec_dot(tlm_dtype w, const void *wrow, const void *act, int64_t n);

/* y[r] = dot(W[r], x) for rows r in [r0, r1). W is row-major with `row_bytes`
 * stride. For FP8E4M3_B128 pass the scale tensor in `scales` (f32,
 * ceil(rows/128) x ceil(n/128)) and x as float*. */
void tlm_matvec_rows(tlm_dtype w, const void *W, size_t row_bytes,
                     const void *act, float *y, int64_t n, int64_t r0, int64_t r1,
                     const float *scales);

/* Kernel tier actually selected: "scalar", "avx2", "avx512". */
const char *tlm_quant_tier(void);
/* Re-run dispatch (after tlm_cpu_force). */
void tlm_quant_init(void);

/* ---- additions (tlm_quant owner) -------------------------------------
 *
 * Activation layouts produced by tlm_quantize_act (internal, not GGUF):
 *   Q8_K : 292 B per 256 values, f32 d = amax/127 (never negative),
 *          q = rne(x * 127/amax) clamped to [-127, 127], int16 sums per 16.
 *   Q8_0 : 36 B per 32 values: f32 d, int8 q[32]. Same rule per 32. NOTE the
 *          activation block carries an f32 scale, unlike the 34-byte weight
 *          block (F-quant-01); size buffers with tlm_qact_bytes().
 *   F32  : plain copy of x.
 * A trailing partial block (n not a multiple of the block) is zero padded.
 * A block holding a NaN or an infinity gets d = NaN / +inf and all-zero
 * codes, so the non-finite input propagates into every dot that uses it.
 *
 * Activation buffers must be 4-byte aligned (any malloc / tlm_alloc_aligned
 * buffer is; 64 is best). Weight rows may start at any byte offset.
 *
 * Q8_K is also accepted as a WEIGHT type (int8 codes, f32 d per 256; the
 * bsums field is ignored on the weight side); it consumes Q8_K activations.
 * Weight-side NaN inputs are not detected by the reference quantisers
 * (they encode as code 0); validate tensors before quantising.
 *
 * Numerics contract (P-43, P-56):
 *   - Integer paths (Q8_0, Q4_K, Q5_K, Q6_K, Q8_K, MXFP4): exact int32 sum
 *     inside a block, then per block acc = fmaf(d_w * d_a, (float)isum, acc)
 *     (and acc = fmaf(-(dmin_w * d_a), (float)msum, acc) for Q4_K/Q5_K),
 *     blocks in ascending order, acc starting at +0. Every tier and both the
 *     single-row and the multi-row (matvec) kernels return the same bits.
 *     Activation codes must lie in [-127, 127] (the sign trick, P-45); the
 *     quantisers above guarantee it.
 *   - Float paths (F32, F16, BF16, FP8): element i accumulates into lane
 *     i mod 32 with fmaf, lanes are reduced with one fixed tree. Every tier
 *     returns the same bits here too (stronger than the 1e-6 we promise).
 *     FP8: lanes restart per 128-column block; block partials are added in
 *     double as part * scale, the row result is rounded to float once.
 *   - Kernels assume the default floating-point environment (round to
 *     nearest even, no FTZ/DAZ); FP8 subnormal decoding relies on it.
 */
#define TLM_QACT_Q8_0_BLOCK_BYTES 36

/* Number of SIMD kernels that failed the bit-exact start-up self-test
 * against the scalar reference and were replaced by the next lower tier's
 * kernel (P-55). 0 on a healthy build; each failure is also logged. */
int tlm_quant_selftest_failures(void);

/* FP8 E4M3 encode: round to nearest even, saturating to +-448; NaN -> 0x7F
 * (sign kept). */
uint8_t tlm_f32_to_e4m3(float f);

/* FP8E4M3_B128 row helpers. `col_scales` is this row's slice of the scale
 * tensor (ceil(n/128) f32 values, scales[(row/128) * ceil(n/128) + j]) or
 * NULL for 1.0. x is float*. */
float tlm_vec_dot_fp8(const uint8_t *wrow, const float *x, int64_t n, const float *col_scales);
int   tlm_dequantize_fp8_row(const uint8_t *src, float *y, int64_t n, const float *col_scales);

/* ---- internal: kernel tier tables (tlm_quant.c <-> tlm_kern_*.c) -----
 * Only visible when TLM_QUANT_INTERNAL is defined before inclusion. */
#ifdef TLM_QUANT_INTERNAL
typedef struct { float d; int8_t q[256]; int16_t bsums[16]; } tlm_blk_q8k; /* 292 B */
typedef struct { float d; int8_t q[32]; } tlm_blk_a8;                      /* 36 B  */

enum {
    TLM_QK_F32 = 0, TLM_QK_F16, TLM_QK_BF16, TLM_QK_Q8_0, TLM_QK_Q4_K,
    TLM_QK_Q5_K, TLM_QK_Q6_K, TLM_QK_MXFP4, TLM_QK_FP8, TLM_QK_Q8_K, TLM_QK_COUNT
};
/* One row dot. `aux` = per-128-column scales for FP8 (NULL = 1.0), else unused. */
typedef float (*tlm_qdot_fn)(const void *wrow, const void *act, int64_t n, const float *aux);
/* y[0..nrows) for rows starting at W (stride row_bytes). Optional (NULL = loop
 * dot). Must return, for every row, the same bits as dot[] on that row. */
typedef void  (*tlm_qmv_fn)(const void *W, size_t row_bytes, const void *act, float *y,
                            int64_t n, int64_t nrows);
typedef struct {
    const char *name;
    tlm_qdot_fn dot[TLM_QK_COUNT];
    tlm_qmv_fn  mv[TLM_QK_COUNT];
} tlm_quant_kernels;
/* Overwrite the entries a tier implements; return how many (0 = tier not
 * compiled in). Call only when tlm_cpu() reports the tier's features. */
int tlm_kern_avx2_fill(tlm_quant_kernels *k);
int tlm_kern_avx512_fill(tlm_quant_kernels *k);
#endif

#ifdef __cplusplus
}
#endif
#endif
