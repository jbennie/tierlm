/* tlm_kern_avx512.c - AVX-512 (F/BW/VL) + VNNI dot / matvec kernels (Ice
 * Lake, Sapphire Rapids, Zen 4/5).
 *
 * Patterns: P-43 bit-identical to the scalar reference, P-44 offsets and
 * mins folded through activation sums, P-45 sign trick for signed x signed
 * int8 on vpdpbusd, P-50 weights stay packed (nibbles / bit planes merged in
 * registers), P-56 explicit fma, fixed lane mapping. Pseudocode sources:
 * fragments/quant_kernels.md F-quant-05, -09, -11, -12, -15, -21.
 *
 * vpdpbusd multiplies unsigned bytes (first operand) by signed bytes
 * (activations, always in [-127, 127]) and adds groups of four products
 * into int32 lanes without an int16 stage, so there is no saturation to
 * budget for. vpdpwssd multiplies int16 pairs and accumulates; we use it to
 * apply per-group scales to dpbusd lane sums that provably fit int16 (the
 * scale sits in the low word of each lane, the high word is 0). Each block
 * still collapses to one exact int32 and the float update is the same op
 * sequence as the scalar reference in tlm_quant.c.
 *
 * Signed weights (Q8_0, Q8_K, MXFP4) in the single-row kernels use the sign
 * trick. The 8-row tiles instead bias the weights to unsigned (P-44, the
 * weight-side mirror of F-quant-09's activation bias): Q8_0/Q8_K bytes are
 * XORed with 0x80 (w + 128), MXFP4 codes map to doubled E2M1 + 12, and the
 * accumulator of every row starts at -bias * sum(x) for that 4-byte lane,
 * computed once per block and shared by the 8 rows. Integer-exact, so the
 * results equal the sign-trick ones.
 *
 * The 8-row tiles (mv_*) follow the AVX2 file: lane r of one ymm
 * accumulator is row r; per lane the float ops are exactly the scalar
 * sequence. The 16-row VNNI-transposed repack of F-quant-09 is not used:
 * these kernels read the GGUF byte layout as stored (P-40), so packets need
 * no repack pass; a repacked path can be added behind the same table.
 *
 * Float formats: element i feeds lane i mod 32 (two zmm accumulators), the
 * < 32-element tail uses masked loads and a masked fma (one fmaf per lane),
 * and the fixed reduce32 tree runs in registers.
 *
 * Rule for this file: no local arrays that GCC would align to 64 bytes
 * (__m512i tables, >= 64-byte float buffers touched with zmm moves). GCC
 * turns zmm accesses to such locals into aligned moves, and ASan's fake
 * stack (use-after-return detection) only aligns them to 32 bytes, which
 * faults. Constants are loop-carried vectors or set1 values instead.
 *
 * Built with -mavx512f -mavx512bw -mavx512vl -mavx512vnni -mavx512dq -mfma
 * -mf16c; dispatch requires AVX512F/BW/VL/VNNI plus AVX2/FMA/F16C (every
 * CPU with AVX512-VNNI also has DQ, which the compiler may use).
 */
#define TLM_QUANT_INTERNAL 1
#include "tlm_quant.h"

#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512BW__) && \
    defined(__AVX512VL__) && defined(__AVX512VNNI__) && defined(__FMA__) && defined(__F16C__)

#include <immintrin.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#define AINLINE static inline __attribute__((always_inline))

/* Keep a loop rolled: used for the 8-row tiles of the large K-quant cores,
 * where full unrolling makes the hoisted shuffle/mask constants and eight
 * live results exceed the register file (measured: heavy spilling). */
#if defined(__clang__)
#define NO_UNROLL _Pragma("clang loop unroll(disable)")
#elif defined(__GNUC__)
#define NO_UNROLL _Pragma("GCC unroll 1")
#else
#define NO_UNROLL
#endif

AINLINE float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
AINLINE float ld_f16(const uint8_t *p) { uint16_t h; memcpy(&h, p, 2); return _cvtsh_ss(h); }
AINLINE __m512i ld512(const void *p) { return _mm512_loadu_si512(p); }
AINLINE __m256i ld256(const void *p) { return _mm256_loadu_si256((const __m256i *)p); }
AINLINE __m128i ld128(const void *p) { return _mm_loadu_si128((const __m128i *)p); }

/* 2^(e-128): E8M0 scale with E2M1's 0.5 folded in (same bits as tlm_quant.c). */
AINLINE float e8m0_half(uint8_t e) {
    uint32_t b;
    if (e == 255) b = 0x7FC00000u;
    else if (e >= 2) b = (uint32_t)(e - 1) << 23;
    else b = e ? 0x00400000u : 0x00200000u;
    return u2f(b);
}
/* Vector form for 8 exponent bytes in int32 lanes. */
AINLINE __m256 e8m0_half8(__m256i e) {
    __m256i nrm = _mm256_slli_epi32(_mm256_sub_epi32(e, _mm256_set1_epi32(1)), 23);  /* e >= 2 */
    __m256i sub = _mm256_sllv_epi32(_mm256_set1_epi32(0x00200000), e);               /* e = 0, 1 */
    __m256i bits = _mm256_mask_blend_epi32(_mm256_cmplt_epi32_mask(e, _mm256_set1_epi32(2)), nrm, sub);
    bits = _mm256_mask_blend_epi32(_mm256_cmpeq_epi32_mask(e, _mm256_set1_epi32(255)), bits,
                                   _mm256_set1_epi32(0x7FC00000));
    return _mm256_castsi256_ps(bits);
}

/* 12-byte 6-bit scale/min block (F-quant-02) -> 16 bytes [sc0..7, mn0..7]
 * in registers (reads 16 bytes; the extra 4 are quant data). Same word
 * algebra as the scalar kq_unpack_scales(). */
AINLINE __m128i kq_scales_v(const uint8_t *s12) {
    __m128i a = ld128(s12);
    __m128i A = _mm_shuffle_epi32(a, _MM_SHUFFLE(2, 1, 2, 0));   /* a0 a2 a1 a2 */
    __m128i B = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 1, 0, 0));   /* a0 a0 a1 a1 */
    A = _mm_srlv_epi32(A, _mm_setr_epi32(0, 0, 0, 4));
    A = _mm_and_si128(A, _mm_setr_epi32(0x3F3F3F3F, 0x0F0F0F0F, 0x3F3F3F3F, 0x0F0F0F0F));
    B = _mm_and_si128(_mm_srli_epi32(B, 6), _mm_setr_epi32(0, 0x03030303, 0, 0x03030303));
    return _mm_or_si128(A, _mm_slli_epi32(B, 4));
}

AINLINE int32_t hsum256_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
AINLINE __m256i fold512(__m512i v) {   /* 16 lanes -> 8 lanes with the same total */
    return _mm256_add_epi32(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1));
}

/* Horizontal sums of 8 vectors -> [sum v0, ..., sum v7] (see tlm_kern_avx2.c). */
AINLINE __m256i hadd4(__m256i v0, __m256i v1, __m256i v2, __m256i v3) {
    return _mm256_hadd_epi32(_mm256_hadd_epi32(v0, v1), _mm256_hadd_epi32(v2, v3));
}
AINLINE __m256i hjoin(__m256i h0123, __m256i h4567) {
    return _mm256_add_epi32(_mm256_permute2x128_si256(h0123, h4567, 0x20),
                            _mm256_permute2x128_si256(h0123, h4567, 0x31));
}
AINLINE __m256i hsum8x8(const __m256i v[8]) {
    return hjoin(hadd4(v[0], v[1], v[2], v[3]), hadd4(v[4], v[5], v[6], v[7]));
}

/* f16 in the low 16 bits of each int32 lane -> 8 floats (exact). */
AINLINE __m256 f16lo_to_ps(__m256i v) {
    return _mm256_cvtph_ps(_mm256_cvtepi32_epi16(_mm256_and_si256(v, _mm256_set1_epi32(0xFFFF))));
}

AINLINE int row_index8(size_t rb, __m256i *idx) {
    if (rb > (size_t)INT_MAX / 8) return 0;
    int r = (int)rb;
    *idx = _mm256_setr_epi32(0, r, 2 * r, 3 * r, 4 * r, 5 * r, 6 * r, 7 * r);
    return 1;
}
AINLINE __m256i gather8(const uint8_t *p, __m256i idx) {
    return _mm256_i32gather_epi32((const int *)(const void *)p, idx, 1);
}

AINLINE float flush8(float acc, __m256i isum, const float *dw, const float *dx) {
    float t[8], f[8];
    _mm256_storeu_ps(t, _mm256_mul_ps(_mm256_loadu_ps(dw), _mm256_loadu_ps(dx)));
    _mm256_storeu_ps(f, _mm256_cvtepi32_ps(isum));
    for (int j = 0; j < 8; j++) acc = fmaf(t[j], f[j], acc);
    return acc;
}

/* Software prefetch for the 8 row streams of a tile: `lines` cache lines
 * starting `ahead` bytes past the current block of row j. Measured on this
 * project's dev machine: without it the hardware stride prefetcher does not
 * follow eight interleaved row streams read from one load site, and the
 * Q6_K tile ran slower than the single-row kernel. Prefetches never fault,
 * so running past the last row is harmless. */
AINLINE void pf_row(const uint8_t *p, size_t ahead, int lines) {
    for (int l = 0; l < lines; l++) _mm_prefetch((const char *)(p + ahead + 64 * (size_t)l), _MM_HINT_T0);
}

/* Signed x signed on vpdpbusd (P-45): |w| unsigned, x negated where w < 0.
 * x in [-127, 127] so the negation never wraps; w may be -128 (|w| = 128
 * read unsigned). */
AINLINE __m512i dot_s8_512(__m512i w, __m512i x) {
    __mmask64 neg = _mm512_movepi8_mask(w);
    __m512i xs = _mm512_mask_sub_epi8(x, neg, _mm512_setzero_si512(), x);
    return _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(w), xs);
}
AINLINE __m256i dot_s8_256(__m256i w, __m256i x) {
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
}
/* -bias * (sum of each 4 activation bytes), the seed of a biased-weight dot. */
AINLINE __m256i neg_bias256(__m256i x, __m256i bias8) {
    return _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_dpbusd_epi32(_mm256_setzero_si256(), bias8, x));
}

AINLINE __m512i join256(__m256i lo, __m256i hi) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

/* ======================================================================
 * Q8_0 x Q8_0-activation. Single row: two 32-element blocks per zmm.
 * ==================================================================== */
static float dot_q8_0(const void *wv, const void *av, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const tlm_blk_a8 *a = av;
    const int64_t nb = n / 32;
    int64_t b = 0;
    float acc = 0.0f;
    for (; b + 8 <= nb; b += 8) {
        __m256i v[8];
        float dw[8], dx[8];
        for (int j = 0; j < 8; j += 2) {
            const uint8_t *b0 = w + 34 * (b + j), *b1 = b0 + 34;
            __m512i t = dot_s8_512(join256(ld256(b0 + 2), ld256(b1 + 2)),
                                   join256(ld256(a[b + j].q), ld256(a[b + j + 1].q)));
            v[j] = _mm512_castsi512_si256(t);
            v[j + 1] = _mm512_extracti64x4_epi64(t, 1);
            dw[j] = ld_f16(b0); dw[j + 1] = ld_f16(b1);
            dx[j] = a[b + j].d; dx[j + 1] = a[b + j + 1].d;
        }
        acc = flush8(acc, hsum8x8(v), dw, dx);
    }
    for (; b < nb; b++) {
        const uint8_t *blk = w + 34 * b;
        int32_t s = hsum256_i32(dot_s8_256(ld256(blk + 2), ld256(a[b].q)));
        acc = fmaf(ld_f16(blk) * a[b].d, (float)s, acc);
    }
    return acc;
}

/* 8-row tile: per row one XOR and one vpdpbusd seeded with -128*sum(x). */
static void mv_q8_0(const void *W, size_t rb, const void *av, float *y, int64_t n, int64_t nrows) {
    const uint8_t *base = W;
    const tlm_blk_a8 *a = av;
    const __m256i x80 = _mm256_set1_epi8((char)0x80);
    const int64_t nb = n / 32;
    int64_t r = 0;
    __m256i idx;
    if (row_index8(rb, &idx)) {
        for (; r + 8 <= nrows; r += 8) {
            const uint8_t *w0 = base + rb * (size_t)r;
            __m256 acc = _mm256_setzero_ps();
            for (int64_t b = 0; b < nb; b++) {
                const uint8_t *bp = w0 + 34 * b;
                const __m256i xq = ld256(a[b].q);
                const __m256i seed = neg_bias256(xq, x80);
                __m256i v[8];
                for (int j = 0; j < 8; j++) {
                    pf_row(bp + rb * j, 512, 1);   /* 64 B per 34 B block */
                    v[j] = _mm256_dpbusd_epi32(seed, _mm256_xor_si256(ld256(bp + rb * j + 2), x80), xq);
                }
                __m256 t = _mm256_mul_ps(f16lo_to_ps(gather8(bp, idx)), _mm256_set1_ps(a[b].d));
                acc = _mm256_fmadd_ps(t, _mm256_cvtepi32_ps(hsum8x8(v)), acc);
            }
            _mm256_storeu_ps(y + r, acc);
        }
    }
    for (; r < nrows; r++) y[r] = dot_q8_0(base + rb * (size_t)r, av, n, NULL);
}

/* ======================================================================
 * MXFP4 x Q8_0-activation. Single row: two blocks per zmm, 128-bit lanes
 * [block0 low nibbles | block0 high | block1 low | block1 high]; one masked
 * 16-bit shift moves the high nibbles down, one byte shuffle maps codes to
 * doubled signed E2M1 ints (|w| <= 12).
 * ==================================================================== */
#define MXFP4_LUT128 _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
/* the same table biased by +12 (unsigned 0..24) for the tile kernel */
#define MXFP4_ULUT128 _mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0)

AINLINE __m512i mxfp4_w64(const uint8_t *q0, const uint8_t *q1, __m512i lut, __m512i m4) {
    __m512i v = join256(_mm256_broadcastsi128_si256(ld128(q0)), _mm256_broadcastsi128_si256(ld128(q1)));
    v = _mm512_mask_srli_epi16(v, (__mmask32)0xFF00FF00u, v, 4);
    return _mm512_shuffle_epi8(lut, _mm512_and_si512(v, m4));
}
/* 16 nibble bytes -> 32 table bytes in element order (low lane = low nibbles). */
AINLINE __m256i mxfp4_w32(const uint8_t *q, __m256i lut, __m256i m4) {
    __m256i v = _mm256_broadcastsi128_si256(ld128(q));
    v = _mm256_mask_srli_epi16(v, (__mmask16)0xFF00u, v, 4);
    return _mm256_shuffle_epi8(lut, _mm256_and_si256(v, m4));
}

static float dot_mxfp4(const void *wv, const void *av, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const tlm_blk_a8 *a = av;
    const __m512i lut = _mm512_broadcast_i32x4(MXFP4_LUT128);
    const __m256i lut256 = _mm256_broadcastsi128_si256(MXFP4_LUT128);
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m256i m4y = _mm256_set1_epi8(0x0F);
    const int64_t nb = n / 32;
    int64_t b = 0;
    float acc = 0.0f;
    for (; b + 8 <= nb; b += 8) {
        __m256i v[8];
        float dw[8], dx[8];
        for (int j = 0; j < 8; j += 2) {
            const uint8_t *b0 = w + 17 * (b + j), *b1 = b0 + 17;
            __m512i t = dot_s8_512(mxfp4_w64(b0 + 1, b1 + 1, lut, m4),
                                   join256(ld256(a[b + j].q), ld256(a[b + j + 1].q)));
            v[j] = _mm512_castsi512_si256(t);
            v[j + 1] = _mm512_extracti64x4_epi64(t, 1);
            dw[j] = e8m0_half(b0[0]); dw[j + 1] = e8m0_half(b1[0]);
            dx[j] = a[b + j].d; dx[j + 1] = a[b + j + 1].d;
        }
        acc = flush8(acc, hsum8x8(v), dw, dx);
    }
    for (; b < nb; b++) {
        const uint8_t *blk = w + 17 * b;
        int32_t s = hsum256_i32(dot_s8_256(mxfp4_w32(blk + 1, lut256, m4y), ld256(a[b].q)));
        acc = fmaf(e8m0_half(blk[0]) * a[b].d, (float)s, acc);
    }
    return acc;
}

static void mv_mxfp4(const void *W, size_t rb, const void *av, float *y, int64_t n, int64_t nrows) {
    const uint8_t *base = W;
    const tlm_blk_a8 *a = av;
    const __m256i ulut = _mm256_broadcastsi128_si256(MXFP4_ULUT128);
    const __m256i m4 = _mm256_set1_epi8(0x0F), k12 = _mm256_set1_epi8(12), mff = _mm256_set1_epi32(0xFF);
    const int64_t nb = n / 32;
    int64_t r = 0;
    __m256i idx;
    if (row_index8(rb, &idx)) {
        for (; r + 8 <= nrows; r += 8) {
            const uint8_t *w0 = base + rb * (size_t)r;
            __m256 acc = _mm256_setzero_ps();
            for (int64_t b = 0; b < nb; b++) {
                const uint8_t *bp = w0 + 17 * b;
                const __m256i xq = ld256(a[b].q);
                const __m256i seed = neg_bias256(xq, k12);
                __m256i v[8];
                for (int j = 0; j < 8; j++) {
                    if (!(b & 1)) pf_row(bp + rb * j, 512, 1);   /* 64 B per two 17 B blocks */
                    v[j] = _mm256_dpbusd_epi32(seed, mxfp4_w32(bp + rb * j + 1, ulut, m4), xq);
                }
                __m256 dw = e8m0_half8(_mm256_and_si256(gather8(bp, idx), mff));
                __m256 t = _mm256_mul_ps(dw, _mm256_set1_ps(a[b].d));
                acc = _mm256_fmadd_ps(t, _mm256_cvtepi32_ps(hsum8x8(v)), acc);
            }
            _mm256_storeu_ps(y + r, acc);
        }
    }
    for (; r < nrows; r++) y[r] = dot_mxfp4(base + rb * (size_t)r, av, n, NULL);
}

/* ======================================================================
 * Q4_K / Q5_K x Q8_K. Per chunk c the 32 quant bytes are broadcast to both
 * zmm halves; the upper half is shifted right by 4 (masked 16-bit shift)
 * so after AND 0x0F the zmm holds elements 64c..64c+63 in order and pairs
 * with one 64-byte activation load. vpdpbusd lanes 0..7 belong to
 * sub-block 2c, lanes 8..15 to 2c+1. Lane sums are <= 4*31*127 = 15748, so
 * the 6-bit scale is applied with vpdpwssd (scale in the low word).
 * Q5_K: bit 2c of qh[l] feeds the lower half, bit 2c+1 the upper half; a
 * byte test-mask selects the bytes that get +16.
 * ==================================================================== */
AINLINE void kq_core(const uint8_t *blk, const tlm_blk_q8k *x, const int q5, __m256i *iv, __m256i *mvv) {
    const __m512i m4 = _mm512_set1_epi8(0x0F), b16 = _mm512_set1_epi8(16);
    const __m128i dup = _mm_setr_epi8(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    const uint8_t *qs = q5 ? blk + 48 : blk + 16;
    const __m128i sm = kq_scales_v(blk + 4);
    const __m512i scm = _mm512_cvtepu8_epi32(sm);      /* lanes 0..7 scales, 8..15 mins */
    /* Q5_K 5th-bit plane in both halves; chunk c tests bit 0 (lower half,
     * sub-block 2c) / bit 1 (upper half, 2c+1) after HH >>= 2 per chunk. A
     * 16-bit shift drags bits across bytes only into bits 6..7, which reach
     * bit 1 after three more shifts at the earliest, i.e. never. */
    __m512i HH = q5 ? _mm512_broadcast_i64x4(ld256(blk + 16)) : _mm512_setzero_si512();
    const __m512i hbit = join256(_mm256_set1_epi8(0x01), _mm256_set1_epi8(0x02));
    /* scale lane of each dpbusd lane: sub-block 2c (lanes 0..7), 2c+1 (8..15); +2 per chunk.
     * (Loop-carried vectors rather than local __m512i tables: GCC's ASan fake
     * stack does not honour 64-byte alignment of such locals.) */
    __m512i sidx = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m512i two = _mm512_set1_epi32(2);
    __m512i acc = _mm512_setzero_si512();
    for (int c = 0; c < 4; c++) {
        __m512i q = _mm512_broadcast_i64x4(ld256(qs + 32 * c));
        q = _mm512_and_si512(_mm512_mask_srli_epi16(q, (__mmask32)0xFFFF0000u, q, 4), m4);
        if (q5) {
            q = _mm512_mask_add_epi8(q, _mm512_test_epi8_mask(HH, hbit), q, b16);
            HH = _mm512_srli_epi16(HH, 2);
        }
        __m512i t = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q, ld512(x->q + 64 * c));
        acc = _mm512_dpwssd_epi32(acc, t, _mm512_permutexvar_epi32(sidx, scm));
        sidx = _mm512_add_epi32(sidx, two);
    }
    *iv = fold512(acc);
    *mvv = _mm256_madd_epi16(ld256(x->bsums), _mm256_cvtepu8_epi16(_mm_shuffle_epi8(sm, dup)));
}

AINLINE float dot_kq(const uint8_t *w, const tlm_blk_q8k *a, int64_t n, const int q5) {
    const int64_t nb = n / 256;
    const size_t bb = q5 ? 176 : 144;
    float acc = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + bb * i;
        __m256i iv, mvv;
        kq_core(blk, a + i, q5, &iv, &mvv);
        int32_t isum = hsum256_i32(iv), msum = hsum256_i32(mvv);
        float dA = ld_f16(blk) * a[i].d, dM = ld_f16(blk + 2) * a[i].d;
        acc = fmaf(dA, (float)isum, acc);
        acc = fmaf(-dM, (float)msum, acc);
    }
    return acc;
}

AINLINE void mv_kq(const uint8_t *base, size_t rb, const tlm_blk_q8k *a, float *y, int64_t n,
                   int64_t nrows, const int q5) {
    const int64_t nb = n / 256;
    const size_t bb = q5 ? 176 : 144;
    int64_t r = 0;
    __m256i idx;
    if (row_index8(rb, &idx)) {
        for (; r + 8 <= nrows; r += 8) {
            const uint8_t *w0 = base + rb * (size_t)r;
            __m256 acc = _mm256_setzero_ps();
            for (int64_t i = 0; i < nb; i++) {
                const uint8_t *bp = w0 + bb * i;
                __m256i iv[8], mv8[8];
                NO_UNROLL
                for (int j = 0; j < 8; j++) {
                    pf_row(bp + rb * (size_t)j, 2 * bb, 3);
                    kq_core(bp + rb * (size_t)j, a + i, q5, &iv[j], &mv8[j]);
                }
                const __m256i dd = gather8(bp, idx);                 /* d | dmin << 16 */
                const __m256 xd = _mm256_set1_ps(a[i].d);
                __m256 dA = _mm256_mul_ps(f16lo_to_ps(dd), xd);
                __m256 dM = _mm256_mul_ps(f16lo_to_ps(_mm256_srli_epi32(dd, 16)), xd);
                acc = _mm256_fmadd_ps(dA, _mm256_cvtepi32_ps(hsum8x8(iv)), acc);
                acc = _mm256_fnmadd_ps(dM, _mm256_cvtepi32_ps(hsum8x8(mv8)), acc);
            }
            _mm256_storeu_ps(y + r, acc);
        }
    }
    for (; r < nrows; r++) y[r] = dot_kq(base + rb * (size_t)r, a, n, q5);
}

static float dot_q4_k(const void *w, const void *a, int64_t n, const float *aux) {
    (void)aux;
    return dot_kq(w, a, n, 0);
}
static float dot_q5_k(const void *w, const void *a, int64_t n, const float *aux) {
    (void)aux;
    return dot_kq(w, a, n, 1);
}
static void mv_q4_k(const void *W, size_t rb, const void *a, float *y, int64_t n, int64_t nrows) {
    mv_kq(W, rb, a, y, n, nrows, 0);
}
static void mv_q5_k(const void *W, size_t rb, const void *a, float *y, int64_t n, int64_t nrows) {
    mv_kq(W, rb, a, y, n, nrows, 1);
}

/* ======================================================================
 * Q6_K x Q8_K. Per 128-element half: AB = ql[0..63] (one zmm), C =
 * qh[0..31] and CC = [C | C >> 2] so one mask pattern serves both 32-byte
 * halves:
 *   w01 = (AB & 15)      | (CC & 0x03) << 4   -> elements   0..63
 *   w23 = (AB >> 4 & 15) | (CC & 0x30)        -> elements  64..127
 * vpdpbusd lanes cover 4 consecutive elements, so lanes 4g..4g+3 of a zmm
 * belong to one 16-element group. Each lane sum is |.| <= 4*63*127 = 32004
 * (fits int16), so the int8 scale is applied with vpdpwssd against a vector
 * holding the scale in the low word and 0 in the high word. The -32 offset
 * is folded out as 32 * sum_g sc[g] * S[g] (P-44).
 * ==================================================================== */
/* One 128-element half. g01 / g23 give the group (scale lane) of every
 * dpbusd lane of the two 64-element pieces of this half. */
AINLINE __m512i q6_half(__m512i acc, const uint8_t *ql, const uint8_t *qh, const int8_t *xq, __m512i scw,
                        __m512i g01, __m512i g23) {
    const __m512i m4 = _mm512_set1_epi8(0x0F), m03 = _mm512_set1_epi8(0x03), m30 = _mm512_set1_epi8(0x30);
    const __m512i AB = ld512(ql);
    const __m256i C = ld256(qh);
    const __m512i CC = join256(C, _mm256_srli_epi16(C, 2));
    const __m512i w01 = _mm512_or_si512(_mm512_and_si512(AB, m4),
                                        _mm512_slli_epi16(_mm512_and_si512(CC, m03), 4));
    const __m512i w23 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(AB, 4), m4),
                                        _mm512_and_si512(CC, m30));
    __m512i t01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ld512(xq));
    __m512i t23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ld512(xq + 64));
    acc = _mm512_dpwssd_epi32(acc, t01, _mm512_permutexvar_epi32(g01, scw));
    return _mm512_dpwssd_epi32(acc, t23, _mm512_permutexvar_epi32(g23, scw));
}

AINLINE __m256i q6_core(const uint8_t *ql, const tlm_blk_q8k *x) {
    const __m128i sc8 = ld128(ql + 192);
    const __m512i scw = _mm512_and_si512(_mm512_cvtepi8_epi32(sc8), _mm512_set1_epi32(0xFFFF));
    /* half 0: pieces cover groups 0..3 and 4..7; half 1: 8..11 and 12..15 */
    __m512i acc = q6_half(_mm512_setzero_si512(), ql, ql + 128, x->q, scw,
                          _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
                          _mm512_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7));
    acc = q6_half(acc, ql + 64, ql + 160, x->q + 128, scw,
                  _mm512_setr_epi32(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11),
                  _mm512_setr_epi32(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15));
    const __m256i off = _mm256_madd_epi16(ld256(x->bsums), _mm256_cvtepi8_epi16(sc8));
    return _mm256_sub_epi32(fold512(acc), _mm256_slli_epi32(off, 5));
}

static float dot_q6_k(const void *wv, const void *av, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const tlm_blk_q8k *a = av;
    const int64_t nb = n / 256;
    float acc = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *ql = w + 210 * i;
        int32_t isum = hsum256_i32(q6_core(ql, a + i));
        acc = fmaf(ld_f16(ql + 208) * a[i].d, (float)isum, acc);
    }
    return acc;
}

static void mv_q6_k(const void *W, size_t rb, const void *av, float *y, int64_t n, int64_t nrows) {
    const uint8_t *base = W;
    const tlm_blk_q8k *a = av;
    const int64_t nb = n / 256;
    int64_t r = 0;
    __m256i idx;
    if (row_index8(rb, &idx)) {
        for (; r + 8 <= nrows; r += 8) {
            const uint8_t *w0 = base + rb * (size_t)r;
            __m256 acc = _mm256_setzero_ps();
            for (int64_t i = 0; i < nb; i++) {
                const uint8_t *bp = w0 + 210 * i;
                __m256i v[8];
                NO_UNROLL
                for (int j = 0; j < 8; j++) {
                    pf_row(bp + rb * (size_t)j, 2 * 210, 4);
                    v[j] = q6_core(bp + rb * (size_t)j, a + i);
                }
                /* d sits in bytes 208..209: gather the word at 206 (in bounds), take its high half */
                __m256 dw = f16lo_to_ps(_mm256_srli_epi32(gather8(bp + 206, idx), 16));
                __m256 t = _mm256_mul_ps(dw, _mm256_set1_ps(a[i].d));
                acc = _mm256_fmadd_ps(t, _mm256_cvtepi32_ps(hsum8x8(v)), acc);
            }
            _mm256_storeu_ps(y + r, acc);
        }
    }
    for (; r < nrows; r++) y[r] = dot_q6_k(base + rb * (size_t)r, av, n, NULL);
}

/* ======================================================================
 * Q8_K weights x Q8_K activations (f32 d, int8 q[256]; weight-side bsums
 * unused). Single row: sign trick on four zmm. Tile: XOR-0x80 bias with the
 * accumulator seeded by -128*sum(x) once per block.
 * ==================================================================== */
AINLINE __m256i q8k_core(const uint8_t *blk, const tlm_blk_q8k *x) {
    __m512i acc = dot_s8_512(ld512(blk + 4), ld512(x->q));
    for (int c = 1; c < 4; c++)
        acc = _mm512_add_epi32(acc, dot_s8_512(ld512(blk + 4 + 64 * c), ld512(x->q + 64 * c)));
    return fold512(acc);
}

static float dot_q8_k(const void *wv, const void *av, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const tlm_blk_q8k *a = av;
    const int64_t nb = n / 256;
    float acc = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + 292 * i;
        float dw;
        memcpy(&dw, blk, 4);
        acc = fmaf(dw * a[i].d, (float)hsum256_i32(q8k_core(blk, a + i)), acc);
    }
    return acc;
}

static void mv_q8_k(const void *W, size_t rb, const void *av, float *y, int64_t n, int64_t nrows) {
    const uint8_t *base = W;
    const tlm_blk_q8k *a = av;
    const __m512i x80 = _mm512_set1_epi8((char)0x80);
    const int64_t nb = n / 256;
    int64_t r = 0;
    __m256i idx;
    if (row_index8(rb, &idx)) {
        for (; r + 8 <= nrows; r += 8) {
            const uint8_t *w0 = base + rb * (size_t)r;
            __m256 acc = _mm256_setzero_ps();
            for (int64_t i = 0; i < nb; i++) {
                const uint8_t *bp = w0 + 292 * i;
                const __m512i x0 = ld512(a[i].q), x1 = ld512(a[i].q + 64);
                const __m512i x2 = ld512(a[i].q + 128), x3 = ld512(a[i].q + 192);
                __m512i bsum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), x80, x0);
                bsum = _mm512_dpbusd_epi32(bsum, x80, x1);
                bsum = _mm512_dpbusd_epi32(bsum, x80, x2);
                bsum = _mm512_dpbusd_epi32(bsum, x80, x3);
                const __m512i seed = _mm512_sub_epi32(_mm512_setzero_si512(), bsum);
                __m256i v[8];
                for (int j = 0; j < 8; j++) {
                    const uint8_t *q = bp + rb * j + 4;
                    pf_row(bp + rb * j, 2 * 292, 5);
                    __m512i s = _mm512_dpbusd_epi32(seed, _mm512_xor_si512(ld512(q), x80), x0);
                    s = _mm512_dpbusd_epi32(s, _mm512_xor_si512(ld512(q + 64), x80), x1);
                    s = _mm512_dpbusd_epi32(s, _mm512_xor_si512(ld512(q + 128), x80), x2);
                    s = _mm512_dpbusd_epi32(s, _mm512_xor_si512(ld512(q + 192), x80), x3);
                    v[j] = fold512(s);
                }
                __m256 t = _mm256_mul_ps(_mm256_castsi256_ps(gather8(bp, idx)), _mm256_set1_ps(a[i].d));
                acc = _mm256_fmadd_ps(t, _mm256_cvtepi32_ps(hsum8x8(v)), acc);
            }
            _mm256_storeu_ps(y + r, acc);
        }
    }
    for (; r < nrows; r++) y[r] = dot_q8_k(base + rb * (size_t)r, av, n, NULL);
}

/* ======================================================================
 * float weights x f32 activations. Two zmm accumulators = lanes 0..15 and
 * 16..31 of the 32-lane contract.
 * ==================================================================== */
enum { FT_F32, FT_F16, FT_BF16, FT_FP8 };

/* Weights for elements [i, i+16), only lanes in k loaded (others 0; a
 * masked load never touches masked-off bytes). */
AINLINE __m512 ld16(const int ft, const uint8_t *w, int64_t i, __mmask16 k) {
    switch (ft) {
    case FT_F32: return _mm512_maskz_loadu_ps(k, w + 4 * i);
    case FT_F16: return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(k, w + 2 * i));
    case FT_BF16:
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, w + 2 * i)), 16));
    default: {   /* FP8 E4M3 via shift + 2^120 rebias, NaN codes patched (F-quant-12) */
        __m512i b = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(k, w + i));
        __m512i m7f = _mm512_set1_epi32(0x7F);
        __m512i lo7 = _mm512_and_si512(b, m7f);
        __m512i bits = _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x80)), 24),
                                       _mm512_slli_epi32(lo7, 20));
        __m512 v = _mm512_mul_ps(_mm512_castsi512_ps(bits), _mm512_set1_ps(0x1p120f));
        return _mm512_mask_mov_ps(v, _mm512_cmpeq_epi32_mask(lo7, m7f), _mm512_set1_ps(u2f(0x7FC00000u)));
    }
    }
}

/* reduce32() of tlm_quant.c done in registers: z0 = lanes 0..15, z1 = 16..31.
 *   s16 = r[l] + r[l+16]; s8 = s16[l] + s16[l+8]; s4 = s8[l] + s8[l+4];
 *   result = (s4[0] + s4[2]) + (s4[1] + s4[3])   -- the same additions. */
AINLINE float reduce32_zmm(__m512 z0, __m512 z1) {
    __m512 s16 = _mm512_add_ps(z0, z1);
    __m256 s8 = _mm256_add_ps(_mm512_castps512_ps256(s16),
                              _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(s16), 1)));
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    __m128 t = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}

/* Lanes over elements [i0, i1); i0 must be a multiple of 32. The tail (< 32
 * elements, at most one per lane) uses masked loads and a masked fma, which
 * per lane is the reference's single fmaf; lanes without a tail element keep
 * their value. No stack buffer is involved. */
AINLINE float lanes_dot(const int ft, const uint8_t *w, const float *x, int64_t i0, int64_t i1) {
    __m512 z0 = _mm512_setzero_ps(), z1 = _mm512_setzero_ps();
    int64_t i = i0;
    for (; i + 32 <= i1; i += 32) {
        z0 = _mm512_fmadd_ps(ld16(ft, w, i, 0xFFFF), _mm512_loadu_ps(x + i), z0);
        z1 = _mm512_fmadd_ps(ld16(ft, w, i + 16, 0xFFFF), _mm512_loadu_ps(x + i + 16), z1);
    }
    const int64_t rem = i1 - i;
    if (rem > 0) {
        const __mmask16 k0 = rem >= 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << rem) - 1u);
        z0 = _mm512_mask3_fmadd_ps(ld16(ft, w, i, k0), _mm512_maskz_loadu_ps(k0, x + i), z0, k0);
        if (rem > 16) {
            const __mmask16 k1 = (__mmask16)((1u << (rem - 16)) - 1u);
            z1 = _mm512_mask3_fmadd_ps(ld16(ft, w, i + 16, k1), _mm512_maskz_loadu_ps(k1, x + i + 16), z1, k1);
        }
    }
    return reduce32_zmm(z0, z1);
}

static float dot_f32(const void *w, const void *x, int64_t n, const float *aux) {
    (void)aux;
    return lanes_dot(FT_F32, w, x, 0, n);
}
static float dot_f16(const void *w, const void *x, int64_t n, const float *aux) {
    (void)aux;
    return lanes_dot(FT_F16, w, x, 0, n);
}
static float dot_bf16(const void *w, const void *x, int64_t n, const float *aux) {
    (void)aux;
    return lanes_dot(FT_BF16, w, x, 0, n);
}
static float dot_fp8(const void *w, const void *x, int64_t n, const float *cs) {
    double acc = 0.0;
    for (int64_t b0 = 0, cb = 0; b0 < n; b0 += 128, cb++) {
        float part = lanes_dot(FT_FP8, w, x, b0, b0 + 128 < n ? b0 + 128 : n);
        acc += (double)part * (double)(cs ? cs[cb] : 1.0f);
    }
    return (float)acc;
}

int tlm_kern_avx512_fill(tlm_quant_kernels *k) {
    k->dot[TLM_QK_F32] = dot_f32;
    k->dot[TLM_QK_F16] = dot_f16;
    k->dot[TLM_QK_BF16] = dot_bf16;
    k->dot[TLM_QK_FP8] = dot_fp8;
    k->dot[TLM_QK_Q8_0] = dot_q8_0;
    k->dot[TLM_QK_MXFP4] = dot_mxfp4;
    k->dot[TLM_QK_Q4_K] = dot_q4_k;
    k->dot[TLM_QK_Q5_K] = dot_q5_k;
    k->dot[TLM_QK_Q6_K] = dot_q6_k;
    k->dot[TLM_QK_Q8_K] = dot_q8_k;
    k->mv[TLM_QK_Q8_0] = mv_q8_0;
    k->mv[TLM_QK_MXFP4] = mv_mxfp4;
    k->mv[TLM_QK_Q4_K] = mv_q4_k;
    k->mv[TLM_QK_Q5_K] = mv_q5_k;
    k->mv[TLM_QK_Q6_K] = mv_q6_k;
    k->mv[TLM_QK_Q8_K] = mv_q8_k;
    return 16;
}

#else  /* not built for AVX-512 VNNI */

int tlm_kern_avx512_fill(tlm_quant_kernels *k) {
    (void)k;
    return 0;
}

#endif
