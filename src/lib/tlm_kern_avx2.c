/* tlm_kern_avx2.c - AVX2 + FMA + F16C dot / matvec kernels (Haswell, Zen
 * and later).
 *
 * Patterns: P-43 bit-identical to the scalar reference, P-44 offsets and
 * mins folded through activation group sums, P-45 sign trick for signed x
 * signed int8, P-50 weights stay packed (nibbles / bit planes are merged in
 * registers only), P-56 explicit fma, fixed lane mapping and reduction
 * tree. Pseudocode sources: fragments/quant_kernels.md F-quant-02, -05,
 * -08, -11, -12, -15, -21.
 *
 * Integer formats: every block is reduced to one exact int32 (vpmaddubsw +
 * vpmaddwd; saturation budgets are given per format), then the float update
 * is the same op sequence as the scalar reference in tlm_quant.c, in
 * ascending block order. Lane order inside a block never matters because
 * integer addition is associative.
 *
 * Two kernel shapes per integer format:
 *   dot_*  one row; per block an int32 vector is reduced horizontally and
 *          the scalar fmaf chain runs over blocks.
 *   mv_*   8-row tile (F-quant-21 rule 6): the same per-block integer core
 *          runs for 8 rows, one transposing horizontal add yields the 8
 *          block sums as one vector (lane r = row r), and one vector fma per
 *          block updates 8 independent accumulators. Per lane this is
 *          exactly the scalar sequence (vfmadd is a correctly rounded fused
 *          multiply-add per lane), so rows come out bit-identical to dot_*;
 *          the 8 chains hide the fma latency that bounds a single row.
 *          The per-row scale fields are fetched with one 32-bit gather.
 *
 * Float formats: element i feeds lane i mod 32 (four ymm accumulators of 8),
 * the tail is finished with scalar fmaf on the stored lanes, and reduce32()
 * is a copy of the reference tree.
 *
 * Built with -mavx2 -mfma -mf16c (see Makefile); only ever called when
 * tlm_cpu() reports those features.
 */
#define TLM_QUANT_INTERNAL 1
#include "tlm_quant.h"

#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)

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

/* Vector form of e8m0_half for 8 exponent bytes held in int32 lanes. */
AINLINE __m256 e8m0_half8(__m256i e) {
    __m256i nrm = _mm256_slli_epi32(_mm256_sub_epi32(e, _mm256_set1_epi32(1)), 23);  /* e >= 2 */
    __m256i sub = _mm256_sllv_epi32(_mm256_set1_epi32(0x00200000), e);               /* e = 0, 1 */
    __m256i small = _mm256_cmpgt_epi32(_mm256_set1_epi32(2), e);
    __m256i bits = _mm256_blendv_epi8(nrm, sub, small);
    __m256i nan = _mm256_cmpeq_epi32(e, _mm256_set1_epi32(255));
    bits = _mm256_blendv_epi8(bits, _mm256_set1_epi32(0x7FC00000), nan);
    return _mm256_castsi256_ps(bits);
}

/* 12-byte 6-bit scale/min block (F-quant-02) unpacked in registers into 16
 * bytes [sc0..sc7, mn0..mn7]. Reads 16 bytes at s12; inside a Q4_K/Q5_K
 * block the 4 extra bytes are quant data. Same word algebra as the scalar
 * kq_unpack_scales():
 *   out word 0 = a0 & 0x3F..          out word 2 = a1 & 0x3F..
 *   out word 1 = (a2 & 0x0F..) | ((a0 >> 6) & 0x03..) << 4
 *   out word 3 = ((a2 >> 4) & 0x0F..) | ((a1 >> 6) & 0x03..) << 4 */
AINLINE __m128i kq_scales_v(const uint8_t *s12) {
    __m128i a = ld128(s12);
    __m128i A = _mm_shuffle_epi32(a, _MM_SHUFFLE(2, 1, 2, 0));   /* a0 a2 a1 a2 */
    __m128i B = _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 1, 0, 0));   /* a0 a0 a1 a1 */
    A = _mm_srlv_epi32(A, _mm_setr_epi32(0, 0, 0, 4));
    A = _mm_and_si128(A, _mm_setr_epi32(0x3F3F3F3F, 0x0F0F0F0F, 0x3F3F3F3F, 0x0F0F0F0F));
    B = _mm_and_si128(_mm_srli_epi32(B, 6), _mm_setr_epi32(0, 0x03030303, 0, 0x03030303));
    return _mm_or_si128(A, _mm_slli_epi32(B, 4));
}

/* Fixed 32-lane tree; identical to reduce32() in tlm_quant.c. */
static float reduce32(const float *r) {
    float s16[16], s8[8], s4[4];
    for (int l = 0; l < 16; l++) s16[l] = r[l] + r[l + 16];
    for (int l = 0; l < 8; l++) s8[l] = s16[l] + s16[l + 8];
    for (int l = 0; l < 4; l++) s4[l] = s8[l] + s8[l + 4];
    return (s4[0] + s4[2]) + (s4[1] + s4[3]);
}

AINLINE int32_t hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

/* Two horizontal int32 sums with one shuffle network. */
AINLINE void hsum2_i32(__m256i a, __m256i b, int32_t *sa, int32_t *sb) {
    __m256i h = _mm256_hadd_epi32(a, b);
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
    s = _mm_hadd_epi32(s, s);
    *sa = _mm_cvtsi128_si32(s);
    *sb = _mm_extract_epi32(s, 1);
}

/* Horizontal sums of 8 vectors as one vector [sum v0, ..., sum v7], built in
 * two halves so only four inputs need to be live at a time:
 *   hadd4(v0..v3) -> low 128: low-half sums of v0..v3, high 128: high-half sums
 *   hjoin(h0123, h4567) adds the halves and orders the lanes 0..7. */
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

/* f16 held in the low 16 bits of each int32 lane -> 8 floats (exact). */
AINLINE __m256 f16lo_to_ps(__m256i v) {
    __m256i m = _mm256_and_si256(v, _mm256_set1_epi32(0xFFFF));
    __m256i p = _mm256_packus_epi32(m, m);              /* per 128: w0..w3 w0..w3 */
    p = _mm256_permute4x64_epi64(p, 0x08);              /* low 128 = lanes 0..7 */
    return _mm256_cvtph_ps(_mm256_castsi256_si128(p));
}

/* Gather index [0, rb, ..., 7rb] for the 8-row tiles; 0 if rb overflows the
 * 32-bit gather offsets (the caller then falls back to single-row dots). */
AINLINE int row_index8(size_t rb, __m256i *idx) {
    if (rb > (size_t)INT_MAX / 8) return 0;
    int r = (int)rb;
    *idx = _mm256_setr_epi32(0, r, 2 * r, 3 * r, 4 * r, 5 * r, 6 * r, 7 * r);
    return 1;
}
/* The 32-bit word at byte offset `off` of the current block in each of the 8 rows. */
AINLINE __m256i gather8(const uint8_t *p, __m256i idx) {
    return _mm256_i32gather_epi32((const int *)(const void *)p, idx, 1);
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

/* Signed x signed 32-byte dot via the sign trick (F-quant-11, P-45):
 * |w| is the unsigned operand, x takes w's sign. x must be in [-127, 127]
 * (our activation quantisers guarantee it); w may be -128. Worst pair sum
 * 2*128*127 = 32512 fits int16. */
AINLINE __m256i dot32_s8(__m256i w, __m256i x) {
    __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w));
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
}

/* Float update for 8 consecutive blocks of one row: t = d_w * d_a (vector
 * multiply, lane-wise identical to scalar), f = (float)isum (RNE), then the
 * scalar fmaf chain in block order. */
AINLINE float flush8(float acc, __m256i isum, const float *dw, const float *dx) {
    float t[8], f[8];
    _mm256_storeu_ps(t, _mm256_mul_ps(_mm256_loadu_ps(dw), _mm256_loadu_ps(dx)));
    _mm256_storeu_ps(f, _mm256_cvtepi32_ps(isum));
    for (int j = 0; j < 8; j++) acc = fmaf(t[j], f[j], acc);
    return acc;
}

/* ======================================================================
 * Q8_0 x Q8_0-activation. Block: f16 d, int8 q[32] (34 bytes).
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
        for (int j = 0; j < 8; j++) {
            const uint8_t *blk = w + 34 * (b + j);
            v[j] = dot32_s8(ld256(blk + 2), ld256(a[b + j].q));
            dw[j] = ld_f16(blk);
            dx[j] = a[b + j].d;
        }
        acc = flush8(acc, hsum8x8(v), dw, dx);
    }
    for (; b < nb; b++) {
        const uint8_t *blk = w + 34 * b;
        int32_t s = hsum_i32(dot32_s8(ld256(blk + 2), ld256(a[b].q)));
        acc = fmaf(ld_f16(blk) * a[b].d, (float)s, acc);
    }
    return acc;
}

static void mv_q8_0(const void *W, size_t rb, const void *av, float *y, int64_t n, int64_t nrows) {
    const uint8_t *base = W;
    const tlm_blk_a8 *a = av;
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
                __m256i v[8];
                for (int j = 0; j < 8; j++) {
                    pf_row(bp + rb * j, 512, 1);   /* 64 B per 34 B block */
                    v[j] = dot32_s8(ld256(bp + rb * j + 2), xq);
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
 * MXFP4 x Q8_0-activation (F-quant-15 AVX2 integer form). Block: E8M0
 * byte, 16 nibble bytes; low nibbles are elements 0..15, high nibbles
 * 16..31 (our "split halves" order, F-quant-14). One byte shuffle maps the
 * 4-bit code to the doubled signed E2M1 integer (|w| <= 12, pair sums
 * <= 2*12*127, no saturation); the 0.5 is folded into the scale.
 * ==================================================================== */
AINLINE __m256i mxfp4_w32(const uint8_t *qs, __m256i lut) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m128i q = ld128(qs);
    __m128i lo = _mm_and_si128(q, m4), hi = _mm_and_si128(_mm_srli_epi16(q, 4), m4);
    return _mm256_shuffle_epi8(lut, _mm256_set_m128i(hi, lo));
}
#define MXFP4_LUT _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12, \
                                   0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

static float dot_mxfp4(const void *wv, const void *av, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const tlm_blk_a8 *a = av;
    const __m256i lut = MXFP4_LUT;
    const int64_t nb = n / 32;
    int64_t b = 0;
    float acc = 0.0f;
    for (; b + 8 <= nb; b += 8) {
        __m256i v[8];
        float dw[8], dx[8];
        for (int j = 0; j < 8; j++) {
            const uint8_t *blk = w + 17 * (b + j);
            v[j] = dot32_s8(mxfp4_w32(blk + 1, lut), ld256(a[b + j].q));
            dw[j] = e8m0_half(blk[0]);
            dx[j] = a[b + j].d;
        }
        acc = flush8(acc, hsum8x8(v), dw, dx);
    }
    for (; b < nb; b++) {
        const uint8_t *blk = w + 17 * b;
        int32_t s = hsum_i32(dot32_s8(mxfp4_w32(blk + 1, lut), ld256(a[b].q)));
        acc = fmaf(e8m0_half(blk[0]) * a[b].d, (float)s, acc);
    }
    return acc;
}

static void mv_mxfp4(const void *W, size_t rb, const void *av, float *y, int64_t n, int64_t nrows) {
    const uint8_t *base = W;
    const tlm_blk_a8 *a = av;
    const __m256i lut = MXFP4_LUT, mff = _mm256_set1_epi32(0xFF);
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
                __m256i v[8];
                for (int j = 0; j < 8; j++) {
                    if (!(b & 1)) pf_row(bp + rb * j, 512, 1);   /* 64 B per two 17 B blocks */
                    v[j] = dot32_s8(mxfp4_w32(bp + rb * j + 1, lut), xq);
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
 * Q4_K / Q5_K x Q8_K (F-quant-03, -04, -08). Block: f16 d, f16 dmin,
 * 12 bytes of 6-bit scales/mins, [Q5_K: 32-byte 5th-bit plane], 128 nibble
 * bytes. Per 64-element chunk c the 32 quant bytes give, after AND 0x0F,
 * elements 64c..64c+31 (sub-block 2c) and, after >>4, 64c+32..64c+63
 * (sub-block 2c+1). Q5_K: bit 2c / 2c+1 of qh[l] becomes bit 4 (qh is
 * shifted right by 2 per chunk; the bits a 16-bit shift drags across a byte
 * boundary only ever land in bits that are not read later).
 * Saturation: maddubs pair sum <= 2*31*127 = 7874; madd with the 6-bit
 * scale stays far inside int32. Mins (P-44): sum_s mn[s]*(S[2s]+S[2s+1])
 * with one vpmaddwd against [mn0,mn0,mn1,mn1,...].
 * ==================================================================== */
AINLINE void kq_core(const uint8_t *blk, const tlm_blk_q8k *x, const int q5, __m256i *iv, __m256i *mvv) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), b16 = _mm256_set1_epi8(16);
    const __m128i dup = _mm_setr_epi8(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);
    const uint8_t *qs = q5 ? blk + 48 : blk + 16;
    const __m128i sm = kq_scales_v(blk + 4);
    const __m256i s16 = _mm256_cvtepu8_epi16(sm);                  /* sc0..7 | mn0..7 as int16 */
    const __m256i sc = _mm256_permute2x128_si256(s16, s16, 0x00);  /* sc0..7 in both lanes */
    __m256i H = q5 ? ld256(blk + 16) : _mm256_setzero_si256();
    __m256i acc = _mm256_setzero_si256();
    for (int c = 0; c < 4; c++) {
        __m256i q = ld256(qs + 32 * c);
        __m256i lo = _mm256_and_si256(q, m4);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
        if (q5) {
            lo = _mm256_or_si256(lo, _mm256_and_si256(_mm256_slli_epi16(H, 4), b16));
            hi = _mm256_or_si256(hi, _mm256_and_si256(_mm256_slli_epi16(H, 3), b16));
            H = _mm256_srli_epi16(H, 2);
        }
        /* broadcast int16 scale k: byte pair (2k, 2k+1) of each 128-bit lane */
        const __m256i bA = _mm256_set1_epi16((short)(0x0100 * (4 * c + 1) + 4 * c));
        const __m256i bB = _mm256_set1_epi16((short)(0x0100 * (4 * c + 3) + 4 * c + 2));
        __m256i p0 = _mm256_maddubs_epi16(lo, ld256(x->q + 64 * c));
        __m256i p1 = _mm256_maddubs_epi16(hi, ld256(x->q + 64 * c + 32));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p0, _mm256_shuffle_epi8(sc, bA)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p1, _mm256_shuffle_epi8(sc, bB)));
    }
    *iv = acc;
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
        int32_t isum, msum;
        hsum2_i32(iv, mvv, &isum, &msum);
        float dA = ld_f16(blk) * a[i].d, dM = ld_f16(blk + 2) * a[i].d;
        acc = fmaf(dA, (float)isum, acc);
        acc = fmaf(-dM, (float)msum, acc);
    }
    return acc;
}

/* 8-row tile; acc lane r: acc = fma(dA_r, isum_r, acc), then
 * acc = fma(-dM_r, msum_r, acc) (vfnmadd computes -(a*b)+c, which is the
 * same correctly rounded value as fma(-a, b, c)). */
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
 * Q6_K x Q8_K (F-quant-05 unpack, -32 folded via group sums, P-44).
 * Block: ql[128] nibble plane, qh[64] 2-bit plane, int8 scales[16], f16 d.
 * Per 128-element half: A = ql[0..31], B = ql[32..63], C = qh[0..31]:
 *   u0 = (A & 15) | (C & 0x03) << 4   elements  0..31
 *   u1 = (B & 15) | (C & 0x0C) << 2   elements 32..63
 *   u2 = (A >> 4) | (C & 0x30)        elements 64..95
 *   u3 = (B >> 4) | (C & 0xC0) >> 2   elements 96..127
 * Each 32-byte u covers two 16-element groups: the low 128-bit lane is the
 * first group, the high lane the second, so the madd scale vector holds
 * scale(g) in the low lane and scale(g+1) in the high lane. Saturation:
 * maddubs pair <= 2*63*127 = 16002.
 * ==================================================================== */
/* One 128-element half: ql/qh point at the half's planes, xq at its 128
 * activations, s holds the half's 8 int16 scales in both 128-bit lanes. */
AINLINE __m256i q6_half(const uint8_t *ql, const uint8_t *qh, const int8_t *xq, __m256i s) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), m03 = _mm256_set1_epi8(0x03);
    const __m256i m0C = _mm256_set1_epi8(0x0C), m30 = _mm256_set1_epi8(0x30);
    const __m256i mC0 = _mm256_set1_epi8((char)0xC0);
    /* byte-shuffle controls broadcasting int16 scale 2k (low lane) / 2k+1 (high lane) */
    const __m256i ctl0 = _mm256_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                                          2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3);
    const __m256i ctl1 = _mm256_setr_epi8(4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
                                          6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7);
    const __m256i ctl2 = _mm256_setr_epi8(8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
                                          10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11);
    const __m256i ctl3 = _mm256_setr_epi8(12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
                                          14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15);
    const __m256i A = ld256(ql), B = ld256(ql + 32), C = ld256(qh);
    const __m256i u0 = _mm256_or_si256(_mm256_and_si256(A, m4), _mm256_slli_epi16(_mm256_and_si256(C, m03), 4));
    const __m256i u1 = _mm256_or_si256(_mm256_and_si256(B, m4), _mm256_slli_epi16(_mm256_and_si256(C, m0C), 2));
    const __m256i u2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(A, 4), m4), _mm256_and_si256(C, m30));
    const __m256i u3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(B, 4), m4),
                                       _mm256_srli_epi16(_mm256_and_si256(C, mC0), 2));
    __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(u0, ld256(xq)), _mm256_shuffle_epi8(s, ctl0));
    __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(u1, ld256(xq + 32)), _mm256_shuffle_epi8(s, ctl1));
    __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(u2, ld256(xq + 64)), _mm256_shuffle_epi8(s, ctl2));
    __m256i p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(u3, ld256(xq + 96)), _mm256_shuffle_epi8(s, ctl3));
    return _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3));
}

AINLINE __m256i q6_core(const uint8_t *ql, const tlm_blk_q8k *x) {
    const __m256i sc16 = _mm256_cvtepi8_epi16(ld128(ql + 192));
    __m256i acc = q6_half(ql, ql + 128, x->q, _mm256_permute2x128_si256(sc16, sc16, 0x00));
    acc = _mm256_add_epi32(acc, q6_half(ql + 64, ql + 160, x->q + 128,
                                        _mm256_permute2x128_si256(sc16, sc16, 0x11)));
    const __m256i off = _mm256_madd_epi16(ld256(x->bsums), sc16);   /* sc[g] * S[g], pairs */
    return _mm256_sub_epi32(acc, _mm256_slli_epi32(off, 5));
}

static float dot_q6_k(const void *wv, const void *av, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const tlm_blk_q8k *a = av;
    const int64_t nb = n / 256;
    float acc = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *ql = w + 210 * i;
        int32_t isum = hsum_i32(q6_core(ql, a + i));
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
                /* d sits in bytes 208..209: gather the word at 206 (in bounds) and take its high half */
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
 * Q8_K weights x Q8_K activations: f32 d, int8 q[256] (bsums unused on the
 * weight side). Sign trick per 32 bytes.
 * ==================================================================== */
AINLINE __m256i q8k_core(const uint8_t *blk, const tlm_blk_q8k *x) {
    __m256i acc = dot32_s8(ld256(blk + 4), ld256(x->q));
    for (int c = 1; c < 8; c++)
        acc = _mm256_add_epi32(acc, dot32_s8(ld256(blk + 4 + 32 * c), ld256(x->q + 32 * c)));
    return acc;
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
        acc = fmaf(dw * a[i].d, (float)hsum_i32(q8k_core(blk, a + i)), acc);
    }
    return acc;
}

static void mv_q8_k(const void *W, size_t rb, const void *av, float *y, int64_t n, int64_t nrows) {
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
                const uint8_t *bp = w0 + 292 * i;
                __m256i v[8];
                NO_UNROLL
                for (int j = 0; j < 8; j++) {
                    pf_row(bp + rb * (size_t)j, 2 * 292, 5);
                    v[j] = q8k_core(bp + rb * (size_t)j, a + i);
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
 * float weights x f32 activations
 * ==================================================================== */
enum { FT_F32, FT_F16, FT_BF16, FT_FP8 };

AINLINE __m256 ld8(const int ft, const uint8_t *w, int64_t i) {
    switch (ft) {
    case FT_F32: return _mm256_loadu_ps((const float *)(const void *)(w + 4 * i));
    case FT_F16: return _mm256_cvtph_ps(ld128(w + 2 * i));
    case FT_BF16: return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(ld128(w + 2 * i)), 16));
    default: {   /* FP8 E4M3: shift fields into f32 position, rebias by 2^120 (F-quant-12) */
        __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(w + i)));
        __m256i m7f = _mm256_set1_epi32(0x7F);
        __m256i lo7 = _mm256_and_si256(b, m7f);
        __m256i bits = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x80)), 24),
                                       _mm256_slli_epi32(lo7, 20));
        __m256 v = _mm256_mul_ps(_mm256_castsi256_ps(bits), _mm256_set1_ps(0x1p120f));
        __m256 nanm = _mm256_castsi256_ps(_mm256_cmpeq_epi32(lo7, m7f));
        return _mm256_blendv_ps(v, _mm256_set1_ps(u2f(0x7FC00000u)), nanm);
    }
    }
}

AINLINE float ld1(const int ft, const uint8_t *w, int64_t i) {
    switch (ft) {
    case FT_F32: { float f; memcpy(&f, w + 4 * i, 4); return f; }
    case FT_F16: return ld_f16(w + 2 * i);
    case FT_BF16: { uint16_t h; memcpy(&h, w + 2 * i, 2); return u2f((uint32_t)h << 16); }
    default: {
        uint8_t v = w[i];
        if ((v & 0x7Fu) == 0x7Fu) return u2f(0x7FC00000u);
        return u2f(((uint32_t)(v & 0x80u) << 24) | ((uint32_t)(v & 0x7Fu) << 20)) * 0x1p120f;
    }
    }
}

/* Lanes over elements [i0, i1); i0 must be a multiple of 32. */
AINLINE float lanes_dot(const int ft, const uint8_t *w, const float *x, int64_t i0, int64_t i1) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    int64_t i = i0;
    for (; i + 32 <= i1; i += 32) {
        a0 = _mm256_fmadd_ps(ld8(ft, w, i), _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(ld8(ft, w, i + 8), _mm256_loadu_ps(x + i + 8), a1);
        a2 = _mm256_fmadd_ps(ld8(ft, w, i + 16), _mm256_loadu_ps(x + i + 16), a2);
        a3 = _mm256_fmadd_ps(ld8(ft, w, i + 24), _mm256_loadu_ps(x + i + 24), a3);
    }
    float r[32];
    _mm256_storeu_ps(r, a0);
    _mm256_storeu_ps(r + 8, a1);
    _mm256_storeu_ps(r + 16, a2);
    _mm256_storeu_ps(r + 24, a3);
    for (; i < i1; i++) r[i & 31] = fmaf(ld1(ft, w, i), x[i], r[i & 31]);
    return reduce32(r);
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
/* FP8: float lanes restart per 128-column block; partial * scale in double. */
static float dot_fp8(const void *w, const void *x, int64_t n, const float *cs) {
    double acc = 0.0;
    for (int64_t b0 = 0, cb = 0; b0 < n; b0 += 128, cb++) {
        float part = lanes_dot(FT_FP8, w, x, b0, b0 + 128 < n ? b0 + 128 : n);
        acc += (double)part * (double)(cs ? cs[cb] : 1.0f);
    }
    return (float)acc;
}

int tlm_kern_avx2_fill(tlm_quant_kernels *k) {
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

#else  /* not built for AVX2 */

int tlm_kern_avx2_fill(tlm_quant_kernels *k) {
    (void)k;
    return 0;
}

#endif
