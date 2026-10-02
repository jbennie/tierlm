/* tlm_quant.c - block formats, conversions, reference quantisers, scalar
 * reference kernels and runtime kernel dispatch.
 *
 * Patterns: P-40 GGUF block layouts, P-42 quantise activations once, P-43
 * SIMD == scalar bit for bit, P-44 offsets/mins folded through activation
 * group sums, P-47 FP8 E4M3 with 128x128 scales, P-48 MXFP4 (E2M1 + E8M0),
 * P-55 runtime dispatch with a start-up self-test, P-56 explicit fma and
 * fixed reduction trees. Written from fragments/quant_kernels.md
 * (F-quant-01..07, 12..14, 21, 22) and the public format definitions.
 *
 * This file is portable C11 (compiled without ISA flags). The SIMD tiers
 * live in tlm_kern_avx2.c / tlm_kern_avx512.c and are selected at run time.
 *
 * Numerics, shared by every tier (see the contract in tlm_quant.h):
 *   integer formats: exact int32 sums inside a block, then one explicit
 *     fmaf per block (two for Q4_K/Q5_K: scale term, then min term), blocks
 *     in ascending order;
 *   float formats: element i -> lane i mod 32, lanes updated with fmaf in
 *     element order, reduced with reduce32() below.
 */
#define TLM_QUANT_INTERNAL 1
#include "tlm_quant.h"
#include "tlm_platform.h"

#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <strings.h>

#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
#error "tlm_quant requires FLT_EVAL_METHOD == 0 for its bit-exact numerics contract"
#endif
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "tlm_quant assumes a little-endian host (block fields are little-endian)"
#endif

_Static_assert(sizeof(tlm_blk_q8k) == 292, "Q8_K activation block must be 292 bytes");
_Static_assert(sizeof(tlm_blk_a8) == TLM_QACT_Q8_0_BLOCK_BYTES, "Q8_0 activation block must be 36 bytes");

/* ======================================================================
 * dtype table
 * ==================================================================== */
typedef struct {
    tlm_dtype   t;
    const char *name;
    int         block;
    size_t      bytes;
} dt_info;

static const dt_info k_dt[] = {
    { TLM_DT_F32,          "f32",           1,   4 },
    { TLM_DT_F16,          "f16",           1,   2 },
    { TLM_DT_BF16,         "bf16",          1,   2 },
    { TLM_DT_Q8_0,         "q8_0",          32,  34 },
    { TLM_DT_Q4_K,         "q4_k",          256, 144 },
    { TLM_DT_Q5_K,         "q5_k",          256, 176 },
    { TLM_DT_Q6_K,         "q6_k",          256, 210 },
    { TLM_DT_Q8_K,         "q8_k",          256, 292 },
    { TLM_DT_MXFP4,        "mxfp4",         32,  17 },
    { TLM_DT_FP8E4M3_B128, "fp8_e4m3_b128", 1,   1 },
};
#define N_DT ((int)(sizeof k_dt / sizeof k_dt[0]))

static const dt_info *dt_find(tlm_dtype t) {
    for (int i = 0; i < N_DT; i++)
        if (k_dt[i].t == t) return &k_dt[i];
    return NULL;
}

const char *tlm_dtype_name(tlm_dtype t) {
    const dt_info *d = dt_find(t);
    return d ? d->name : "unknown";
}

int tlm_dtype_from_name(const char *name, tlm_dtype *out) {
    static const struct { const char *alias; tlm_dtype t; } aliases[] = {
        { "fp32", TLM_DT_F32 }, { "float32", TLM_DT_F32 }, { "float", TLM_DT_F32 },
        { "fp16", TLM_DT_F16 }, { "float16", TLM_DT_F16 }, { "half", TLM_DT_F16 },
        { "bfloat16", TLM_DT_BF16 },
        { "fp8", TLM_DT_FP8E4M3_B128 }, { "e4m3", TLM_DT_FP8E4M3_B128 },
        { "fp8_e4m3", TLM_DT_FP8E4M3_B128 }, { "fp8e4m3", TLM_DT_FP8E4M3_B128 },
        { "f8_e4m3", TLM_DT_FP8E4M3_B128 }, { "fp8e4m3_b128", TLM_DT_FP8E4M3_B128 },
        { "float8_e4m3fn", TLM_DT_FP8E4M3_B128 }, { "mxfp4_e2m1", TLM_DT_MXFP4 },
    };
    if (!name) return tlm_seterr(TLM_EINVAL, "dtype name is NULL");
    for (int i = 0; i < N_DT; i++)
        if (strcasecmp(name, k_dt[i].name) == 0) { if (out) *out = k_dt[i].t; return TLM_OK; }
    for (size_t i = 0; i < sizeof aliases / sizeof aliases[0]; i++)
        if (strcasecmp(name, aliases[i].alias) == 0) { if (out) *out = aliases[i].t; return TLM_OK; }
    return tlm_seterr(TLM_EINVAL, "unknown dtype '%s'", name);
}

int tlm_dtype_block(tlm_dtype t) {
    const dt_info *d = dt_find(t);
    return d ? d->block : 0;
}

size_t tlm_dtype_block_bytes(tlm_dtype t) {
    const dt_info *d = dt_find(t);
    return d ? d->bytes : 0;
}

/* 0 when the dtype is unknown or n is negative / not a multiple of the block. */
size_t tlm_row_bytes(tlm_dtype t, int64_t n) {
    const dt_info *d = dt_find(t);
    if (!d || n < 0 || n % d->block) return 0;
    return (size_t)(n / d->block) * d->bytes;
}

/* ======================================================================
 * scalar conversions (exact; RNE on the way in)
 * ==================================================================== */
static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float    u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

float tlm_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0x1F)  /* inf / NaN (NaN made quiet, payload kept) */
        return u2f(sign | 0x7F800000u | (m << 13) | (m ? 0x00400000u : 0));
    if (e == 0) {   /* zero / subnormal: m * 2^-24, exact in f32 */
        float v = (float)m * 0x1p-24f;
        return u2f(f2u(v) | sign);
    }
    return u2f(sign | ((e + 112u) << 23) | (m << 13));
}

uint16_t tlm_f32_to_f16(float f) {
    uint32_t x = f2u(f);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t ax = x & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) {                    /* inf / NaN */
        if (ax == 0x7F800000u) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | 0x7E00u | ((ax >> 13) & 0x3FFu));
    }
    if (ax >= 0x477FF000u) return (uint16_t)(sign | 0x7C00u);   /* >= 65520 rounds to inf */
    if (ax < 0x38800000u) {                     /* below 2^-14: f16 subnormal or zero */
        if (ax <= 0x33000000u) return (uint16_t)sign;           /* <= 2^-25 ties to 0 */
        uint32_t e = ax >> 23, mant = (ax & 0x7FFFFFu) | 0x800000u;
        uint32_t shift = 126u - e;                               /* 14..24 */
        uint32_t q = mant >> shift, rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (q & 1u))) q++;
        return (uint16_t)(sign | q);
    }
    uint32_t r = ax - 0x38000000u;              /* rebias exponent 127 -> 15 */
    uint32_t q = r >> 13, rem = r & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (q & 1u))) q++;
    return (uint16_t)(sign | q);
}

float tlm_bf16_to_f32(uint16_t h) { return u2f((uint32_t)h << 16); }

uint16_t tlm_f32_to_bf16(float f) {
    uint32_t x = f2u(f);
    if ((x & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((x >> 16) | 0x0040u);  /* quiet NaN */
    x += 0x7FFFu + ((x >> 16) & 1u);
    return (uint16_t)(x >> 16);
}

/* E4M3 (FN variant): 1 sign, 4 exponent (bias 7), 3 mantissa, no inf,
 * 0x7F/0xFF are NaN, max finite 448. Built from integer fields so it is
 * exact regardless of FTZ/DAZ. */
float tlm_e4m3_to_f32(uint8_t v) {
    uint32_t sign = (uint32_t)(v & 0x80u) << 24;
    uint32_t e = (v >> 3) & 0xFu, m = v & 7u;
    if ((v & 0x7Fu) == 0x7Fu) return u2f(sign | 0x7FC00000u);
    if (e == 0) return u2f(f2u((float)m * 0x1p-9f) | sign);
    return u2f(sign | ((e + 120u) << 23) | (m << 20));
}

uint8_t tlm_f32_to_e4m3(float f) {
    uint32_t x = f2u(f);
    uint8_t sign = (uint8_t)((x >> 24) & 0x80u);
    float a = fabsf(f);
    if (a != a) return (uint8_t)(sign | 0x7Fu);
    if (a > 448.0f) return (uint8_t)(sign | 0x7Eu);          /* saturate (incl. inf) */
    if (a < 0x1p-6f) {                                         /* subnormal range: m * 2^-9 */
        float s = a * 512.0f;                                  /* exact, < 8 */
        uint32_t q = (uint32_t)s;
        float rem = s - (float)q;
        if (rem > 0.5f || (rem == 0.5f && (q & 1u))) q++;      /* q == 8 is the min normal 0x08 */
        return (uint8_t)(sign | q);
    }
    int ex;
    float fr = frexpf(a, &ex);                                 /* a = fr * 2^ex, fr in [0.5, 1) */
    int e = ex - 1;                                            /* a = (2fr) * 2^e, 2fr in [1, 2) */
    float s = (2.0f * fr - 1.0f) * 8.0f;                       /* exact, in [0, 8) */
    uint32_t q = (uint32_t)s;
    float rem = s - (float)q;
    if (rem > 0.5f || (rem == 0.5f && (q & 1u))) q++;
    if (q == 8) { q = 0; e++; }
    uint32_t code = ((uint32_t)(e + 7) << 3) | q;
    if (code > 0x7Eu) code = 0x7Eu;
    return (uint8_t)(sign | code);
}

/* E8M0: 2^(e-127); 255 = NaN (OCP MX); 0 = 2^-127 (an f32 subnormal). */
float tlm_e8m0_to_f32(uint8_t e) {
    if (e == 255) return u2f(0x7FC00000u);
    if (e == 0) return u2f(0x00400000u);
    return u2f((uint32_t)e << 23);
}

/* 2^(e-128): the E8M0 scale with E2M1's 0.5 folded in, so the doubled
 * integer codes can be used inside the integer dot (F-quant-14/15). Every
 * tier computes it with the same bit construction. */
static inline float e8m0_half(uint8_t e) {
    uint32_t b;
    if (e == 255) b = 0x7FC00000u;
    else if (e >= 2) b = (uint32_t)(e - 1) << 23;
    else b = e ? 0x00400000u : 0x00200000u;
    return u2f(b);
}

/* E4M3 decode used inside the FP8 kernels: the same shift-and-rebias trick
 * the SIMD tiers use (F-quant-12), so every tier decodes identically. */
static inline float e4m3_kern(uint8_t v) {
    if ((v & 0x7Fu) == 0x7Fu) return u2f(0x7FC00000u);
    uint32_t b = ((uint32_t)(v & 0x80u) << 24) | ((uint32_t)(v & 0x7Fu) << 20);
    return u2f(b) * 0x1p120f;
}

static inline float ld_f16(const uint8_t *p) { uint16_t h; memcpy(&h, p, 2); return tlm_f16_to_f32(h); }
static inline void  st_f16(uint8_t *p, float f) { uint16_t h = tlm_f32_to_f16(f); memcpy(p, &h, 2); }

/* ======================================================================
 * rounding helpers for the quantisers
 * ==================================================================== */
/* Round to nearest, ties to even, valid for |v| < 2^22. Requires no
 * -ffast-math (the Makefile never uses it). */
static inline float rne_small(float v) {
    const float magic = 12582912.0f;  /* 1.5 * 2^23 */
    return (v + magic) - magic;
}
/* RNE then clamp to [lo, hi]; NaN -> 0. */
static inline int qround(float v, int lo, int hi) {
    if (v != v) return 0;
    if (v <= (float)lo) return lo;
    if (v >= (float)hi) return hi;
    return (int)rne_small(v);
}

/* ======================================================================
 * K-quant 6-bit scale/min packing (F-quant-02)
 *
 * 12 bytes S: bytes 0..3 carry scales 0..3 in bits 0..5 and the top two
 * bits of scales 4..7 in bits 6..7; bytes 4..7 do the same for the mins;
 * bytes 8..11 carry the low nibble of scale 4+i (bits 0..3) and of min 4+i
 * (bits 4..7). Unpacked word-wise into out[0..7] = scales, out[8..15] = mins.
 * ==================================================================== */
static inline void kq_unpack_scales(const uint8_t *s12, uint8_t out[16]) {
    uint32_t a0, a1, a2;
    memcpy(&a0, s12, 4); memcpy(&a1, s12 + 4, 4); memcpy(&a2, s12 + 8, 4);
    uint32_t w0 = a0 & 0x3F3F3F3Fu;
    uint32_t w1 = (a2 & 0x0F0F0F0Fu) | (((a0 >> 6) & 0x03030303u) << 4);
    uint32_t w2 = a1 & 0x3F3F3F3Fu;
    uint32_t w3 = ((a2 >> 4) & 0x0F0F0F0Fu) | (((a1 >> 6) & 0x03030303u) << 4);
    memcpy(out, &w0, 4); memcpy(out + 4, &w1, 4); memcpy(out + 8, &w2, 4); memcpy(out + 12, &w3, 4);
}

static void kq_pack_scales(const uint8_t sc[8], const uint8_t mn[8], uint8_t *s12) {
    for (int j = 0; j < 4; j++) {
        s12[j]     = (uint8_t)((sc[j] & 0x3F) | ((sc[j + 4] >> 4) << 6));
        s12[j + 4] = (uint8_t)((mn[j] & 0x3F) | ((mn[j + 4] >> 4) << 6));
        s12[j + 8] = (uint8_t)((sc[j + 4] & 0x0F) | ((mn[j + 4] & 0x0F) << 4));
    }
}

/* ======================================================================
 * reference quantisers (deterministic: fixed loops, fixed op order)
 * ==================================================================== */

/* Q8_0 weights: per 32, d = amax/127 stored as f16, q = rne(x/d). */
static void quant_q8_0(const float *x, uint8_t *dst, int64_t nb) {
    for (int64_t b = 0; b < nb; b++, x += 32, dst += 34) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
        float d = amax / 127.0f;
        if (d > 65504.0f) d = 65504.0f;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        st_f16(dst, d);
        int8_t *q = (int8_t *)(dst + 2);
        for (int i = 0; i < 32; i++) q[i] = (int8_t)qround(x[i] * id, -127, 127);
    }
}

/* Fit x[0..n) ~ scale * L - mn with L in [0, nmax] and mn >= 0 (the GGML
 * K-quant convention: the offset is a non-positive minimum). Search a grid
 * of 21 inverse scales around nmax / (max - min); for each candidate code
 * vector solve the 2-parameter least-squares fit and keep the lowest error. */
static float fit_scale_min(const float *x, int n, int nmax, float *mn_out) {
    float lo = x[0], hi = x[0];
    for (int i = 1; i < n; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
    if (lo > 0.0f) lo = 0.0f;
    if (!(hi > lo)) { *mn_out = -lo; return 0.0f; }
    float range = hi - lo;
    float sx = 0.0f;
    for (int i = 0; i < n; i++) sx += x[i];

    float best_a = (float)nmax > 0 ? range / (float)nmax : 0.0f, best_b = lo, best_err;
    {
        float is = (float)nmax / range;
        best_err = 0.0f;
        for (int i = 0; i < n; i++) {
            int l = qround(is * (x[i] - lo), 0, nmax);
            float e = best_a * (float)l + best_b - x[i];
            best_err += e * e;
        }
    }
    for (int k = 0; k <= 20; k++) {
        float is = ((float)nmax - 1.0f + 0.1f * (float)k) / range;
        float sl = 0.0f, sl2 = 0.0f, slx = 0.0f;
        uint8_t L[32];
        for (int i = 0; i < n; i++) {
            int l = qround(is * (x[i] - lo), 0, nmax);
            L[i] = (uint8_t)l;
            sl += (float)l; sl2 += (float)(l * l); slx += (float)l * x[i];
        }
        float D = (float)n * sl2 - sl * sl;
        if (!(D > 0.0f)) continue;
        float a = ((float)n * slx - sl * sx) / D;
        float b = (sl2 * sx - sl * slx) / D;
        if (b > 0.0f) { b = 0.0f; a = slx / sl2; }
        if (!(a > 0.0f)) continue;
        float err = 0.0f;
        for (int i = 0; i < n; i++) { float e = a * (float)L[i] + b - x[i]; err += e * e; }
        if (err < best_err) { best_err = err; best_a = a; best_b = b; }
    }
    *mn_out = -best_b;
    return best_a;
}

/* Q4_K (nmax 15) and Q5_K (nmax 31): 8 sub-blocks of 32, each with a 6-bit
 * scale and a 6-bit min under f16 super-scales d and dmin (F-quant-02..04).
 * Element mapping per 64-element chunk c: elements 64c..64c+31 are the low
 * nibbles of qs[32c..32c+31], elements 64c+32..64c+63 the high nibbles;
 * Q5_K adds the 5th bit of element 32k+l as bit k of qh[l]. */
static void quant_kq_affine(const float *x, uint8_t *blk, int nmax) {
    float scale[8], minv[8];
    for (int j = 0; j < 8; j++) scale[j] = fit_scale_min(x + 32 * j, 32, nmax, &minv[j]);
    float max_s = 0.0f, max_m = 0.0f;
    for (int j = 0; j < 8; j++) { if (scale[j] > max_s) max_s = scale[j]; if (minv[j] > max_m) max_m = minv[j]; }
    float inv_s = max_s > 0.0f ? 63.0f / max_s : 0.0f;
    float inv_m = max_m > 0.0f ? 63.0f / max_m : 0.0f;
    uint8_t sc[8], mn[8];
    for (int j = 0; j < 8; j++) {
        sc[j] = (uint8_t)qround(inv_s * scale[j], 0, 63);
        mn[j] = (uint8_t)qround(inv_m * minv[j], 0, 63);
    }
    float dsup = max_s / 63.0f, msup = max_m / 63.0f;
    if (dsup > 65504.0f) dsup = 65504.0f;
    if (msup > 65504.0f) msup = 65504.0f;
    st_f16(blk, dsup);
    st_f16(blk + 2, msup);
    kq_pack_scales(sc, mn, blk + 4);
    float d = ld_f16(blk), dmin = ld_f16(blk + 2);

    uint8_t L[256];
    for (int j = 0; j < 8; j++) {
        float dl = d * (float)sc[j], ml = dmin * (float)mn[j];
        for (int i = 0; i < 32; i++)
            L[32 * j + i] = (uint8_t)(dl > 0.0f ? qround((x[32 * j + i] + ml) / dl, 0, nmax) : 0);
    }
    if (nmax == 15) {
        uint8_t *qs = blk + 16;
        for (int c = 0; c < 4; c++)
            for (int l = 0; l < 32; l++)
                qs[32 * c + l] = (uint8_t)(L[64 * c + l] | (L[64 * c + 32 + l] << 4));
    } else {
        uint8_t *qh = blk + 16, *qs = blk + 48;
        memset(qh, 0, 32);
        for (int c = 0; c < 4; c++)
            for (int l = 0; l < 32; l++) {
                uint8_t a = L[64 * c + l], b = L[64 * c + 32 + l];
                qs[32 * c + l] = (uint8_t)((a & 0x0F) | ((b & 0x0F) << 4));
                qh[l] |= (uint8_t)(((a >> 4) << (2 * c)) | ((b >> 4) << (2 * c + 1)));
            }
    }
}

/* Symmetric fit for a Q6_K group of 16: codes in [-32, 31] with a signed
 * scale (the extreme value maps to -32, using the full negative range).
 * Tries 19 inverse scales and keeps the one maximising (sum xL)^2 / sum L^2,
 * i.e. the least-squares optimum for its code vector. */
static float fit_symmetric(const float *x, int n) {
    float amax = 0.0f, mx = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) { amax = a; mx = x[i]; } }
    if (!(amax > 1e-30f)) return 0.0f;
    float best_score = -1.0f, best_scale = 0.0f;
    for (int k = -9; k <= 9; k++) {
        float is = -(32.0f + 0.1f * (float)k) / mx;
        float slx = 0.0f, sl2 = 0.0f;
        for (int i = 0; i < n; i++) {
            int l = qround(is * x[i], -32, 31);
            slx += x[i] * (float)l; sl2 += (float)(l * l);
        }
        if (!(sl2 > 0.0f)) continue;
        float score = slx * slx / sl2;
        if (score > best_score) { best_score = score; best_scale = slx / sl2; }
    }
    return best_scale;
}

/* Q6_K: 16 groups of 16 with int8 scales under one f16 d (stored last).
 * Half h (128 elements) uses ql[64h..64h+63] and qh[32h..32h+31]; for
 * l < 32: element 128h+l = low nibble of ql[64h+l] + bits 0..1 of qh,
 * +32 = low nibble of ql[64h+32+l] + bits 2..3, +64 = high nibble of
 * ql[64h+l] + bits 4..5, +96 = high nibble of ql[64h+32+l] + bits 6..7.
 * Stored code u = q + 32 (0..63). (F-quant-05) */
static void quant_q6_k(const float *x, uint8_t *blk) {
    float scale[16];
    float max_abs = 0.0f, max_scale = 0.0f;
    for (int g = 0; g < 16; g++) {
        scale[g] = fit_symmetric(x + 16 * g, 16);
        if (fabsf(scale[g]) > max_abs) { max_abs = fabsf(scale[g]); max_scale = scale[g]; }
    }
    uint8_t *ql = blk, *qh = blk + 128;
    int8_t *sc = (int8_t *)(blk + 192);
    uint8_t U[256];
    if (!(max_abs > 0.0f)) {
        memset(sc, 0, 16);
        st_f16(blk + 208, 0.0f);
        memset(U, 32, sizeof U);
    } else {
        float iscale = -128.0f / max_scale;
        float dsup = 1.0f / iscale;
        if (dsup > 65504.0f) dsup = 65504.0f;
        if (dsup < -65504.0f) dsup = -65504.0f;
        st_f16(blk + 208, dsup);
        float d = ld_f16(blk + 208);
        for (int g = 0; g < 16; g++) sc[g] = (int8_t)qround(iscale * scale[g], -128, 127);
        for (int g = 0; g < 16; g++) {
            float dl = d * (float)sc[g];
            for (int i = 0; i < 16; i++) {
                int q = dl != 0.0f ? qround(x[16 * g + i] / dl, -32, 31) : 0;
                U[16 * g + i] = (uint8_t)(q + 32);
            }
        }
    }
    for (int h = 0; h < 2; h++) {
        const uint8_t *u = U + 128 * h;
        for (int l = 0; l < 32; l++) {
            ql[64 * h + l]      = (uint8_t)((u[l] & 0x0F) | ((u[l + 64] & 0x0F) << 4));
            ql[64 * h + l + 32] = (uint8_t)((u[l + 32] & 0x0F) | ((u[l + 96] & 0x0F) << 4));
            qh[32 * h + l] = (uint8_t)((u[l] >> 4) | ((u[l + 32] >> 4) << 2) |
                                       ((u[l + 64] >> 4) << 4) | ((u[l + 96] >> 4) << 6));
        }
    }
}

/* Q8_K / Q8_0 activation rule (F-quant-01, F-quant-06 positive-d variant):
 * d = amax/127, q = rne(x * 127/amax) clamped to [-127, 127] (never -128,
 * so the sign trick of P-45 is always safe). Codes past nvalid are 0.
 * Non-finite input: d = NaN (any NaN) or +inf (an infinity), codes 0, so
 * the dot product of that block is NaN instead of a silently wrong number. */
static void quant_act_block(const float *x, int nvalid, int qk, float *d_out, int8_t *q) {
    float amax = 0.0f;
    int has_nan = 0, has_inf = 0;
    for (int i = 0; i < nvalid; i++) {
        float a = fabsf(x[i]);
        if (a != a) has_nan = 1;
        else if (a == INFINITY) has_inf = 1;
        else if (a > amax) amax = a;
    }
    memset(q, 0, (size_t)qk);
    if (has_nan || has_inf) { *d_out = has_nan ? NAN : INFINITY; return; }
    if (!(amax > 0.0f)) { *d_out = 0.0f; return; }
    *d_out = amax / 127.0f;
    if (amax >= 0x1p-100f) {
        float id = 127.0f / amax;
        for (int i = 0; i < nvalid; i++) q[i] = (int8_t)qround(x[i] * id, -127, 127);
    } else {   /* 127/amax would overflow float: go through double (rare, tiny blocks) */
        double id = 127.0 / (double)amax;
        for (int i = 0; i < nvalid; i++) q[i] = (int8_t)qround((float)((double)x[i] * id), -127, 127);
    }
}

/* Q8_K blocks (activation layout, also usable as weights). Written through
 * a local block + memcpy so `dst` may sit at any byte offset (weight rows
 * inside packets need not be 4-byte aligned). */
static void quant_q8_k(const float *x, int64_t n, void *dst) {
    int64_t nb = (n + 255) / 256;
    uint8_t *out = dst;
    for (int64_t b = 0; b < nb; b++) {
        tlm_blk_q8k blk;
        int64_t rem = n - b * 256;
        int nv = rem >= 256 ? 256 : (int)rem;
        quant_act_block(x + b * 256, nv, 256, &blk.d, blk.q);
        for (int g = 0; g < 16; g++) {
            int s = 0;
            for (int i = 0; i < 16; i++) s += blk.q[16 * g + i];
            blk.bsums[g] = (int16_t)s;
        }
        memcpy(out + sizeof blk * (size_t)b, &blk, sizeof blk);
    }
}

static void quant_a8(const float *x, int64_t n, tlm_blk_a8 *y) {
    int64_t nb = (n + 31) / 32;
    for (int64_t b = 0; b < nb; b++) {
        int64_t rem = n - b * 32;
        quant_act_block(x + b * 32, rem >= 32 ? 32 : (int)rem, 32, &y[b].d, y[b].q);
    }
}

/* MXFP4 (OCP MX, E2M1 elements + one E8M0 scale per 32). Our internal
 * nibble order is "split halves": byte j holds element j (low nibble) and
 * element j+16 (high nibble) (F-quant-14 layout A). */
static const int8_t k_e2m1_x2[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

/* Nearest E2M1 code for v (already divided by the block scale), ties to
 * the even code, saturating at +-6. Magnitudes: 0 .5 1 1.5 2 3 4 6. */
static uint8_t e2m1_encode(float v) {
    uint8_t s = 0;
    if (v != v) return 0;
    if (v < 0.0f) { s = 8; v = -v; }
    uint8_t c;
    if (v <= 0.25f) c = 0;
    else if (v < 0.75f) c = 1;
    else if (v <= 1.25f) c = 2;
    else if (v < 1.75f) c = 3;
    else if (v <= 2.5f) c = 4;
    else if (v < 3.5f) c = 5;
    else if (v <= 5.0f) c = 6;
    else c = 7;
    return c ? (uint8_t)(s | c) : 0;
}

/* Shared scale: try the OCP choice 2^(floor(log2 amax) - 2) and the next
 * power of two (which never clips) and keep the lower squared error. */
static void quant_mxfp4(const float *x, uint8_t *dst, int64_t nb) {
    for (int64_t b = 0; b < nb; b++, x += 32, dst += 17) {
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
        if (!(amax > 0.0f)) { dst[0] = 127; memset(dst + 1, 0, 16); continue; }
        int k;
        if (amax == INFINITY) k = 128; else (void)frexpf(amax, &k);   /* amax in [2^(k-1), 2^k) */
        int e0 = k + 124;                                    /* floor(log2 amax) - 2 + 127 */
        uint8_t best_codes[32], codes[32];
        int best_e = -1;
        float best_err = 0.0f;
        for (int cand = 0; cand < 2; cand++) {
            int e = e0 + cand;
            if (e < 0) e = 0;
            if (e > 254) e = 254;
            if (cand == 1 && e == best_e) break;
            float inv = ldexpf(1.0f, 127 - e), sc = tlm_e8m0_to_f32((uint8_t)e);
            float err = 0.0f;
            for (int i = 0; i < 32; i++) {
                codes[i] = e2m1_encode(x[i] * inv);
                float r = 0.5f * (float)k_e2m1_x2[codes[i]] * sc - x[i];
                err += r * r;
            }
            if (best_e < 0 || err < best_err) { best_err = err; best_e = e; memcpy(best_codes, codes, 32); }
        }
        dst[0] = (uint8_t)best_e;
        for (int j = 0; j < 16; j++) dst[1 + j] = (uint8_t)(best_codes[j] | (best_codes[j + 16] << 4));
    }
}

int tlm_quantize_row(tlm_dtype t, const float *x, void *dst, int64_t n) {
    const dt_info *d = dt_find(t);
    if (!d) return tlm_seterr(TLM_EINVAL, "quantize_row: unknown dtype %d", (int)t);
    if (n < 0 || n % d->block)
        return tlm_seterr(TLM_EINVAL, "quantize_row: n=%lld is not a multiple of %d for %s",
                          (long long)n, d->block, d->name);
    int64_t nb = n / d->block;
    uint8_t *o = dst;
    switch (t) {
    case TLM_DT_F32:  memcpy(dst, x, (size_t)n * 4); return TLM_OK;
    case TLM_DT_F16:  for (int64_t i = 0; i < n; i++) st_f16(o + 2 * i, x[i]); return TLM_OK;
    case TLM_DT_BF16:
        for (int64_t i = 0; i < n; i++) { uint16_t h = tlm_f32_to_bf16(x[i]); memcpy(o + 2 * i, &h, 2); }
        return TLM_OK;
    case TLM_DT_Q8_0: quant_q8_0(x, o, nb); return TLM_OK;
    case TLM_DT_Q4_K: for (int64_t b = 0; b < nb; b++) quant_kq_affine(x + 256 * b, o + 144 * b, 15); return TLM_OK;
    case TLM_DT_Q5_K: for (int64_t b = 0; b < nb; b++) quant_kq_affine(x + 256 * b, o + 176 * b, 31); return TLM_OK;
    case TLM_DT_Q6_K: for (int64_t b = 0; b < nb; b++) quant_q6_k(x + 256 * b, o + 210 * b); return TLM_OK;
    case TLM_DT_Q8_K: quant_q8_k(x, n, dst); return TLM_OK;
    case TLM_DT_MXFP4: quant_mxfp4(x, o, nb); return TLM_OK;
    case TLM_DT_FP8E4M3_B128:
        return tlm_seterr(TLM_EUNSUP, "quantize_row: FP8 needs 128x128 block scales (not provided here)");
    }
    return tlm_seterr(TLM_EINVAL, "quantize_row: unsupported dtype");
}

/* ======================================================================
 * dequantisers
 * ==================================================================== */
static void deq_kq_affine(const uint8_t *blk, float *y, int q5) {
    uint8_t sm[16];
    kq_unpack_scales(blk + 4, sm);
    float d = ld_f16(blk), dmin = ld_f16(blk + 2);
    const uint8_t *qh = blk + 16, *qs = q5 ? blk + 48 : blk + 16;
    for (int c = 0; c < 4; c++) {
        float d0 = d * (float)sm[2 * c], m0 = dmin * (float)sm[8 + 2 * c];
        float d1 = d * (float)sm[2 * c + 1], m1 = dmin * (float)sm[8 + 2 * c + 1];
        for (int l = 0; l < 32; l++) {
            int a = qs[32 * c + l] & 0x0F, b = qs[32 * c + l] >> 4;
            if (q5) { a |= ((qh[l] >> (2 * c)) & 1) << 4; b |= ((qh[l] >> (2 * c + 1)) & 1) << 4; }
            y[64 * c + l]      = d0 * (float)a - m0;
            y[64 * c + 32 + l] = d1 * (float)b - m1;
        }
    }
}

static void deq_q6_k(const uint8_t *blk, float *y) {
    const uint8_t *ql = blk, *qh = blk + 128;
    const int8_t *sc = (const int8_t *)(blk + 192);
    float d = ld_f16(blk + 208);
    for (int h = 0; h < 2; h++) {
        for (int l = 0; l < 32; l++) {
            const uint8_t A = ql[64 * h + l], B = ql[64 * h + 32 + l], C = qh[32 * h + l];
            int u0 = (A & 0x0F) | ((C & 3) << 4);
            int u1 = (B & 0x0F) | (((C >> 2) & 3) << 4);
            int u2 = (A >> 4) | (((C >> 4) & 3) << 4);
            int u3 = (B >> 4) | (((C >> 6) & 3) << 4);
            int e = 128 * h + l;
            y[e]      = d * (float)sc[e / 16] * (float)(u0 - 32);
            y[e + 32] = d * (float)sc[(e + 32) / 16] * (float)(u1 - 32);
            y[e + 64] = d * (float)sc[(e + 64) / 16] * (float)(u2 - 32);
            y[e + 96] = d * (float)sc[(e + 96) / 16] * (float)(u3 - 32);
        }
    }
}

int tlm_dequantize_row(tlm_dtype t, const void *src, float *y, int64_t n) {
    const dt_info *d = dt_find(t);
    if (!d) return tlm_seterr(TLM_EINVAL, "dequantize_row: unknown dtype %d", (int)t);
    if (n < 0 || n % d->block)
        return tlm_seterr(TLM_EINVAL, "dequantize_row: n=%lld is not a multiple of %d for %s",
                          (long long)n, d->block, d->name);
    int64_t nb = n / d->block;
    const uint8_t *s = src;
    switch (t) {
    case TLM_DT_F32: memcpy(y, src, (size_t)n * 4); return TLM_OK;
    case TLM_DT_F16: for (int64_t i = 0; i < n; i++) y[i] = ld_f16(s + 2 * i); return TLM_OK;
    case TLM_DT_BF16:
        for (int64_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, s + 2 * i, 2); y[i] = tlm_bf16_to_f32(h); }
        return TLM_OK;
    case TLM_DT_Q8_0:
        for (int64_t b = 0; b < nb; b++) {
            float dd = ld_f16(s + 34 * b);
            const int8_t *q = (const int8_t *)(s + 34 * b + 2);
            for (int i = 0; i < 32; i++) y[32 * b + i] = dd * (float)q[i];
        }
        return TLM_OK;
    case TLM_DT_Q4_K: for (int64_t b = 0; b < nb; b++) deq_kq_affine(s + 144 * b, y + 256 * b, 0); return TLM_OK;
    case TLM_DT_Q5_K: for (int64_t b = 0; b < nb; b++) deq_kq_affine(s + 176 * b, y + 256 * b, 1); return TLM_OK;
    case TLM_DT_Q6_K: for (int64_t b = 0; b < nb; b++) deq_q6_k(s + 210 * b, y + 256 * b); return TLM_OK;
    case TLM_DT_Q8_K:
        for (int64_t b = 0; b < nb; b++) {
            float dd;
            memcpy(&dd, s + 292 * b, 4);
            const int8_t *q = (const int8_t *)(s + 292 * b + 4);
            for (int i = 0; i < 256; i++) y[256 * b + i] = dd * (float)q[i];
        }
        return TLM_OK;
    case TLM_DT_MXFP4:
        for (int64_t b = 0; b < nb; b++) {
            const uint8_t *blk = s + 17 * b;
            float sc = tlm_e8m0_to_f32(blk[0]);
            for (int j = 0; j < 16; j++) {
                y[32 * b + j]      = 0.5f * (float)k_e2m1_x2[blk[1 + j] & 15] * sc;
                y[32 * b + j + 16] = 0.5f * (float)k_e2m1_x2[blk[1 + j] >> 4] * sc;
            }
        }
        return TLM_OK;
    case TLM_DT_FP8E4M3_B128:
        return tlm_seterr(TLM_EUNSUP, "dequantize_row: FP8 needs block scales; use tlm_dequantize_fp8_row");
    }
    return tlm_seterr(TLM_EINVAL, "dequantize_row: unsupported dtype");
}

int tlm_dequantize_fp8_row(const uint8_t *src, float *y, int64_t n, const float *col_scales) {
    if (n < 0) return tlm_seterr(TLM_EINVAL, "dequantize_fp8_row: negative n");
    for (int64_t i = 0; i < n; i++) {
        float s = col_scales ? col_scales[i / 128] : 1.0f;
        y[i] = tlm_e4m3_to_f32(src[i]) * s;
    }
    return TLM_OK;
}

/* ======================================================================
 * activation quantisation (P-42: once per input vector, shared)
 * ==================================================================== */
size_t tlm_qact_bytes(tlm_dtype kind, int64_t n) {
    if (n < 0) return 0;
    switch (kind) {
    case TLM_DT_Q8_K: return (size_t)((n + 255) / 256) * sizeof(tlm_blk_q8k);
    case TLM_DT_Q8_0: return (size_t)((n + 31) / 32) * sizeof(tlm_blk_a8);
    case TLM_DT_F32:  return (size_t)n * sizeof(float);
    default:          return 0;
    }
}

void tlm_quantize_act(tlm_dtype kind, const float *x, int64_t n, void *dst) {
    if (n <= 0) return;
    switch (kind) {
    case TLM_DT_Q8_K: quant_q8_k(x, n, dst); break;
    case TLM_DT_Q8_0: quant_a8(x, n, dst); break;
    case TLM_DT_F32:  memcpy(dst, x, (size_t)n * sizeof(float)); break;
    default: tlm_seterr(TLM_EINVAL, "quantize_act: unsupported activation kind %d", (int)kind); break;
    }
}

tlm_dtype tlm_act_kind_for(tlm_dtype weight) {
    switch (weight) {
    case TLM_DT_Q4_K: case TLM_DT_Q5_K: case TLM_DT_Q6_K: case TLM_DT_Q8_K: return TLM_DT_Q8_K;
    case TLM_DT_Q8_0: case TLM_DT_MXFP4: return TLM_DT_Q8_0;
    default: return TLM_DT_F32;   /* F32, F16, BF16, FP8 (and anything else) */
    }
}

/* ======================================================================
 * scalar reference kernels (the numerics contract, F-quant-07)
 * ==================================================================== */

/* Fixed 32-lane reduction tree, mirrored exactly by every SIMD tier:
 *   s16[l] = r[l] + r[l+16]; s8[l] = s16[l] + s16[l+8];
 *   s4[l] = s8[l] + s8[l+4]; result = (s4[0] + s4[2]) + (s4[1] + s4[3]). */
static float reduce32(const float *r) {
    float s16[16], s8[8], s4[4];
    for (int l = 0; l < 16; l++) s16[l] = r[l] + r[l + 16];
    for (int l = 0; l < 8; l++) s8[l] = s16[l] + s16[l + 8];
    for (int l = 0; l < 4; l++) s4[l] = s8[l] + s8[l + 4];
    return (s4[0] + s4[2]) + (s4[1] + s4[3]);
}

static float dot_f32_ref(const void *wv, const void *xv, int64_t n, const float *aux) {
    (void)aux;
    const float *w = wv, *x = xv;
    float r[32] = { 0 };
    for (int64_t i = 0; i < n; i++) r[i & 31] = fmaf(w[i], x[i], r[i & 31]);
    return reduce32(r);
}

static float dot_f16_ref(const void *wv, const void *xv, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const float *x = xv;
    float r[32] = { 0 };
    for (int64_t i = 0; i < n; i++) r[i & 31] = fmaf(ld_f16(w + 2 * i), x[i], r[i & 31]);
    return reduce32(r);
}

static float dot_bf16_ref(const void *wv, const void *xv, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const float *x = xv;
    float r[32] = { 0 };
    for (int64_t i = 0; i < n; i++) {
        uint16_t h; memcpy(&h, w + 2 * i, 2);
        r[i & 31] = fmaf(tlm_bf16_to_f32(h), x[i], r[i & 31]);
    }
    return reduce32(r);
}

/* FP8 x f32 (F-quant-13 reference, P-47): float lanes inside each 128-column
 * block, block partial times its scale added in double (the product of two
 * floats is exact in double, so contraction cannot change it). */
static float dot_fp8_ref(const void *wv, const void *xv, int64_t n, const float *cs) {
    const uint8_t *w = wv;
    const float *x = xv;
    double acc = 0.0;
    for (int64_t b0 = 0, cb = 0; b0 < n; b0 += 128, cb++) {
        int64_t e = b0 + 128 < n ? b0 + 128 : n;
        float r[32] = { 0 };
        for (int64_t i = b0; i < e; i++) r[i & 31] = fmaf(e4m3_kern(w[i]), x[i], r[i & 31]);
        float part = reduce32(r);
        acc += (double)part * (double)(cs ? cs[cb] : 1.0f);
    }
    return (float)acc;
}

static float dot_q8_0_ref(const void *wv, const void *av, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const tlm_blk_a8 *a = av;
    int64_t nb = n / 32;
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *blk = w + 34 * b;
        const int8_t *q = (const int8_t *)(blk + 2);
        int32_t s = 0;
        for (int i = 0; i < 32; i++) s += (int32_t)q[i] * a[b].q[i];
        acc = fmaf(ld_f16(blk) * a[b].d, (float)s, acc);
    }
    return acc;
}

static float dot_mxfp4_ref(const void *wv, const void *av, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const tlm_blk_a8 *a = av;
    int64_t nb = n / 32;
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *blk = w + 17 * b;
        int32_t s = 0;
        for (int j = 0; j < 16; j++) {
            s += (int32_t)k_e2m1_x2[blk[1 + j] & 15] * a[b].q[j];
            s += (int32_t)k_e2m1_x2[blk[1 + j] >> 4] * a[b].q[j + 16];
        }
        acc = fmaf(e8m0_half(blk[0]) * a[b].d, (float)s, acc);
    }
    return acc;
}

/* Q4_K / Q5_K x Q8_K: isum = sum_s sc[s] * sum(q*x), msum = sum_s mn[s] *
 * (S[2s] + S[2s+1]) (P-44), then acc += dA*isum, acc -= dM*msum. */
static float dot_kq_affine_ref(const uint8_t *w, const tlm_blk_q8k *a, int64_t n, int q5) {
    int64_t nb = n / 256;
    size_t bb = q5 ? 176 : 144;
    float acc = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *blk = w + bb * i;
        const uint8_t *qh = blk + 16, *qs = q5 ? blk + 48 : blk + 16;
        uint8_t sm[16];
        kq_unpack_scales(blk + 4, sm);
        const int8_t *x = a[i].q;
        int32_t isum = 0, msum = 0;
        for (int c = 0; c < 4; c++) {
            int32_t s0 = 0, s1 = 0;
            for (int l = 0; l < 32; l++) {
                int lo = qs[32 * c + l] & 0x0F, hi = qs[32 * c + l] >> 4;
                if (q5) { lo |= ((qh[l] >> (2 * c)) & 1) << 4; hi |= ((qh[l] >> (2 * c + 1)) & 1) << 4; }
                s0 += lo * x[64 * c + l];
                s1 += hi * x[64 * c + 32 + l];
            }
            isum += (int32_t)sm[2 * c] * s0 + (int32_t)sm[2 * c + 1] * s1;
        }
        for (int s = 0; s < 8; s++)
            msum += (int32_t)sm[8 + s] * ((int32_t)a[i].bsums[2 * s] + a[i].bsums[2 * s + 1]);
        float dA = ld_f16(blk) * a[i].d, dM = ld_f16(blk + 2) * a[i].d;
        acc = fmaf(dA, (float)isum, acc);
        acc = fmaf(-dM, (float)msum, acc);
    }
    return acc;
}

static float dot_q4_k_ref(const void *w, const void *a, int64_t n, const float *aux) {
    (void)aux;
    return dot_kq_affine_ref(w, a, n, 0);
}
static float dot_q5_k_ref(const void *w, const void *a, int64_t n, const float *aux) {
    (void)aux;
    return dot_kq_affine_ref(w, a, n, 1);
}

/* Q6_K x Q8_K: unsigned codes u (0..63) in the integer dot, the -32 folded
 * out through the activation group sums: isum = sum_g sc[g] * (sum u*x -
 * 32*S[g]) (F-quant-05/07, P-44). */
static float dot_q6_k_ref(const void *wv, const void *av, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const tlm_blk_q8k *a = av;
    int64_t nb = n / 256;
    float acc = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *ql = w + 210 * i, *qh = ql + 128;
        const int8_t *sc = (const int8_t *)(ql + 192);
        const int8_t *x = a[i].q;
        int32_t gs[16] = { 0 };
        for (int h = 0; h < 2; h++) {
            for (int l = 0; l < 32; l++) {
                const uint8_t A = ql[64 * h + l], B = ql[64 * h + 32 + l], C = qh[32 * h + l];
                int e = 128 * h + l;
                gs[e / 16]        += ((A & 0x0F) | ((C & 3) << 4)) * x[e];
                gs[(e + 32) / 16] += ((B & 0x0F) | (((C >> 2) & 3) << 4)) * x[e + 32];
                gs[(e + 64) / 16] += ((A >> 4) | (((C >> 4) & 3) << 4)) * x[e + 64];
                gs[(e + 96) / 16] += ((B >> 4) | (((C >> 6) & 3) << 4)) * x[e + 96];
            }
        }
        int32_t isum = 0;
        for (int g = 0; g < 16; g++) isum += (int32_t)sc[g] * (gs[g] - 32 * (int32_t)a[i].bsums[g]);
        acc = fmaf(ld_f16(ql + 208) * a[i].d, (float)isum, acc);
    }
    return acc;
}

/* Q8_K weights x Q8_K activations: one int32 per 256, one fmaf. The weight
 * block is read byte-wise (rows may sit at any offset inside a packet). */
static float dot_q8_k_ref(const void *wv, const void *av, int64_t n, const float *aux) {
    (void)aux;
    const uint8_t *w = wv;
    const tlm_blk_q8k *a = av;
    int64_t nb = n / 256;
    float acc = 0.0f;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *blk = w + 292 * b;
        const int8_t *q = (const int8_t *)(blk + 4);
        float dw;
        memcpy(&dw, blk, 4);
        int32_t s = 0;
        for (int i = 0; i < 256; i++) s += (int32_t)q[i] * a[b].q[i];
        acc = fmaf(dw * a[b].d, (float)s, acc);
    }
    return acc;
}

static void fill_scalar(tlm_quant_kernels *k) {
    memset(k, 0, sizeof *k);
    k->name = "scalar";
    k->dot[TLM_QK_F32] = dot_f32_ref;
    k->dot[TLM_QK_F16] = dot_f16_ref;
    k->dot[TLM_QK_BF16] = dot_bf16_ref;
    k->dot[TLM_QK_Q8_0] = dot_q8_0_ref;
    k->dot[TLM_QK_Q4_K] = dot_q4_k_ref;
    k->dot[TLM_QK_Q5_K] = dot_q5_k_ref;
    k->dot[TLM_QK_Q6_K] = dot_q6_k_ref;
    k->dot[TLM_QK_MXFP4] = dot_mxfp4_ref;
    k->dot[TLM_QK_FP8] = dot_fp8_ref;
    k->dot[TLM_QK_Q8_K] = dot_q8_k_ref;
}

/* ======================================================================
 * dispatch (P-55): pick the best tier the (possibly masked) CPU supports,
 * self-test each SIMD kernel against the scalar reference on fixed data,
 * and fall back one tier for any kernel that does not match bit for bit.
 * ==================================================================== */
static tlm_quant_kernels g_tab[3];      /* scalar, avx2, avx512 */
static int g_tab_state[3];              /* 0 = not built, 1 = ok, -1 = unavailable */
static _Atomic(const tlm_quant_kernels *) g_kern;
static pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER;

static int qk_index(tlm_dtype t) {
    switch (t) {
    case TLM_DT_F32: return TLM_QK_F32;
    case TLM_DT_F16: return TLM_QK_F16;
    case TLM_DT_BF16: return TLM_QK_BF16;
    case TLM_DT_Q8_0: return TLM_QK_Q8_0;
    case TLM_DT_Q4_K: return TLM_QK_Q4_K;
    case TLM_DT_Q5_K: return TLM_QK_Q5_K;
    case TLM_DT_Q6_K: return TLM_QK_Q6_K;
    case TLM_DT_MXFP4: return TLM_QK_MXFP4;
    case TLM_DT_FP8E4M3_B128: return TLM_QK_FP8;
    case TLM_DT_Q8_K: return TLM_QK_Q8_K;
    default: return -1;
    }
}

static const tlm_dtype k_qk_dtype[TLM_QK_COUNT] = {
    TLM_DT_F32, TLM_DT_F16, TLM_DT_BF16, TLM_DT_Q8_0, TLM_DT_Q4_K,
    TLM_DT_Q5_K, TLM_DT_Q6_K, TLM_DT_MXFP4, TLM_DT_FP8E4M3_B128, TLM_DT_Q8_K
};

/* Self-test data: ST_ROWS rows of 1024 values (4 super-blocks) with random
 * block bytes (scale fields replaced by finite values). ST_ROWS covers one
 * full 8-row matvec tile plus a tail row. */
#define ST_N    1024
#define ST_NF   1003        /* odd length for the float paths (exercises tails) */
#define ST_ROWS 9
#define ST_ROWB (ST_N * 4)  /* max bytes per row over all formats (f32 rows) */
static uint32_t st_rng(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s >> 8; }

static void st_fix_f16(uint8_t *p, uint32_t *s) {
    float v = ((float)(st_rng(s) & 0xFFFF) / 65536.0f - 0.5f) * 0.02f;
    st_f16(p, v);
}

static void st_make_row(int qk, uint8_t *w, uint32_t *s) {
    tlm_dtype t = k_qk_dtype[qk];
    int64_t n = (qk <= TLM_QK_BF16 || qk == TLM_QK_FP8) ? ST_NF : ST_N;
    size_t bytes = qk == TLM_QK_FP8 ? (size_t)n : tlm_row_bytes(t, n);
    for (size_t i = 0; i < bytes; i++) w[i] = (uint8_t)st_rng(s);
    switch (qk) {
    case TLM_QK_F32:
        for (int64_t i = 0; i < n; i++) {
            float v = ((float)(st_rng(s) & 0xFFFF) / 32768.0f - 1.0f);
            memcpy(w + 4 * i, &v, 4);
        }
        break;
    case TLM_QK_F16:
        for (int64_t i = 0; i < n; i++) st_f16(w + 2 * i, (float)(st_rng(s) & 0xFFFF) / 32768.0f - 1.0f);
        break;
    case TLM_QK_BF16:
        for (int64_t i = 0; i < n; i++) {
            uint16_t h = tlm_f32_to_bf16((float)(st_rng(s) & 0xFFFF) / 32768.0f - 1.0f);
            memcpy(w + 2 * i, &h, 2);
        }
        break;
    case TLM_QK_Q8_0: for (int64_t b = 0; b < n / 32; b++) st_fix_f16(w + 34 * b, s); break;
    case TLM_QK_Q4_K:
        for (int64_t b = 0; b < n / 256; b++) { st_fix_f16(w + 144 * b, s); st_fix_f16(w + 144 * b + 2, s); }
        break;
    case TLM_QK_Q5_K:
        for (int64_t b = 0; b < n / 256; b++) { st_fix_f16(w + 176 * b, s); st_fix_f16(w + 176 * b + 2, s); }
        break;
    case TLM_QK_Q6_K: for (int64_t b = 0; b < n / 256; b++) st_fix_f16(w + 210 * b + 208, s); break;
    case TLM_QK_MXFP4: for (int64_t b = 0; b < n / 32; b++) w[17 * b] = (uint8_t)(110 + st_rng(s) % 30); break;
    case TLM_QK_FP8: for (int64_t i = 0; i < n; i++) if ((w[i] & 0x7F) == 0x7F) w[i] = 0x3C; break;
    case TLM_QK_Q8_K:
        for (int64_t b = 0; b < n / 256; b++) {
            float d = (float)(st_rng(s) & 0xFFFF) * 1e-7f;
            memcpy(w + 292 * b, &d, 4);
        }
        break;
    default: break;
    }
}

static atomic_int g_selftest_failures;

int tlm_quant_selftest_failures(void) { return atomic_load(&g_selftest_failures); }

static int st_same(float a, float b) {
    if (a != a || b != b) return (a != a) && (b != b);
    return f2u(a) == f2u(b);
}

static void selftest_tier(tlm_quant_kernels *k, const tlm_quant_kernels *fallback) {
    static float xf[ST_N];
    static tlm_blk_q8k xk[ST_N / 256];
    static tlm_blk_a8 xa[ST_N / 32];
    static uint8_t w[ST_ROWS * ST_ROWB];
    static const float fp8_scales[(ST_N + 127) / 128] = { 0.5f, 2.0f, 0.125f, 1.0f, 3.0f, 0.25f, 1.5f, 8.0f };
    uint32_t s = 0x2545F491u;
    for (int i = 0; i < ST_N; i++) xf[i] = (float)(st_rng(&s) & 0xFFFF) / 32768.0f - 1.0f;
    xf[5] = 7.5f;   /* one outlier per block family so codes hit +-127 */
    xf[700] = -7.5f;
    quant_q8_k(xf, ST_N, xk);
    quant_a8(xf, ST_N, xa);
    for (int qk = 0; qk < TLM_QK_COUNT; qk++) {
        if (k->dot[qk] == fallback->dot[qk] && k->mv[qk] == fallback->mv[qk]) continue;
        tlm_dtype ak = tlm_act_kind_for(k_qk_dtype[qk]);
        const void *act = ak == TLM_DT_Q8_0 ? (const void *)xa : ak == TLM_DT_Q8_K ? (const void *)xk
                                                                                      : (const void *)xf;
        int64_t n = (qk <= TLM_QK_BF16 || qk == TLM_QK_FP8) ? ST_NF : ST_N;
        const float *aux = qk == TLM_QK_FP8 ? fp8_scales : NULL;
        size_t rb = qk == TLM_QK_FP8 ? (size_t)n : tlm_row_bytes(k_qk_dtype[qk], n);
        int ok = 1;
        for (int rep = 0; rep < 2 && ok; rep++) {
            float ref[ST_ROWS], y[ST_ROWS];
            for (int r = 0; r < ST_ROWS; r++) {
                st_make_row(qk, w + rb * (size_t)r, &s);
                ref[r] = g_tab[0].dot[qk](w + rb * (size_t)r, act, n, aux);
                if (!st_same(ref[r], k->dot[qk](w + rb * (size_t)r, act, n, aux))) ok = 0;
            }
            if (ok && k->mv[qk]) {
                k->mv[qk](w, rb, act, y, n, ST_ROWS);
                for (int r = 0; r < ST_ROWS; r++)
                    if (!st_same(ref[r], y[r])) ok = 0;
            }
        }
        if (!ok) {
            atomic_fetch_add(&g_selftest_failures, 1);
            tlm_log(TLM_LOG_WARN, "tlm_quant: %s kernel for %s failed the bit-exact self-test; using %s",
                    k->name, tlm_dtype_name(k_qk_dtype[qk]), fallback->name);
            k->dot[qk] = fallback->dot[qk];
            k->mv[qk] = fallback->mv[qk];
        }
    }
}

static void build_tier(int idx) {
    if (g_tab_state[idx]) return;
    if (idx == 0) { fill_scalar(&g_tab[0]); g_tab_state[0] = 1; return; }
    g_tab[idx] = g_tab[idx - 1];
    int got = idx == 1 ? tlm_kern_avx2_fill(&g_tab[idx]) : tlm_kern_avx512_fill(&g_tab[idx]);
    if (got <= 0) { g_tab_state[idx] = -1; return; }
    g_tab[idx].name = idx == 1 ? "avx2" : "avx512";
    selftest_tier(&g_tab[idx], &g_tab[idx - 1]);
    g_tab_state[idx] = 1;
}

void tlm_quant_init(void) {
    pthread_mutex_lock(&g_init_mu);
    build_tier(0);
    uint32_t f = tlm_cpu()->features;
    const uint32_t need2 = TLM_CPU_AVX2 | TLM_CPU_FMA | TLM_CPU_F16C;
    const uint32_t need512 = need2 | TLM_CPU_AVX512F | TLM_CPU_AVX512BW | TLM_CPU_AVX512VL | TLM_CPU_AVX512VNNI;
    int sel = 0;
    if ((f & need2) == need2) {
        build_tier(1);
        if (g_tab_state[1] > 0) {
            sel = 1;
            if ((f & need512) == need512) {
                build_tier(2);
                if (g_tab_state[2] > 0) sel = 2;
            }
        }
    }
    atomic_store_explicit(&g_kern, &g_tab[sel], memory_order_release);
    pthread_mutex_unlock(&g_init_mu);
}

static inline const tlm_quant_kernels *kern(void) {
    const tlm_quant_kernels *k = atomic_load_explicit(&g_kern, memory_order_acquire);
    if (!k) {
        tlm_quant_init();
        k = atomic_load_explicit(&g_kern, memory_order_acquire);
    }
    return k;
}

const char *tlm_quant_tier(void) { return kern()->name; }

/* ======================================================================
 * public dot / matvec
 * ==================================================================== */
float tlm_vec_dot(tlm_dtype w, const void *wrow, const void *act, int64_t n) {
    int qk = qk_index(w);
    if (qk < 0) { tlm_seterr(TLM_EUNSUP, "vec_dot: no kernel for %s weights", tlm_dtype_name(w)); return NAN; }
    if (n < 0 || n % tlm_dtype_block(w)) {
        tlm_seterr(TLM_EINVAL, "vec_dot: n=%lld not a multiple of the %s block", (long long)n, tlm_dtype_name(w));
        return NAN;
    }
    return kern()->dot[qk](wrow, act, n, NULL);
}

float tlm_vec_dot_fp8(const uint8_t *wrow, const float *x, int64_t n, const float *col_scales) {
    if (n < 0) { tlm_seterr(TLM_EINVAL, "vec_dot_fp8: negative n"); return NAN; }
    return kern()->dot[TLM_QK_FP8](wrow, x, n, col_scales);
}

void tlm_matvec_rows(tlm_dtype w, const void *W, size_t row_bytes,
                     const void *act, float *y, int64_t n, int64_t r0, int64_t r1,
                     const float *scales) {
    int qk = qk_index(w);
    if (qk < 0 || n < 0 || n % tlm_dtype_block(w)) {
        tlm_seterr(TLM_EINVAL, "matvec_rows: bad dtype %s or n=%lld", tlm_dtype_name(w), (long long)n);
        for (int64_t r = r0; r < r1; r++) y[r] = NAN;
        return;
    }
    if (r1 <= r0) return;
    const tlm_quant_kernels *k = kern();
    const uint8_t *base = W;
    if (qk == TLM_QK_FP8) {
        int64_t ncb = (n + 127) / 128;
        for (int64_t r = r0; r < r1; r++)
            y[r] = k->dot[qk](base + (size_t)r * row_bytes, act, n, scales ? scales + (r / 128) * ncb : NULL);
        return;
    }
    if (k->mv[qk]) {
        k->mv[qk](base + (size_t)r0 * row_bytes, row_bytes, act, y + r0, n, r1 - r0);
        return;
    }
    tlm_qdot_fn dot = k->dot[qk];
    for (int64_t r = r0; r < r1; r++) y[r] = dot(base + (size_t)r * row_bytes, act, n, NULL);
}
