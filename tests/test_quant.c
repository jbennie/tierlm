/* test_quant.c - tlm_quant: block layouts, conversions, reference
 * quantisers, dot / matvec kernels, bit-exact equality of the scalar /
 * AVX2 / AVX-512 tiers (P-43) for single-row and 8-row-tile kernels, and a
 * matvec micro-benchmark (printed, never asserted).
 *
 * Set TLM_TEST_NO_BENCH=1 to skip the benchmark. */
#include "tlm_test.h"
#include "tlm_quant.h"
#include "tlm_platform.h"

#include <stdint.h>
#include <string.h>

/* ---------------------------------------------------------------------- */
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
/* identical bits, or both NaN (NaN payloads may differ between decoders) */
static int same_bits(float a, float b) {
    if (a != a || b != b) return (a != a) && (b != b);
    return f2u(a) == f2u(b);
}
static uint32_t rnd32(void) { return (uint32_t)(tlm_test_next() >> 32); }
static float gauss(void) {   /* cheap bell-shaped noise in about [-4, 4] */
    return tlm_test_randf(-1, 1) + tlm_test_randf(-1, 1) + tlm_test_randf(-1, 1) + tlm_test_randf(-1, 1);
}
static void put_f16(uint8_t *p, float f) { uint16_t h = tlm_f32_to_f16(f); memcpy(p, &h, 2); }
static float get_f32(const uint8_t *p) { float f; memcpy(&f, p, 4); return f; }

/* Every format with an integer (exact-sum) kernel. */
static const tlm_dtype k_int_types[] = { TLM_DT_Q8_0, TLM_DT_Q4_K, TLM_DT_Q5_K, TLM_DT_Q6_K,
                                         TLM_DT_Q8_K, TLM_DT_MXFP4 };
#define N_INT ((int)(sizeof k_int_types / sizeof k_int_types[0]))

/* tiers this CPU can run */
static const char *k_tiers[3] = { "scalar", "avx2", "avx512" };
static int g_tier_ok[3] = { -1, -1, -1 };
static int tier_ok(int t) {
    if (g_tier_ok[t] < 0) {
        const uint32_t need2 = TLM_CPU_AVX2 | TLM_CPU_FMA | TLM_CPU_F16C;
        const uint32_t need512 = need2 | TLM_CPU_AVX512F | TLM_CPU_AVX512BW | TLM_CPU_AVX512VL | TLM_CPU_AVX512VNNI;
        tlm_cpu_force("auto");
        uint32_t f = tlm_cpu()->features;
        g_tier_ok[0] = 1;
        g_tier_ok[1] = (f & need2) == need2;
        g_tier_ok[2] = (f & need512) == need512;
    }
    return g_tier_ok[t];
}
static void use_tier(int t) {
    tlm_cpu_force(k_tiers[t]);
    tlm_quant_init();
}
static void use_best_tier(void) {
    tlm_cpu_force("auto");
    tlm_quant_init();
}

/* ---------------------------------------------------------------------- */
static void test_layout(void) {
    CHECK_EQ_INT(tlm_dtype_block_bytes(TLM_DT_Q8_0), 34);
    CHECK_EQ_INT(tlm_dtype_block_bytes(TLM_DT_Q4_K), 144);
    CHECK_EQ_INT(tlm_dtype_block_bytes(TLM_DT_Q5_K), 176);
    CHECK_EQ_INT(tlm_dtype_block_bytes(TLM_DT_Q6_K), 210);
    CHECK_EQ_INT(tlm_dtype_block_bytes(TLM_DT_Q8_K), 292);
    CHECK_EQ_INT(tlm_dtype_block_bytes(TLM_DT_MXFP4), 17);
    CHECK_EQ_INT(tlm_dtype_block_bytes(TLM_DT_F32), 4);
    CHECK_EQ_INT(tlm_dtype_block_bytes(TLM_DT_F16), 2);
    CHECK_EQ_INT(tlm_dtype_block_bytes(TLM_DT_BF16), 2);
    CHECK_EQ_INT(tlm_dtype_block_bytes(TLM_DT_FP8E4M3_B128), 1);
    CHECK_EQ_INT(tlm_dtype_block(TLM_DT_Q8_0), 32);
    CHECK_EQ_INT(tlm_dtype_block(TLM_DT_Q4_K), 256);
    CHECK_EQ_INT(tlm_dtype_block(TLM_DT_Q6_K), 256);
    CHECK_EQ_INT(tlm_dtype_block(TLM_DT_Q8_K), 256);
    CHECK_EQ_INT(tlm_dtype_block(TLM_DT_MXFP4), 32);
    CHECK_EQ_INT(tlm_dtype_block(TLM_DT_F16), 1);
    CHECK_EQ_INT(tlm_dtype_block((tlm_dtype)99), 0);
    CHECK_EQ_INT(tlm_dtype_block_bytes((tlm_dtype)99), 0);
    CHECK_EQ_INT(tlm_row_bytes(TLM_DT_Q6_K, 4096), 16 * 210);
    CHECK_EQ_INT(tlm_row_bytes(TLM_DT_Q4_K, 4096), 16 * 144);
    CHECK_EQ_INT(tlm_row_bytes(TLM_DT_Q5_K, 512), 2 * 176);
    CHECK_EQ_INT(tlm_row_bytes(TLM_DT_Q8_0, 64), 68);
    CHECK_EQ_INT(tlm_row_bytes(TLM_DT_MXFP4, 96), 51);
    CHECK_EQ_INT(tlm_row_bytes(TLM_DT_BF16, 7), 14);
    CHECK_EQ_INT(tlm_row_bytes(TLM_DT_FP8E4M3_B128, 300), 300);
    CHECK_EQ_INT(tlm_row_bytes(TLM_DT_Q4_K, 100), 0);
    CHECK_EQ_INT(tlm_row_bytes(TLM_DT_Q8_0, -32), 0);
    CHECK_EQ_INT(tlm_qact_bytes(TLM_DT_Q8_K, 256), 292);
    CHECK_EQ_INT(tlm_qact_bytes(TLM_DT_Q8_K, 257), 584);
    CHECK_EQ_INT(tlm_qact_bytes(TLM_DT_Q8_0, 64), 2 * TLM_QACT_Q8_0_BLOCK_BYTES);
    CHECK_EQ_INT(tlm_qact_bytes(TLM_DT_Q8_0, 33), 2 * TLM_QACT_Q8_0_BLOCK_BYTES);
    CHECK_EQ_INT(tlm_qact_bytes(TLM_DT_F32, 10), 40);
    CHECK_EQ_INT(tlm_qact_bytes(TLM_DT_Q6_K, 256), 0);

    static const tlm_dtype all[] = { TLM_DT_F32, TLM_DT_F16, TLM_DT_BF16, TLM_DT_Q8_0, TLM_DT_Q4_K,
                                     TLM_DT_Q5_K, TLM_DT_Q6_K, TLM_DT_Q8_K, TLM_DT_MXFP4, TLM_DT_FP8E4M3_B128 };
    for (size_t i = 0; i < sizeof all / sizeof all[0]; i++) {
        tlm_dtype t = (tlm_dtype)-1;
        CHECK(tlm_dtype_from_name(tlm_dtype_name(all[i]), &t) == 0);
        CHECK_EQ_INT(t, all[i]);
    }
    tlm_dtype t;
    CHECK(tlm_dtype_from_name("Q6_K", &t) == 0 && t == TLM_DT_Q6_K);
    CHECK(tlm_dtype_from_name("BF16", &t) == 0 && t == TLM_DT_BF16);
    CHECK(tlm_dtype_from_name("float8_e4m3fn", &t) == 0 && t == TLM_DT_FP8E4M3_B128);
    CHECK(tlm_dtype_from_name("q3_k", &t) == TLM_EINVAL);
    CHECK(tlm_dtype_from_name(NULL, &t) == TLM_EINVAL);
    CHECK(strcmp(tlm_dtype_name((tlm_dtype)99), "unknown") == 0);

    CHECK_EQ_INT(tlm_act_kind_for(TLM_DT_Q4_K), TLM_DT_Q8_K);
    CHECK_EQ_INT(tlm_act_kind_for(TLM_DT_Q5_K), TLM_DT_Q8_K);
    CHECK_EQ_INT(tlm_act_kind_for(TLM_DT_Q6_K), TLM_DT_Q8_K);
    CHECK_EQ_INT(tlm_act_kind_for(TLM_DT_Q8_K), TLM_DT_Q8_K);
    CHECK_EQ_INT(tlm_act_kind_for(TLM_DT_Q8_0), TLM_DT_Q8_0);
    CHECK_EQ_INT(tlm_act_kind_for(TLM_DT_MXFP4), TLM_DT_Q8_0);
    CHECK_EQ_INT(tlm_act_kind_for(TLM_DT_F32), TLM_DT_F32);
    CHECK_EQ_INT(tlm_act_kind_for(TLM_DT_F16), TLM_DT_F32);
    CHECK_EQ_INT(tlm_act_kind_for(TLM_DT_BF16), TLM_DT_F32);
    CHECK_EQ_INT(tlm_act_kind_for(TLM_DT_FP8E4M3_B128), TLM_DT_F32);
}

/* ---------------------------------------------------------------------- */
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
__attribute__((target("f16c"))) static uint16_t hw_f32_to_f16(float f) {
    return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
}
__attribute__((target("f16c"))) static float hw_f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
#define HAVE_HW_F16 1
#endif

static void test_f16(void) {
    /* exhaustive: every f16 survives f16 -> f32 -> f16 (NaNs stay NaN) */
    int bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        float f = tlm_f16_to_f32((uint16_t)h);
        uint16_t back = tlm_f32_to_f16(f);
        int is_nan = ((h & 0x7C00) == 0x7C00) && (h & 0x3FF);
        if (is_nan) { if (!(f != f) || (back & 0x7C00) != 0x7C00 || !(back & 0x3FF)) bad++; }
        else if (back != h) bad++;
    }
    CHECK_EQ_INT(bad, 0);
    CHECK_EQ_INT(tlm_f32_to_f16(1.0f), 0x3C00);
    CHECK_EQ_INT(tlm_f32_to_f16(-2.0f), 0xC000);
    CHECK_EQ_INT(tlm_f32_to_f16(65504.0f), 0x7BFF);
    CHECK_EQ_INT(tlm_f32_to_f16(65519.996f), 0x7BFF);
    CHECK_EQ_INT(tlm_f32_to_f16(65520.0f), 0x7C00);          /* tie -> even = inf */
    CHECK_EQ_INT(tlm_f32_to_f16(0x1p-24f), 0x0001);          /* smallest subnormal */
    CHECK_EQ_INT(tlm_f32_to_f16(0x1p-25f), 0x0000);          /* tie -> even = 0 */
    CHECK_EQ_INT(tlm_f32_to_f16(0x1.8p-24f), 0x0002);        /* 1.5 ulp tie -> even 2 */
    CHECK_EQ_INT(tlm_f32_to_f16(0x1.ff8p-15f), 0x03FF);      /* 1023 subnormal ulps, exact */
    CHECK_EQ_INT(tlm_f32_to_f16(0x1.ffcp-15f), 0x0400);      /* 1023.5 tie -> even 1024 = min normal */
    /* f16 ulp at 1.0 is 2^-10 = 0x1.004p0 - 1; half an ulp is 0x1.002p0 - 1 */
    CHECK_EQ_INT(tlm_f32_to_f16(0x1.004p0f), 0x3C01);        /* exactly one ulp */
    CHECK_EQ_INT(tlm_f32_to_f16(0x1.002p0f), 0x3C00);        /* half ulp tie -> even (down) */
    CHECK_EQ_INT(tlm_f32_to_f16(0x1.006p0f), 0x3C02);        /* 1.5 ulp tie -> even (up) */
    CHECK_EQ_INT(tlm_f32_to_f16(0x1.0021p0f), 0x3C01);       /* just above half -> up */
    CHECK_EQ_INT(tlm_f32_to_f16(-0.0f), 0x8000);
    CHECK_EQ_INT(tlm_f32_to_f16(INFINITY), 0x7C00);
    CHECK_EQ_INT(tlm_f32_to_f16(-INFINITY), 0xFC00);
    CHECK_EQ_INT(tlm_f32_to_f16(1e10f), 0x7C00);
    CHECK((tlm_f32_to_f16(NAN) & 0x7C00) == 0x7C00 && (tlm_f32_to_f16(NAN) & 0x3FF));
    CHECK(tlm_f16_to_f32(0x0001) == 0x1p-24f);
    CHECK(tlm_f16_to_f32(0x7BFF) == 65504.0f);
    CHECK(tlm_f16_to_f32(0x7C00) == INFINITY);
    CHECK(f2u(tlm_f16_to_f32(0x8000)) == 0x80000000u);
#ifdef HAVE_HW_F16
    if (tlm_cpu()->features & TLM_CPU_F16C) {
        int mism = 0;
        for (int i = 0; i < 2000000; i++) {
            float f = u2f(rnd32());
            if (f != f) continue;
            if (tlm_f32_to_f16(f) != hw_f32_to_f16(f)) mism++;
        }
        for (uint32_t h = 0; h < 65536; h++) {
            float a = tlm_f16_to_f32((uint16_t)h), b = hw_f16_to_f32((uint16_t)h);
            if (!same_bits(a, b)) mism++;
        }
        CHECK_EQ_INT(mism, 0);
    }
#endif
}

static void test_bf16(void) {
    int bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        float f = tlm_bf16_to_f32((uint16_t)h);
        uint16_t back = tlm_f32_to_bf16(f);
        if (f != f) { if (!((back & 0x7F80) == 0x7F80 && (back & 0x7F))) bad++; }
        else if (back != h) bad++;
        if (f2u(f) != (h << 16)) bad++;
    }
    CHECK_EQ_INT(bad, 0);
    CHECK_EQ_INT(tlm_f32_to_bf16(u2f(0x3F808000u)), 0x3F80);   /* tie, even stays */
    CHECK_EQ_INT(tlm_f32_to_bf16(u2f(0x3F818000u)), 0x3F82);   /* tie, odd rounds up */
    CHECK_EQ_INT(tlm_f32_to_bf16(u2f(0x3F808001u)), 0x3F81);
    CHECK_EQ_INT(tlm_f32_to_bf16(u2f(0x3F807FFFu)), 0x3F80);
    CHECK_EQ_INT(tlm_f32_to_bf16(u2f(0x7F7FFFFFu)), 0x7F80);   /* FLT_MAX rounds to inf */
    CHECK_EQ_INT(tlm_f32_to_bf16(u2f(0xFF800000u)), 0xFF80);   /* -inf */
    CHECK_EQ_INT(tlm_f32_to_bf16(u2f(0x00000001u)), 0x0000);   /* tiny subnormal -> 0 */
    {   /* signalling NaN whose payload sits only in the low 16 bits stays NaN */
        uint16_t b = tlm_f32_to_bf16(u2f(0x7F800001u));
        CHECK((b & 0x7F80) == 0x7F80 && (b & 0x7F));
    }
    /* random round trip property: |x - bf16(x)| <= half ulp */
    int far = 0;
    for (int i = 0; i < 200000; i++) {
        float x = gauss() * 1000.0f;
        float y = tlm_bf16_to_f32(tlm_f32_to_bf16(x));
        if (fabsf(x - y) > fabsf(x) * 0x1p-8f) far++;
    }
    CHECK_EQ_INT(far, 0);
    CHECK(tlm_bf16_to_f32(0x3F80) == 1.0f);
}

/* e4m3 code nearest to v by brute force (ties to the even code). */
static int e4m3_nearest(float v) {
    int best = -1;
    double bd = 0;
    for (int c = 0; c < 0x7F; c++) {
        double d = fabs((double)tlm_e4m3_to_f32((uint8_t)c) - fabs((double)v));
        if (best < 0 || d < bd || (d == bd && !(c & 1))) { best = c; bd = d; }
    }
    return (v < 0 || (v == 0 && signbit(v))) ? (best | 0x80) : best;
}

static void test_e4m3_e8m0(void) {
    CHECK(tlm_e4m3_to_f32(0x00) == 0.0f);
    CHECK(tlm_e4m3_to_f32(0x01) == 0x1p-9f);
    CHECK(tlm_e4m3_to_f32(0x07) == 7 * 0x1p-9f);
    CHECK(tlm_e4m3_to_f32(0x08) == 0x1p-6f);
    CHECK(tlm_e4m3_to_f32(0x38) == 1.0f);
    CHECK(tlm_e4m3_to_f32(0x7E) == 448.0f);
    CHECK(tlm_e4m3_to_f32(0xFE) == -448.0f);
    CHECK(tlm_e4m3_to_f32(0x77) == 240.0f);        /* exponent 14: no infinity encodings */
    CHECK(tlm_e4m3_to_f32(0x78) == 256.0f);
    CHECK(tlm_e4m3_to_f32(0x7F) != tlm_e4m3_to_f32(0x7F));
    CHECK(tlm_e4m3_to_f32(0xFF) != tlm_e4m3_to_f32(0xFF));
    CHECK(f2u(tlm_e4m3_to_f32(0x80)) == 0x80000000u);
    int bad = 0;
    for (int c = 0; c < 256; c++) {
        if ((c & 0x7F) == 0x7F) continue;
        int e = (c >> 3) & 15, m = c & 7;
        double ref = e ? ldexp(1.0 + m / 8.0, e - 7) : ldexp(m, -9);
        if (c & 0x80) ref = -ref;
        float v = tlm_e4m3_to_f32((uint8_t)c);
        if ((double)v != ref) bad++;
        if (tlm_f32_to_e4m3(v) != c) bad++;          /* round trip of every code */
    }
    CHECK_EQ_INT(bad, 0);
    /* encoder = brute-force nearest (RNE) over normal, subnormal and tie values */
    int mism = 0;
    for (int i = 0; i < 20000; i++) {
        float v;
        switch (i % 4) {
        case 0: v = tlm_test_randf(-448, 448); break;
        case 1: v = tlm_test_randf(-0.03f, 0.03f); break;                      /* subnormal region */
        case 2: {                                                              /* exact midpoints */
            int c = (int)(rnd32() % 0x7E);
            v = 0.5f * (tlm_e4m3_to_f32((uint8_t)c) + tlm_e4m3_to_f32((uint8_t)(c + 1)));
            if (rnd32() & 1) v = -v;
            break;
        }
        default: v = gauss() * 10.0f; break;
        }
        if (tlm_f32_to_e4m3(v) != e4m3_nearest(v)) mism++;
    }
    CHECK_EQ_INT(mism, 0);
    CHECK_EQ_INT(tlm_f32_to_e4m3(1000.0f), 0x7E);      /* saturating */
    CHECK_EQ_INT(tlm_f32_to_e4m3(460.0f), 0x7E);
    CHECK_EQ_INT(tlm_f32_to_e4m3(-INFINITY), 0xFE);
    CHECK_EQ_INT(tlm_f32_to_e4m3(NAN), 0x7F);
    CHECK_EQ_INT(tlm_f32_to_e4m3(-NAN) & 0x7F, 0x7F);
    CHECK_EQ_INT(tlm_f32_to_e4m3(1.0625f), 0x38);   /* tie between 1 and 1.125 -> even */
    CHECK_EQ_INT(tlm_f32_to_e4m3(1.1875f), 0x3A);   /* tie between 1.125 and 1.25 -> even */
    CHECK_EQ_INT(tlm_f32_to_e4m3(0x1p-10f), 0x00);  /* half of min subnormal -> 0 */
    CHECK_EQ_INT(tlm_f32_to_e4m3(0x1.8p-9f), 0x02); /* 1.5 subnormal ulps -> 2 */
    CHECK_EQ_INT(tlm_f32_to_e4m3(0x1.Fp-7f), 0x08); /* 7.75 subnormal ulps -> min normal */
    CHECK_EQ_INT(tlm_f32_to_e4m3(-0.0f), 0x80);

    CHECK(tlm_e8m0_to_f32(127) == 1.0f);
    CHECK(tlm_e8m0_to_f32(128) == 2.0f);
    CHECK(tlm_e8m0_to_f32(126) == 0.5f);
    CHECK(tlm_e8m0_to_f32(254) == 0x1p127f);
    CHECK(tlm_e8m0_to_f32(1) == 0x1p-126f);
    CHECK(tlm_e8m0_to_f32(0) == 0x1p-127f);          /* f32 subnormal, not flushed */
    CHECK(tlm_e8m0_to_f32(255) != tlm_e8m0_to_f32(255));
    for (int e = 1; e < 255; e++) CHECK(tlm_e8m0_to_f32((uint8_t)e) == ldexpf(1.0f, e - 127));
}

/* ---------------------------------------------------------------------- */
/* Distributions: 0 uniform, 1 small bell, 2 bell with rare 8x outliers,
 * 3 positive offset (exercises K-quant mins). */
static void fill_row(float *x, int64_t n, int kind) {
    for (int64_t i = 0; i < n; i++) {
        switch (kind) {
        case 0: x[i] = tlm_test_randf(-1, 1); break;
        case 1: x[i] = 0.05f * gauss(); break;
        case 2: x[i] = gauss() + ((rnd32() % 97) == 0 ? 8.0f * gauss() : 0.0f); break;
        default: x[i] = 3.0f + 0.1f * gauss(); break;
        }
    }
}

static double rel_rmse(const float *a, const float *b, int64_t n) {
    double e = 0, s = 0;
    for (int64_t i = 0; i < n; i++) { double d = (double)a[i] - b[i]; e += d * d; s += (double)a[i] * a[i]; }
    return s > 0 ? sqrt(e / s) : sqrt(e);
}

static void test_quant_roundtrip(void) {
    const int64_t n = 4096;
    float *x = malloc(n * 4), *y = malloc(n * 4);
    uint8_t *q = malloc(n * 4), *q2 = malloc(n * 4);
    /* Relative RMSE bounds (dequant vs input), ~25% above measured worst
     * over 20 rows, for the regular distributions and the outlier one.
     * Theory check: Q8_0 on uniform input is step/sqrt(12)/rms = 0.0038. */
    static const struct { tlm_dtype t; double reg, outl; } tb[] = {
        { TLM_DT_Q8_0, 0.007, 0.012 }, { TLM_DT_Q8_K, 0.008, 0.028 }, { TLM_DT_Q6_K, 0.022, 0.034 },
        { TLM_DT_Q5_K, 0.044, 0.058 }, { TLM_DT_Q4_K, 0.088, 0.120 }, { TLM_DT_MXFP4, 0.140, 0.200 },
        { TLM_DT_BF16, 0.0025, 0.0025 }, { TLM_DT_F16, 0.0003, 0.0003 },
    };
    for (size_t k = 0; k < sizeof tb / sizeof tb[0]; k++) {
        double worst[4] = { 0, 0, 0, 0 };
        for (int rep = 0; rep < 12; rep++) {
            int kind = rep % 4;
            fill_row(x, n, kind);
            CHECK(tlm_quantize_row(tb[k].t, x, q, n) == 0);
            CHECK(tlm_quantize_row(tb[k].t, x, q2, n) == 0);
            CHECK(memcmp(q, q2, tlm_row_bytes(tb[k].t, n)) == 0);   /* deterministic */
            CHECK(tlm_dequantize_row(tb[k].t, q, y, n) == 0);
            double e = rel_rmse(x, y, n), bound = kind == 2 ? tb[k].outl : tb[k].reg;
            if (e > worst[kind]) worst[kind] = e;
            if (!(e <= bound))
                fprintf(stderr, "  %s kind %d rel rmse %.5f > %.5f\n", tlm_dtype_name(tb[k].t), kind, e, bound);
            CHECK(e <= bound);
            int nonfinite = 0;
            for (int64_t i = 0; i < n; i++) nonfinite += !isfinite(y[i]);
            CHECK_EQ_INT(nonfinite, 0);
        }
        printf("  quant %-6s rel-rmse uniform %.5f bell %.5f outliers %.5f offset %.5f\n",
               tlm_dtype_name(tb[k].t), worst[0], worst[1], worst[2], worst[3]);
        /* zero row -> zero */
        memset(x, 0, n * 4);
        CHECK(tlm_quantize_row(tb[k].t, x, q, n) == 0);
        CHECK(tlm_dequantize_row(tb[k].t, q, y, n) == 0);
        int nz = 0;
        for (int64_t i = 0; i < n; i++) nz += y[i] != 0.0f;
        CHECK_EQ_INT(nz, 0);
        /* constant rows (positive, negative) reproduce the constant closely */
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            for (int64_t i = 0; i < 256; i++) x[i] = 0.75f * (float)sgn;
            CHECK(tlm_quantize_row(tb[k].t, x, q, 256) == 0);
            CHECK(tlm_dequantize_row(tb[k].t, q, y, 256) == 0);
            double e = rel_rmse(x, y, 256);
            if (!(e < 0.01)) fprintf(stderr, "  %s constant %+d: rel rmse %.5f\n", tlm_dtype_name(tb[k].t), sgn, e);
            CHECK(e < 0.01);
        }
    }
    /* Q8_0 elementwise bound: |x - y| <= d/2 + |q| * |d - f16(d)| */
    fill_row(x, 256, 2);
    tlm_quantize_row(TLM_DT_Q8_0, x, q, 256);
    tlm_dequantize_row(TLM_DT_Q8_0, q, y, 256);
    for (int b = 0; b < 8; b++) {
        float d = tlm_f16_to_f32((uint16_t)(q[34 * b] | (q[34 * b + 1] << 8)));
        for (int i = 0; i < 32; i++) {
            CHECK((int8_t)q[34 * b + 2 + i] >= -127);
            CHECK(fabsf(x[32 * b + i] - y[32 * b + i]) <= 0.5f * d + 128 * d * 0x1p-11f + 1e-12f);
        }
    }
    /* Q8_K elementwise bound: |x - d*q| <= d/2 (tiny slack for the float reciprocal) */
    fill_row(x, 512, 2);
    tlm_quantize_row(TLM_DT_Q8_K, x, q, 512);
    tlm_dequantize_row(TLM_DT_Q8_K, q, y, 512);
    for (int b = 0; b < 2; b++) {
        float d = get_f32(q + 292 * b);
        for (int i = 0; i < 256; i++) CHECK(fabsf(x[256 * b + i] - y[256 * b + i]) <= d * 0.5001f);
    }
    /* MXFP4: values exactly on the E2M1 grid survive unchanged */
    {
        static const float grid[8] = { 0, 0.5f, 1, 1.5f, 2, 3, 4, 6 };
        for (int i = 0; i < 32; i++) x[i] = grid[i % 8] * ((i & 8) ? -0.25f : 0.25f);
        x[7] = 1.5f;   /* amax = 1.5 = 6 * 0.25 -> E8M0 125 */
        tlm_quantize_row(TLM_DT_MXFP4, x, q, 32);
        tlm_dequantize_row(TLM_DT_MXFP4, q, y, 32);
        CHECK_EQ_INT(q[0], 125);
        for (int i = 0; i < 32; i++) CHECK(x[i] == y[i]);
        /* nibble order: element j in the low nibble of byte j, element j+16 in its high nibble */
        memset(x, 0, 32 * 4);
        x[3] = 1.0f;     /* code 2 */
        x[19] = -6.0f;   /* code 15 */
        tlm_quantize_row(TLM_DT_MXFP4, x, q, 32);
        CHECK_EQ_INT(q[0], 127);
        CHECK_EQ_INT(q[1 + 3], 0xF2);
        for (int j = 0; j < 16; j++) if (j != 3) CHECK_EQ_INT(q[1 + j], 0);
    }
    /* Q4_K / Q5_K / Q6_K element ordering, checked through the dequantiser
     * against a hand-built block (layout documented in tlm_quant.h). */
    {
        uint8_t blk[210];
        memset(blk, 0, sizeof blk);
        put_f16(blk, 1.0f);                  /* Q4_K: d = 1, dmin = 0 */
        put_f16(blk + 2, 0.0f);
        blk[4 + 1] = 2;                      /* scale of sub-block 1 = 2 */
        blk[4 + 8 + 2] = 3;                  /* low nibble of scale 6 (bytes 8..11 hold sc4..7 / mn4..7) */
        blk[16 + 32 * 0 + 5] = 0x70;         /* chunk 0 byte 5: high nibble 7 -> element 32+5, sub 1 */
        blk[16 + 32 * 3 + 9] = 0x04;         /* chunk 3 byte 9: low nibble 4 -> element 192+9, sub 6 */
        tlm_dequantize_row(TLM_DT_Q4_K, blk, y, 256);
        int nz = 0;
        for (int i = 0; i < 256; i++) nz += y[i] != 0.0f;
        CHECK_EQ_INT(nz, 2);
        CHECK(y[37] == 14.0f);
        CHECK(y[201] == 12.0f);

        memset(blk, 0, sizeof blk);          /* Q6_K: d = 1 (last), all scales 1 */
        put_f16(blk + 208, 1.0f);
        for (int g = 0; g < 16; g++) blk[192 + g] = 1;
        for (int i = 0; i < 128; i++) blk[i] = 0x00;
        for (int i = 0; i < 64; i++) blk[128 + i] = 0xAA;   /* every 2-bit field = 2 -> u = 32 -> value 0 */
        blk[64 + 32 + 7] = 0x50;             /* half 1, byte 32+7: high nibble 5 -> element 128+96+7 */
        tlm_dequantize_row(TLM_DT_Q6_K, blk, y, 256);
        nz = 0;
        for (int i = 0; i < 256; i++) nz += y[i] != 0.0f;
        CHECK_EQ_INT(nz, 1);
        CHECK(y[231] == 5.0f);
    }
    CHECK_EQ_INT(tlm_quantize_row(TLM_DT_Q4_K, x, q, 100), TLM_EINVAL);
    CHECK_EQ_INT(tlm_quantize_row(TLM_DT_FP8E4M3_B128, x, q, 128), TLM_EUNSUP);
    CHECK_EQ_INT(tlm_dequantize_row(TLM_DT_FP8E4M3_B128, q, y, 128), TLM_EUNSUP);
    CHECK_EQ_INT(tlm_quantize_row((tlm_dtype)99, x, q, 32), TLM_EINVAL);
    CHECK_EQ_INT(tlm_dequantize_row(TLM_DT_MXFP4, q, y, 33), TLM_EINVAL);
    free(x); free(y); free(q); free(q2);
}

static void test_act_quant(void) {
    float x[512];
    fill_row(x, 512, 2);
    x[3] = -100.0f;   /* the block extreme: must map to code -127 */
    {   /* Q8_K: positive d = amax/127, codes in [-127, 127], an extreme hits +-127, bsums per 16 */
        uint8_t buf[2 * 292];
        tlm_quantize_act(TLM_DT_Q8_K, x, 512, buf);
        for (int b = 0; b < 2; b++) {
            const uint8_t *blk = buf + 292 * b;
            float d = get_f32(blk);
            const int8_t *q = (const int8_t *)(blk + 4);
            float amax = 0;
            for (int i = 0; i < 256; i++) amax = fmaxf(amax, fabsf(x[256 * b + i]));
            CHECK(d == amax / 127.0f);
            CHECK(d > 0);
            int hit = 0, out = 0;
            for (int i = 0; i < 256; i++) { out += q[i] < -127; hit |= (q[i] == 127 || q[i] == -127); }
            CHECK_EQ_INT(out, 0);
            CHECK(hit);
            for (int g = 0; g < 16; g++) {
                int s = 0;
                int16_t bs;
                for (int i = 0; i < 16; i++) s += q[16 * g + i];
                memcpy(&bs, blk + 260 + 2 * g, 2);
                CHECK_EQ_INT(bs, s);
            }
        }
        CHECK((int8_t)buf[4 + 3] == -127);
        /* the activation layout equals the Q8_K weight quantiser output */
        uint8_t wbuf[2 * 292];
        CHECK(tlm_quantize_row(TLM_DT_Q8_K, x, wbuf, 512) == 0);
        CHECK(memcmp(buf, wbuf, sizeof buf) == 0);
    }
    {   /* Q8_0 activation (36-byte blocks, f32 d) */
        uint8_t buf[16 * 36];
        tlm_quantize_act(TLM_DT_Q8_0, x, 512, buf);
        float d = get_f32(buf + 36);
        float amax = 0;
        for (int i = 32; i < 64; i++) amax = fmaxf(amax, fabsf(x[i]));
        CHECK(d == amax / 127.0f);
        /* zero block and zero-padded partial block */
        float z[40] = { 0 };
        z[33] = 1.0f;
        uint8_t zb[2 * 36];
        memset(zb, 0x55, sizeof zb);
        tlm_quantize_act(TLM_DT_Q8_0, z, 34, zb);
        CHECK(get_f32(zb) == 0.0f);
        CHECK((int8_t)zb[4 + 5] == 0);
        CHECK(get_f32(zb + 36) == 1.0f / 127.0f);
        CHECK((int8_t)zb[36 + 4 + 1] == 127);
        CHECK((int8_t)zb[36 + 4 + 2] == 0 && (int8_t)zb[36 + 4 + 31] == 0);
    }
    {   /* Q8_K partial block: tail codes and their group sums are zero */
        uint8_t kb[292];
        memset(kb, 0x55, sizeof kb);
        tlm_quantize_act(TLM_DT_Q8_K, x, 100, kb);
        int nzq = 0;
        for (int i = 100; i < 256; i++) nzq += kb[4 + i] != 0;
        CHECK_EQ_INT(nzq, 0);
        int16_t bs;
        memcpy(&bs, kb + 260 + 2 * 15, 2);
        CHECK_EQ_INT(bs, 0);
    }
    {   /* non-finite input poisons only its own block */
        float v[512];
        for (int i = 0; i < 512; i++) v[i] = gauss();
        v[10] = NAN;
        v[300] = INFINITY;
        uint8_t kb[2 * 292], ab[16 * 36];
        tlm_quantize_act(TLM_DT_Q8_K, v, 512, kb);
        CHECK(isnan(get_f32(kb)));
        CHECK(get_f32(kb + 292) == INFINITY);
        tlm_quantize_act(TLM_DT_Q8_0, v, 512, ab);
        CHECK(isnan(get_f32(ab)));
        CHECK(isfinite(get_f32(ab + 36)) && get_f32(ab + 36) > 0);
        CHECK(get_f32(ab + 9 * 36) == INFINITY);
        /* and the dot product of a finite weight row with it is NaN on every tier */
        float w[512];
        uint8_t wq[2 * 292];
        fill_row(w, 512, 0);
        tlm_quantize_row(TLM_DT_Q8_K, w, wq, 512);
        for (int t = 0; t < 3; t++) {
            if (!tier_ok(t)) continue;
            use_tier(t);
            CHECK(isnan(tlm_vec_dot(TLM_DT_Q8_K, wq, kb, 512)));
        }
        use_best_tier();
    }
    {   /* tiny blocks (127/amax overflows float) still quantise sensibly */
        float v[32];
        for (int i = 0; i < 32; i++) v[i] = (float)(i - 16) * 0x1p-140f;   /* subnormal values */
        uint8_t ab[36];
        tlm_quantize_act(TLM_DT_Q8_0, v, 32, ab);
        CHECK((int8_t)ab[4 + 0] == -127);
        CHECK((int8_t)ab[4 + 16] == 0);
        CHECK((int8_t)ab[4 + 24] == 64);   /* 8/16 * 127 = 63.5 -> ties to even 64 */
    }
    {   /* F32 activation kind is a copy */
        float c[7];
        tlm_quantize_act(TLM_DT_F32, x, 7, c);
        CHECK(memcmp(c, x, sizeof c) == 0);
    }
}

/* ---------------------------------------------------------------------- */
/* Float reference: dequantised weights . dequantised activations, in double. */
static double ref_dot(tlm_dtype t, const void *w, const void *act, int64_t n, double *sumabs) {
    float *wd = malloc(n * 4), *xd = malloc(n * 4);
    tlm_dequantize_row(t, w, wd, n);
    tlm_dtype ak = tlm_act_kind_for(t);
    if (ak == TLM_DT_Q8_K) tlm_dequantize_row(TLM_DT_Q8_K, act, xd, n);
    else if (ak == TLM_DT_Q8_0) {
        const uint8_t *a = act;
        for (int64_t b = 0; b < n / 32; b++) {
            float d = get_f32(a + 36 * b);
            for (int i = 0; i < 32; i++) xd[32 * b + i] = d * (float)(int8_t)a[36 * b + 4 + i];
        }
    } else memcpy(xd, act, n * 4);
    double s = 0, sa = 0;
    for (int64_t i = 0; i < n; i++) { s += (double)wd[i] * xd[i]; sa += fabs((double)wd[i] * xd[i]); }
    free(wd); free(xd);
    *sumabs = sa;
    return s;
}

static void test_dot_vs_ref(void) {
    static const tlm_dtype types[] = { TLM_DT_Q8_0, TLM_DT_Q4_K, TLM_DT_Q5_K, TLM_DT_Q6_K, TLM_DT_Q8_K,
                                       TLM_DT_MXFP4, TLM_DT_F32, TLM_DT_F16, TLM_DT_BF16 };
    const int64_t n = 2048;
    float *x = malloc(n * 4), *wf = malloc(n * 4);
    uint8_t *w = malloc(n * 4), *act = malloc(tlm_qact_bytes(TLM_DT_Q8_K, n) + tlm_qact_bytes(TLM_DT_F32, n));
    for (int ti = 0; ti < 3; ti++) {
        if (!tier_ok(ti)) continue;
        use_tier(ti);
        CHECK(strcmp(tlm_quant_tier(), k_tiers[ti]) == 0);
        for (size_t k = 0; k < sizeof types / sizeof types[0]; k++) {
            double worst = 0;
            for (int rep = 0; rep < 20; rep++) {
                fill_row(wf, n, rep % 4);
                fill_row(x, n, (rep + 1) % 3);
                tlm_quantize_row(types[k], wf, w, n);
                tlm_quantize_act(tlm_act_kind_for(types[k]), x, n, act);
                double sa, ref = ref_dot(types[k], w, act, n, &sa);
                float got = tlm_vec_dot(types[k], w, act, n);
                double err = fabs((double)got - ref) / (sa + 1e-30);
                if (err > worst) worst = err;
                CHECK(err < 2e-6);
            }
            if (ti == 0) printf("  dot %-5s vs double reference: worst |err|/sum|w*x| = %.2e\n",
                                tlm_dtype_name(types[k]), worst);
        }
        /* FP8 with 128x128 block scales vs a double reference; 260 rows so
         * rows 128.. and 256.. use the second / third scale row. */
        {
            int64_t rows = 260, cols = 300, ncb = 3;
            uint8_t *fw = malloc(rows * cols);
            float sc[9];
            for (int i = 0; i < 9; i++) sc[i] = ldexpf(1.0f, (int)(rnd32() % 9) - 4) * tlm_test_randf(0.5f, 1.0f);
            for (int64_t i = 0; i < rows * cols; i++) fw[i] = tlm_f32_to_e4m3(gauss() * 50.0f);
            fill_row(x, cols, 0);
            float *y = malloc(rows * 4);
            tlm_matvec_rows(TLM_DT_FP8E4M3_B128, fw, cols, x, y, cols, 0, rows, sc);
            int bad = 0, badbits = 0;
            for (int64_t r = 0; r < rows; r++) {
                double ref = 0, sa = 0;
                for (int64_t c = 0; c < cols; c++) {
                    double v = (double)tlm_e4m3_to_f32(fw[r * cols + c]) * sc[(r / 128) * ncb + c / 128] * x[c];
                    ref += v; sa += fabs(v);
                }
                if (!(fabs(y[r] - ref) <= 1e-6 * sa)) bad++;
                float y1 = tlm_vec_dot_fp8(fw + r * cols, x, cols, sc + (r / 128) * ncb);
                if (!same_bits(y1, y[r])) badbits++;
            }
            CHECK_EQ_INT(bad, 0);
            CHECK_EQ_INT(badbits, 0);
            fw[5] = 0x7F;   /* NaN code poisons that row only */
            tlm_matvec_rows(TLM_DT_FP8E4M3_B128, fw, cols, x, y, cols, 0, 2, sc);
            CHECK(y[0] != y[0]);
            CHECK(y[1] == y[1]);
            float deq[300];
            CHECK(tlm_dequantize_fp8_row(fw + cols, deq, cols, sc) == 0);
            CHECK(deq[200] == tlm_e4m3_to_f32(fw[cols + 200]) * sc[1]);
            /* NULL scales = 1.0 */
            float y0 = tlm_vec_dot_fp8(fw + cols, x, cols, NULL);
            double ref0 = 0;
            for (int64_t c = 0; c < cols; c++) ref0 += (double)tlm_e4m3_to_f32(fw[cols + c]) * x[c];
            CHECK(fabs(y0 - ref0) <= 1e-5 * (fabs(ref0) + 1));
            free(fw); free(y);
        }
    }
    use_best_tier();
    free(x); free(wf); free(w); free(act);
}

/* ---------------------------------------------------------------------- */
/* Weight rows with random code bits and finite scale fields, plus the edge
 * patterns named by mode:
 *   0 random, 1 all-zero, 2 max magnitude, 3 quantised from floats,
 *   4 random codes with extreme scales (f16 max / min subnormal, E8M0 0/1/254/255). */
static void make_wrow(tlm_dtype t, uint8_t *w, int64_t n, int mode) {
    size_t rb = tlm_row_bytes(t, n);
    int64_t nb = n / tlm_dtype_block(t);
    if (mode == 3) {
        float *x = malloc(n * 4);
        fill_row(x, n, (int)(rnd32() % 4));
        tlm_quantize_row(t, x, w, n);
        free(x);
        return;
    }
    for (size_t i = 0; i < rb; i++) w[i] = mode == 1 ? 0 : (uint8_t)rnd32();
    for (int64_t b = 0; b < nb; b++) {
        float dv = mode == 4 ? ((rnd32() & 1) ? 65504.0f : 0x1p-24f) * ((rnd32() & 2) ? -1.0f : 1.0f)
                             : tlm_test_randf(-0.05f, 0.05f);
        if (mode == 1) dv = 0;
        switch (t) {
        case TLM_DT_Q8_0: {
            uint8_t *blk = w + 34 * b;
            put_f16(blk, mode == 2 ? 0.01f : dv);
            if (mode == 2) for (int i = 0; i < 32; i++) blk[2 + i] = (rnd32() & 1) ? 0x80 : 0x7F;   /* -128 / 127 */
            break;
        }
        case TLM_DT_Q4_K: case TLM_DT_Q5_K: {
            size_t bb = tlm_dtype_block_bytes(t);
            uint8_t *blk = w + bb * b;
            put_f16(blk, mode == 2 ? 0.01f : dv);
            put_f16(blk + 2, mode == 2 ? 0.01f : (mode == 4 ? dv : tlm_test_randf(0, 0.05f)));
            if (mode == 2) memset(blk + 4, 0xFF, bb - 4);   /* scales/mins 63, codes 15/31 */
            break;
        }
        case TLM_DT_Q6_K: {
            uint8_t *blk = w + 210 * b;
            put_f16(blk + 208, mode == 2 ? 0.01f : dv);
            if (mode == 2) {
                uint8_t fill = (rnd32() & 1) ? 0xFF : 0x00;   /* u = 63 or 0 everywhere */
                memset(blk, fill, 192);
                for (int g = 0; g < 16; g++) blk[192 + g] = (g & 1) ? 0x80 : 0x7F;   /* -128 / 127 */
            }
            break;
        }
        case TLM_DT_Q8_K: {
            uint8_t *blk = w + 292 * b;
            float d = mode == 2 ? 1e30f : mode == 4 ? dv * 1e3f : dv;
            memcpy(blk, &d, 4);
            if (mode == 2) for (int i = 0; i < 256; i++) blk[4 + i] = (rnd32() & 1) ? 0x80 : 0x7F;
            break;
        }
        case TLM_DT_MXFP4: {
            uint8_t *blk = w + 17 * b;
            if (mode == 2) { blk[0] = 254; memset(blk + 1, (rnd32() & 1) ? 0x77 : 0xFF, 16); }   /* +-6 * 2^127 */
            else if (mode == 4) { static const uint8_t ee[4] = { 0, 1, 254, 255 }; blk[0] = ee[rnd32() & 3]; }
            else if (mode != 1) blk[0] = (uint8_t)(100 + rnd32() % 50);
            break;
        }
        default: break;
        }
    }
}

/* Activations: 0 bell, 1 zeros, 2 +-1 only (codes +-127), 3 one spike per 77, 4 one NaN. */
static void make_act(tlm_dtype t, void *act, int64_t n, int mode) {
    float *x = malloc(n * 4);
    for (int64_t i = 0; i < n; i++) {
        switch (mode) {
        case 0: x[i] = gauss(); break;
        case 1: x[i] = 0; break;
        case 2: x[i] = (rnd32() & 1) ? 1.0f : -1.0f; break;
        default: x[i] = (i % 77 == 0) ? 1000.0f : 0.001f * gauss(); break;
        }
    }
    if (mode == 4) x[rnd32() % n] = NAN;
    tlm_quantize_act(tlm_act_kind_for(t), x, n, act);
    free(x);
}

static void test_tiers_bitexact(void) {
    static const int64_t nblocks[] = { 1, 2, 3, 7, 8, 9, 16, 17, 33, 64 };
    const int64_t maxn = 64 * 256;
    uint8_t *w = malloc(maxn * 4 + 64), *act = malloc(tlm_qact_bytes(TLM_DT_Q8_K, maxn) + maxn * 4);
    int ntier = 0;
    for (int ti = 0; ti < 3; ti++) ntier += tier_ok(ti);
    printf("  tiers available here: scalar%s%s\n", tier_ok(1) ? " avx2" : "", tier_ok(2) ? " avx512" : "");
    if (ntier < 3) printf("  NOTE: not every SIMD tier can run on this CPU; those comparisons are skipped\n");

    long compared = 0, mismatches = 0, nan_results = 0;
    for (int k = 0; k < N_INT; k++) {
        tlm_dtype t = k_int_types[k];
        int qk = tlm_dtype_block(t);
        for (size_t nbi = 0; nbi < sizeof nblocks / sizeof nblocks[0]; nbi++) {
            int64_t n = nblocks[nbi] * qk;
            for (int rep = 0; rep < 60; rep++) {
                int wmode = rep < 5 ? rep : (rep % 7 == 0 ? 2 : (rep % 11 == 0 ? 4 : (rep % 5 == 0 ? 3 : 0)));
                int amode = rep % 13 == 1 ? 1 : rep % 13 == 2 ? 2 : rep % 13 == 3 ? 3 : rep % 29 == 4 ? 4 : 0;
                if (rep < 5) amode = rep % 3 == 0 ? 2 : 0;
                make_wrow(t, w, n, wmode);
                make_act(t, act, n, amode);
                /* sprinkle an all-zero block into some random rows */
                if (wmode == 0 && (rep & 1)) {
                    size_t bb = tlm_dtype_block_bytes(t);
                    memset(w + bb * (rnd32() % nblocks[nbi]), 0, bb);
                }
                float r[3] = { 0, 0, 0 };
                for (int ti = 0; ti < 3; ti++) {
                    if (!tier_ok(ti)) continue;
                    use_tier(ti);
                    r[ti] = tlm_vec_dot(t, w, act, n);
                }
                nan_results += r[0] != r[0];
                for (int ti = 1; ti < 3; ti++) {
                    if (!tier_ok(ti)) continue;
                    compared++;
                    if (!same_bits(r[0], r[ti])) {
                        mismatches++;
                        if (mismatches < 10)
                            fprintf(stderr, "  MISMATCH %s n=%lld wmode=%d amode=%d: scalar %.9g (%08x) %s %.9g (%08x)\n",
                                    tlm_dtype_name(t), (long long)n, wmode, amode, r[0], f2u(r[0]),
                                    k_tiers[ti], r[ti], f2u(r[ti]));
                    }
                }
            }
        }
    }
    /* float paths: same contract (lane i mod 32 + fixed tree), so also exact */
    static const tlm_dtype ftypes[] = { TLM_DT_F32, TLM_DT_F16, TLM_DT_BF16 };
    for (size_t k = 0; k < 3; k++) {
        for (int rep = 0; rep < 120; rep++) {
            int64_t n = rep < 64 ? rep + 1 : (int64_t)(rnd32() % 5000) + 1;
            float *x = (float *)act;
            for (int64_t i = 0; i < n; i++) x[i] = gauss();
            float *wf = malloc(n * 4);
            fill_row(wf, n, rep % 4);
            if (rep == 70) wf[n / 2] = INFINITY;
            tlm_quantize_row(ftypes[k], wf, w, n);
            free(wf);
            float r[3] = { 0, 0, 0 };
            for (int ti = 0; ti < 3; ti++) {
                if (!tier_ok(ti)) continue;
                use_tier(ti);
                r[ti] = tlm_vec_dot(ftypes[k], w, x, n);
            }
            for (int ti = 1; ti < 3; ti++) {
                if (!tier_ok(ti)) continue;
                compared++;
                if (!same_bits(r[0], r[ti])) {
                    mismatches++;
                    fprintf(stderr, "  MISMATCH %s n=%lld: scalar %.9g %s %.9g\n", tlm_dtype_name(ftypes[k]),
                            (long long)n, r[0], k_tiers[ti], r[ti]);
                }
            }
        }
    }
    /* FP8 rows with scales, odd widths, occasional NaN codes and subnormal codes */
    for (int rep = 0; rep < 60; rep++) {
        int64_t n = (int64_t)(rnd32() % 1000) + 1, ncb = (n + 127) / 128;
        float *x = (float *)act, sc[8];
        for (int64_t i = 0; i < n; i++) {
            x[i] = gauss();
            w[i] = (uint8_t)rnd32();
            if ((w[i] & 0x7F) == 0x7F && rep % 3) w[i] = (uint8_t)(w[i] & 0x87);   /* keep NaN in 1/3 of rows */
        }
        for (int j = 0; j < ncb; j++) sc[j] = tlm_test_randf(0.001f, 2.0f);
        float r[3] = { 0, 0, 0 };
        for (int ti = 0; ti < 3; ti++) {
            if (!tier_ok(ti)) continue;
            use_tier(ti);
            r[ti] = tlm_vec_dot_fp8(w, x, n, (rep & 1) ? sc : NULL);
        }
        for (int ti = 1; ti < 3; ti++) {
            if (!tier_ok(ti)) continue;
            compared++;
            if (!same_bits(r[0], r[ti])) { mismatches++; fprintf(stderr, "  MISMATCH fp8 n=%lld\n", (long long)n); }
        }
    }
    printf("  bit-exact single-row comparisons: %ld, mismatches: %ld (NaN results among them: %ld)\n",
           compared, mismatches, nan_results);
    CHECK_EQ_INT(mismatches, 0);
    CHECK(nan_results > 0);   /* the NaN edge cases were really exercised */
    use_best_tier();
    free(w); free(act);
}

/* matvec (8-row tiles + tails) == per-row scalar dot, bit for bit, on every
 * tier; absolute row indexing, padded row stride, unaligned base pointer. */
static void test_matvec_tiles(void) {
    static const struct { int64_t r0, r1; } ranges[] = {
        { 0, 1 }, { 0, 7 }, { 0, 8 }, { 3, 11 }, { 3, 30 }, { 0, 40 }, { 5, 45 }, { 13, 13 },
    };
    const int64_t rows = 48;
    long compared = 0, mismatches = 0;
    for (int k = 0; k < N_INT; k++) {
        tlm_dtype t = k_int_types[k];
        for (int shape = 0; shape < 3; shape++) {
            int64_t n = (shape == 0 ? 1 : shape == 1 ? 3 : 16) * tlm_dtype_block(t);
            size_t tight = tlm_row_bytes(t, n);
            size_t rb = tight + (shape == 1 ? 6 : 0);          /* padded stride */
            size_t off = shape == 2 ? 3 : 0;                    /* unaligned base */
            uint8_t *mem = malloc(rb * rows + off + 64);
            uint8_t *W = mem + off;
            memset(mem, 0xA5, rb * rows + off + 64);
            for (int64_t r = 0; r < rows; r++) make_wrow(t, W + rb * r, n, (int)(r % 5));
            uint8_t *act = malloc(tlm_qact_bytes(TLM_DT_Q8_K, n) + 64);
            make_act(t, act, n, shape == 2 ? 2 : 0);
            float ref[48];
            use_tier(0);
            for (int64_t r = 0; r < rows; r++) ref[r] = tlm_vec_dot(t, W + rb * r, act, n);
            for (int ti = 0; ti < 3; ti++) {
                if (!tier_ok(ti)) continue;
                use_tier(ti);
                for (size_t q = 0; q < sizeof ranges / sizeof ranges[0]; q++) {
                    float y[48];
                    for (int64_t r = 0; r < rows; r++) y[r] = -12345.0f;
                    tlm_matvec_rows(t, W, rb, act, y, n, ranges[q].r0, ranges[q].r1, NULL);
                    for (int64_t r = 0; r < rows; r++) {
                        int inside = r >= ranges[q].r0 && r < ranges[q].r1;
                        if (!inside) { CHECK(y[r] == -12345.0f); continue; }
                        compared++;
                        if (!same_bits(y[r], ref[r])) {
                            mismatches++;
                            if (mismatches < 10)
                                fprintf(stderr, "  MV MISMATCH %s %s n=%lld row %lld: %.9g vs scalar %.9g\n",
                                        k_tiers[ti], tlm_dtype_name(t), (long long)n, (long long)r, y[r], ref[r]);
                        }
                    }
                }
            }
            free(mem); free(act);
        }
    }
    printf("  bit-exact matvec rows vs scalar: %ld, mismatches: %ld\n", compared, mismatches);
    CHECK_EQ_INT(mismatches, 0);
    use_best_tier();
}

static void test_api_errors(void) {
    float x[64] = { 0 };
    uint8_t buf[64] = { 0 };
    CHECK(isnan(tlm_vec_dot((tlm_dtype)99, buf, x, 32)));
    CHECK(isnan(tlm_vec_dot(TLM_DT_Q8_0, buf, x, 33)));     /* not a block multiple */
    CHECK(isnan(tlm_vec_dot_fp8(buf, x, -1, NULL)));
    float y[4] = { 1, 1, 1, 1 };
    tlm_matvec_rows(TLM_DT_Q4_K, buf, 144, x, y, 100, 1, 3, NULL);
    CHECK(isnan(y[1]) && isnan(y[2]) && y[0] == 1 && y[3] == 1);
    CHECK(tlm_vec_dot(TLM_DT_F32, buf, x, 0) == 0.0f);      /* empty row */
    CHECK(tlm_dequantize_fp8_row(buf, x, -1, NULL) == TLM_EINVAL);
    /* TIERLM_KERNEL-style forcing downgrades the selected tier */
    tlm_cpu_force("scalar");
    tlm_quant_init();
    CHECK(strcmp(tlm_quant_tier(), "scalar") == 0);
    if (tier_ok(1)) {
        tlm_cpu_force("avx2");
        tlm_quant_init();
        CHECK(strcmp(tlm_quant_tier(), "avx2") == 0);
    }
    use_best_tier();
    CHECK(strcmp(tlm_quant_tier(), tier_ok(2) ? "avx512" : tier_ok(1) ? "avx2" : "scalar") == 0);
    /* No SIMD kernel may have been demoted by the start-up self-test:
     * otherwise the tier comparisons above would compare a tier against
     * its silent fallback. */
    for (int t = 0; t < 3; t++) if (tier_ok(t)) use_tier(t);
    CHECK_EQ_INT(tlm_quant_selftest_failures(), 0);
    use_best_tier();
}

/* ---------------------------------------------------------------------- */
static void bench_one(tlm_dtype t, int64_t rows, int64_t cols, double budget_s) {
    size_t rb = tlm_row_bytes(t, cols);
    size_t bytes = rb * (size_t)rows;
    uint8_t *W = tlm_alloc_aligned(bytes, 64);
    void *act = tlm_alloc_aligned(tlm_qact_bytes(TLM_DT_Q8_K, cols) + cols * 4, 64);
    float *y = tlm_alloc_aligned(rows * 4, 64);
    if (!W || !act || !y) { tlm_free_aligned(W); tlm_free_aligned(act); tlm_free_aligned(y); return; }
    for (int64_t r = 0; r < rows; r++) make_wrow(t, W + rb * r, cols, 0);   /* speed is value-independent */
    make_act(t, act, cols, 0);
    printf("  bench %-5s %lldx%lld (%.1f MB):", tlm_dtype_name(t), (long long)rows, (long long)cols, bytes / 1e6);
    for (int ti = 0; ti < 3; ti++) {
        if (!tier_ok(ti)) { printf("  %s n/a", k_tiers[ti]); continue; }
        use_tier(ti);
        tlm_matvec_rows(t, W, rb, act, y, cols, 0, rows, NULL);   /* warm */
        double best = 1e30, spent = 0;
        int it = 0;
        while ((spent < budget_s && it < 400) || it < 2) {
            uint64_t t0 = tlm_now_ns();
            tlm_matvec_rows(t, W, rb, act, y, cols, 0, rows, NULL);
            double dt = (double)(tlm_now_ns() - t0) * 1e-9;
            if (dt < best) best = dt;
            spent += dt;
            it++;
        }
        printf("  %s %.2f GB/s", k_tiers[ti], (double)bytes / best / 1e9);
    }
    printf("\n");
    tlm_free_aligned(W); tlm_free_aligned(act); tlm_free_aligned(y);
}

static void bench_matvec(void) {
    printf("  single-thread tlm_matvec_rows, GB/s of weight bytes (best of N; matrices this size\n"
           "  may sit in a large L3, so these are kernel, not DRAM, numbers):\n");
    bench_one(TLM_DT_Q6_K, 4096, 4096, 0.5);
    bench_one(TLM_DT_Q4_K, 4096, 4096, 0.25);
    bench_one(TLM_DT_Q5_K, 4096, 4096, 0.25);
    bench_one(TLM_DT_Q8_0, 4096, 4096, 0.25);
    bench_one(TLM_DT_Q8_K, 4096, 4096, 0.25);
    bench_one(TLM_DT_MXFP4, 4096, 4096, 0.25);
    bench_one(TLM_DT_BF16, 2048, 4096, 0.25);
    use_best_tier();
}

int main(void) {
    tlm_quant_init();
    printf("  kernel tier: %s\n", tlm_quant_tier());
    RUN(test_layout);
    RUN(test_f16);
    RUN(test_bf16);
    RUN(test_e4m3_e8m0);
    RUN(test_quant_roundtrip);
    RUN(test_act_quant);
    RUN(test_dot_vs_ref);
    RUN(test_tiers_bitexact);
    RUN(test_matvec_tiles);
    RUN(test_api_errors);
    const char *nb = getenv("TLM_TEST_NO_BENCH");
    if (!nb || !*nb || *nb == '0') RUN(bench_matvec);
    TEST_MAIN_END();
}
