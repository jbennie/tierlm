/* tlm_test.h - tiny test harness: CHECK macros and a test runner. */
#ifndef TLM_TEST_H
#define TLM_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tlm_test_failures = 0;
static const char *tlm_test_current = "";

#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "  FAIL %s:%d [%s] %s\n", __FILE__, __LINE__, tlm_test_current, #cond); \
    tlm_test_failures++; } } while (0)
#define CHECK_EQ_INT(a, b) do { long long _a = (long long)(a), _b = (long long)(b); if (_a != _b) { \
    fprintf(stderr, "  FAIL %s:%d [%s] %s == %s (%lld vs %lld)\n", __FILE__, __LINE__, \
            tlm_test_current, #a, #b, _a, _b); tlm_test_failures++; } } while (0)
#define CHECK_NEAR(a, b, tol) do { double _a = (double)(a), _b = (double)(b); \
    if (!(fabs(_a - _b) <= (tol))) { \
    fprintf(stderr, "  FAIL %s:%d [%s] %s ~ %s (%.9g vs %.9g, tol %g)\n", __FILE__, __LINE__, \
            tlm_test_current, #a, #b, _a, _b, (double)(tol)); tlm_test_failures++; } } while (0)

#define RUN(fn) do { tlm_test_current = #fn; fn(); } while (0)
#define TEST_MAIN_END() do { if (tlm_test_failures) { \
    fprintf(stderr, "%d failure(s)\n", tlm_test_failures); return 1; } \
    printf("  ok\n"); return 0; } while (0)

/* deterministic RNG for tests */
static unsigned long long tlm_test_rng = 0x9E3779B97F4A7C15ull;
static inline unsigned long long tlm_test_next(void) {
    tlm_test_rng ^= tlm_test_rng << 13; tlm_test_rng ^= tlm_test_rng >> 7; tlm_test_rng ^= tlm_test_rng << 17;
    return tlm_test_rng;
}
static inline float tlm_test_randf(float lo, float hi) {
    return lo + (hi - lo) * (float)((tlm_test_next() >> 11) * (1.0 / 9007199254740992.0));
}

/* Base directory for test scratch files: $TMPDIR when set and short, else /tmp. */
static inline const char *tlm_test_tmpbase(void) {
    const char *t = getenv("TMPDIR");
    return t && *t && strlen(t) < 80 ? t : "/tmp";
}

#endif
