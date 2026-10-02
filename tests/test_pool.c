/* test_pool.c - worker pool: coverage, deterministic static partition (P-43),
 * caller participation, nested and concurrent calls, start-up race (P-23). */
#include "tlm_platform.h"
#include "tlm_pool.h"
#include "tlm_test.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>

#define MAXN 100000

typedef struct {
    _Atomic int hits[MAXN];
    int64_t     cb[64], ce[64];     /* chunk recorded per worker id */
    _Atomic int calls;
    pthread_t   caller;
    _Atomic int caller_ran;
    _Atomic int bad_worker;
    int         nworkers;
} cover_ctx;

static void cover_fn(void *ctx, int64_t b, int64_t e, int worker) {
    cover_ctx *c = (cover_ctx *)ctx;
    if (worker < 0 || worker >= 64 || worker >= c->nworkers) { atomic_store(&c->bad_worker, 1); return; }
    c->cb[worker] = b;
    c->ce[worker] = e;
    atomic_fetch_add(&c->calls, 1);
    if (worker == 0 && pthread_equal(pthread_self(), c->caller)) atomic_store(&c->caller_ran, 1);
    for (int64_t i = b; i < e; i++) atomic_fetch_add(&c->hits[i], 1);
}

static cover_ctx g_c;

static void reset_ctx(int nworkers) {
    for (int i = 0; i < MAXN; i++) atomic_store(&g_c.hits[i], 0);
    for (int i = 0; i < 64; i++) g_c.cb[i] = g_c.ce[i] = -1;
    atomic_store(&g_c.calls, 0);
    atomic_store(&g_c.caller_ran, 0);
    atomic_store(&g_c.bad_worker, 0);
    g_c.caller = pthread_self();
    g_c.nworkers = nworkers;
}

static int all_once(int64_t n) {
    for (int64_t i = 0; i < n; i++)
        if (atomic_load(&g_c.hits[i]) != 1) return 0;
    for (int64_t i = n; i < MAXN; i++)
        if (atomic_load(&g_c.hits[i]) != 0) return 0;
    return 1;
}

static void test_coverage(void) {
    const int sizes[] = {1, 2, 3, 4, 7, 16};   /* 16 > CPUs here: late workers, caller steals */
    const int64_t ns[] = {1, 2, 5, 16, 17, 100, 1000, 4097, MAXN};
    const int64_t grains[] = {1, 3, 16, 64, 1000};
    for (size_t a = 0; a < sizeof sizes / sizeof sizes[0]; a++) {
        tlm_pool *p = tlm_pool_new(sizes[a], 0);
        CHECK(p != NULL);
        if (!p) continue;
        CHECK_EQ_INT(tlm_pool_size(p), sizes[a]);
        for (size_t b = 0; b < sizeof ns / sizeof ns[0]; b++)
            for (size_t c = 0; c < sizeof grains / sizeof grains[0]; c++) {
                reset_ctx(sizes[a]);
                tlm_pool_for(p, ns[b], grains[c], cover_fn, &g_c);
                CHECK(all_once(ns[b]));
                CHECK(!atomic_load(&g_c.bad_worker));
                /* chunk boundaries are multiples of the grain (except the end) */
                for (int w = 0; w < sizes[a]; w++)
                    if (g_c.cb[w] >= 0) {
                        CHECK(g_c.cb[w] % grains[c] == 0);
                        CHECK(g_c.ce[w] == ns[b] || g_c.ce[w] % grains[c] == 0);
                    }
                int64_t units = (ns[b] + grains[c] - 1) / grains[c];
                int expect = units < sizes[a] ? (int)units : sizes[a];
                CHECK_EQ_INT(atomic_load(&g_c.calls), expect);
                /* the caller runs chunk 0 itself */
                CHECK(atomic_load(&g_c.caller_ran));
            }
        tlm_pool_free(p);
    }
}

/* The split is a pure function of (n, grain, size): repeated runs and the
 * serial path (NULL pool) produce identical chunk tables. */
static void test_deterministic_split(void) {
    tlm_pool *p = tlm_pool_new(4, 0);
    CHECK(p != NULL);
    if (!p) return;
    int64_t ref_b[64], ref_e[64];
    reset_ctx(4);
    tlm_pool_for(p, 1234, 16, cover_fn, &g_c);
    memcpy(ref_b, g_c.cb, sizeof ref_b);
    memcpy(ref_e, g_c.ce, sizeof ref_e);
    for (int rep = 0; rep < 200; rep++) {
        reset_ctx(4);
        tlm_pool_for(p, 1234, 16, cover_fn, &g_c);
        CHECK(memcmp(ref_b, g_c.cb, sizeof ref_b) == 0);
        CHECK(memcmp(ref_e, g_c.ce, sizeof ref_e) == 0);
        CHECK(all_once(1234));
    }
    /* chunk sizes differ by at most one grain */
    int64_t mn = INT64_MAX, mx = 0;
    for (int w = 0; w < 4; w++) {
        int64_t len = ref_e[w] - ref_b[w];
        if (len < mn) mn = len;
        if (len > mx) mx = len;
    }
    CHECK(mx - mn <= 16);
    tlm_pool_free(p);

    reset_ctx(1);
    tlm_pool_for(NULL, 500, 7, cover_fn, &g_c);   /* NULL pool = caller only */
    CHECK(all_once(500));
    CHECK_EQ_INT(atomic_load(&g_c.calls), 1);
    CHECK_EQ_INT(g_c.cb[0], 0);
    CHECK_EQ_INT(g_c.ce[0], 500);

    reset_ctx(1);
    tlm_pool_for(NULL, 0, 7, cover_fn, &g_c);      /* empty range: no calls */
    CHECK_EQ_INT(atomic_load(&g_c.calls), 0);
}

/* A deterministic float reduction: each worker owns whole outputs, so the
 * result is bit-identical for every pool size (P-43). */
typedef struct { const float *x; float *y; int64_t cols; } rows_ctx;
static void rows_fn(void *ctx, int64_t b, int64_t e, int worker) {
    (void)worker;
    rows_ctx *c = (rows_ctx *)ctx;
    for (int64_t r = b; r < e; r++) {
        float s = 0.0f;
        for (int64_t j = 0; j < c->cols; j++) s += c->x[r * c->cols + j] * (float)(j + 1);
        c->y[r] = s;
    }
}

static void test_bit_identical_rows(void) {
    enum { R = 333, C = 77 };
    static float x[R * C], y1[R], yn[R];
    for (int i = 0; i < R * C; i++) x[i] = tlm_test_randf(-1.0f, 1.0f);
    rows_ctx c = {x, y1, C};
    tlm_pool_for(NULL, R, 1, rows_fn, &c);
    for (int t = 2; t <= 5; t++) {
        tlm_pool *p = tlm_pool_new(t, 0);
        CHECK(p != NULL);
        if (!p) continue;
        c.y = yn;
        memset(yn, 0, sizeof yn);
        tlm_pool_for(p, R, 5, rows_fn, &c);
        CHECK(memcmp(y1, yn, sizeof y1) == 0);
        tlm_pool_free(p);
    }
}

/* Nested parallel-for from inside a worker runs serially (no deadlock). */
typedef struct { tlm_pool *p; _Atomic int inner_items; } nest_ctx;
static void inner_fn(void *ctx, int64_t b, int64_t e, int worker) {
    (void)worker;
    nest_ctx *c = (nest_ctx *)ctx;
    atomic_fetch_add(&c->inner_items, (int)(e - b));
}
static void outer_fn(void *ctx, int64_t b, int64_t e, int worker) {
    (void)worker;
    nest_ctx *c = (nest_ctx *)ctx;
    for (int64_t i = b; i < e; i++) tlm_pool_for(c->p, 10, 1, inner_fn, c);
}

static void test_nested(void) {
    tlm_pool *p = tlm_pool_new(3, 0);
    CHECK(p != NULL);
    if (!p) return;
    nest_ctx c;
    c.p = p;
    atomic_store(&c.inner_items, 0);
    tlm_pool_for(p, 9, 1, outer_fn, &c);
    CHECK_EQ_INT(atomic_load(&c.inner_items), 90);
    tlm_pool_free(p);
}

/* Two threads calling the same pool concurrently are serialised. */
typedef struct { tlm_pool *p; _Atomic int sum; } conc_ctx;
static void add_fn(void *ctx, int64_t b, int64_t e, int worker) {
    (void)worker;
    conc_ctx *c = (conc_ctx *)ctx;
    atomic_fetch_add(&c->sum, (int)(e - b));
}
static void *conc_thread(void *arg) {
    conc_ctx *c = (conc_ctx *)arg;
    for (int i = 0; i < 300; i++) tlm_pool_for(c->p, 64, 4, add_fn, c);
    return NULL;
}

static void test_concurrent_callers(void) {
    tlm_pool *p = tlm_pool_new(3, 0);
    CHECK(p != NULL);
    if (!p) return;
    conc_ctx c;
    c.p = p;
    atomic_store(&c.sum, 0);
    pthread_t t[2];
    for (int i = 0; i < 2; i++) pthread_create(&t[i], NULL, conc_thread, &c);
    for (int i = 0; i < 2; i++) pthread_join(t[i], NULL);
    CHECK_EQ_INT(atomic_load(&c.sum), 2 * 300 * 64);
    tlm_pool_free(p);
}

/* A job published immediately after creation (workers possibly not yet
 * scheduled) must still complete; repeated create/use/free cycles. */
static void test_startup_race(void) {
    for (int i = 0; i < 200; i++) {
        tlm_pool *p = tlm_pool_new(2 + (i % 3), 0);
        CHECK(p != NULL);
        if (!p) return;
        conc_ctx c;
        c.p = p;
        atomic_store(&c.sum, 0);
        tlm_pool_for(p, 100, 1, add_fn, &c);
        CHECK_EQ_INT(atomic_load(&c.sum), 100);
        tlm_pool_free(p);
    }
    /* free without ever running a job */
    tlm_pool_free(tlm_pool_new(4, 0));
    tlm_pool_free(NULL);
}

/* Pinning may fail (cpusets, containers) but the pool must still work and
 * the caller's affinity must not be narrowed. */
static void test_pinning(void) {
    cpu_set_t before, after;
    CPU_ZERO(&before);
    CPU_ZERO(&after);
    int have = sched_getaffinity(0, sizeof before, &before) == 0;
    tlm_pool *p = tlm_pool_new(0, 1);   /* physical cores, pinned */
    CHECK(p != NULL);
    if (!p) return;
    CHECK(tlm_pool_size(p) >= 1);
    reset_ctx(tlm_pool_size(p));
    tlm_pool_for(p, 5000, 8, cover_fn, &g_c);
    CHECK(all_once(5000));
    if (have && sched_getaffinity(0, sizeof after, &after) == 0) CHECK(CPU_EQUAL(&before, &after));
    tlm_pool_free(p);
}

/* Oversubscribed pool: results stay bit-identical and every chunk runs once
 * even when most workers are descheduled and the caller runs their chunks. */
static void test_oversubscribed(void) {
    enum { R = 4096, C = 33 };
    static float x[R * C], ya[R], yb[R];
    for (int i = 0; i < R * C; i++) x[i] = tlm_test_randf(-1.0f, 1.0f);
    rows_ctx c = {x, ya, C};
    tlm_pool_for(NULL, R, 7, rows_fn, &c);
    tlm_pool *p = tlm_pool_new(24, 0);
    CHECK(p != NULL);
    if (!p) return;
    for (int rep = 0; rep < 60; rep++) {
        c.y = yb;
        memset(yb, 0, sizeof yb);
        tlm_pool_for(p, R, 7, rows_fn, &c);
        if (memcmp(ya, yb, sizeof ya) != 0) { CHECK(0); break; }
        reset_ctx(24);
        tlm_pool_for(p, 777, 3, cover_fn, &g_c);
        if (!all_once(777)) { CHECK(0); break; }
    }
    tlm_pool_free(p);
}

int main(void) {
    RUN(test_coverage);
    RUN(test_oversubscribed);
    RUN(test_deterministic_split);
    RUN(test_bit_identical_rows);
    RUN(test_nested);
    RUN(test_concurrent_callers);
    RUN(test_startup_race);
    RUN(test_pinning);
    TEST_MAIN_END();
}
