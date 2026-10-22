/* test_io.c - pread worker pool: concurrent reads, replica choice, error
 * fallback, short reads, drain on close, callbacks that resubmit. */
#include "tlm_io.h"
#include "tlm_platform.h"
#include "tlm_test.h"

#include <ftw.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <unistd.h>

#define FILE_BYTES (4u << 20)
#define SHIFT      8192u          /* file 1 = same bytes as file 0, shifted */
#define SHORT_BYTES (FILE_BYTES / 2)

static char g_root[128];
static char g_f[3][200];
/* TLM_TEST_IO_BACKEND=uring runs the whole suite on io_uring (make URING=1). */
static tlm_io_backend g_backend = TLM_IO_PREAD;          /* 0: full, 1: shifted copy, 2: truncated copy */

static uint8_t pat(uint64_t off) {
    uint64_t x = off * 0x9E3779B97F4A7C15ull;
    x ^= x >> 29;
    return (uint8_t)(x >> 24);
}

static int rm_cb(const char *p, const struct stat *sb, int flag, struct FTW *ftw) {
    (void)sb; (void)flag; (void)ftw;
    return remove(p);
}

static int make_file(const char *path, uint64_t prefix, uint64_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint8_t *b = (uint8_t *)calloc(1, prefix + n);
    for (uint64_t i = 0; i < n; i++) b[prefix + i] = pat(i);
    size_t w = fwrite(b, 1, prefix + n, f);
    free(b);
    fclose(f);
    return w == prefix + n ? 0 : -1;
}

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    int done, errors;
} waiter;

typedef struct {
    waiter    *w;
    tlm_io    *io;
    uint8_t   *mem, *buf;     /* buf = mem + skew */
    uint64_t   off, len;
    int        status;
    atomic_int calls;
    int        chain;         /* remaining resubmits from inside the callback */
} job;

static void waiter_init(waiter *w) {
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cv, NULL);
    w->done = w->errors = 0;
}
static void waiter_destroy(waiter *w) {
    pthread_mutex_destroy(&w->mu);
    pthread_cond_destroy(&w->cv);
}
static void waiter_wait(waiter *w, int n) {
    pthread_mutex_lock(&w->mu);
    while (w->done < n) pthread_cond_wait(&w->cv, &w->mu);
    pthread_mutex_unlock(&w->mu);
}

static void on_done(void *user, int status) {
    job *j = (job *)user;
    j->status = status;
    atomic_fetch_add(&j->calls, 1);
    pthread_mutex_lock(&j->w->mu);
    j->w->done++;
    if (status) j->w->errors++;
    pthread_cond_broadcast(&j->w->cv);
    pthread_mutex_unlock(&j->w->mu);
}

static void on_done_chain(void *user, int status) {
    job *j = (job *)user;
    if (status == 0 && j->chain > 0) {   /* resubmit from the I/O thread */
        j->chain--;
        j->off = (j->off + 4096) % (FILE_BYTES - j->len);
        if (tlm_io_submit(j->io, 0, j->off, -1, 0, j->buf, j->len, on_done_chain, j) == 0) return;
        status = TLM_EIO;
    }
    on_done(user, status);
}

static int check_bytes(const uint8_t *b, uint64_t off, uint64_t len) {
    for (uint64_t i = 0; i < len; i++)
        if (b[i] != pat(off + i)) return 0;
    return 1;
}

static uint64_t rnd(uint64_t n) { return tlm_test_next() % n; }

static tlm_io *open_io(int a, int b, int direct, int threads) {
    const char *p[2] = {g_f[a], g_f[b]};
    tlm_io_opts o = {g_backend, direct, threads, 16};
    return tlm_io_open(p, 2, &o);
}

static void run_concurrent(int direct) {
    tlm_io *io = open_io(0, 1, direct, 4);
    CHECK(io != NULL);
    if (!io) return;
#ifndef HAVE_LIBURING
    CHECK(strcmp(tlm_io_backend_name(io), "pread") == 0);
#endif
    enum { N = 1000 };
    job *jobs = (job *)calloc(N, sizeof *jobs);
    waiter w;
    waiter_init(&w);
    uint64_t want_bytes[2] = {0, 0};
    for (int i = 0; i < N; i++) {
        job *j = &jobs[i];
        j->w = &w;
        int aligned = i % 2 == 0;
        j->len = aligned ? 4096 * (1 + rnd(16)) : 1 + rnd(20000);
        j->off = aligned ? 4096 * rnd((FILE_BYTES - j->len) / 4096) : rnd(FILE_BYTES - j->len);
        j->mem = (uint8_t *)tlm_alloc_aligned((size_t)j->len + 4096, 4096);
        j->buf = j->mem + (aligned ? 0 : 1 + rnd(63));
        atomic_init(&j->calls, 0);
        int file = i % 3 == 0 ? 1 : 0;
        want_bytes[file] += j->len;
        CHECK_EQ_INT(tlm_io_submit(io, file, j->off + (file ? SHIFT : 0), -1, 0, j->buf, j->len,
                                   on_done, j), 0);
    }
    waiter_wait(&w, N);
    CHECK_EQ_INT(w.errors, 0);
    int bad = 0;
    for (int i = 0; i < N; i++) {
        bad += atomic_load(&jobs[i].calls) != 1 || jobs[i].status != 0 ||
               !check_bytes(jobs[i].buf, jobs[i].off, jobs[i].len);
        tlm_free_aligned(jobs[i].mem);
    }
    CHECK_EQ_INT(bad, 0);
    tlm_io_file_stats st0, st1;
    tlm_io_stats(io, 0, &st0);
    tlm_io_stats(io, 1, &st1);
    CHECK_EQ_INT(st0.reads + st1.reads, N);
    CHECK_EQ_INT(st0.bytes, want_bytes[0]);
    CHECK_EQ_INT(st1.bytes, want_bytes[1]);
    CHECK_EQ_INT(st0.queued, 0);
    CHECK_EQ_INT(st1.queued, 0);
    tlm_io_close(io);
    waiter_destroy(&w);
    free(jobs);
}

static void test_concurrent_direct_off(void) { run_concurrent(0); }
static void test_concurrent_direct_on(void) { run_concurrent(1); }

static void test_alt_file_selection(void) {
    tlm_io *io = open_io(0, 1, 1, 1);   /* one worker per file so queues build up */
    CHECK(io != NULL);
    if (!io) return;
    enum { N = 200 };
    job *jobs = (job *)calloc(N, sizeof *jobs);
    waiter w;
    waiter_init(&w);
    for (int i = 0; i < N; i++) {
        job *j = &jobs[i];
        j->w = &w;
        j->len = 65536;
        j->off = 4096 * rnd((FILE_BYTES - j->len) / 4096);
        j->mem = j->buf = (uint8_t *)tlm_alloc_aligned((size_t)j->len, 4096);
        atomic_init(&j->calls, 0);
        CHECK_EQ_INT(tlm_io_submit(io, 0, j->off, 1, j->off + SHIFT, j->buf, j->len, on_done, j), 0);
    }
    waiter_wait(&w, N);
    CHECK_EQ_INT(w.errors, 0);
    int bad = 0;
    for (int i = 0; i < N; i++) {
        bad += !check_bytes(jobs[i].buf, jobs[i].off, jobs[i].len);
        tlm_free_aligned(jobs[i].mem);
    }
    CHECK_EQ_INT(bad, 0);   /* whichever copy served it, bytes are the same */
    tlm_io_file_stats st0, st1;
    tlm_io_stats(io, 0, &st0);
    tlm_io_stats(io, 1, &st1);
    CHECK_EQ_INT(st0.reads + st1.reads, N);
    CHECK(st0.reads > 0);   /* ties go to the primary, so it always serves at least one */

    /* P-31 decision, deterministically: tlm_io_submit counts the new job in
     * the file's load before returning, so while a near-full-file read
     * occupies f0, the next request naming both copies must go to idle f1. */
    waiter_init(&w);
    job ja, jb;
    memset(&ja, 0, sizeof ja);
    ja.w = &w;
    ja.len = FILE_BYTES - 8192;
    ja.off = 0;
    ja.mem = ja.buf = (uint8_t *)tlm_alloc_aligned(ja.len, 4096);
    atomic_init(&ja.calls, 0);
    memset(&jb, 0, sizeof jb);
    jb.w = &w;
    jb.len = 4096;
    jb.off = 4096;
    jb.mem = jb.buf = (uint8_t *)tlm_alloc_aligned(4096, 4096);
    atomic_init(&jb.calls, 0);
    CHECK_EQ_INT(tlm_io_submit(io, 0, ja.off, -1, 0, ja.buf, ja.len, on_done, &ja), 0);
    tlm_io_file_stats s0x, s1a, s1b;
    tlm_io_stats(io, 0, &s0x);
    CHECK(s0x.queued > 0);   /* load is counted at submit time, so it holds now */
    tlm_io_stats(io, 1, &s1a);
    CHECK_EQ_INT(tlm_io_submit(io, 0, jb.off, 1, jb.off + SHIFT, jb.buf, jb.len, on_done, &jb), 0);
    waiter_wait(&w, 2);
    CHECK_EQ_INT(w.errors, 0);
    CHECK(check_bytes(ja.buf, ja.off, ja.len));
    CHECK(check_bytes(jb.buf, jb.off, jb.len));
    tlm_io_stats(io, 1, &s1b);
    CHECK_EQ_INT(s1b.reads - s1a.reads, 1);   /* the probe went to the idle copy (P-31) */
    tlm_free_aligned(ja.mem);
    tlm_free_aligned(jb.mem);

    /* alt == file means "no replica". */
    waiter_init(&w);
    job j1;
    memset(&j1, 0, sizeof j1);
    j1.w = &w;
    j1.len = 4096;
    j1.off = 8192;
    j1.mem = j1.buf = (uint8_t *)tlm_alloc_aligned(4096, 4096);
    atomic_init(&j1.calls, 0);
    CHECK_EQ_INT(tlm_io_submit(io, 0, j1.off, 0, 0, j1.buf, j1.len, on_done, &j1), 0);
    waiter_wait(&w, 1);
    CHECK(j1.status == 0 && check_bytes(j1.buf, j1.off, j1.len));
    tlm_free_aligned(j1.mem);
    tlm_io_close(io);
    waiter_destroy(&w);
    free(jobs);
}

static void test_short_read_and_fallback(void) {
    tlm_log_level(TLM_LOG_ERROR);   /* the replica fallback warns by design */
    tlm_io *io = open_io(2, 0, 1, 2);   /* file 0 = truncated, file 1 = full */
    CHECK(io != NULL);
    if (!io) return;
    uint8_t *buf = (uint8_t *)tlm_alloc_aligned(65536, 4096);
    /* Reads that cross EOF are errors, never partial successes (P-30). */
    CHECK_EQ_INT(tlm_io_read_sync(io, 0, SHORT_BYTES - 100, buf, 4096), TLM_EIO);
    CHECK_EQ_INT(tlm_io_read_sync(io, 1, FILE_BYTES - 4096, buf, 8192), TLM_EIO);
    CHECK_EQ_INT(tlm_io_read_sync(io, 1, FILE_BYTES - 4096, buf, 4096), 0);
    CHECK(check_bytes(buf, FILE_BYTES - 4096, 4096));
    CHECK_EQ_INT(tlm_io_read_sync(io, 2, 0, buf, 4096), TLM_EINVAL);

    waiter w;
    waiter_init(&w);
    job j;
    memset(&j, 0, sizeof j);
    j.w = &w;
    j.buf = buf;
    atomic_init(&j.calls, 0);
    CHECK_EQ_INT(tlm_io_submit(io, 0, SHORT_BYTES - 100, -1, 0, buf, 4096, on_done, &j), 0);
    waiter_wait(&w, 1);
    CHECK_EQ_INT(j.status, TLM_EIO);
    CHECK_EQ_INT(atomic_load(&j.calls), 1);

    /* Same read with the full copy as alternative: the idle tie goes to the
     * truncated primary, which fails, and the replica serves it. */
    tlm_io_file_stats before, after;
    tlm_io_stats(io, 1, &before);
    j.off = SHORT_BYTES - 4096;
    memset(buf, 0, 65536);
    CHECK_EQ_INT(tlm_io_submit(io, 0, j.off, 1, j.off, buf, 16384, on_done, &j), 0);
    waiter_wait(&w, 2);
    CHECK_EQ_INT(j.status, 0);
    CHECK(check_bytes(buf, j.off, 16384));
    tlm_io_stats(io, 1, &after);
    CHECK_EQ_INT(after.reads, before.reads + 1);

    /* Argument errors. */
    CHECK_EQ_INT(tlm_io_submit(io, 2, 0, -1, 0, buf, 10, on_done, &j), TLM_EINVAL);
    CHECK_EQ_INT(tlm_io_submit(io, 0, 0, 5, 0, buf, 10, on_done, &j), TLM_EINVAL);
    CHECK_EQ_INT(tlm_io_submit(io, 0, 0, -1, 0, NULL, 10, on_done, &j), TLM_EINVAL);
    tlm_io_close(io);
    waiter_destroy(&w);
    tlm_free_aligned(buf);
    tlm_log_level(TLM_LOG_WARN);

    const char *missing[1] = {"/nonexistent-tlm-test/definitely-not-here.tlmp"};
    CHECK(tlm_io_open(missing, 1, NULL) == NULL);
}

static void test_close_drains(void) {
    tlm_io *io = open_io(0, 1, 0, 2);
    CHECK(io != NULL);
    if (!io) return;
    enum { N = 300 };
    job *jobs = (job *)calloc(N, sizeof *jobs);
    waiter w;
    waiter_init(&w);
    for (int i = 0; i < N; i++) {
        job *j = &jobs[i];
        j->w = &w;
        j->len = 32768;
        j->off = rnd(FILE_BYTES - j->len);
        j->mem = j->buf = (uint8_t *)tlm_alloc_aligned((size_t)j->len, 4096);
        atomic_init(&j->calls, 0);
        CHECK_EQ_INT(tlm_io_submit(io, i % 2, j->off + (i % 2 ? SHIFT : 0), -1, 0, j->buf, j->len,
                                   on_done, j), 0);
    }
    tlm_io_close(io);   /* must run every accepted request before returning */
    CHECK_EQ_INT(w.done, N);
    int bad = 0;
    for (int i = 0; i < N; i++) {
        bad += atomic_load(&jobs[i].calls) != 1 || jobs[i].status != 0 ||
               !check_bytes(jobs[i].buf, jobs[i].off, jobs[i].len);
        tlm_free_aligned(jobs[i].mem);
    }
    CHECK_EQ_INT(bad, 0);
    waiter_destroy(&w);
    free(jobs);
}

static void test_resubmit_from_callback(void) {
    tlm_io *io = open_io(0, 1, 1, 2);
    CHECK(io != NULL);
    if (!io) return;
    enum { N = 8 };
    job jobs[N];
    waiter w;
    waiter_init(&w);
    for (int i = 0; i < N; i++) {
        job *j = &jobs[i];
        memset(j, 0, sizeof *j);
        j->w = &w;
        j->io = io;
        j->len = 8192;
        j->off = 4096 * (uint64_t)i;
        j->chain = 25;
        j->mem = j->buf = (uint8_t *)tlm_alloc_aligned((size_t)j->len, 4096);
        atomic_init(&j->calls, 0);
        CHECK_EQ_INT(tlm_io_submit(io, 0, j->off, -1, 0, j->buf, j->len, on_done_chain, j), 0);
    }
    waiter_wait(&w, N);
    CHECK_EQ_INT(w.errors, 0);
    for (int i = 0; i < N; i++) {
        CHECK_EQ_INT(jobs[i].chain, 0);
        CHECK(check_bytes(jobs[i].buf, jobs[i].off, jobs[i].len));
        tlm_free_aligned(jobs[i].mem);
    }
    tlm_io_file_stats st;
    tlm_io_stats(io, 0, &st);
    CHECK_EQ_INT(st.reads, N * 26);
    tlm_io_close(io);
    waiter_destroy(&w);
}

/* Callbacks keep resubmitting while tlm_io_close runs: each accepted request
 * gets exactly one callback, and once close has begun a resubmit is refused
 * with TLM_EBUSY (and gets no callback). */
typedef struct {
    tlm_io    *io;
    uint8_t   *buf;
    uint64_t   off, last;         /* off = request in flight, last = last completed */
    atomic_int accepted, callbacks, busy, other;
} close_job;

static void on_done_close(void *user, int status) {
    close_job *j = (close_job *)user;
    atomic_fetch_add(&j->callbacks, 1);
    if (status != 0) atomic_fetch_add(&j->other, 1);
    j->last = j->off;
    j->off = (j->off + 4096) % (FILE_BYTES - 4096);
    int rc = tlm_io_submit(j->io, 0, j->off, 1, j->off + SHIFT, j->buf, 4096, on_done_close, j);
    if (rc == 0) atomic_fetch_add(&j->accepted, 1);
    else if (rc == TLM_EBUSY) atomic_fetch_add(&j->busy, 1);
    else atomic_fetch_add(&j->other, 1);
}

static void test_close_refuses_resubmits(void) {
    tlm_io *io = open_io(0, 1, 1, 2);
    CHECK(io != NULL);
    if (!io) return;
    enum { N = 6 };
    close_job jobs[N];
    for (int i = 0; i < N; i++) {
        close_job *j = &jobs[i];
        j->io = io;
        j->buf = (uint8_t *)tlm_alloc_aligned(4096, 4096);
        j->off = 4096 * (uint64_t)i;
        atomic_init(&j->accepted, 1);
        atomic_init(&j->callbacks, 0);
        atomic_init(&j->busy, 0);
        atomic_init(&j->other, 0);
        CHECK_EQ_INT(tlm_io_submit(io, 0, j->off, 1, j->off + SHIFT, j->buf, 4096, on_done_close, j), 0);
    }
    usleep(20000);      /* let the chains run for a while */
    tlm_io_close(io);   /* drains; every chain ends in exactly one refused resubmit */
    for (int i = 0; i < N; i++) {
        close_job *j = &jobs[i];
        CHECK_EQ_INT(atomic_load(&j->other), 0);
        CHECK_EQ_INT(atomic_load(&j->busy), 1);
        CHECK_EQ_INT(atomic_load(&j->callbacks), atomic_load(&j->accepted));
        CHECK(atomic_load(&j->callbacks) > 1);
        CHECK(check_bytes(j->buf, j->last, 4096));   /* same bytes from either copy */
        tlm_free_aligned(j->buf);
    }
}

static void test_uring_request_falls_back(void) {
    const char *p[1] = {g_f[0]};
    tlm_io_opts o = {TLM_IO_URING, 1, 2, 8};
    tlm_io *io = tlm_io_open(p, 1, &o);
    CHECK(io != NULL);
    if (!io) return;
#ifndef HAVE_LIBURING
    CHECK(strcmp(tlm_io_backend_name(io), "pread") == 0);
#endif
    uint8_t *buf = (uint8_t *)tlm_alloc_aligned(8192, 4096);
    waiter w;
    waiter_init(&w);
    job j;
    memset(&j, 0, sizeof j);
    j.w = &w;
    atomic_init(&j.calls, 0);
    CHECK_EQ_INT(tlm_io_submit(io, 0, 12288, -1, 0, buf, 8192, on_done, &j), 0);
    waiter_wait(&w, 1);
    CHECK_EQ_INT(j.status, 0);
    CHECK(check_bytes(buf, 12288, 8192));
    tlm_io_close(io);
    waiter_destroy(&w);
    tlm_free_aligned(buf);
    CHECK(strcmp(tlm_io_backend_name(NULL), "none") == 0);
}

int main(void) {
    snprintf(g_root, sizeof g_root, "%s/tlm-test-io-%ld", tlm_test_tmpbase(), (long)getpid());
    const char *be = getenv("TLM_TEST_IO_BACKEND");
    if (be && strcmp(be, "uring") == 0) {
        g_backend = TLM_IO_URING;
        tlm_log_level(TLM_LOG_INFO);   /* show which backend actually ran */
    }
    nftw(g_root, rm_cb, 16, FTW_DEPTH | FTW_PHYS);
    CHECK_EQ_INT(tlm_mkdirs(g_root, 0700), 0);
    for (int i = 0; i < 3; i++) snprintf(g_f[i], sizeof g_f[i], "%s/f%d.bin", g_root, i);
    CHECK_EQ_INT(make_file(g_f[0], 0, FILE_BYTES), 0);
    CHECK_EQ_INT(make_file(g_f[1], SHIFT, FILE_BYTES), 0);
    CHECK_EQ_INT(make_file(g_f[2], 0, SHORT_BYTES), 0);
    RUN(test_concurrent_direct_off);
    RUN(test_concurrent_direct_on);
    RUN(test_alt_file_selection);
    RUN(test_short_read_and_fallback);
    RUN(test_close_drains);
    RUN(test_resubmit_from_callback);
    RUN(test_close_refuses_resubmits);
    RUN(test_uring_request_falls_back);
    nftw(g_root, rm_cb, 16, FTW_DEPTH | FTW_PHYS);
    TEST_MAIN_END();
}
