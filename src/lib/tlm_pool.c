/* tlm_pool.c - fixed worker pool with a deterministic static parallel-for.
 *
 * Patterns: P-23 persistent workers (threads created once, parked between
 * jobs), P-43 fixed reduction order (the split of [0, n) into chunks is a
 * pure function of (n, grain, pool size); every chunk is processed exactly
 * once, by one thread, so no float is ever summed across threads), CCD/L3
 * grouping for pinning.
 *
 * Dispatch. A job is published under a sequence lock (seq odd while the
 * caller writes the job fields, even when stable; job generation = seq/2).
 * Chunk i "belongs" to worker i, but chunks are claimed with a CAS on a
 * per-chunk word holding the last generation that claimed it. The caller
 * runs chunk 0, then claims any chunk whose worker has not claimed it yet
 * (a descheduled or still-sleeping worker never stalls the job), then waits
 * until every chunk has finished. A worker that wakes late finds its chunk
 * already claimed and just goes back to waiting. The `worker` argument passed
 * to fn is the chunk index, so per-index scratch stays exclusive whoever runs
 * it. Results never depend on which thread ran a chunk.
 *
 * Why claims are safe: a job is complete only when all its chunks have been
 * claimed and finished, so a successful claim for generation g proves job g
 * is still live (its ctx valid); the seqlock proves the fields read belong to
 * g. Job fields are atomics so racing reads are well defined.
 *
 * Calls from inside a pool thread (nested parallel-for) run serially on that
 * thread; concurrent callers from different threads are serialised.
 *
 * Pinning (pin != 0): workers 1..n-1 are pinned one per physical core, cores
 * listed L3 group by L3 group (CCD order), so neighbouring chunks share an
 * L3. The calling thread is deliberately NOT pinned: threads it creates
 * later (I/O workers, other pools) would inherit a one-CPU mask. Pinning
 * failures (containers, cpusets) are tolerated and only logged.
 */
#include "tlm_pool.h"
#include "tlm_platform.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TLM_POOL_MAX     256
#define TLM_POOL_SPIN    4000

struct tlm_pool;

typedef struct {
    struct tlm_pool *pool;
    int              id;      /* worker index, 1..n-1 (0 is the caller) */
    int              cpu;     /* pinned CPU or -1 */
} tlm_worker;

struct tlm_pool {
    int              n;               /* participants including the caller */
    pthread_t       *threads;
    tlm_worker      *workers;
    pthread_mutex_t  mu;
    pthread_cond_t   cv_start, cv_done;
    pthread_mutex_t  call_mu;         /* serialises concurrent callers */
    _Atomic uint64_t seq;             /* odd while a job is being written */
    _Atomic int      stop;
    _Atomic int      done;            /* chunks finished in the current job */
    _Atomic int      sleepers;        /* workers blocked on cv_start */
    _Atomic int      caller_waits;    /* caller blocked on cv_done */
    /* current job (written by the caller between the two seq bumps) */
    _Atomic(uintptr_t) fn, ctx;
    _Atomic int64_t  n_items, grain, units;
    _Atomic int      nchunks;
    _Atomic uint64_t claim[TLM_POOL_MAX];   /* generation that last claimed chunk i */
};

static _Thread_local const tlm_pool *t_in_pool = NULL;

static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

/* Chunk i of the static partition. Boundaries are multiples of `grain`
 * (except the end of the range); chunk sizes differ by at most one grain. */
static void chunk_range(int64_t n, int64_t grain, int64_t units, int nchunks, int i,
                        int64_t *b, int64_t *e) {
    int64_t ub = units * i / nchunks, ue = units * (i + 1) / nchunks;
    int64_t lo = ub * grain, hi = ue * grain;
    if (lo > n) lo = n;
    if (hi > n) hi = n;
    *b = lo;
    *e = hi;
}

typedef struct {
    tlm_range_fn fn;
    void        *ctx;
    int64_t      n, grain, units;
    int          nchunks;
    uint64_t     gen;
} job_view;

/* Seqlock read of the current job; returns 0 if a write raced (retry). */
static int read_job(tlm_pool *p, job_view *j) {
    uint64_t s1 = atomic_load_explicit(&p->seq, memory_order_acquire);
    if (s1 & 1) return 0;
    /* acquire loads keep the second seq read after the field reads (no
     * fences: ThreadSanitizer does not model them) */
    j->fn = (tlm_range_fn)atomic_load_explicit(&p->fn, memory_order_acquire);
    j->ctx = (void *)atomic_load_explicit(&p->ctx, memory_order_acquire);
    j->n = atomic_load_explicit(&p->n_items, memory_order_acquire);
    j->grain = atomic_load_explicit(&p->grain, memory_order_acquire);
    j->units = atomic_load_explicit(&p->units, memory_order_acquire);
    j->nchunks = atomic_load_explicit(&p->nchunks, memory_order_acquire);
    uint64_t s2 = atomic_load_explicit(&p->seq, memory_order_relaxed);
    if (s1 != s2) return 0;
    j->gen = s1 / 2;
    return 1;
}

/* Claim chunk i for generation gen; 1 if this thread now owns it. */
static int claim_chunk(tlm_pool *p, int i, uint64_t gen) {
    uint64_t c = atomic_load_explicit(&p->claim[i], memory_order_acquire);
    while (c < gen) {
        if (atomic_compare_exchange_weak_explicit(&p->claim[i], &c, gen, memory_order_acq_rel,
                                                  memory_order_acquire))
            return 1;
    }
    return 0;
}

static void finish_chunk(tlm_pool *p, int nchunks) {
    if (atomic_fetch_add_explicit(&p->done, 1, memory_order_seq_cst) + 1 == nchunks &&
        atomic_load_explicit(&p->caller_waits, memory_order_seq_cst)) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv_done);
        pthread_mutex_unlock(&p->mu);
    }
}

static void run_chunk(const job_view *j, int i) {
    int64_t b, e;
    chunk_range(j->n, j->grain, j->units, j->nchunks, i, &b, &e);
    if (e > b) j->fn(j->ctx, b, e, i);
}

static void *worker_main(void *arg) {
    tlm_worker *w = (tlm_worker *)arg;
    tlm_pool *p = w->pool;
    t_in_pool = p;
    if (w->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(w->cpu, &set);
        if (sched_setaffinity(0, sizeof set, &set) != 0)
            tlm_log(TLM_LOG_DEBUG, "pool: pinning worker %d to cpu %d failed", w->id, w->cpu);
    }
    uint64_t seen = 0;   /* generation 0 = no job yet */
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&p->seq, memory_order_acquire) / 2 <= seen &&
               !atomic_load_explicit(&p->stop, memory_order_acquire)) {
            if (spins < TLM_POOL_SPIN) {
                cpu_relax();
                spins++;
                continue;
            }
            pthread_mutex_lock(&p->mu);
            atomic_fetch_add_explicit(&p->sleepers, 1, memory_order_seq_cst);
            while (atomic_load_explicit(&p->seq, memory_order_seq_cst) / 2 <= seen &&
                   !atomic_load_explicit(&p->stop, memory_order_acquire))
                pthread_cond_wait(&p->cv_start, &p->mu);
            atomic_fetch_sub_explicit(&p->sleepers, 1, memory_order_seq_cst);
            pthread_mutex_unlock(&p->mu);
        }
        if (atomic_load_explicit(&p->stop, memory_order_acquire)) break;
        job_view j;
        if (!read_job(p, &j)) { cpu_relax(); continue; }   /* a newer job is being written */
        seen = j.gen;
        if (w->id < j.nchunks && claim_chunk(p, w->id, j.gen)) {
            run_chunk(&j, w->id);
            finish_chunk(p, j.nchunks);
        }
    }
    return NULL;
}

/* Build a CPU order for pinning: one logical CPU per physical core,
 * grouped by L3 domain (CCD), groups in order of their first CPU. CPUs
 * outside the process's current affinity mask (cpusets) are skipped. */
static int pin_order(int *cpus, int cap) {
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu <= 0) return 0;
    if (ncpu > 1024) ncpu = 1024;
    char (*l3)[64] = (char (*)[64])calloc((size_t)ncpu, 64);
    int *primary = (int *)calloc((size_t)ncpu, sizeof(int));
    int *done = (int *)calloc((size_t)ncpu, sizeof(int));
    if (!l3 || !primary || !done) { free(l3); free(primary); free(done); return 0; }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    int have_mask = sched_getaffinity(0, sizeof allowed, &allowed) == 0;
    int n = 0;
    for (int c = 0; c < ncpu; c++) {
        char path[128], buf[64] = {0};
        primary[c] = 1;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        FILE *f = fopen(path, "r");
        if (f) {
            if (fgets(buf, sizeof buf, f)) {
                int first = atoi(buf);   /* list starts with the lowest sibling */
                if (first != c) primary[c] = 0;
            }
            fclose(f);
        }
        l3[c][0] = 0;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", c);
        f = fopen(path, "r");
        if (f) {
            if (fgets(l3[c], sizeof l3[c], f)) l3[c][strcspn(l3[c], "\n")] = 0;
            fclose(f);
        }
        if (have_mask && c < CPU_SETSIZE && !CPU_ISSET(c, &allowed)) primary[c] = 0;
    }
    for (int c = 0; c < ncpu && n < cap; c++) {
        if (done[c] || !primary[c]) continue;
        /* emit every primary CPU sharing c's L3 group, in order */
        for (int d = c; d < ncpu && n < cap; d++) {
            if (done[d] || !primary[d] || strcmp(l3[d], l3[c]) != 0) continue;
            done[d] = 1;
            cpus[n++] = d;
        }
    }
    free(l3);
    free(primary);
    free(done);
    return n;
}

tlm_pool *tlm_pool_new(int nthreads, int pin) {
    if (nthreads <= 0) nthreads = tlm_cpu()->physical_cores;
    if (nthreads <= 0) nthreads = 1;
    if (nthreads > TLM_POOL_MAX) nthreads = TLM_POOL_MAX;
    tlm_pool *p = (tlm_pool *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->n = nthreads;
    pthread_mutex_init(&p->mu, NULL);
    pthread_mutex_init(&p->call_mu, NULL);
    pthread_cond_init(&p->cv_start, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    atomic_init(&p->seq, 0);
    atomic_init(&p->stop, 0);
    atomic_init(&p->done, 0);
    atomic_init(&p->sleepers, 0);
    atomic_init(&p->caller_waits, 0);
    atomic_init(&p->fn, 0);
    atomic_init(&p->ctx, 0);
    atomic_init(&p->n_items, 0);
    atomic_init(&p->grain, 1);
    atomic_init(&p->units, 0);
    atomic_init(&p->nchunks, 0);
    for (int i = 0; i < TLM_POOL_MAX; i++) atomic_init(&p->claim[i], 0);

    int cpus[TLM_POOL_MAX];
    int ncpus = pin ? pin_order(cpus, TLM_POOL_MAX) : 0;
    if (pin && ncpus < nthreads)   /* more threads than cores: do not stack workers */
        ncpus = 0;
    if (nthreads > 1) {
        p->threads = (pthread_t *)calloc((size_t)nthreads, sizeof *p->threads);
        p->workers = (tlm_worker *)calloc((size_t)nthreads, sizeof *p->workers);
        if (!p->threads || !p->workers) { tlm_pool_free(p); return NULL; }
        for (int i = 1; i < nthreads; i++) {
            p->workers[i].pool = p;
            p->workers[i].id = i;
            /* worker i takes core i; core 0 is left to the (unpinned) caller */
            p->workers[i].cpu = ncpus > 0 ? cpus[i] : -1;
            if (pthread_create(&p->threads[i], NULL, worker_main, &p->workers[i]) != 0) {
                tlm_log(TLM_LOG_WARN, "pool: could not start worker %d; using %d threads", i, i);
                p->n = i;
                break;
            }
        }
    }
    return p;
}

void tlm_pool_free(tlm_pool *p) {
    if (!p) return;
    if (p->threads) {
        pthread_mutex_lock(&p->mu);
        atomic_store_explicit(&p->stop, 1, memory_order_release);
        pthread_cond_broadcast(&p->cv_start);
        pthread_mutex_unlock(&p->mu);
        for (int i = 1; i < p->n; i++) pthread_join(p->threads[i], NULL);
    }
    pthread_mutex_destroy(&p->mu);
    pthread_mutex_destroy(&p->call_mu);
    pthread_cond_destroy(&p->cv_start);
    pthread_cond_destroy(&p->cv_done);
    free(p->threads);
    free(p->workers);
    free(p);
}

int tlm_pool_size(const tlm_pool *p) { return p ? p->n : 1; }

void tlm_pool_for(tlm_pool *p, int64_t n, int64_t grain, tlm_range_fn fn, void *ctx) {
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    int64_t units = (n + grain - 1) / grain;
    int size = p ? p->n : 1;
    int nchunks = units < size ? (int)units : size;
    if (!p || nchunks <= 1 || t_in_pool == p) {
        /* serial path: same chunk boundaries and indices as the parallel path */
        for (int i = 0; i < nchunks; i++) {
            int64_t b, e;
            chunk_range(n, grain, units, nchunks, i, &b, &e);
            if (e > b) fn(ctx, b, e, i);
        }
        return;
    }
    pthread_mutex_lock(&p->call_mu);
    /* publish (seqlock: odd while writing) */
    uint64_t s = atomic_load_explicit(&p->seq, memory_order_relaxed);
    atomic_store_explicit(&p->seq, s + 1, memory_order_relaxed);
    /* release stores: a reader that sees any new field also sees seq odd */
    atomic_store_explicit(&p->fn, (uintptr_t)fn, memory_order_release);
    atomic_store_explicit(&p->ctx, (uintptr_t)ctx, memory_order_release);
    atomic_store_explicit(&p->n_items, n, memory_order_release);
    atomic_store_explicit(&p->grain, grain, memory_order_release);
    atomic_store_explicit(&p->units, units, memory_order_release);
    atomic_store_explicit(&p->nchunks, nchunks, memory_order_release);
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    atomic_store_explicit(&p->seq, s + 2, memory_order_seq_cst);
    if (atomic_load_explicit(&p->sleepers, memory_order_seq_cst) > 0) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv_start);
        pthread_mutex_unlock(&p->mu);
    }
    job_view j = {fn, ctx, n, grain, units, nchunks, (s + 2) / 2};

    const tlm_pool *prev = t_in_pool;
    t_in_pool = p;
    if (claim_chunk(p, 0, j.gen)) {
        run_chunk(&j, 0);
        finish_chunk(p, nchunks);
    }
    /* help: run every chunk no worker has claimed yet (P-23 without stalls) */
    for (int i = 1; i < nchunks; i++)
        if (claim_chunk(p, i, j.gen)) {
            run_chunk(&j, i);
            finish_chunk(p, nchunks);
        }
    t_in_pool = prev;

    int spins = 0;
    while (atomic_load_explicit(&p->done, memory_order_acquire) < nchunks && spins < TLM_POOL_SPIN) {
        cpu_relax();
        spins++;
    }
    if (atomic_load_explicit(&p->done, memory_order_acquire) < nchunks) {
        pthread_mutex_lock(&p->mu);
        atomic_store_explicit(&p->caller_waits, 1, memory_order_seq_cst);
        while (atomic_load_explicit(&p->done, memory_order_seq_cst) < nchunks)
            pthread_cond_wait(&p->cv_done, &p->mu);
        atomic_store_explicit(&p->caller_waits, 0, memory_order_relaxed);
        pthread_mutex_unlock(&p->mu);
    }
    pthread_mutex_unlock(&p->call_mu);
}
