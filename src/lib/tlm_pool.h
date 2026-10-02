/* tlm_pool.h - fixed worker pool and parallel-for.
 *
 * Patterns: P-23 persistent workers, P-43 deterministic reductions (each
 * worker owns whole output rows; no cross-thread float sums), CCD grouping.
 */
#ifndef TLM_POOL_H
#define TLM_POOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tlm_pool tlm_pool;

/* fn(ctx, begin, end, worker_id) processes items [begin, end). */
typedef void (*tlm_range_fn)(void *ctx, int64_t begin, int64_t end, int worker);

/* nthreads <= 0 -> physical cores. pin != 0 pins workers to cores, grouped by L3. */
tlm_pool *tlm_pool_new(int nthreads, int pin);
void      tlm_pool_free(tlm_pool *p);
int       tlm_pool_size(const tlm_pool *p);
/* Split [0, n) into contiguous chunks of at least `grain` items, run on all
 * workers (the caller participates), return when all are done. The split is
 * a pure function of (n, grain, size) so results never depend on timing. */
void      tlm_pool_for(tlm_pool *p, int64_t n, int64_t grain, tlm_range_fn fn, void *ctx);

/* ---- notes (tlm_pool.c) --------------------------------------------------
 * - `worker` passed to fn is the CHUNK index in [0, min(size, ceil(n/grain))):
 *   chunk i normally runs on pool thread i, but the caller runs chunk 0 and
 *   any chunk whose thread has not claimed it yet (a descheduled worker never
 *   stalls a job). Each chunk runs exactly once per call, so per-index
 *   scratch is exclusive and results never depend on which thread ran it.
 * - p == NULL, a single chunk, or a call from inside a pool thread (nested)
 *   runs serially on the caller with the same chunk boundaries.
 * - Concurrent callers from different threads are serialised.
 * - pin != 0 pins workers 1..n-1 one per physical core in L3-group order
 *   (skipping CPUs outside the affinity mask); the caller is never pinned.
 *   Pinning failures are ignored (logged at debug level).
 */

#ifdef __cplusplus
}
#endif
#endif
