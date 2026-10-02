/* tlm_io.h - asynchronous positioned reads from the drive files.
 *
 * Patterns: P-23 persistent pread worker pool (default), P-24 io_uring
 * (built when HAVE_LIBURING; opt-in), P-25 O_DIRECT switch, P-30 full reads,
 * P-31 replica reads (shortest queue wins).
 */
#ifndef TLM_IO_H
#define TLM_IO_H

#include <stdint.h>
#include "tlm_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tlm_io tlm_io;

typedef enum { TLM_IO_PREAD = 0, TLM_IO_URING = 1 } tlm_io_backend;

typedef struct {
    tlm_io_backend backend;
    int  direct;           /* O_DIRECT when aligned */
    int  threads_per_drive;/* pread backend; default 4 */
    int  queue_depth;      /* io_uring; default 32 */
} tlm_io_opts;

/* Completion callback, called on an I/O thread. status is 0 or TLM_E*. */
typedef void (*tlm_io_done_fn)(void *user, int status);

tlm_io *tlm_io_open(const char *const *paths, int n_files, const tlm_io_opts *o);
void    tlm_io_close(tlm_io *io);
/* Queue a read of `len` bytes at `off` from file `file` into `buf`.
 * If alt_file >= 0, the backend may read from (alt_file, alt_off) instead,
 * choosing the file with the shorter queue. Never blocks the caller. */
int     tlm_io_submit(tlm_io *io, int file, uint64_t off, int alt_file, uint64_t alt_off,
                      void *buf, uint64_t len, tlm_io_done_fn done, void *user);
/* Synchronous convenience read (used at load). */
int     tlm_io_read_sync(tlm_io *io, int file, uint64_t off, void *buf, uint64_t len);
/* Notes (tlm_io.c): if the chosen copy fails to read, the other copy is read
 * before the callback reports an error. Callbacks run on an I/O thread with no
 * tlm_io lock held and may call tlm_io_submit. tlm_io_close drains: every
 * accepted request gets its callback; submits during close get TLM_EBUSY.
 * Requesting TLM_IO_URING without liburing (or with io_uring disabled) falls
 * back to the pread pool with an info log; tlm_io_backend_name tells which. */
/* Per-file statistics. busy_ns sums per-request service time (overlapping
 * requests overlap), queued = requests waiting or in service on that file. */
typedef struct { uint64_t reads, bytes, busy_ns; int queued; } tlm_io_file_stats;
void    tlm_io_stats(tlm_io *io, int file, tlm_io_file_stats *out);
const char *tlm_io_backend_name(const tlm_io *io);

#ifdef __cplusplus
}
#endif
#endif
