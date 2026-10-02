/* tlm_io.c - asynchronous positioned reads from the drive files.
 *
 * Patterns: P-23 persistent blocking pread worker pool (default backend),
 * P-24 io_uring with forced-async SQEs (only when built with HAVE_LIBURING),
 * P-25 O_DIRECT as a switch, P-30 full-read discipline (via tlm_pread_full),
 * P-31 replicas: each request may name a second copy; the copy whose file has
 * the shorter queue is read, and a failed read is retried from the other copy.
 *
 * Model: one FIFO queue per file. The pread backend runs threads_per_drive
 * workers per file; the io_uring backend runs one ring-owning thread per file
 * that keeps up to queue_depth reads in flight. Completion callbacks run on
 * the I/O thread without any tlm_io lock held, so a callback may submit again.
 * tlm_io_close drains: every request accepted by tlm_io_submit gets exactly
 * one callback before tlm_io_close returns. Submissions racing with close are
 * refused with TLM_EBUSY (no callback).
 */
#include "tlm_io.h"

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef HAVE_LIBURING
#include <liburing.h>
#include <sys/eventfd.h>
#endif

#define IO_MAX_THREADS 64
#define IO_CHUNK       (1ull << 30)   /* largest single read, same cap as tlm_pread_full */

typedef struct io_req {
    struct io_req *next;
    int            file, alt_file;    /* file = copy chosen at submit; alt = other copy or -1 */
    uint64_t       off, alt_off;
    uint8_t       *buf;
    uint64_t       len, done_bytes;
    unsigned       cur_n;             /* io_uring: bytes requested by the SQE in flight */
    uint64_t       t0;
    tlm_io_done_fn done;
    void          *user;
} io_req;

typedef struct io_file {
    struct tlm_io  *io;
    tlm_file        f;
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    io_req         *head, *tail;      /* FIFO of requests not yet started */
    int             stop;
    atomic_int      load;             /* queued + in service; read without the lock (P-31) */
    uint64_t        reads, bytes, busy_ns;
    int             n_threads, started;
    pthread_t      *threads;
    int             sync_ok;          /* mutex/cond initialised */
#ifdef HAVE_LIBURING
    struct io_uring ring;
    int             ring_ok, efd;
    uint64_t        efd_buf;          /* heap-resident: an armed read may land after the thread exits */
#endif
} io_file;

struct tlm_io {
    tlm_io_backend backend;
    int            direct, qd, n_files;
    atomic_int     closing;
    io_file       *files;
};

/* ---- helpers --------------------------------------------------------------- */
static void account(io_file *fl, uint64_t bytes, uint64_t ns) {
    pthread_mutex_lock(&fl->mu);
    fl->reads++;
    fl->bytes += bytes;
    fl->busy_ns += ns;
    pthread_mutex_unlock(&fl->mu);
}

static io_req *pop_locked(io_file *fl) {
    io_req *r = fl->head;
    if (r) {
        fl->head = r->next;
        if (!fl->head) fl->tail = NULL;
        r->next = NULL;
    }
    return r;
}

/* Finish r synchronously from byte r->done_bytes on the chosen copy; on error,
 * re-read the whole range from the other copy (P-31 fallback). */
static int read_rest(tlm_io *io, io_req *r) {
    io_file *a = &io->files[r->file];
    uint64_t t0 = tlm_now_ns(), n = r->len - r->done_bytes;
    int rc = tlm_pread_full(&a->f, r->buf + r->done_bytes, n, r->off + r->done_bytes, io->direct);
    account(a, rc == TLM_OK ? n : 0, tlm_now_ns() - t0);
    if (rc != TLM_OK && r->alt_file >= 0) {
        io_file *b = &io->files[r->alt_file];
        tlm_log(TLM_LOG_WARN, "%s; retrying from replica %s", tlm_errmsg(), b->f.path);
        t0 = tlm_now_ns();
        rc = tlm_pread_full(&b->f, r->buf, r->len, r->alt_off, io->direct);
        account(b, rc == TLM_OK ? r->len : 0, tlm_now_ns() - t0);
    }
    if (rc != TLM_OK) tlm_log(TLM_LOG_INFO, "io: %s", tlm_errmsg());
    return rc;
}

static void complete(tlm_io *io, io_req *r, int status) {
    atomic_fetch_sub(&io->files[r->file].load, 1);
    if (r->done) r->done(r->user, status);
    free(r);
}

/* ---- pread backend (P-23) ------------------------------------------------- */
static void *pread_worker(void *arg) {
    io_file *fl = (io_file *)arg;
    tlm_io *io = fl->io;
    for (;;) {
        pthread_mutex_lock(&fl->mu);
        while (!fl->head && !fl->stop) pthread_cond_wait(&fl->cv, &fl->mu);
        io_req *r = pop_locked(fl);
        pthread_mutex_unlock(&fl->mu);
        if (!r) break;   /* stop requested and the queue is drained */
        complete(io, r, read_rest(io, r));
    }
    return NULL;
}

/* ---- io_uring backend (P-24, adopt-tested: opt-in) ------------------------- */
#ifdef HAVE_LIBURING
static char g_efd_tag;   /* user_data of the eventfd wake-up read */

static void uring_prep(tlm_io *io, io_file *fl, struct io_uring_sqe *sqe, io_req *r) {
    uint64_t left = r->len - r->done_bytes;
    r->cur_n = (unsigned)(left < IO_CHUNK ? left : IO_CHUNK);
    uint8_t *dst = r->buf + r->done_bytes;
    uint64_t off = r->off + r->done_bytes;
    int fd = fl->f.fd;
    if (io->direct && fl->f.fd_direct >= 0 && ((uintptr_t)dst % TLM_ALIGN_IO) == 0 &&
        off % TLM_ALIGN_IO == 0 && r->cur_n % TLM_ALIGN_IO == 0)
        fd = fl->f.fd_direct;
    io_uring_prep_read(sqe, fd, dst, r->cur_n, off);
    /* Without forced async, a cold buffered read can execute inline inside
     * io_uring_enter and serialise this thread (F-io-15). */
    io_uring_sqe_set_flags(sqe, IOSQE_ASYNC);
    io_uring_sqe_set_data(sqe, r);
}

/* One thread owns the ring of one file. New submissions wake it through an
 * eventfd whose read is kept armed inside the ring, so the thread can sleep
 * in a single io_uring wait for either a completion or new work. */
static void *uring_worker(void *arg) {
    io_file *fl = (io_file *)arg;
    tlm_io *io = fl->io;
    struct io_uring *ring = &fl->ring;
    int efd_armed = 0, inflight = 0;
    for (;;) {
        pthread_mutex_lock(&fl->mu);
        int stopping = fl->stop;
        if (!efd_armed && !stopping) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
            if (sqe) {
                io_uring_prep_read(sqe, fl->efd, &fl->efd_buf, sizeof fl->efd_buf, 0);
                io_uring_sqe_set_data(sqe, &g_efd_tag);
                efd_armed = 1;
            }
        }
        while (fl->head && inflight < io->qd) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
            if (!sqe) break;
            io_req *r = pop_locked(fl);
            if (!r->t0) r->t0 = tlm_now_ns();
            uring_prep(io, fl, sqe, r);
            inflight++;
        }
        int finished = stopping && !fl->head && inflight == 0;
        pthread_mutex_unlock(&fl->mu);
        if (finished) break;
        if (stopping && inflight == 0) continue;   /* more queued work: loop to submit it */

        int rc = io_uring_submit_and_wait(ring, 1);
        if (rc < 0 && rc != -EINTR && rc != -EAGAIN && rc != -EBUSY)
            tlm_log(TLM_LOG_WARN, "io_uring %s: submit failed (%s)", fl->f.path, strerror(-rc));

        struct io_uring_cqe *cqe;
        while (io_uring_peek_cqe(ring, &cqe) == 0) {
            void *tag = io_uring_cqe_get_data(cqe);
            int res = cqe->res;
            io_uring_cqe_seen(ring, cqe);
            if (tag == (void *)&g_efd_tag) {
                efd_armed = 0;
                continue;
            }
            io_req *r = (io_req *)tag;
            inflight--;
            if (res > 0 && (unsigned)res == r->cur_n) {
                r->done_bytes += (uint64_t)res;
                if (r->done_bytes < r->len) {   /* next 1 GiB chunk: back to the queue head */
                    pthread_mutex_lock(&fl->mu);
                    r->next = fl->head;
                    fl->head = r;
                    if (!fl->tail) fl->tail = r;
                    pthread_mutex_unlock(&fl->mu);
                    continue;
                }
                account(fl, r->len, tlm_now_ns() - r->t0);
                complete(io, r, TLM_OK);
                continue;
            }
            /* Error, EOF or short read: finish synchronously with the full-read
             * loop, which also handles misaligned remainders and the replica. */
            if (res > 0) r->done_bytes += (uint64_t)res;
            complete(io, r, read_rest(io, r));
        }
    }
    return NULL;
}

static int uring_setup(tlm_io *io) {
    for (int i = 0; i < io->n_files; i++) {
        io_file *fl = &io->files[i];
        fl->efd = eventfd(0, EFD_CLOEXEC);
        if (fl->efd < 0) return -errno;
        int rc = io_uring_queue_init((unsigned)io->qd + 1, &fl->ring, 0);
        if (rc < 0) {
            close(fl->efd);
            fl->efd = -1;
            return rc;
        }
        fl->ring_ok = 1;
    }
    return 0;
}

static void uring_teardown(io_file *fl) {
    if (fl->ring_ok) io_uring_queue_exit(&fl->ring);
    if (fl->efd >= 0) close(fl->efd);
    fl->ring_ok = 0;
    fl->efd = -1;
}

static void uring_wake(io_file *fl) {
    if (fl->ring_ok) {
        uint64_t one = 1;
        if (write(fl->efd, &one, sizeof one) < 0) { /* counter saturated: thread is awake anyway */ }
    }
}
#endif /* HAVE_LIBURING */

/* ---- public API ------------------------------------------------------------ */
static void io_destroy(tlm_io *io) {
    if (!io) return;
    for (int i = 0; i < io->n_files; i++) {
        io_file *fl = &io->files[i];
        if (fl->sync_ok) {
            pthread_mutex_lock(&fl->mu);
            fl->stop = 1;
            pthread_cond_broadcast(&fl->cv);
            pthread_mutex_unlock(&fl->mu);
#ifdef HAVE_LIBURING
            uring_wake(fl);
#endif
        }
    }
    for (int i = 0; i < io->n_files; i++) {
        io_file *fl = &io->files[i];
        for (int t = 0; t < fl->started; t++) pthread_join(fl->threads[t], NULL);
    }
    for (int i = 0; i < io->n_files; i++) {
        io_file *fl = &io->files[i];
#ifdef HAVE_LIBURING
        uring_teardown(fl);
#endif
        tlm_file_close(&fl->f);
        if (fl->sync_ok) {
            pthread_mutex_destroy(&fl->mu);
            pthread_cond_destroy(&fl->cv);
        }
        free(fl->threads);
    }
    free(io->files);
    free(io);
}

tlm_io *tlm_io_open(const char *const *paths, int n_files, const tlm_io_opts *o) {
    if (!paths || n_files < 1) {
        tlm_seterr(TLM_EINVAL, "io: need at least one file");
        return NULL;
    }
    tlm_io_opts def = {TLM_IO_PREAD, 0, 4, 32};
    if (!o) o = &def;
    tlm_io *io = (tlm_io *)calloc(1, sizeof *io);
    if (!io) {
        tlm_seterr(TLM_ENOMEM, "io: out of memory");
        return NULL;
    }
    io->backend = o->backend == TLM_IO_URING ? TLM_IO_URING : TLM_IO_PREAD;
    io->direct = o->direct ? 1 : 0;
    io->qd = o->queue_depth > 0 ? (o->queue_depth > 4096 ? 4096 : o->queue_depth) : 32;
    int tpd = o->threads_per_drive > 0 ? o->threads_per_drive : 4;
    if (tpd > IO_MAX_THREADS) tpd = IO_MAX_THREADS;
    atomic_init(&io->closing, 0);
    io->files = (io_file *)calloc((size_t)n_files, sizeof *io->files);
    if (!io->files) {
        free(io);
        tlm_seterr(TLM_ENOMEM, "io: out of memory");
        return NULL;
    }
    io->n_files = n_files;
    /* The CRC table is built lazily by tlm_crc32c; build it here, before any
     * I/O thread exists, so completion callbacks never race on its init. */
    (void)tlm_crc32c(0, "", 0);

    for (int i = 0; i < n_files; i++) {
        io_file *fl = &io->files[i];
        fl->io = io;
        fl->f.fd = fl->f.fd_direct = -1;
#ifdef HAVE_LIBURING
        fl->efd = -1;
#endif
        atomic_init(&fl->load, 0);
        if (!paths[i]) {
            tlm_seterr(TLM_EINVAL, "io: file %d path is NULL", i);
            goto fail;
        }
        if (tlm_file_open(&fl->f, paths[i], io->direct ? TLM_F_DIRECT : TLM_F_READ) != TLM_OK)
            goto fail;
        pthread_mutex_init(&fl->mu, NULL);
        pthread_cond_init(&fl->cv, NULL);
        fl->sync_ok = 1;
    }

    if (io->backend == TLM_IO_URING) {
#ifdef HAVE_LIBURING
        int rc = uring_setup(io);
        if (rc < 0) {
            tlm_log(TLM_LOG_INFO, "io_uring unavailable (%s); using the pread worker pool",
                    strerror(-rc));
            for (int i = 0; i < n_files; i++) uring_teardown(&io->files[i]);
            io->backend = TLM_IO_PREAD;
        }
#else
        tlm_log(TLM_LOG_INFO, "built without liburing (make URING=1); using the pread worker pool");
        io->backend = TLM_IO_PREAD;
#endif
    }

    for (int i = 0; i < n_files; i++) {
        io_file *fl = &io->files[i];
        void *(*fn)(void *) = pread_worker;
        fl->n_threads = tpd;
#ifdef HAVE_LIBURING
        if (io->backend == TLM_IO_URING) {
            fn = uring_worker;
            fl->n_threads = 1;
        }
#endif
        fl->threads = (pthread_t *)calloc((size_t)fl->n_threads, sizeof *fl->threads);
        if (!fl->threads) {
            tlm_seterr(TLM_ENOMEM, "io: out of memory");
            goto fail;
        }
        for (int t = 0; t < fl->n_threads; t++) {
            int rc = pthread_create(&fl->threads[t], NULL, fn, fl);
            if (rc != 0) {
                tlm_seterr(TLM_EIO, "io: pthread_create: %s", strerror(rc));
                goto fail;
            }
            fl->started++;
        }
    }
    tlm_log(TLM_LOG_INFO, "io: %d file(s), backend %s, direct %s, %d thread(s) per file",
            n_files, tlm_io_backend_name(io), io->direct ? "on" : "off", io->files[0].n_threads);
    return io;
fail:
    io_destroy(io);
    return NULL;
}

void tlm_io_close(tlm_io *io) {
    if (!io) return;
    atomic_store(&io->closing, 1);   /* refuse new work; workers drain what was accepted */
    io_destroy(io);
}

int tlm_io_submit(tlm_io *io, int file, uint64_t off, int alt_file, uint64_t alt_off, void *buf,
                  uint64_t len, tlm_io_done_fn done, void *user) {
    if (!io || !buf || file < 0 || file >= io->n_files || alt_file >= io->n_files)
        return tlm_seterr(TLM_EINVAL, "io submit: bad file index or buffer");
    if (alt_file < 0 || alt_file == file) alt_file = -1;
    if (atomic_load(&io->closing)) return tlm_seterr(TLM_EBUSY, "io submit: closing");
    /* P-31: read whichever copy has less work queued; ties go to the primary.
     * The loads are sampled without a lock: a stale value only costs balance. */
    if (alt_file >= 0 &&
        atomic_load(&io->files[alt_file].load) < atomic_load(&io->files[file].load)) {
        int tf = file;
        uint64_t to = off;
        file = alt_file;
        off = alt_off;
        alt_file = tf;
        alt_off = to;
    }
    io_req *r = (io_req *)calloc(1, sizeof *r);
    if (!r) return tlm_seterr(TLM_ENOMEM, "io submit: out of memory");
    r->file = file;
    r->alt_file = alt_file;
    r->off = off;
    r->alt_off = alt_off;
    r->buf = (uint8_t *)buf;
    r->len = len;
    r->done = done;
    r->user = user;
    io_file *fl = &io->files[file];
    atomic_fetch_add(&fl->load, 1);
    pthread_mutex_lock(&fl->mu);
    if (fl->stop) {
        pthread_mutex_unlock(&fl->mu);
        atomic_fetch_sub(&fl->load, 1);
        free(r);
        return tlm_seterr(TLM_EBUSY, "io submit: closing");
    }
    if (fl->tail) fl->tail->next = r;
    else fl->head = r;
    fl->tail = r;
    pthread_cond_signal(&fl->cv);
    pthread_mutex_unlock(&fl->mu);
#ifdef HAVE_LIBURING
    uring_wake(fl);
#endif
    return TLM_OK;
}

int tlm_io_read_sync(tlm_io *io, int file, uint64_t off, void *buf, uint64_t len) {
    if (!io || !buf || file < 0 || file >= io->n_files)
        return tlm_seterr(TLM_EINVAL, "io read: bad file index or buffer");
    io_file *fl = &io->files[file];
    uint64_t t0 = tlm_now_ns();
    int rc = tlm_pread_full(&fl->f, buf, len, off, io->direct);
    account(fl, rc == TLM_OK ? len : 0, tlm_now_ns() - t0);
    return rc;
}

void tlm_io_stats(tlm_io *io, int file, tlm_io_file_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    if (!io || file < 0 || file >= io->n_files) return;
    io_file *fl = &io->files[file];
    pthread_mutex_lock(&fl->mu);
    out->reads = fl->reads;
    out->bytes = fl->bytes;
    out->busy_ns = fl->busy_ns;
    pthread_mutex_unlock(&fl->mu);
    out->queued = atomic_load(&fl->load);
}

const char *tlm_io_backend_name(const tlm_io *io) {
    if (!io) return "none";
    return io->backend == TLM_IO_URING ? "io_uring" : "pread";
}
