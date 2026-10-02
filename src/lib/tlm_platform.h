/* tlm_platform.h - OS and CPU services used by every libtlm module.
 *
 * Patterns: P-30 full-read discipline, P-25 cache-bypass reads as a switch,
 * P-55 runtime CPU feature dispatch, P-107 memory snapshot (MemAvailable + cgroup).
 * See docs/DISTILLED.md and fragments/CONSENSUS.md.
 */
#ifndef TLM_PLATFORM_H
#define TLM_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TLM_ALIGN_IO   4096u          /* O_DIRECT offset/length/buffer alignment */
#define TLM_ALIGN_SIMD 64u            /* AVX-512 friendly */

#define TLM_ALIGN_UP(x, a)   (((x) + ((a) - 1)) & ~((uint64_t)(a) - 1))

/* ---- errors ---------------------------------------------------------- */
/* All fallible libtlm functions return 0 on success or a negative TLM_E*
 * code, and fill a thread-local message readable with tlm_errmsg(). */
enum {
    TLM_OK        = 0,
    TLM_EIO       = -1,   /* read/write failure or short read */
    TLM_ENOMEM    = -2,
    TLM_EINVAL    = -3,   /* bad argument or malformed input */
    TLM_EFORMAT   = -4,   /* file format / magic / CRC mismatch */
    TLM_ERANGE    = -5,   /* value out of allowed range */
    TLM_ENOTFOUND = -6,
    TLM_EUNSUP    = -7,   /* feature not built or not supported by CPU */
    TLM_EBUSY     = -8
};
const char *tlm_errmsg(void);
/* printf-style; always returns `code` so callers can `return tlm_seterr(...)`. */
int tlm_seterr(int code, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/* ---- logging --------------------------------------------------------- */
enum { TLM_LOG_ERROR = 0, TLM_LOG_WARN = 1, TLM_LOG_INFO = 2, TLM_LOG_DEBUG = 3 };
void tlm_log_level(int level);
void tlm_log(int level, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/* ---- memory ---------------------------------------------------------- */
void *tlm_alloc_aligned(size_t bytes, size_t align);   /* zeroed; NULL on failure */
void  tlm_free_aligned(void *p);
/* Large arena backed by mmap; tries MADV_HUGEPAGE when `huge` != 0 and
 * mlock when `lock` != 0 (failure to lock is a warning, not an error). */
void *tlm_map_anon(size_t bytes, int huge, int lock);
void  tlm_unmap_anon(void *p, size_t bytes);

/* ---- time ------------------------------------------------------------ */
uint64_t tlm_now_ns(void);                 /* CLOCK_MONOTONIC */

/* ---- files ----------------------------------------------------------- */
typedef struct {
    int      fd;          /* buffered descriptor, always valid when open */
    int      fd_direct;   /* O_DIRECT descriptor or -1 */
    uint64_t size;
    char     path[512];
} tlm_file;

/* flags for tlm_file_open */
enum { TLM_F_READ = 0, TLM_F_DIRECT = 1 };
int  tlm_file_open(tlm_file *f, const char *path, int flags);
void tlm_file_close(tlm_file *f);
/* Read exactly `len` bytes at `off` (loops over EINTR and short reads; a
 * short read at EOF is TLM_EIO). When `direct` != 0 and the descriptor,
 * buffer, offset and length are all TLM_ALIGN_IO aligned, uses fd_direct;
 * otherwise falls back to the buffered fd. */
int  tlm_pread_full(const tlm_file *f, void *buf, uint64_t len, uint64_t off, int direct);
/* Write all bytes (loops); used by writers. */
int  tlm_write_full(int fd, const void *buf, uint64_t len);
int  tlm_pwrite_full(int fd, const void *buf, uint64_t len, uint64_t off);
/* Atomic replace: write to "<path>.tmp.<pid>", fsync, rename (P-08). */
int  tlm_write_file_atomic(const char *path, const void *buf, size_t len);
int  tlm_read_file(const char *path, char **out, size_t *out_len);   /* malloc'd, NUL-terminated */
int  tlm_mkdirs(const char *path, int mode);

/* ---- CPU ------------------------------------------------------------- */
enum {
    TLM_CPU_SSE42     = 1u << 0,
    TLM_CPU_AVX2      = 1u << 1,
    TLM_CPU_FMA       = 1u << 2,
    TLM_CPU_F16C      = 1u << 3,
    TLM_CPU_AVX512F   = 1u << 4,
    TLM_CPU_AVX512BW  = 1u << 5,
    TLM_CPU_AVX512VL  = 1u << 6,
    TLM_CPU_AVX512VNNI= 1u << 7,
    TLM_CPU_AVX512BF16= 1u << 8,
    TLM_CPU_AVX512VBMI= 1u << 9,
    TLM_CPU_AVXVNNI   = 1u << 10,
    TLM_CPU_NEON      = 1u << 16,
    TLM_CPU_DOTPROD   = 1u << 17
};
typedef struct {
    uint32_t features;        /* TLM_CPU_* bits actually usable (OS-enabled) */
    int      logical_cpus;
    int      physical_cores;
    int      l3_groups;       /* distinct L3 domains (CCDs on Zen), >= 1 */
    int      cores_per_l3;
    char     model_name[96];
} tlm_cpu_info;
const tlm_cpu_info *tlm_cpu(void);         /* probed once, cached */
/* Mask features for testing / forcing a kernel tier ("scalar", "avx2", "avx512"). */
void tlm_cpu_force(const char *tier);

/* ---- memory snapshot ------------------------------------------------- */
typedef struct {
    uint64_t total_bytes;
    uint64_t available_bytes;   /* min(MemAvailable, cgroup limit - usage) */
    uint64_t cgroup_limit;      /* 0 if none */
    uint64_t hugepage_bytes;    /* Hugepagesize */
} tlm_mem_info;
int tlm_mem_snapshot(tlm_mem_info *out);

/* ---- misc ------------------------------------------------------------ */
uint32_t tlm_crc32c(uint32_t crc, const void *buf, size_t len);  /* Castagnoli, init 0 */
/* Parse "12G", "512M", "4096", "0.5" (fraction returned via *frac when it has a '.') */
int tlm_parse_size(const char *s, uint64_t *bytes);

#ifdef __cplusplus
}
#endif
#endif
