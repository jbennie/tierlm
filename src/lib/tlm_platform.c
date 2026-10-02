/* tlm_platform.c - OS and CPU services. See tlm_platform.h. */
#include "tlm_platform.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

/* ---- errors ------------------------------------------------------------ */
static _Thread_local char g_errmsg[512];

const char *tlm_errmsg(void) { return g_errmsg[0] ? g_errmsg : "no error"; }

int tlm_seterr(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_errmsg, sizeof g_errmsg, fmt, ap);
    va_end(ap);
    return code;
}

/* ---- logging ----------------------------------------------------------- */
static int g_log_level = TLM_LOG_WARN;
void tlm_log_level(int level) { g_log_level = level; }

void tlm_log(int level, const char *fmt, ...) {
    static const char *tag[] = {"error", "warn", "info", "debug"};
    if (level > g_log_level) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "tierlm %s: ", tag[level < 0 ? 0 : level > 3 ? 3 : level]);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

/* ---- memory ------------------------------------------------------------ */
void *tlm_alloc_aligned(size_t bytes, size_t align) {
    void *p = NULL;
    if (align < sizeof(void *)) align = sizeof(void *);
    if (bytes == 0) bytes = align;
    if (posix_memalign(&p, align, TLM_ALIGN_UP(bytes, align)) != 0) return NULL;
    memset(p, 0, bytes);
    return p;
}
void tlm_free_aligned(void *p) { free(p); }

void *tlm_map_anon(size_t bytes, int huge, int lock) {
    if (bytes == 0) return NULL;
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
#ifdef MADV_HUGEPAGE
    if (huge) (void)madvise(p, bytes, MADV_HUGEPAGE);
#else
    (void)huge;
#endif
    if (lock && mlock(p, bytes) != 0)
        tlm_log(TLM_LOG_WARN, "mlock of %zu bytes failed (%s); raise the memlock limit",
                bytes, strerror(errno));
    return p;
}
void tlm_unmap_anon(void *p, size_t bytes) { if (p) munmap(p, bytes); }

/* ---- time -------------------------------------------------------------- */
uint64_t tlm_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* ---- files ------------------------------------------------------------- */
int tlm_file_open(tlm_file *f, const char *path, int flags) {
    memset(f, 0, sizeof *f);
    f->fd = f->fd_direct = -1;
    snprintf(f->path, sizeof f->path, "%s", path);
    f->fd = open(path, O_RDONLY | O_CLOEXEC);
    if (f->fd < 0) return tlm_seterr(TLM_ENOTFOUND, "open %s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(f->fd, &st) != 0) {
        close(f->fd);
        f->fd = -1;
        return tlm_seterr(TLM_EIO, "stat %s: %s", path, strerror(errno));
    }
    f->size = (uint64_t)st.st_size;
#ifdef O_DIRECT
    if (flags & TLM_F_DIRECT) {
        f->fd_direct = open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (f->fd_direct < 0)
            tlm_log(TLM_LOG_INFO, "O_DIRECT unavailable for %s (%s); using buffered reads",
                    path, strerror(errno));
    }
#endif
#ifdef POSIX_FADV_RANDOM
    (void)posix_fadvise(f->fd, 0, 0, POSIX_FADV_RANDOM);
#endif
    return TLM_OK;
}

void tlm_file_close(tlm_file *f) {
    if (f->fd >= 0) close(f->fd);
    if (f->fd_direct >= 0) close(f->fd_direct);
    f->fd = f->fd_direct = -1;
}

static int pread_loop(int fd, void *buf, uint64_t len, uint64_t off) {
    uint8_t *p = (uint8_t *)buf;
    while (len > 0) {
        size_t chunk = len > (1ull << 30) ? (size_t)(1ull << 30) : (size_t)len;
        ssize_t r = pread(fd, p, chunk, (off_t)off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return -errno;
        }
        if (r == 0) return -EIO; /* unexpected EOF: a short read is an error (P-30) */
        p += r;
        off += (uint64_t)r;
        len -= (uint64_t)r;
    }
    return 0;
}

int tlm_pread_full(const tlm_file *f, void *buf, uint64_t len, uint64_t off, int direct) {
    if (off > f->size || len > f->size - off)
        return tlm_seterr(TLM_EIO, "read past end of %s (off %llu len %llu size %llu)", f->path,
                          (unsigned long long)off, (unsigned long long)len,
                          (unsigned long long)f->size);
    int use_direct = direct && f->fd_direct >= 0 && ((uintptr_t)buf % TLM_ALIGN_IO) == 0 &&
                     (off % TLM_ALIGN_IO) == 0 && (len % TLM_ALIGN_IO) == 0;
    int rc = pread_loop(use_direct ? f->fd_direct : f->fd, buf, len, off);
    if (rc != 0 && use_direct) rc = pread_loop(f->fd, buf, len, off); /* fall back */
    if (rc != 0)
        return tlm_seterr(TLM_EIO, "pread %s off %llu len %llu: %s", f->path,
                          (unsigned long long)off, (unsigned long long)len, strerror(-rc));
    return TLM_OK;
}

int tlm_write_full(int fd, const void *buf, uint64_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        size_t chunk = len > (1ull << 30) ? (size_t)(1ull << 30) : (size_t)len;
        ssize_t r = write(fd, p, chunk);
        if (r < 0) {
            if (errno == EINTR) continue;
            return tlm_seterr(TLM_EIO, "write: %s", strerror(errno));
        }
        p += r;
        len -= (uint64_t)r;
    }
    return TLM_OK;
}

int tlm_pwrite_full(int fd, const void *buf, uint64_t len, uint64_t off) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        size_t chunk = len > (1ull << 30) ? (size_t)(1ull << 30) : (size_t)len;
        ssize_t r = pwrite(fd, p, chunk, (off_t)off);
        if (r < 0) {
            if (errno == EINTR) continue;
            return tlm_seterr(TLM_EIO, "pwrite: %s", strerror(errno));
        }
        p += r;
        off += (uint64_t)r;
        len -= (uint64_t)r;
    }
    return TLM_OK;
}

int tlm_write_file_atomic(const char *path, const void *buf, size_t len) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid());
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return tlm_seterr(TLM_EIO, "create %s: %s", tmp, strerror(errno));
    int rc = tlm_write_full(fd, buf, len);
    if (rc == 0 && fsync(fd) != 0) rc = tlm_seterr(TLM_EIO, "fsync %s: %s", tmp, strerror(errno));
    close(fd);
    if (rc == 0 && rename(tmp, path) != 0)
        rc = tlm_seterr(TLM_EIO, "rename %s -> %s: %s", tmp, path, strerror(errno));
    if (rc != 0) unlink(tmp);
    return rc;
}

int tlm_read_file(const char *path, char **out, size_t *out_len) {
    *out = NULL;
    if (out_len) *out_len = 0;
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return tlm_seterr(TLM_ENOTFOUND, "open %s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return tlm_seterr(TLM_EIO, "stat %s", path); }
    size_t n = (size_t)st.st_size;
    char *buf = (char *)malloc(n + 1);
    if (!buf) { close(fd); return tlm_seterr(TLM_ENOMEM, "read %s: out of memory", path); }
    int rc = n ? pread_loop(fd, buf, n, 0) : 0;
    close(fd);
    if (rc != 0) { free(buf); return tlm_seterr(TLM_EIO, "read %s: %s", path, strerror(-rc)); }
    buf[n] = 0;
    *out = buf;
    if (out_len) *out_len = n;
    return TLM_OK;
}

int tlm_mkdirs(const char *path, int mode) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s", path);
    size_t n = strlen(tmp);
    if (n == 0) return TLM_EINVAL;
    for (size_t i = 1; i <= n; i++) {
        if (tmp[i] == '/' || tmp[i] == 0) {
            char c = tmp[i];
            tmp[i] = 0;
            if (mkdir(tmp, (mode_t)mode) != 0 && errno != EEXIST)
                return tlm_seterr(TLM_EIO, "mkdir %s: %s", tmp, strerror(errno));
            tmp[i] = c;
        }
    }
    return TLM_OK;
}

/* ---- CPU --------------------------------------------------------------- */
static tlm_cpu_info g_cpu;
static int g_cpu_probed;
static uint32_t g_cpu_mask = 0xFFFFFFFFu;

static int read_int_file(const char *path, int *out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int ok = fscanf(f, "%d", out) == 1;
    fclose(f);
    return ok ? 0 : -1;
}

static void probe_topology(tlm_cpu_info *c) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    c->logical_cpus = n > 0 ? (int)n : 1;
    /* physical cores = distinct (package, core_id); L3 groups = distinct
     * shared_cpu_list strings of cache index3. Bounded to 1024 CPUs. */
    int max = c->logical_cpus > 1024 ? 1024 : c->logical_cpus;
    static int core_key[1024];
    static char l3_keys[64][128];
    int ncores = 0, nl3 = 0;
    for (int cpu = 0; cpu < max; cpu++) {
        char path[256];
        int pkg = 0, core = cpu;
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        (void)read_int_file(path, &pkg);
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        (void)read_int_file(path, &core);
        int key = pkg * 65536 + core, seen = 0;
        for (int i = 0; i < ncores; i++)
            if (core_key[i] == key) { seen = 1; break; }
        if (!seen && ncores < 1024) core_key[ncores++] = key;

        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", cpu);
        FILE *f = fopen(path, "r");
        if (f) {
            char buf[128] = {0};
            if (fgets(buf, sizeof buf, f)) {
                buf[strcspn(buf, "\n")] = 0;
                int found = 0;
                for (int i = 0; i < nl3; i++)
                    if (strcmp(l3_keys[i], buf) == 0) { found = 1; break; }
                if (!found && nl3 < 64) snprintf(l3_keys[nl3++], sizeof l3_keys[0], "%s", buf);
            }
            fclose(f);
        }
    }
    c->physical_cores = ncores > 0 ? ncores : c->logical_cpus;
    c->l3_groups = nl3 > 0 ? nl3 : 1;
    c->cores_per_l3 = c->physical_cores / c->l3_groups;
    if (c->cores_per_l3 < 1) c->cores_per_l3 = 1;

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof line, f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char *p = strchr(line, ':');
                if (p) {
                    p++;
                    while (*p == ' ') p++;
                    p[strcspn(p, "\n")] = 0;
                    snprintf(c->model_name, sizeof c->model_name, "%s", p);
                }
                break;
            }
        }
        fclose(f);
    }
}

static void probe_features(tlm_cpu_info *c) {
    uint32_t feat = 0;
#if defined(__x86_64__) || defined(__i386__)
    unsigned a, b, cc, d;
    if (__get_cpuid(1, &a, &b, &cc, &d)) {
        int osxsave = (cc >> 27) & 1;
        uint64_t xcr0 = 0;
        if (osxsave) {
            unsigned lo, hi;
            __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
            xcr0 = ((uint64_t)hi << 32) | lo;
        }
        int ymm_ok = (xcr0 & 0x6) == 0x6;
        int zmm_ok = ymm_ok && (xcr0 & 0xE0) == 0xE0;
        if ((cc >> 20) & 1) feat |= TLM_CPU_SSE42;
        if (ymm_ok && ((cc >> 12) & 1)) feat |= TLM_CPU_FMA;
        if (ymm_ok && ((cc >> 29) & 1)) feat |= TLM_CPU_F16C;
        if (__get_cpuid_count(7, 0, &a, &b, &cc, &d)) {
            if (ymm_ok && ((b >> 5) & 1)) feat |= TLM_CPU_AVX2;
            if (zmm_ok && ((b >> 16) & 1)) feat |= TLM_CPU_AVX512F;
            if (zmm_ok && ((b >> 30) & 1)) feat |= TLM_CPU_AVX512BW;
            if (zmm_ok && ((b >> 31) & 1)) feat |= TLM_CPU_AVX512VL;
            if (zmm_ok && ((cc >> 11) & 1)) feat |= TLM_CPU_AVX512VNNI;
            if (zmm_ok && ((cc >> 1) & 1)) feat |= TLM_CPU_AVX512VBMI;
            unsigned a1, b1, c1, d1;
            if (__get_cpuid_count(7, 1, &a1, &b1, &c1, &d1)) {
                if (zmm_ok && ((a1 >> 5) & 1)) feat |= TLM_CPU_AVX512BF16;
                if (ymm_ok && ((a1 >> 4) & 1)) feat |= TLM_CPU_AVXVNNI;
            }
        }
    }
#elif defined(__aarch64__)
    feat |= TLM_CPU_NEON;
#endif
    c->features = feat;
}

const tlm_cpu_info *tlm_cpu(void) {
    if (!g_cpu_probed) {
        memset(&g_cpu, 0, sizeof g_cpu);
        probe_features(&g_cpu);
        probe_topology(&g_cpu);
        g_cpu_probed = 1;
        const char *force = getenv("TIERLM_KERNEL");
        if (force && *force) tlm_cpu_force(force);
    }
    static tlm_cpu_info masked;
    masked = g_cpu;
    masked.features &= g_cpu_mask;
    return &masked;
}

void tlm_cpu_force(const char *tier) {
    if (!tier || strcmp(tier, "auto") == 0) g_cpu_mask = 0xFFFFFFFFu;
    else if (strcmp(tier, "scalar") == 0) g_cpu_mask = 0;
    else if (strcmp(tier, "avx2") == 0)
        g_cpu_mask = TLM_CPU_SSE42 | TLM_CPU_AVX2 | TLM_CPU_FMA | TLM_CPU_F16C | TLM_CPU_AVXVNNI;
    else g_cpu_mask = 0xFFFFFFFFu;
}

/* ---- memory snapshot ----------------------------------------------------- */
static uint64_t read_u64_file(const char *path, int *ok) {
    FILE *f = fopen(path, "r");
    *ok = 0;
    if (!f) return 0;
    char buf[64] = {0};
    uint64_t v = 0;
    if (fgets(buf, sizeof buf, f) && strncmp(buf, "max", 3) != 0) {
        v = strtoull(buf, NULL, 10);
        *ok = 1;
    }
    fclose(f);
    return v;
}

int tlm_mem_snapshot(tlm_mem_info *out) {
    memset(out, 0, sizeof *out);
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return tlm_seterr(TLM_EUNSUP, "no /proc/meminfo");
    char key[64];
    unsigned long long val;
    char unit[16];
    while (fscanf(f, "%63s %llu %15[^\n]\n", key, &val, unit) >= 2) {
        if (strcmp(key, "MemTotal:") == 0) out->total_bytes = val * 1024ull;
        else if (strcmp(key, "MemAvailable:") == 0) out->available_bytes = val * 1024ull;
        else if (strcmp(key, "Hugepagesize:") == 0) out->hugepage_bytes = val * 1024ull;
    }
    fclose(f);
    int ok1 = 0, ok2 = 0;
    uint64_t lim = read_u64_file("/sys/fs/cgroup/memory.max", &ok1);       /* cgroup v2 */
    uint64_t use = read_u64_file("/sys/fs/cgroup/memory.current", &ok2);
    if (!ok1) {
        lim = read_u64_file("/sys/fs/cgroup/memory/memory.limit_in_bytes", &ok1); /* v1 */
        use = read_u64_file("/sys/fs/cgroup/memory/memory.usage_in_bytes", &ok2);
    }
    if (ok1 && lim > 0 && lim < out->total_bytes) {
        out->cgroup_limit = lim;
        uint64_t avail = ok2 && use < lim ? lim - use : 0;
        if (avail < out->available_bytes) out->available_bytes = avail;
    }
    return TLM_OK;
}

/* ---- CRC32C (Castagnoli, reflected poly 0x82F63B78), slice-by-1 table ----- */
static uint32_t g_crc_table[256];
static int g_crc_init;
static void crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
        g_crc_table[i] = c;
    }
    g_crc_init = 1;
}

uint32_t tlm_crc32c(uint32_t crc, const void *buf, size_t len) {
    if (!g_crc_init) crc_init();
    const uint8_t *p = (const uint8_t *)buf;
    crc = ~crc;
#if defined(__SSE4_2__) && defined(__x86_64__)
    while (len >= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        crc = (uint32_t)__builtin_ia32_crc32di(crc, v);
        p += 8;
        len -= 8;
    }
#endif
    while (len--) crc = g_crc_table[(crc ^ *p++) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

int tlm_parse_size(const char *s, uint64_t *bytes) {
    if (!s || !*s) return TLM_EINVAL;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || v < 0) return tlm_seterr(TLM_EINVAL, "not a size: '%s'", s);
    while (*end == ' ') end++;
    double mul = 1;
    switch (*end) {
    case 'k': case 'K': mul = 1024.0; end++; break;
    case 'm': case 'M': mul = 1024.0 * 1024; end++; break;
    case 'g': case 'G': mul = 1024.0 * 1024 * 1024; end++; break;
    case 't': case 'T': mul = 1024.0 * 1024 * 1024 * 1024; end++; break;
    default: break;
    }
    if (*end == 'i' || *end == 'I') end++;
    if (*end == 'b' || *end == 'B') end++;
    if (*end != 0) return tlm_seterr(TLM_EINVAL, "not a size: '%s'", s);
    *bytes = (uint64_t)(v * mul + 0.5);
    return TLM_OK;
}
