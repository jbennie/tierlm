/* tierlm-setup.c - probe the machine, recommend a value for every engine
 * preference with a one-line reason, let the user accept or change each one,
 * and write <etc>/tierlm.conf and <etc>/hardware.conf (etc = --etc DIR, else
 * $TIERLM_ETC, else /etc/tierlm).
 *
 * Design: research/03-two-stage-design.md ("Setup tool: tierlm-setup and
 * /etc/tierlm"), research/04-decision-engine.md ("Setup: experts and
 * content"), research/02-wrx90-build-profile.md ("Runtime defaults").
 * Patterns: F-HW-01 memory snapshot, F-HW-02 CPU inventory, F-HW-03 disk
 * probe that does not measure RAM, F-HW-04 L2-defeating bandwidth probe,
 * F-CFG-01 strict validation and origins, P-08 tmp + rename writes.
 *
 * Flow: probe (read-only, no root needed) -> recommend -> select (prompts,
 * --auto or --answers) -> validate with the runtime's own parser and range
 * checks -> show the changed keys -> write atomically, keeping a timestamped
 * backup -> print what the system still needs (--apply-system writes those
 * snippets under <etc>/system/ only; it never touches /etc/sysctl.d).
 */
#include "tlm_cfg.h"
#include "tlm_platform.h"

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <glob.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define SETUP_VERSION "1.0"
#define KIB_U 1024ull
#define MIB_U (1024ull * 1024ull)
#define GIB_U (1024ull * 1024ull * 1024ull)

enum { SEC_ENGINE = 1, SEC_CONTENT = 2, SEC_ALL = 3 };

/* Default taxonomy, identical to etc/content-taxonomy.conf (tests/setup.sh
 * compares them). Used when no copy is found next to the binary. */
static const char k_default_taxonomy[] =
    "# content-taxonomy.conf -- content types offered by tierlm-setup's\n"
    "# \"Experts and content\" section and used by tlmc for calibration labelling.\n"
    "# Installed as /etc/tierlm/content-taxonomy.conf; edit that copy to add types.\n"
    "#\n"
    "# Format: <type> = <description>\n"
    "#   - a type is a lowercase dotted name; <parent>.<child> is a child type,\n"
    "#     and a child inherits its parent's depth unless it is set itself\n"
    "#   - every type becomes the key content.<type> in tierlm.conf, set to\n"
    "#     exclude | shallow | standard | deep; content.default covers the rest\n"
    "#   - descriptions say what a type covers: typed decision models follow the\n"
    "#     label more than its definition, so keep labels meaningful\n"
    "\n"
    "code                    = Programming and source code\n"
    "code.python             = Python\n"
    "code.c_cpp              = C and C++\n"
    "code.rust               = Rust\n"
    "code.go                 = Go\n"
    "code.js_ts              = JavaScript and TypeScript\n"
    "code.java               = Java and other JVM languages\n"
    "code.sql                = SQL and database queries\n"
    "code.shell              = Shell scripts and command lines\n"
    "code.other              = Other programming languages\n"
    "\n"
    "math                    = Mathematics\n"
    "science                 = Natural sciences\n"
    "engineering             = Engineering\n"
    "\n"
    "law                     = Law and legal text\n"
    "medicine                = Medicine and health\n"
    "finance                 = Finance and accounting\n"
    "\n"
    "lang                    = Natural languages\n"
    "lang.en                 = English\n"
    "lang.zh                 = Chinese\n"
    "lang.ja                 = Japanese\n"
    "lang.ko                 = Korean\n"
    "lang.de                 = German\n"
    "lang.fr                 = French\n"
    "lang.es                 = Spanish\n"
    "lang.other              = Other natural languages\n"
    "\n"
    "creative                = Creative writing\n"
    "chat                    = Conversation and chit-chat\n"
    "\n"
    "tools                   = Tool use and structured output\n"
    "tools.json              = JSON and other structured output\n"
    "tools.function_calling  = Function and tool calling\n"
    "tools.agentic_traces    = Multi-step agentic traces\n"
    "\n"
    "reasoning               = Long chain-of-thought reasoning\n";

/* ======================================================================
 * small utilities
 * ====================================================================== */
static void oom(void) {
    fputs("tierlm-setup: out of memory\n", stderr);
    exit(1);
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) oom();
    return q;
}

static char *xstrdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (!p) oom();
    memcpy(p, s, n);
    return p;
}

static char *xvprintf(const char *fmt, va_list ap) {
    va_list cp;
    va_copy(cp, ap);
    int n = vsnprintf(NULL, 0, fmt, cp);
    va_end(cp);
    if (n < 0) oom();
    char *p = (char *)malloc((size_t)n + 1);
    if (!p) oom();
    vsnprintf(p, (size_t)n + 1, fmt, ap);
    return p;
}

static char *xprintf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static char *xprintf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *p = xvprintf(fmt, ap);
    va_end(ap);
    return p;
}

static void set_str(char **dst, const char *s) {
    char *n = s ? xstrdup(s) : NULL;
    free(*dst);
    *dst = n;
}

static int same_str(const char *a, const char *b) {
    if (!a || !b) return a == b;
    return strcmp(a, b) == 0;
}

/* Human-readable size: GiB with one decimal, else whole MiB / KiB / bytes. */
static const char *gib(uint64_t b, char *buf, size_t cap) {
    if (b >= GIB_U) snprintf(buf, cap, "%.1f GiB", (double)b / (double)GIB_U);
    else if (b >= MIB_U) snprintf(buf, cap, "%.0f MiB", (double)b / (double)MIB_U);
    else if (b >= KIB_U) snprintf(buf, cap, "%.0f KiB", (double)b / (double)KIB_U);
    else snprintf(buf, cap, "%llu bytes", (unsigned long long)b);
    return buf;
}

static int read_small(const char *path, char *buf, size_t cap) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t n = fread(buf, 1, cap - 1, f);
    fclose(f);
    buf[n] = 0;
    buf[strcspn(buf, "\n")] = 0;
    return 0;
}

static int is_dir(const char *p) {
    struct stat st;
    return p && stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

static void sleep_seconds(double s) {
    struct timespec ts;
    ts.tv_sec = (time_t)s;
    ts.tv_nsec = (long)((s - (double)ts.tv_sec) * 1e9);
    while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
    }
}

static void stamp_local(char *out, size_t cap) {
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(out, cap, "%Y%m%d-%H%M%S", &tm);
}

static void date_local(char *out, size_t cap) {
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    strftime(out, cap, "%Y-%m-%d", &tm);
}

/* ---- key/value and string lists --------------------------------------- */
typedef struct { char *key, *val; int line; } kv_t;
typedef struct { kv_t *v; int n, cap; } kvlist;

static kv_t *kv_find(const kvlist *l, const char *key) {
    for (int i = 0; i < l->n; i++)
        if (strcmp(l->v[i].key, key) == 0) return &l->v[i];
    return NULL;
}

static void kv_set(kvlist *l, const char *key, const char *val, int line) {
    kv_t *e = kv_find(l, key);
    if (e) {
        set_str(&e->val, val);
        e->line = line;
        return;
    }
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 32;
        l->v = (kv_t *)xrealloc(l->v, (size_t)l->cap * sizeof *l->v);
    }
    l->v[l->n].key = xstrdup(key);
    l->v[l->n].val = xstrdup(val);
    l->v[l->n].line = line;
    l->n++;
}

static void kv_free(kvlist *l) {
    for (int i = 0; i < l->n; i++) {
        free(l->v[i].key);
        free(l->v[i].val);
    }
    free(l->v);
    memset(l, 0, sizeof *l);
}

typedef struct { char **v; int n, cap; } strlist;

static void sl_add(strlist *l, const char *s) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 8;
        l->v = (char **)xrealloc(l->v, (size_t)l->cap * sizeof *l->v);
    }
    l->v[l->n++] = xstrdup(s);
}

static void sl_free(strlist *l) {
    for (int i = 0; i < l->n; i++) free(l->v[i]);
    free(l->v);
    memset(l, 0, sizeof *l);
}

/* ---- config lines (tlm_cfg_write input) -------------------------------- */
typedef struct { tlm_cfg_line *v; int n, cap; } lines_t;

static void ln_add(lines_t *L, const char *key, const char *val, const char *comment) {
    if (L->n == L->cap) {
        L->cap = L->cap ? L->cap * 2 : 64;
        L->v = (tlm_cfg_line *)xrealloc(L->v, (size_t)L->cap * sizeof *L->v);
    }
    L->v[L->n].key = xstrdup(key);
    L->v[L->n].value = xstrdup(val);
    L->v[L->n].comment = xstrdup(comment);
    L->n++;
}

static void ln_addf(lines_t *L, const char *key, const char *comment, const char *fmt, ...)
    __attribute__((format(printf, 4, 5)));
static void ln_addf(lines_t *L, const char *key, const char *comment, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *v = xvprintf(fmt, ap);
    va_end(ap);
    ln_add(L, key, v, comment);
    free(v);
}

static void ln_group(lines_t *L, const char *title) {
    char buf[128];
    int n = snprintf(buf, sizeof buf, "-- %s ", title);
    while (n < 70 && n < (int)sizeof buf - 1) buf[n++] = '-';
    buf[n] = 0;
    if (L->n) ln_add(L, NULL, NULL, NULL);
    ln_add(L, NULL, NULL, buf);
}

static void ln_free(lines_t *L) {
    for (int i = 0; i < L->n; i++) {
        free((char *)L->v[i].key);
        free((char *)L->v[i].value);
        free((char *)L->v[i].comment);
    }
    free(L->v);
    memset(L, 0, sizeof *L);
}

/* Render lines to text; without_includes drops `include` lines (used to
 * validate the body without following the conf.d glob). */
static char *render(const char *header, const lines_t *L, int without_includes, size_t *len) {
    char *buf = NULL;
    size_t n = 0;
    FILE *f = open_memstream(&buf, &n);
    if (!f) oom();
    tlm_cfg_line *tmp = (tlm_cfg_line *)xrealloc(NULL, (size_t)(L->n + 1) * sizeof *tmp);
    int m = 0;
    for (int i = 0; i < L->n; i++) {
        if (without_includes && L->v[i].key && strcmp(L->v[i].key, "include") == 0) continue;
        tmp[m++] = L->v[i];
    }
    int rc = tlm_cfg_write(f, header, tmp, m);
    free(tmp);
    fclose(f);
    if (rc != 0) {
        free(buf);
        return NULL;
    }
    if (len) *len = n;
    return buf;
}

/* ======================================================================
 * existing files: tolerant line-by-line read
 * ====================================================================== */
typedef struct {
    int exists;
    kvlist pairs;   /* valid keys, canonical values; last wins */
    strlist incl;   /* include patterns, in order */
    strlist kept;   /* lines that cannot be carried over (shown as comments) */
} oldfile;

typedef struct { oldfile *o; int line; } old_ctx;

static int old_line_cb(void *ctx, const char *key, const char *value, int line) {
    (void)line;
    old_ctx *c = (old_ctx *)ctx;
    if (strcmp(key, "include") == 0) {
        sl_add(&c->o->incl, value);
        return 0;
    }
    size_t cap = strlen(value) * 2 + 64;
    char *norm = (char *)malloc(cap);
    if (!norm) oom();
    char why[256];
    int rc = tlm_cfg_validate(key, value, norm, cap, why, sizeof why);
    if (rc == 0) {
        kv_set(&c->o->pairs, key, norm, c->line);
    } else {
        char *s = xprintf("line %d (%s): %s = %s", c->line, why, key, value);
        sl_add(&c->o->kept, s);
        free(s);
    }
    free(norm);
    return 0;
}

static void read_old(const char *path, oldfile *o) {
    memset(o, 0, sizeof *o);
    char *text = NULL;
    size_t n = 0;
    if (tlm_read_file(path, &text, &n) != 0) return;
    o->exists = 1;
    old_ctx c = {o, 0};
    char *p = text;
    while (p && *p) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = 0;
        size_t l = strlen(p);
        if (l && p[l - 1] == '\r') p[l - 1] = 0;
        c.line++;
        if (tlm_cfg_parse_text(p, path, old_line_cb, &c) != 0) {
            char *s = xprintf("line %d (syntax error): %s", c.line, p);
            sl_add(&o->kept, s);
            free(s);
        }
        p = nl ? nl + 1 : NULL;
    }
    free(text);
}

static void old_free(oldfile *o) {
    kv_free(&o->pairs);
    sl_free(&o->incl);
    sl_free(&o->kept);
}

/* ======================================================================
 * options
 * ====================================================================== */
typedef struct {
    char etc[PATH_MAX];
    int probe_only, dry_run, auto_yes, show, apply_system, strict, print_taxonomy;
    const char *answers;
    int section;
    int seconds;
    strlist drives;
    int interactive;
} opts_t;

static void usage(FILE *out) {
    fprintf(out,
        "usage: tierlm-setup [options]\n"
        "\n"
        "Probe this machine, recommend a value for every tierlm preference with the\n"
        "reason, and write <etc>/tierlm.conf and <etc>/hardware.conf.\n"
        "\n"
        "  --etc DIR            configuration directory (default $TIERLM_ETC or %s)\n"
        "  --probe              only probe: write hardware.conf, show recommendations and\n"
        "                       flag preferences that no longer fit; tierlm.conf untouched\n"
        "  --dry-run, -n        show recommendations and changes; write nothing\n"
        "  --auto, -y           accept every recommendation without prompting\n"
        "  --answers FILE       unattended: take values from a key = value file; other\n"
        "                       keys get the recommendation (content.<type> = inherit\n"
        "                       removes a depth)\n"
        "  --section S          all | engine | content (default all)\n"
        "  --drive DIR          drive mount point to measure (repeatable); becomes io.drives\n"
        "                       (default: the drives already in io.drives)\n"
        "  --probe-seconds N    seconds for the DRAM test and for each drive (default 10)\n"
        "  --apply-system       also write sysctl / limits / tuned snippets to <etc>/system/\n"
        "                       and print the commands that install them\n"
        "  --show               print the effective configuration with the origin of every\n"
        "                       value, and exit (non-zero if the files have errors)\n"
        "  --strict             unknown keys are errors when reading existing files\n"
        "  --print-default-taxonomy   print the built-in content taxonomy and exit\n"
        "  --version, -h/--help\n"
        "\n"
        "Interactive prompts: Enter accepts the recommendation, '=' keeps the current\n"
        "value, '-' unsets (content types: inherit), '?' explains the key, '*' accepts\n"
        "all remaining recommendations of the section. Without a terminal on stdin and\n"
        "without --auto/--answers the tool behaves like --dry-run.\n",
        TLM_ETC_DEFAULT);
}

static void usage_error(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));
static void usage_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("tierlm-setup: ", stderr);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputs("\nTry 'tierlm-setup --help'.\n", stderr);
    exit(2);
}

static void parse_args(int argc, char **argv, opts_t *o) {
    static const struct option lo[] = {
        {"etc", required_argument, 0, 'e'},
        {"probe", no_argument, 0, 'p'},
        {"dry-run", no_argument, 0, 'n'},
        {"auto", no_argument, 0, 'y'},
        {"answers", required_argument, 0, 'a'},
        {"section", required_argument, 0, 's'},
        {"apply-system", no_argument, 0, 'S'},
        {"drive", required_argument, 0, 'd'},
        {"probe-seconds", required_argument, 0, 't'},
        {"show", no_argument, 0, 'w'},
        {"strict", no_argument, 0, 'x'},
        {"print-default-taxonomy", no_argument, 0, 'T'},
        {"help", no_argument, 0, 'h'},
        {"version", no_argument, 0, 'V'},
        {0, 0, 0, 0}};
    memset(o, 0, sizeof *o);
    o->section = SEC_ALL;
    o->seconds = 10;
    const char *env = getenv("TIERLM_ETC");
    snprintf(o->etc, sizeof o->etc, "%s", env && *env ? env : TLM_ETC_DEFAULT);
    int ch;
    opterr = 0;
    while ((ch = getopt_long(argc, argv, "hny", lo, NULL)) != -1) {
        switch (ch) {
        case 'e':
            if (!*optarg) usage_error("--etc needs a directory");
            snprintf(o->etc, sizeof o->etc, "%s", optarg);
            break;
        case 'p': o->probe_only = 1; break;
        case 'n': o->dry_run = 1; break;
        case 'y': o->auto_yes = 1; break;
        case 'a': o->answers = optarg; break;
        case 's':
            if (strcmp(optarg, "all") == 0) o->section = SEC_ALL;
            else if (strcmp(optarg, "engine") == 0) o->section = SEC_ENGINE;
            else if (strcmp(optarg, "content") == 0) o->section = SEC_CONTENT;
            else usage_error("--section must be all, engine or content (got '%s')", optarg);
            break;
        case 'S': o->apply_system = 1; break;
        case 'd': {
            char rp[PATH_MAX];
            if (!realpath(optarg, rp) || !is_dir(rp)) usage_error("--drive %s: not an existing directory", optarg);
            if (strchr(rp, ',')) usage_error("--drive %s: paths containing ',' cannot be listed in io.drives", rp);
            sl_add(&o->drives, rp);
            break;
        }
        case 't': {
            int64_t v;
            if (tlm_cfg_parse_int(optarg, &v) != 0 || v < 1 || v > 600)
                usage_error("--probe-seconds must be an integer in [1, 600] (got '%s')", optarg);
            o->seconds = (int)v;
            break;
        }
        case 'w': o->show = 1; break;
        case 'x': o->strict = 1; break;
        case 'T': o->print_taxonomy = 1; break;
        case 'h': usage(stdout); exit(0);
        case 'V': printf("tierlm-setup %s\n", SETUP_VERSION); exit(0);
        default:
            if (optopt) usage_error("option -%c needs an argument or is unknown", optopt);
            usage_error("unknown option '%s'", argv[optind - 1]);
        }
    }
    if (optind < argc) usage_error("unexpected argument '%s'", argv[optind]);
    size_t el = strlen(o->etc);
    while (el > 1 && o->etc[el - 1] == '/') o->etc[--el] = 0;
    if (o->probe_only && (o->auto_yes || o->answers))
        usage_error("--probe only refreshes hardware.conf; drop --probe to also write tierlm.conf");
}

/* ======================================================================
 * probe
 * ====================================================================== */
typedef struct {
    char path[PATH_MAX];
    char note[200];
    int tested;                  /* the read test ran */
    int direct_ok;               /* O_DIRECT reads worked */
    uint64_t capacity, free_bytes, test_bytes;
    double buffered_gbs, direct_gbs;
} drive_t;

typedef struct { int card; unsigned vendor; uint64_t vram_total, vram_used; } gpu_t;

#define MAX_GPUS 16
typedef struct {
    int ran, seconds;
    char date[32];
    tlm_cpu_info cpu;
    tlm_mem_info mem;
    int mem_ok;
    uint64_t ram;                /* MemTotal, capped by a cgroup limit */
    char thp[16];                /* always | madvise | never | "" */
    uint64_t hugetlb_bytes;      /* HugePages_Total x Hugepagesize */
    double dram_gbs;
    int dram_threads;
    uint64_t dram_buf;
    char dram_note[160];
    int uring_sysctl;            /* -1: the sysctl does not exist */
    int uring_kernel, uring_errno, uring_lib;
    drive_t *drives;
    int ndrives;
    gpu_t gpus[MAX_GPUS];
    int ngpus, namd;             /* namd: AMD GPUs with >= 4 GiB VRAM, hip:0.. in card order */
    int hip_runtime;
    char hip_path[PATH_MAX];
    uint64_t memlock;            /* soft RLIMIT_MEMLOCK; UINT64_MAX = unlimited */
    int nvme_ps_latency;         /* -1 unknown */
    char tuned[64];
} probe_t;

/* -- DRAM read bandwidth: one thread per physical core, each streaming its
 *    own slice of a buffer far larger than any cache. */
typedef struct {
    const uint64_t *p;
    size_t n;
    atomic_int *go, *stop;
    uint64_t bytes, end_ns, sink;
} dram_job;

static uint64_t sum_words(const uint64_t *p, size_t n) {
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 += p[i];
        a1 += p[i + 1];
        a2 += p[i + 2];
        a3 += p[i + 3];
    }
    for (; i < n; i++) a0 += p[i];
    return a0 ^ a1 ^ a2 ^ a3;
}

static void *dram_main(void *arg) {
    dram_job *j = (dram_job *)arg;
    while (!atomic_load(j->go)) sched_yield();
    uint64_t acc = 0, bytes = 0;
    do {
        acc += sum_words(j->p, j->n);
        bytes += (uint64_t)j->n * 8u;
    } while (!atomic_load_explicit(j->stop, memory_order_relaxed));
    j->end_ns = tlm_now_ns();
    j->bytes = bytes;
    j->sink = acc;
    return NULL;
}

#define MAX_DRAM_THREADS 256
static void probe_dram(probe_t *P) {
    int nt = P->cpu.physical_cores;
    if (nt < 1) nt = 1;
    if (nt > MAX_DRAM_THREADS) nt = MAX_DRAM_THREADS;
    uint64_t cap = 512 * MIB_U;
    if (P->mem_ok && P->mem.available_bytes / 4 < cap) cap = P->mem.available_bytes / 4;
    uint64_t unit = (uint64_t)nt * 4096u;
    cap = cap / unit * unit;
    if (cap < 16 * MIB_U) {
        char b[32];
        snprintf(P->dram_note, sizeof P->dram_note, "skipped: only %s of RAM available",
                 gib(P->mem.available_bytes, b, sizeof b));
        return;
    }
    void *buf = tlm_map_anon((size_t)cap, 1, 0);
    if (!buf) {
        snprintf(P->dram_note, sizeof P->dram_note, "skipped: cannot map a %llu MiB buffer",
                 (unsigned long long)(cap / MIB_U));
        return;
    }
    memset(buf, 0x5a, (size_t)cap);                 /* fault every page in before timing */
    dram_job jobs[MAX_DRAM_THREADS];
    pthread_t th[MAX_DRAM_THREADS];
    atomic_int go, stop;
    atomic_init(&go, 0);
    atomic_init(&stop, 0);
    size_t per = (size_t)(cap / 8u / (uint64_t)nt);
    int started = 0;
    for (int i = 0; i < nt; i++) {
        jobs[i].p = (const uint64_t *)buf + (size_t)i * per;
        jobs[i].n = per;
        jobs[i].go = &go;
        jobs[i].stop = &stop;
        jobs[i].bytes = jobs[i].end_ns = jobs[i].sink = 0;
        if (pthread_create(&th[i], NULL, dram_main, &jobs[i]) != 0) break;
        started++;
    }
    if (started == 0) {
        tlm_unmap_anon(buf, (size_t)cap);
        snprintf(P->dram_note, sizeof P->dram_note, "skipped: cannot start threads");
        return;
    }
    uint64_t t0 = tlm_now_ns();
    atomic_store(&go, 1);
    sleep_seconds((double)P->seconds);
    atomic_store(&stop, 1);
    uint64_t bytes = 0, tend = t0;
    for (int i = 0; i < started; i++) {
        pthread_join(th[i], NULL);
        bytes += jobs[i].bytes;
        if (jobs[i].end_ns > tend) tend = jobs[i].end_ns;
    }
    tlm_unmap_anon(buf, (size_t)cap);
    if (tend > t0) P->dram_gbs = (double)bytes / (double)(tend - t0);   /* bytes/ns = GB/s */
    P->dram_threads = started;
    P->dram_buf = (uint64_t)per * 8u * (uint64_t)started;
}

/* -- drive read test: random 4 MiB reads of a temp file from several threads,
 *    buffered (pages dropped after every read, so RAM is never measured) and
 *    O_DIRECT. */
#define DRIVE_BLOCK (4u * 1024u * 1024u)
#define DRIVE_THREADS 4

typedef struct {
    int fd, direct;
    uint64_t fsize;
    atomic_int *stop;
    uint64_t seed, bytes, end_ns;
    int err;
} rd_job;

static void *rd_main(void *arg) {
    rd_job *j = (rd_job *)arg;
    void *buf = tlm_alloc_aligned(DRIVE_BLOCK, TLM_ALIGN_IO);
    if (!buf) {
        j->err = ENOMEM;
        j->end_ns = tlm_now_ns();
        return NULL;
    }
    uint64_t nblocks = j->fsize / DRIVE_BLOCK, x = j->seed | 1u;
    while (!atomic_load_explicit(j->stop, memory_order_relaxed)) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        off_t off = (off_t)((x % nblocks) * DRIVE_BLOCK);
        ssize_t r = pread(j->fd, buf, DRIVE_BLOCK, off);
        if (r < 0) {
            if (errno == EINTR) continue;
            j->err = errno;
            break;
        }
        if ((size_t)r != DRIVE_BLOCK) {
            j->err = EIO;
            break;
        }
        j->bytes += (uint64_t)r;
        if (!j->direct) (void)posix_fadvise(j->fd, off, (off_t)DRIVE_BLOCK, POSIX_FADV_DONTNEED);
    }
    j->end_ns = tlm_now_ns();
    tlm_free_aligned(buf);
    return NULL;
}

/* Returns GB/s, or a negative value with *err set. */
static double run_reads(int fd, int direct, uint64_t fsize, double secs, int *err) {
    rd_job jobs[DRIVE_THREADS];
    pthread_t th[DRIVE_THREADS];
    atomic_int stop;
    atomic_init(&stop, 0);
    int started = 0;
    *err = 0;
    uint64_t t0 = tlm_now_ns();
    for (int i = 0; i < DRIVE_THREADS; i++) {
        jobs[i] = (rd_job){fd, direct, fsize, &stop, 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1 + direct * 8), 0, 0, 0};
        if (pthread_create(&th[i], NULL, rd_main, &jobs[i]) != 0) break;
        started++;
    }
    if (!started) {
        *err = EAGAIN;
        return -1;
    }
    sleep_seconds(secs);
    atomic_store(&stop, 1);
    uint64_t bytes = 0, tend = t0;
    for (int i = 0; i < started; i++) {
        pthread_join(th[i], NULL);
        bytes += jobs[i].bytes;
        if (jobs[i].end_ns > tend) tend = jobs[i].end_ns;
        if (jobs[i].err && !*err) *err = jobs[i].err;
    }
    if (*err && bytes == 0) return -1;
    return tend > t0 ? (double)bytes / (double)(tend - t0) : -1;
}

static void probe_drive(drive_t *d, int seconds) {
    struct statvfs sv;
    if (statvfs(d->path, &sv) == 0) {
        d->capacity = (uint64_t)sv.f_blocks * (uint64_t)sv.f_frsize;
        d->free_bytes = (uint64_t)sv.f_bavail * (uint64_t)sv.f_frsize;
    }
    if (access(d->path, W_OK) != 0) {
        snprintf(d->note, sizeof d->note, "not tested: the directory is not writable by this user");
        return;
    }
    uint64_t want = (uint64_t)seconds * 64u * MIB_U;
    if (want < 64 * MIB_U) want = 64 * MIB_U;
    if (want > GIB_U) want = GIB_U;
    if (d->free_bytes / 8 < want) want = d->free_bytes / 8;
    want = want / DRIVE_BLOCK * DRIVE_BLOCK;
    if (want < 4u * DRIVE_BLOCK) {
        snprintf(d->note, sizeof d->note, "not tested: not enough free space for a test file");
        return;
    }
    char tmpl[PATH_MAX + 32];
    snprintf(tmpl, sizeof tmpl, "%s/.tierlm-probe-XXXXXX", d->path);
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        snprintf(d->note, sizeof d->note, "not tested: cannot create a test file: %s", strerror(errno));
        return;
    }
    uint64_t *wbuf = (uint64_t *)tlm_alloc_aligned(DRIVE_BLOCK, TLM_ALIGN_IO);
    int ok = wbuf != NULL;
    uint64_t x = 0x2545F4914F6CDD1Dull;
    for (uint64_t off = 0; ok && off < want; off += DRIVE_BLOCK) {
        for (size_t i = 0; i < DRIVE_BLOCK / 8; i++) {   /* incompressible content */
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            wbuf[i] = x;
        }
        if (tlm_write_full(fd, wbuf, DRIVE_BLOCK) != 0) ok = 0;
    }
    tlm_free_aligned(wbuf);
    if (ok && fsync(fd) != 0) ok = 0;
    int dfd = -1;
#ifdef O_DIRECT
    if (ok) dfd = open(tmpl, O_RDONLY | O_CLOEXEC | O_DIRECT);
#endif
    int derr = dfd < 0 ? errno : 0;
    unlink(tmpl);                                  /* nothing is left behind, even on a crash */
    if (!ok) {
        snprintf(d->note, sizeof d->note, "not tested: writing the test file failed: %s", strerror(errno));
        close(fd);
        if (dfd >= 0) close(dfd);
        return;
    }
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    d->test_bytes = want;
    double half = seconds / 2.0;
    int err = 0;
    double g = run_reads(fd, 0, want, half, &err);
    d->tested = 1;
    if (g > 0) d->buffered_gbs = g;
    if (dfd >= 0) {
        double gd = run_reads(dfd, 1, want, half, &err);
        if (gd > 0) {
            d->direct_gbs = gd;
            d->direct_ok = 1;
        } else {
            snprintf(d->note, sizeof d->note, "O_DIRECT reads failed: %s", strerror(err ? err : EIO));
        }
        close(dfd);
    } else {
        snprintf(d->note, sizeof d->note, "O_DIRECT not supported here: %s", strerror(derr));
    }
    close(fd);
}

static void probe_uring(probe_t *P) {
    char b[32];
    P->uring_sysctl = -1;
    if (read_small("/proc/sys/kernel/io_uring_disabled", b, sizeof b) == 0) P->uring_sysctl = atoi(b);
#ifdef __NR_io_uring_setup
    unsigned char params[256];                     /* struct io_uring_params is 120 bytes */
    memset(params, 0, sizeof params);
    long fd = syscall(__NR_io_uring_setup, 1u, params);
    if (fd >= 0) {
        close((int)fd);
        P->uring_kernel = 1;
    } else {
        P->uring_errno = errno;
    }
#else
    P->uring_errno = ENOSYS;
#endif
#ifdef HAVE_LIBURING
    P->uring_lib = 1;
#endif
}

static void probe_gpus(probe_t *P) {
    glob_t g;
    memset(&g, 0, sizeof g);
    if (glob("/sys/class/drm/card[0-9]*", 0, NULL, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc && P->ngpus < MAX_GPUS; i++) {
            const char *name = strrchr(g.gl_pathv[i], '/');
            name = name ? name + 1 : g.gl_pathv[i];
            char *end = NULL;
            long card = strtol(name + 4, &end, 10);
            if (!end || *end) continue;              /* card0-DP-1 etc. are connectors */
            char path[PATH_MAX], b[64];
            snprintf(path, sizeof path, "%s/device/vendor", g.gl_pathv[i]);
            if (read_small(path, b, sizeof b) != 0) continue;
            gpu_t *gp = &P->gpus[P->ngpus];
            memset(gp, 0, sizeof *gp);
            gp->card = (int)card;
            gp->vendor = (unsigned)strtoul(b, NULL, 16);
            snprintf(path, sizeof path, "%s/device/mem_info_vram_total", g.gl_pathv[i]);
            if (read_small(path, b, sizeof b) == 0) gp->vram_total = strtoull(b, NULL, 10);
            snprintf(path, sizeof path, "%s/device/mem_info_vram_used", g.gl_pathv[i]);
            if (read_small(path, b, sizeof b) == 0) gp->vram_used = strtoull(b, NULL, 10);
            P->ngpus++;
        }
    }
    globfree(&g);
    for (int i = 1; i < P->ngpus; i++) {             /* card order (glob sorts card10 < card2) */
        gpu_t t = P->gpus[i];
        int j = i - 1;
        while (j >= 0 && P->gpus[j].card > t.card) {
            P->gpus[j + 1] = P->gpus[j];
            j--;
        }
        P->gpus[j + 1] = t;
    }
    for (int i = 0; i < P->ngpus; i++)
        if (P->gpus[i].vendor == 0x1002 && P->gpus[i].vram_total >= 4 * GIB_U) P->namd++;
    const char *roots[2] = {getenv("ROCM_PATH"), "/opt/rocm"};
    for (int r = 0; r < 2 && !P->hip_runtime; r++) {
        if (!roots[r] || !*roots[r]) continue;
        static const char *libs[] = {"lib/libamdhip64.so*", "lib64/libamdhip64.so*"};
        for (int k = 0; k < 2 && !P->hip_runtime; k++) {
            char pat[PATH_MAX];
            snprintf(pat, sizeof pat, "%s/%s", roots[r], libs[k]);
            glob_t h;
            memset(&h, 0, sizeof h);
            if (glob(pat, 0, NULL, &h) == 0 && h.gl_pathc > 0) {
                P->hip_runtime = 1;
                snprintf(P->hip_path, sizeof P->hip_path, "%s", h.gl_pathv[0]);
            }
            globfree(&h);
        }
    }
}

static const char *vendor_name(unsigned v) {
    switch (v) {
    case 0x1002: return "amd";
    case 0x10de: return "nvidia";
    case 0x8086: return "intel";
    default: return "other";
    }
}

static void simd_list(uint32_t f, char *out, size_t cap) {
    static const struct { uint32_t bit; const char *name; } k[] = {
        {TLM_CPU_SSE42, "sse4.2"}, {TLM_CPU_AVX2, "avx2"}, {TLM_CPU_FMA, "fma"}, {TLM_CPU_F16C, "f16c"},
        {TLM_CPU_AVX512F, "avx512f"}, {TLM_CPU_AVX512BW, "avx512bw"}, {TLM_CPU_AVX512VL, "avx512vl"},
        {TLM_CPU_AVX512VNNI, "avx512vnni"}, {TLM_CPU_AVX512BF16, "avx512bf16"},
        {TLM_CPU_AVX512VBMI, "avx512vbmi"}, {TLM_CPU_AVXVNNI, "avxvnni"}, {TLM_CPU_NEON, "neon"},
        {TLM_CPU_DOTPROD, "dotprod"}};
    size_t n = 0;
    out[0] = 0;
    for (size_t i = 0; i < sizeof k / sizeof k[0]; i++) {
        if (!(f & k[i].bit)) continue;
        int w = snprintf(out + n, cap - n, "%s%s", n ? ", " : "", k[i].name);
        if (w < 0 || (size_t)w >= cap - n) break;
        n += (size_t)w;
    }
    if (!n) snprintf(out, cap, "none");
}

static void run_probe(probe_t *P, const strlist *drives, int seconds) {
    memset(P, 0, sizeof *P);
    P->ran = 1;
    P->seconds = seconds;
    time_t t = time(NULL);
    struct tm tm;
    gmtime_r(&t, &tm);
    strftime(P->date, sizeof P->date, "%Y-%m-%dT%H:%M:%SZ", &tm);
    P->cpu = *tlm_cpu();
    P->mem_ok = tlm_mem_snapshot(&P->mem) == 0;
    P->ram = P->mem.total_bytes;
    if (P->mem.cgroup_limit && P->mem.cgroup_limit < P->ram) P->ram = P->mem.cgroup_limit;
    char b[256];
    if (read_small("/sys/kernel/mm/transparent_hugepage/enabled", b, sizeof b) == 0) {
        char *l = strchr(b, '['), *r = l ? strchr(l, ']') : NULL;
        if (l && r && r - l - 1 < (long)sizeof P->thp) snprintf(P->thp, sizeof P->thp, "%.*s", (int)(r - l - 1), l + 1);
    }
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        unsigned long long v;
        while (fgets(line, sizeof line, f))
            if (sscanf(line, "HugePages_Total: %llu", &v) == 1) P->hugetlb_bytes = v * P->mem.hugepage_bytes;
        fclose(f);
    }
    struct rlimit rl;
    P->memlock = 0;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0)
        P->memlock = rl.rlim_cur == RLIM_INFINITY ? UINT64_MAX : (uint64_t)rl.rlim_cur;
    P->nvme_ps_latency = -1;
    if (read_small("/sys/module/nvme_core/parameters/default_ps_max_latency_us", b, sizeof b) == 0)
        P->nvme_ps_latency = atoi(b);
    if (read_small("/etc/tuned/active_profile", b, sizeof b) == 0) snprintf(P->tuned, sizeof P->tuned, "%.63s", b);
    probe_uring(P);
    probe_gpus(P);

    printf("Probing DRAM read bandwidth for %d s ...\n", seconds);
    fflush(stdout);
    probe_dram(P);
    if (drives->n) {
        P->drives = (drive_t *)xrealloc(NULL, (size_t)drives->n * sizeof *P->drives);
        memset(P->drives, 0, (size_t)drives->n * sizeof *P->drives);
        P->ndrives = drives->n;
        for (int i = 0; i < drives->n; i++) {
            snprintf(P->drives[i].path, sizeof P->drives[i].path, "%s", drives->v[i]);
            printf("Probing %s for %d s (buffered, then O_DIRECT) ...\n", drives->v[i], seconds);
            fflush(stdout);
            probe_drive(&P->drives[i], seconds);
        }
    }
}

static void print_probe(const probe_t *P) {
    char b1[32], b2[32], b3[256];
    printf("\nProbe results (%s):\n", P->date);
    simd_list(P->cpu.features, b3, sizeof b3);
    printf("  CPU       %s\n", P->cpu.model_name[0] ? P->cpu.model_name : "(unknown model)");
    printf("            %d cores / %d threads, %d L3 group%s x %d cores; SIMD: %s\n", P->cpu.physical_cores,
           P->cpu.logical_cpus, P->cpu.l3_groups, P->cpu.l3_groups == 1 ? "" : "s", P->cpu.cores_per_l3, b3);
    if (P->mem_ok)
        printf("  Memory    %s total, %s available%s; THP %s; %s%s\n", gib(P->ram, b1, sizeof b1),
               gib(P->mem.available_bytes, b2, sizeof b2), P->mem.cgroup_limit ? " (cgroup limit)" : "",
               P->thp[0] ? P->thp : "unknown", P->hugetlb_bytes ? gib(P->hugetlb_bytes, b3, sizeof b3) : "no",
               " huge pages reserved");
    else
        printf("  Memory    unknown (no /proc/meminfo)\n");
    if (P->dram_threads)
        printf("  DRAM      read %.1f GB/s (%d threads over %llu MiB)\n", P->dram_gbs, P->dram_threads,
               (unsigned long long)(P->dram_buf / MIB_U));
    else
        printf("  DRAM      %s\n", P->dram_note);
    printf("  io_uring  kernel.io_uring_disabled = %s; io_uring_setup %s; this build %s liburing\n",
           P->uring_sysctl < 0 ? "(absent)" : (snprintf(b1, sizeof b1, "%d", P->uring_sysctl), b1),
           P->uring_kernel ? "works" : strerror(P->uring_errno), P->uring_lib ? "has" : "has no");
    if (!P->ndrives) printf("  Drives    none tested (use --drive DIR)\n");
    for (int i = 0; i < P->ndrives; i++) {
        const drive_t *d = &P->drives[i];
        printf("  Drive     %s: %s free of %s\n", d->path, gib(d->free_bytes, b1, sizeof b1),
               gib(d->capacity, b2, sizeof b2));
        if (d->tested) {
            printf("            buffered %.2f GB/s", d->buffered_gbs);
            if (d->direct_ok) printf(", O_DIRECT %.2f GB/s", d->direct_gbs);
            printf(" (%llu MiB test file, %d threads x 4 MiB reads)\n", (unsigned long long)(d->test_bytes / MIB_U),
                   DRIVE_THREADS);
        }
        if (d->note[0]) printf("            %s\n", d->note);
    }
    if (!P->ngpus) printf("  GPUs      none found in /sys/class/drm\n");
    for (int i = 0; i < P->ngpus; i++)
        printf("  GPU       card%d %s, VRAM %s (%s used)\n", P->gpus[i].card, vendor_name(P->gpus[i].vendor),
               P->gpus[i].vram_total ? gib(P->gpus[i].vram_total, b1, sizeof b1) : "unknown",
               gib(P->gpus[i].vram_used, b2, sizeof b2));
    if (P->ngpus) printf("            HIP runtime: %s\n", P->hip_runtime ? P->hip_path : "not found (ROCM_PATH, /opt/rocm)");
}

static void build_hw_lines(lines_t *L, const probe_t *P) {
    char v[256], c[64];
    ln_group(L, "Probe");
    ln_add(L, "hardware.probe.date", P->date, NULL);
    ln_add(L, "hardware.probe.tool", "tierlm-setup " SETUP_VERSION, NULL);
    ln_addf(L, "hardware.probe.seconds", NULL, "%d", P->seconds);

    ln_group(L, "CPU");
    if (P->cpu.model_name[0]) ln_add(L, "hardware.cpu.model", P->cpu.model_name, NULL);
    ln_addf(L, "hardware.cpu.logical", NULL, "%d", P->cpu.logical_cpus);
    ln_addf(L, "hardware.cpu.cores", "physical cores", "%d", P->cpu.physical_cores);
    ln_addf(L, "hardware.cpu.l3_groups", "L3 domains (CCDs on Zen)", "%d", P->cpu.l3_groups);
    ln_addf(L, "hardware.cpu.cores_per_l3", NULL, "%d", P->cpu.cores_per_l3);
    simd_list(P->cpu.features, v, sizeof v);
    ln_add(L, "hardware.cpu.simd", v, NULL);

    ln_group(L, "Memory");
    tlm_cfg_fmt_size(P->ram, v, sizeof v);
    ln_add(L, "hardware.mem.total", v, gib(P->ram, c, sizeof c));
    tlm_cfg_fmt_size(P->mem.available_bytes, v, sizeof v);
    ln_add(L, "hardware.mem.available", v, gib(P->mem.available_bytes, c, sizeof c));
    tlm_cfg_fmt_size(P->mem.cgroup_limit, v, sizeof v);
    ln_add(L, "hardware.mem.cgroup_limit", v, P->mem.cgroup_limit ? NULL : "no cgroup limit");
    tlm_cfg_fmt_size(P->mem.hugepage_bytes, v, sizeof v);
    ln_add(L, "hardware.mem.hugepage_size", v, NULL);
    tlm_cfg_fmt_size(P->hugetlb_bytes, v, sizeof v);
    ln_add(L, "hardware.mem.hugetlb_reserved", v, "vm.nr_hugepages x hugepage_size");
    ln_add(L, "hardware.mem.thp", P->thp[0] ? P->thp : "unknown", NULL);
    if (P->dram_threads) {
        ln_addf(L, "hardware.mem.read_gbs", "multi-threaded streaming read, GB/s (1e9 bytes)", "%.1f", P->dram_gbs);
        ln_addf(L, "hardware.mem.read_threads", NULL, "%d", P->dram_threads);
        tlm_cfg_fmt_size(P->dram_buf, v, sizeof v);
        ln_add(L, "hardware.mem.read_buffer", v, NULL);
    } else {
        ln_add(L, "hardware.mem.read_gbs", "unknown", P->dram_note);
    }
    if (P->memlock == UINT64_MAX) ln_add(L, "hardware.mem.memlock_limit", "unlimited", "RLIMIT_MEMLOCK of this user");
    else {
        tlm_cfg_fmt_size(P->memlock, v, sizeof v);
        ln_add(L, "hardware.mem.memlock_limit", v, "RLIMIT_MEMLOCK of this user");
    }

    ln_group(L, "I/O");
    if (P->uring_sysctl < 0) ln_add(L, "hardware.io_uring.sysctl", "absent", "kernel.io_uring_disabled");
    else ln_addf(L, "hardware.io_uring.sysctl", "kernel.io_uring_disabled", "%d", P->uring_sysctl);
    ln_add(L, "hardware.io_uring.kernel", P->uring_kernel ? "yes" : "no",
           P->uring_kernel ? "io_uring_setup works for this user" : strerror(P->uring_errno));
    ln_add(L, "hardware.io_uring.liburing", P->uring_lib ? "yes" : "no", "this build (make URING=1)");
    if (P->nvme_ps_latency >= 0)
        ln_addf(L, "hardware.nvme.ps_max_latency_us", "nvme_core.default_ps_max_latency_us", "%d", P->nvme_ps_latency);
    ln_addf(L, "hardware.drive.count", NULL, "%d", P->ndrives);
    for (int i = 0; i < P->ndrives; i++) {
        const drive_t *d = &P->drives[i];
        char k[96];
#define DK(suffix) (snprintf(k, sizeof k, "hardware.drive.%d.%s", i, suffix), k)
        ln_add(L, DK("path"), d->path, NULL);
        tlm_cfg_fmt_size(d->capacity, v, sizeof v);
        ln_add(L, DK("capacity"), v, gib(d->capacity, c, sizeof c));
        tlm_cfg_fmt_size(d->free_bytes, v, sizeof v);
        ln_add(L, DK("free"), v, gib(d->free_bytes, c, sizeof c));
        if (d->tested) {
            tlm_cfg_fmt_size(d->test_bytes, v, sizeof v);
            ln_add(L, DK("test_size"), v, NULL);
            ln_addf(L, DK("buffered_gbs"), "4 MiB random reads, page cache dropped", "%.2f", d->buffered_gbs);
            if (d->direct_ok) ln_addf(L, DK("direct_gbs"), "4 MiB random reads, O_DIRECT", "%.2f", d->direct_gbs);
            ln_add(L, DK("direct"), d->direct_ok ? "yes" : "unsupported", NULL);
        }
        if (d->note[0]) ln_add(L, DK("note"), d->note, NULL);
#undef DK
    }

    ln_group(L, "GPUs");
    ln_addf(L, "hardware.gpu.count", NULL, "%d", P->ngpus);
    ln_addf(L, "hardware.gpu.hip_devices", "AMD GPUs >= 4 GiB VRAM, hip:0.. in card order", "%d", P->namd);
    ln_add(L, "hardware.gpu.hip_runtime", P->hip_runtime ? "yes" : "no", P->hip_runtime ? P->hip_path : NULL);
    for (int i = 0; i < P->ngpus; i++) {
        const gpu_t *g = &P->gpus[i];
        char k[96];
        snprintf(k, sizeof k, "hardware.gpu.%d.card", i);
        ln_addf(L, k, NULL, "%d", g->card);
        snprintf(k, sizeof k, "hardware.gpu.%d.vendor", i);
        ln_add(L, k, vendor_name(g->vendor), NULL);
        snprintf(k, sizeof k, "hardware.gpu.%d.vram", i);
        tlm_cfg_fmt_size(g->vram_total, v, sizeof v);
        ln_add(L, k, v, g->vram_total ? gib(g->vram_total, c, sizeof c) : "unknown");
        snprintf(k, sizeof k, "hardware.gpu.%d.vram_used", i);
        tlm_cfg_fmt_size(g->vram_used, v, sizeof v);
        ln_add(L, k, v, NULL);
    }
}

/* ======================================================================
 * content taxonomy
 * ====================================================================== */
typedef struct {
    kvlist types;          /* type -> description, file order */
    strlist warnings;
    char *text;            /* the source text (installed when missing) */
    char src[PATH_MAX + 32];
    int from_etc;
} taxonomy_t;

typedef struct { taxonomy_t *t; const char *name; } tax_ctx;

static int tax_cb(void *ctx, const char *key, const char *value, int line) {
    tax_ctx *c = (tax_ctx *)ctx;
    char full[300];
    if (!key || !value) return 0;
    if (strcmp(key, "include") == 0) {
        char *w = xprintf("%s:%d: include is not supported in the taxonomy (ignored)", c->name, line);
        sl_add(&c->t->warnings, w);
        free(w);
        return 0;
    }
    snprintf(full, sizeof full, "content.%s", key);
    if (strcmp(key, "default") == 0 || !tlm_cfg_key_valid(full) || !tlm_cfg_schema_find(full)) {
        char *w = xprintf("%s:%d: '%s' cannot be a content type (ignored)", c->name, line, key);
        sl_add(&c->t->warnings, w);
        free(w);
        return 0;
    }
    kv_set(&c->t->types, key, *value ? value : key, line);
    return 0;
}

static void exe_dir(char *out, size_t cap) {
    char p[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", p, sizeof p - 1);
    out[0] = 0;
    if (n <= 0) return;
    p[n] = 0;
    char *s = strrchr(p, '/');
    if (s) *s = 0;
    snprintf(out, cap, "%s", p);
}

/* <etc>/content-taxonomy.conf, else a copy next to the binary (repo:
 * build/bin/../../etc, install: bin/../share/tierlm), else built in. */
static void load_taxonomy(taxonomy_t *t, const char *etc) {
    memset(t, 0, sizeof *t);
    char path[PATH_MAX + 64], dir[PATH_MAX];
    snprintf(path, sizeof path, "%s/content-taxonomy.conf", etc);
    size_t n = 0;
    if (tlm_read_file(path, &t->text, &n) == 0) {
        t->from_etc = 1;
        snprintf(t->src, sizeof t->src, "%.*s", (int)sizeof t->src - 1, path);
    } else {
        exe_dir(dir, sizeof dir);
        const char *cands[4] = {"../../etc/content-taxonomy.conf", "../etc/content-taxonomy.conf",
                                "../share/tierlm/content-taxonomy.conf", NULL};
        for (int i = 0; dir[0] && cands[i] && !t->text; i++) {
            snprintf(path, sizeof path, "%s/%s", dir, cands[i]);
            char rp[PATH_MAX];
            if (realpath(path, rp) && tlm_read_file(rp, &t->text, &n) == 0) snprintf(t->src, sizeof t->src, "%s", rp);
        }
        if (!t->text && tlm_read_file("/usr/share/tierlm/content-taxonomy.conf", &t->text, &n) == 0)
            snprintf(t->src, sizeof t->src, "/usr/share/tierlm/content-taxonomy.conf");
        if (!t->text) {
            t->text = xstrdup(k_default_taxonomy);
            snprintf(t->src, sizeof t->src, "built-in default");
        }
    }
    tax_ctx c = {t, t->src};
    if (tlm_cfg_parse_text(t->text, t->src, tax_cb, &c) != 0) sl_add(&t->warnings, tlm_errmsg());
    for (int i = 0; i < t->types.n; i++) {          /* a child's parent must exist */
        char parent[256];
        snprintf(parent, sizeof parent, "%s", t->types.v[i].key);
        char *dot = strrchr(parent, '.');
        if (!dot) continue;
        *dot = 0;
        if (!kv_find(&t->types, parent)) {
            char *w = xprintf("%s:%d: '%s' has no parent type '%s'", t->src, t->types.v[i].line, t->types.v[i].key, parent);
            sl_add(&t->warnings, w);
            free(w);
        }
    }
}

static void taxonomy_free(taxonomy_t *t) {
    kv_free(&t->types);
    sl_free(&t->warnings);
    free(t->text);
}

/* ======================================================================
 * items: one per key the tool manages
 * ====================================================================== */
typedef struct {
    char *key;
    int section;
    const char *group;
    char *label;          /* taxonomy description (content types) */
    int is_type;          /* content.<type> */
    char *rec;            /* recommendation; NULL = unset / inherit */
    char *reason;
    int derived;          /* computed from the probe */
    char *warn;           /* the current value no longer fits */
    char *final;          /* chosen value; NULL = unset / inherit */
    int done, answered;
} item_t;

typedef struct { item_t *v; int n, cap; } items_t;

static const struct { const char *key, *group; } k_engine_keys[] = {
    {"cpu.threads", "CPU"}, {"cpu.thread_groups", "CPU"}, {"cpu.pin", "CPU"},
    {"memory.ram_reserve", "Memory"}, {"memory.expert_cache", "Memory"}, {"memory.pin_frac", "Memory"},
    {"memory.hugepages", "Memory"},
    {"io.backend", "I/O"}, {"io.direct", "I/O"}, {"io.queue_depth", "I/O"}, {"io.threads_per_drive", "I/O"},
    {"io.drives", "I/O"}, {"io.prefetch", "I/O"},
    {"kv.device", "KV cache"}, {"kv.vram_frac", "KV cache"}, {"kv.dtype", "KV cache"}, {"kv.page_tokens", "KV cache"},
    {"spec.mode", "Speculative decoding"}, {"spec.shape", "Speculative decoding"}, {"spec.depth", "Speculative decoding"},
    {"spec.union_guard", "Speculative decoding"}, {"spec.restrict_experts", "Speculative decoding"},
    {"quant.experts", "Quantisation and compile defaults"}, {"quant.dense", "Quantisation and compile defaults"},
    {"quant.mtp_head", "Quantisation and compile defaults"}, {"compile.profile", "Quantisation and compile defaults"},
    {"paths.models", "Paths, server and logging"}, {"paths.kv_store", "Paths, server and logging"},
    {"paths.log", "Paths, server and logging"}, {"server.listen", "Paths, server and logging"},
    {"log.level", "Paths, server and logging"},
};

static const struct { const char *key, *group; } k_content_tail[] = {
    {"experts.protect", "Experts and pruning"}, {"experts.exclude", "Experts and pruning"},
    {"prune.max_ratio", "Experts and pruning"}, {"prune.min_keep_per_layer", "Experts and pruning"},
    {"decide.backend", "Decision engine"}, {"decide.encoder_model", "Decision engine"},
    {"decide.auto_threshold", "Decision engine"}, {"decide.ask_threshold", "Decision engine"},
    {"decide.log", "Decision engine"},
};

/* Reasons for keys whose recommendation is the built-in default. */
static const struct { const char *key, *reason; } k_reasons[] = {
    {"cpu.pin", "pin compute threads to physical cores"},
    {"memory.pin_frac", "share of the expert cache kept for the pinned hot set"},
    {"io.queue_depth", "outstanding reads per drive (io_uring only)"},
    {"io.threads_per_drive", "pread workers per drive"},
    {"io.drives", "no drives given (--drive DIR); experts are read from the model's own paths"},
    {"io.prefetch", "run layer l+1's router on layer l's output and prefetch its experts"},
    {"kv.dtype", "f16 is exact; f8_e4m3 doubles the context that fits"},
    {"kv.page_tokens", "tokens per KV page (paged attention, page-wise save and restore)"},
    {"spec.mode", "set mtp for models compiled with an MTP head"},
    {"spec.shape", "chain drafts"},
    {"spec.depth", "draft length chosen per step by measured utility"},
    {"spec.union_guard", "trim drafts that would pull too many experts from disk"},
    {"spec.restrict_experts", "lossy expert-restricted drafting; opt-in only"},
    {"quant.experts", "q5_k when q6_k does not fit; native keeps shipped FP4/FP8"},
    {"quant.dense", "attention, router, shared experts and embeddings never go below q8_0"},
    {"quant.mtp_head", "the MTP head never goes below q8_0"},
    {"compile.profile", "exact is lossless; balanced/aggressive allow mixed precision and pruning"},
    {"content.default", "depth for every content type not set below"},
    {"experts.protect", "layer:expert ids that are never pruned or demoted"},
    {"experts.exclude", "layer:expert ids to remove (still eval-gated)"},
    {"prune.max_ratio", "cap on the share of routed experts removed, whatever is excluded"},
    {"prune.min_keep_per_layer", "share of experts per layer that always survive"},
    {"decide.backend", "rules are built in; the encoder backend is not ported yet"},
    {"decide.encoder_model", "only used by decide.backend = encoder"},
    {"decide.auto_threshold", "confidence at or above which decisions apply automatically"},
    {"decide.ask_threshold", "below this confidence the safe default is taken"},
    {"decide.log", "JSONL audit log of every decision (tlmc explain replays them)"},
};

static const char *static_reason(const char *key) {
    for (size_t i = 0; i < sizeof k_reasons / sizeof k_reasons[0]; i++)
        if (strcmp(k_reasons[i].key, key) == 0) return k_reasons[i].reason;
    return NULL;
}

static item_t *it_find(const items_t *its, const char *key) {
    for (int i = 0; i < its->n; i++)
        if (strcmp(its->v[i].key, key) == 0) return &its->v[i];
    return NULL;
}

static item_t *it_add(items_t *its, const char *key, int section, const char *group) {
    item_t *e = it_find(its, key);
    if (e) return e;
    if (its->n == its->cap) {
        its->cap = its->cap ? its->cap * 2 : 64;
        its->v = (item_t *)xrealloc(its->v, (size_t)its->cap * sizeof *its->v);
    }
    e = &its->v[its->n++];
    memset(e, 0, sizeof *e);
    e->key = xstrdup(key);
    e->section = section;
    e->group = group;
    return e;
}

static void items_free(items_t *its) {
    for (int i = 0; i < its->n; i++) {
        item_t *e = &its->v[i];
        free(e->key);
        free(e->label);
        free(e->rec);
        free(e->reason);
        free(e->warn);
        free(e->final);
    }
    free(its->v);
    memset(its, 0, sizeof *its);
}

static const char *choice_of(const item_t *it) { return it->done ? it->final : it->rec; }

/* Keep each group contiguous (items added later, e.g. content types from an
 * answers file, join their group); stable, so file order is preserved. */
static void items_group_sort(items_t *its) {
    const char *order[64];
    int ng = 0;
    for (int i = 0; i < its->n; i++) {
        int k = 0;
        while (k < ng && strcmp(order[k], its->v[i].group) != 0) k++;
        if (k == ng && ng < 64) order[ng++] = its->v[i].group;
    }
    item_t *tmp = (item_t *)xrealloc(NULL, (size_t)its->n * sizeof *tmp);
    int m = 0;
    for (int g = 0; g < ng; g++)
        for (int i = 0; i < its->n; i++)
            if (strcmp(its->v[i].group, order[g]) == 0) tmp[m++] = its->v[i];
    if (m == its->n) memcpy(its->v, tmp, (size_t)m * sizeof *tmp);
    free(tmp);
}

static void build_items(items_t *its, const taxonomy_t *tax, const oldfile *own) {
    for (size_t i = 0; i < sizeof k_engine_keys / sizeof k_engine_keys[0]; i++)
        it_add(its, k_engine_keys[i].key, SEC_ENGINE, k_engine_keys[i].group);
    it_add(its, "content.default", SEC_CONTENT, "Content depth");
    for (int i = 0; i < tax->types.n; i++) {
        char *k = xprintf("content.%s", tax->types.v[i].key);
        item_t *it = it_add(its, k, SEC_CONTENT, "Content depth");
        it->is_type = 1;
        set_str(&it->label, tax->types.v[i].val);
        free(k);
    }
    for (int i = 0; i < own->pairs.n; i++) {        /* content types set but not in the taxonomy */
        const char *k = own->pairs.v[i].key;
        if (strncmp(k, "content.", 8) != 0 || strcmp(k, "content.default") == 0 || it_find(its, k)) continue;
        item_t *it = it_add(its, k, SEC_CONTENT, "Content depth");
        it->is_type = 1;
        set_str(&it->label, "(not in the taxonomy)");
    }
    for (size_t i = 0; i < sizeof k_content_tail / sizeof k_content_tail[0]; i++)
        it_add(its, k_content_tail[i].key, SEC_CONTENT, k_content_tail[i].group);
    for (const tlm_cfg_key *k = tlm_cfg_schema; k->key; k++)   /* keys added to the schema later */
        if (!strchr(k->key, '*') && !it_find(its, k->key)) it_add(its, k->key, SEC_ENGINE, "Other");
}

/* ======================================================================
 * recommendations
 * ====================================================================== */
static void rec_set(item_t *it, const char *val, int derived, const char *fmt, ...)
    __attribute__((format(printf, 4, 5)));
static void rec_set(item_t *it, const char *val, int derived, const char *fmt, ...) {
    set_str(&it->rec, val);
    free(it->reason);
    va_list ap;
    va_start(ap, fmt);
    it->reason = xvprintf(fmt, ap);
    va_end(ap);
    it->derived = derived;
}

static void rec_keep(item_t *it, const oldfile *own) {
    const kv_t *o = kv_find(&own->pairs, it->key);
    if (o) {
        rec_set(it, o->val, 0, "your setting (tierlm.conf:%d)", o->line);
    } else if (it->is_type) {
        rec_set(it, NULL, 0, "%s", "not set: inherits from its parent type or content.default");
    } else {
        const tlm_cfg_key *k = tlm_cfg_schema_find(it->key);
        const char *r = static_reason(it->key);
        rec_set(it, k ? k->def : NULL, 0, "%s", r ? r : (k && k->help ? k->help : ""));
    }
}

static const char *eff_str(const tlm_cfg *eff, const char *key) { return tlm_cfg_str(eff, key); }

static uint64_t reserve_for(uint64_t ram, int *capped) {
    uint64_t r = ram / 10;
    if (r < 4 * GIB_U) r = 4 * GIB_U;
    r = (r + GIB_U - 1) / GIB_U * GIB_U;
    *capped = 0;
    if (ram && r >= ram) {                       /* a machine with <= 4 GiB: keep half for tierlm */
        r = ram / 2 / MIB_U * MIB_U;
        *capped = 1;
    }
    return r;
}

static void recommend(items_t *its, const probe_t *P, const oldfile *own, const opts_t *o) {
    char b1[64], b2[64], v[512];
    for (int i = 0; i < its->n; i++) rec_keep(&its->v[i], own);
    item_t *it;
    if ((it = it_find(its, "decide.backend")) && !kv_find(&own->pairs, "decide.backend"))
        rec_set(it, "rules", 0, "%s", "deterministic rules; the encoder backend is not ported yet");
    if (!P->ran) return;

    if ((it = it_find(its, "cpu.threads"))) {
        snprintf(v, sizeof v, "%d", P->cpu.physical_cores);
        rec_set(it, v, 1, "1 per physical core (probe: %d cores, %d threads%s)", P->cpu.physical_cores,
                P->cpu.logical_cpus, P->cpu.logical_cpus > P->cpu.physical_cores ? "; SMT siblings left for I/O" : "");
    }
    if ((it = it_find(its, "cpu.thread_groups"))) {
        if (P->cpu.l3_groups > 1)
            rec_set(it, "ccd", 1, "%d L3 domains (CCDs) x %d cores: each routed expert runs inside one CCD",
                    P->cpu.l3_groups, P->cpu.cores_per_l3);
        else
            rec_set(it, "none", 1, "%s", "one L3 domain: nothing to group");
    }
    int capped = 0;
    uint64_t reserve = reserve_for(P->ram, &capped);
    if ((it = it_find(its, "memory.ram_reserve")) && P->ram) {
        tlm_cfg_fmt_size(reserve, v, sizeof v);
        if (capped) rec_set(it, v, 1, "half of %s RAM (max(4G, 10%%) would take all of it)", gib(P->ram, b1, sizeof b1));
        else rec_set(it, v, 1, "max(4G, 10%% of %s RAM) for the OS and other processes", gib(P->ram, b1, sizeof b1));
    }
    if ((it = it_find(its, "memory.expert_cache")) && P->ram) {
        const kv_t *ov = kv_find(&own->pairs, "memory.expert_cache");
        uint64_t sz = 0, room = P->ram > reserve ? P->ram - reserve : 0;
        if (ov && strcmp(ov->val, "auto") != 0 && tlm_cfg_parse_size(ov->val, &sz) == 0 && sz <= room)
            rec_set(it, ov->val, 0, "your setting (tierlm.conf:%d) fits: at most %s after the reserve", ov->line,
                    gib(room, b1, sizeof b1));
        else
            rec_set(it, "auto", 1, "RAM - reserve - dense weights at load (at most %s here)", gib(room, b1, sizeof b1));
    }
    if ((it = it_find(its, "memory.hugepages")) && P->thp[0]) {
        if (strcmp(P->thp, "never") != 0)
            rec_set(it, "thp", 1, "transparent huge pages are '%s': the slot pool is madvised and pre-faulted", P->thp);
        else if (P->hugetlb_bytes >= GIB_U)
            rec_set(it, "explicit", 1, "THP is off but %s of huge pages are reserved", gib(P->hugetlb_bytes, b1, sizeof b1));
        else
            rec_set(it, "off", 1, "%s", "transparent huge pages are disabled ('never') and none are reserved");
    }
    if ((it = it_find(its, "io.backend"))) {
        if (P->uring_kernel && P->uring_lib)
            rec_set(it, "io_uring", 1, "kernel allows io_uring%s and this build has liburing",
                    P->uring_sysctl == 1 ? " (only for kernel.io_uring_group)" : "");
        else if (P->uring_kernel)
            rec_set(it, "pread", 1, "%s", "kernel allows io_uring but this build has no liburing (make URING=1)");
        else if (P->uring_sysctl >= 1)
            rec_set(it, "pread", 1, "io_uring is %s (kernel.io_uring_disabled = %d)",
                    P->uring_sysctl == 1 ? "restricted to kernel.io_uring_group" : "disabled", P->uring_sysctl);
        else
            rec_set(it, "pread", 1, "io_uring_setup failed here: %s", strerror(P->uring_errno));
    }
    if ((it = it_find(its, "io.direct"))) {
        int tested = 0, faster = 0, best = -1;
        for (int i = 0; i < P->ndrives; i++) {
            if (!P->drives[i].tested) continue;
            tested++;
            if (P->drives[i].direct_ok && P->drives[i].direct_gbs > P->drives[i].buffered_gbs) faster++;
            if (best < 0) best = i;
        }
        if (tested) {
            const drive_t *d = &P->drives[best];
            if (faster == tested)
                rec_set(it, "yes", 1, "O_DIRECT faster on %d of %d tested drive%s (%s: %.2f vs %.2f GB/s buffered)",
                        faster, tested, tested == 1 ? "" : "s", d->path, d->direct_gbs, d->buffered_gbs);
            else if (d->direct_ok)
                rec_set(it, "no", 1, "O_DIRECT not faster on %d of %d tested drive%s (%s: %.2f vs %.2f GB/s buffered)",
                        tested - faster, tested, tested == 1 ? "" : "s", d->path, d->direct_gbs, d->buffered_gbs);
            else
                rec_set(it, "no", 1, "O_DIRECT does not work on %s", d->path);
        } else if (!kv_find(&own->pairs, "io.direct")) {
            rec_set(it, "no", 0, "%s", "no drive measured (--drive DIR); buffered reads are the safe default");
        }
    }
    if ((it = it_find(its, "io.drives")) && o->drives.n) {
        size_t n = 0;
        v[0] = 0;
        for (int i = 0; i < o->drives.n; i++) {
            int w = snprintf(v + n, sizeof v - n, "%s%s", i ? ", " : "", o->drives.v[i]);
            if (w < 0 || (size_t)w >= sizeof v - n) break;
            n += (size_t)w;
        }
        rec_set(it, v, 1, "%d drive%s from --drive", o->drives.n, o->drives.n == 1 ? "" : "s");
    }
    if ((it = it_find(its, "kv.device"))) {
        if (P->namd && P->hip_runtime) {
            size_t n = 0;
            v[0] = 0;
            uint64_t minv = UINT64_MAX;
            for (int i = 0, h = 0; i < P->ngpus; i++) {
                if (P->gpus[i].vendor != 0x1002 || P->gpus[i].vram_total < 4 * GIB_U) continue;
                int w = snprintf(v + n, sizeof v - n, "%ship:%d", h ? ", " : "", h);
                if (w > 0 && (size_t)w < sizeof v - n) n += (size_t)w;
                if (P->gpus[i].vram_total < minv) minv = P->gpus[i].vram_total;
                h++;
            }
            rec_set(it, v, 1, "%d AMD GPU%s (%s VRAM%s) and a HIP runtime; KV falls back to RAM if HIP fails",
                    P->namd, P->namd == 1 ? "" : "s", gib(minv, b1, sizeof b1), P->namd > 1 ? " each" : "");
        } else if (P->namd) {
            rec_set(it, "cpu", 1, "%d AMD GPU%s but no HIP runtime (ROCm) found; KV stays in RAM", P->namd,
                    P->namd == 1 ? "" : "s");
        } else {
            rec_set(it, "cpu", 1, "%s", "no AMD GPU with >= 4 GiB VRAM found; KV stays in RAM");
        }
    }
    if ((it = it_find(its, "kv.vram_frac")) && P->namd) {
        const char *val = it->rec ? it->rec : "0.05";
        double f = 0.05;
        tlm_cfg_parse_frac(val, &f);
        uint64_t minv = UINT64_MAX, used = 0, tot = 0;
        for (int i = 0; i < P->ngpus; i++) {
            if (P->gpus[i].vendor != 0x1002 || P->gpus[i].vram_total < 4 * GIB_U) continue;
            if (P->gpus[i].vram_total < minv) minv = P->gpus[i].vram_total;
            used += P->gpus[i].vram_used;
            tot += P->gpus[i].vram_total;
        }
        rec_set(it, val, 0, "%s of %s per card for KV pages; hard cap 0.10 (cards now %d%% used)",
                gib((uint64_t)((double)minv * f), b1, sizeof b1), gib(minv, b2, sizeof b2),
                tot ? (int)(100.0 * (double)used / (double)tot) : 0);
    }
}

/* Flag current (effective) values that no longer fit this machine. */
static void check_fits(items_t *its, const tlm_cfg *eff, const probe_t *P) {
    char b1[64];
    for (int i = 0; i < its->n; i++) {
        free(its->v[i].warn);
        its->v[i].warn = NULL;
    }
    item_t *it;
    const char *v;
    if (P->ran) {
        int64_t n;
        uint64_t sz;
        if ((it = it_find(its, "cpu.threads")) && (v = eff_str(eff, "cpu.threads")) && tlm_cfg_parse_int(v, &n) == 0 &&
            n > P->cpu.logical_cpus)
            it->warn = xprintf("%s exceeds the %d logical CPUs", v, P->cpu.logical_cpus);
        uint64_t reserve = 0;
        if ((v = eff_str(eff, "memory.ram_reserve")) && tlm_cfg_parse_size(v, &sz) == 0) reserve = sz;
        else {
            int capped;
            reserve = reserve_for(P->ram, &capped);
        }
        if ((it = it_find(its, "memory.ram_reserve")) && P->ram && reserve >= P->ram)
            it->warn = xprintf("not less than the RAM (%s)", gib(P->ram, b1, sizeof b1));
        if ((it = it_find(its, "memory.expert_cache")) && P->ram && (v = eff_str(eff, "memory.expert_cache")) &&
            tlm_cfg_parse_size(v, &sz) == 0 && sz + reserve > P->ram)
            it->warn = xprintf("%s is larger than RAM minus the reserve (%s)", v,
                               gib(P->ram > reserve ? P->ram - reserve : 0, b1, sizeof b1));
        if ((it = it_find(its, "memory.hugepages")) && (v = eff_str(eff, "memory.hugepages")) &&
            strcmp(v, "explicit") == 0 && P->hugetlb_bytes == 0)
            it->warn = xprintf("%s", "no huge pages are reserved (vm.nr_hugepages = 0)");
        if ((it = it_find(its, "io.backend")) && (v = eff_str(eff, "io.backend")) && strcmp(v, "io_uring") == 0 &&
            !(P->uring_kernel && P->uring_lib))
            it->warn = xprintf("io_uring is not usable here: %s", !P->uring_lib ? "this build has no liburing"
                                                                               : "the kernel refuses io_uring_setup");
        int dflag = 0;
        if ((it = it_find(its, "io.direct")) && tlm_cfg_bool(eff, "io.direct", 0)) {
            for (int i = 0; i < P->ndrives && !dflag; i++)
                if (P->drives[i].tested && !P->drives[i].direct_ok) {
                    it->warn = xprintf("O_DIRECT does not work on %s", P->drives[i].path);
                    dflag = 1;
                }
        }
        if ((it = it_find(its, "kv.device"))) {
            const char *dev[16];
            int nd = tlm_cfg_list(eff, "kv.device", dev, 16);
            for (int i = 0; i < nd && !it->warn; i++)
                if (strncmp(dev[i], "hip:", 4) == 0 && atoi(dev[i] + 4) >= P->namd)
                    it->warn = xprintf("%s is not present (%d usable AMD GPU%s found)", dev[i], P->namd,
                                       P->namd == 1 ? "" : "s");
        }
    }
    if ((it = it_find(its, "io.drives"))) {
        const char *dr[64];
        int nd = tlm_cfg_list(eff, "io.drives", dr, 64);
        for (int i = 0; i < nd && !it->warn; i++)
            if (!is_dir(dr[i])) it->warn = xprintf("%s is not an existing directory", dr[i]);
    }
    if ((it = it_find(its, "decide.backend")) && (v = eff_str(eff, "decide.backend")) && strcmp(v, "encoder") == 0)
        it->warn = xprintf("%s", "the encoder backend is not ported yet; decisions fall back to rules");
    if ((it = it_find(its, "decide.encoder_model")) && (v = eff_str(eff, "decide.encoder_model")) && *v &&
        access(v, R_OK) != 0)
        it->warn = xprintf("%s is not readable", v);
}

/* ======================================================================
 * display and prompts
 * ====================================================================== */
static void origin_label(const tlm_cfg *eff, const char *key, const char *etc, char *out, size_t cap) {
    const char *where = NULL;
    if (!tlm_cfg_str(eff, key)) {
        snprintf(out, cap, "unset");
        return;
    }
    tlm_cfg_origin og = tlm_cfg_origin_of(eff, key, &where);
    if (og == TLM_ORIGIN_FILE && where) {
        size_t el = strlen(etc);
        if (strncmp(where, etc, el) == 0 && where[el] == '/') where += el + 1;
    }
    snprintf(out, cap, "%s", where && *where ? where : tlm_cfg_origin_name(og));
}

/* Depth a content type inherits: nearest parent with a value, else content.default. */
static const char *inherited_depth(const items_t *its, const char *key, char *src, size_t cap) {
    char k[256];
    snprintf(k, sizeof k, "%s", key);
    for (;;) {
        char *dot = strrchr(k, '.');
        if (!dot) break;
        *dot = 0;
        if (strcmp(k, "content") == 0) break;
        const item_t *p = it_find(its, k);
        const char *v = p ? choice_of(p) : NULL;
        if (v) {
            snprintf(src, cap, "%s", k);
            return v;
        }
    }
    const item_t *d = it_find(its, "content.default");
    snprintf(src, cap, "content.default");
    return d && choice_of(d) ? choice_of(d) : "standard";
}

static void show_value(const items_t *its, const item_t *it, const char *val, char *out, size_t cap) {
    if (!val) {
        if (it->is_type) {
            char src[256];
            const char *d = inherited_depth(its, it->key, src, sizeof src);
            snprintf(out, cap, "inherit (%s from %s)", d, src);
        } else {
            snprintf(out, cap, "(unset)");
        }
    } else if (!*val) {
        snprintf(out, cap, "(empty)");
    } else {
        snprintf(out, cap, "%s", val);
    }
}

static void print_row(const items_t *its, const item_t *it, const tlm_cfg *eff, const char *etc) {
    char shown[600], cur[600], org[400];
    const char *val = choice_of(it);
    const char *ev = eff_str(eff, it->key);
    show_value(its, it, val, shown, sizeof shown);
    origin_label(eff, it->key, etc, org, sizeof org);
    if (ev) snprintf(cur, sizeof cur, "%s (%s)", *ev ? ev : "(empty)", org);
    else snprintf(cur, sizeof cur, "%s", it->is_type ? "inherit" : "unset");
    int change = !same_str(val, ev);
    printf("  %c %-28s %-24s current: %s\n", change ? '*' : ' ', it->key, shown, cur);
    if (it->label) printf("      %s\n", it->label);
    if (it->done && !same_str(it->final, it->rec)) {
        char rs[600];
        show_value(its, it, it->rec, rs, sizeof rs);
        printf("      %s; recommended %s: %s\n", it->answered ? "from the answers file" : "your choice", rs,
               it->reason ? it->reason : "");
    } else if (it->reason && *it->reason && !it->is_type) {
        printf("      %s\n", it->reason);
    }
    if (it->warn) printf("      ! %s\n", it->warn);
}

static void print_group_header(const char *title) {
    char buf[128];
    int n = snprintf(buf, sizeof buf, "-- %s ", title);
    while (n < 72 && n < (int)sizeof buf - 1) buf[n++] = '-';
    buf[n] = 0;
    printf("\n%s\n", buf);
}

static int read_answer(const char *prompt, char *buf, size_t cap) {
    fputs(prompt, stdout);
    fflush(stdout);
    if (!fgets(buf, (int)cap, stdin)) return -1;
    buf[strcspn(buf, "\r\n")] = 0;
    char *s = buf;
    while (*s == ' ' || *s == '\t') s++;
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\t')) s[--n] = 0;
    memmove(buf, s, n + 1);
    return 0;
}

static int yes_no(const char *q, int dflt) {
    char buf[64];
    for (;;) {
        char p[PATH_MAX + 128];
        snprintf(p, sizeof p, "%.*s [%s] ", PATH_MAX + 64, q, dflt ? "Y/n" : "y/N");
        if (read_answer(p, buf, sizeof buf) != 0) return -1;
        if (!buf[0]) return dflt;
        if (strcasecmp(buf, "y") == 0 || strcasecmp(buf, "yes") == 0) return 1;
        if (strcasecmp(buf, "n") == 0 || strcasecmp(buf, "no") == 0) return 0;
    }
}

/* Validate a value typed or answered for an item; on success *out is the
 * canonical value (NULL = unset/inherit). Returns 0 or prints why. */
static int accept_value(const items_t *its, const item_t *it, const char *in, char **out, char *why, size_t whycap) {
    if (strcmp(in, "-") == 0 || (it->is_type && strcasecmp(in, "inherit") == 0)) {
        const tlm_cfg_key *k = tlm_cfg_schema_find(it->key);
        if (!it->is_type && k && k->def) {
            snprintf(why, whycap, "%s needs a value (default %s)", it->key, k->def);
            return -1;
        }
        *out = NULL;
        return 0;
    }
    size_t cap = strlen(in) * 2 + 64;
    char *norm = (char *)malloc(cap);
    if (!norm) oom();
    int rc = tlm_cfg_validate(it->key, in, norm, cap, why, whycap);
    if (rc == 0 && strcmp(it->key, "decide.ask_threshold") == 0) {
        const item_t *a = it_find(its, "decide.auto_threshold");
        double ask = 0, aut = 0.90;
        tlm_cfg_parse_frac(norm, &ask);
        if (a && choice_of(a)) tlm_cfg_parse_frac(choice_of(a), &aut);
        if (ask > aut) {
            snprintf(why, whycap, "must not exceed decide.auto_threshold (%g)", aut);
            rc = TLM_ERANGE;
        }
    }
    if (rc != 0) {
        free(norm);
        return -1;
    }
    *out = norm;
    return 0;
}

static void explain_key(const item_t *it) {
    const tlm_cfg_key *k = tlm_cfg_schema_find(it->key);
    char desc[256];
    tlm_cfg_describe(k, desc, sizeof desc);
    printf("      %s: %s\n", it->key, k && k->help ? k->help : "");
    printf("      allowed: %s%s\n", desc, it->is_type ? ", or inherit" : "");
    printf("      Enter = recommendation, '=' = keep current, '-' = %s, '*' = accept the rest\n",
           it->is_type ? "inherit" : "unset");
}

/* Returns 0, 1 = accept the rest of the section, -1 = input closed. */
static int prompt_item(items_t *its, item_t *it, const tlm_cfg *eff, const char *etc) {
    char shown[600], buf[1024], why[512], org[400];
    print_row(its, it, eff, etc);
    show_value(its, it, it->rec, shown, sizeof shown);
    for (;;) {
        char p[800];
        snprintf(p, sizeof p, "    %s [%s]: ", it->key, shown);
        if (read_answer(p, buf, sizeof buf) != 0) return -1;
        if (!buf[0] || strcmp(buf, "*") == 0) {
            char *chk = NULL;
            if (it->rec && accept_value(its, it, it->rec, &chk, why, sizeof why) != 0) {
                printf("      ! %s\n", why);          /* e.g. ask_threshold above a lowered auto_threshold */
                continue;
            }
            free(chk);
            set_str(&it->final, it->rec);
            it->done = 1;
            return buf[0] ? 1 : 0;
        }
        if (strcmp(buf, "?") == 0) {
            explain_key(it);
            continue;
        }
        if (strcmp(buf, "=") == 0) {
            const char *ev = eff_str(eff, it->key);
            origin_label(eff, it->key, etc, org, sizeof org);
            set_str(&it->final, ev);
            it->done = 1;
            if (ev && strncmp(org, "env ", 4) == 0) printf("      note: the current value comes from %s\n", org);
            return 0;
        }
        char *out = NULL;
        if (accept_value(its, it, buf, &out, why, sizeof why) != 0) {
            printf("      ! %s\n", why);
            continue;
        }
        free(it->final);
        it->final = out;
        it->done = 1;
        return 0;
    }
}

/* ======================================================================
 * answers file
 * ====================================================================== */
typedef struct { items_t *its; const opts_t *o; int errors; const char *path; } ans_ctx;

static int ans_cb(void *ctx, const char *key, const char *value, int line) {
    ans_ctx *c = (ans_ctx *)ctx;
    char why[512];
    if (strcmp(key, "include") == 0) {
        fprintf(stderr, "tierlm-setup: %s:%d: include is not allowed in an answers file\n", c->path, line);
        c->errors++;
        return 0;
    }
    if (strncmp(key, "hardware.", 9) == 0) {
        fprintf(stderr, "tierlm-setup: %s:%d: %s is measured by the probe, not answered\n", c->path, line, key);
        c->errors++;
        return 0;
    }
    item_t *it = it_find(c->its, key);
    if (!it && strncmp(key, "content.", 8) == 0 && tlm_cfg_schema_find(key)) {
        it = it_add(c->its, key, SEC_CONTENT, "Content depth");
        it->is_type = 1;
        set_str(&it->label, "(not in the taxonomy)");
        rec_set(it, NULL, 0, "%s", "not set: inherits from its parent type or content.default");
    }
    if (!it) {
        tlm_cfg_validate(key, value, NULL, 0, why, sizeof why);
        fprintf(stderr, "tierlm-setup: %s:%d: %s\n", c->path, line, why);
        c->errors++;
        return 0;
    }
    if (!(it->section & c->o->section)) {
        printf("note: %s:%d: %s ignored (section %s not selected)\n", c->path, line, key,
               it->section == SEC_ENGINE ? "engine" : "content");
        return 0;
    }
    char *out = NULL;
    if (accept_value(c->its, it, value, &out, why, sizeof why) != 0) {
        fprintf(stderr, "tierlm-setup: %s:%d: %s: %s\n", c->path, line, key, why);
        c->errors++;
        return 0;
    }
    free(it->final);
    it->final = out;
    it->done = it->answered = 1;
    return 0;
}

static int apply_answers(items_t *its, const opts_t *o) {
    char *text = NULL;
    size_t n = 0;
    if (tlm_read_file(o->answers, &text, &n) != 0) {
        fprintf(stderr, "tierlm-setup: --answers: %s\n", tlm_errmsg());
        return -1;
    }
    ans_ctx c = {its, o, 0, o->answers};
    int rc = tlm_cfg_parse_text(text, o->answers, ans_cb, &c);
    free(text);
    if (rc != 0) {
        fprintf(stderr, "tierlm-setup: %s\n", tlm_errmsg());
        return -1;
    }
    return c.errors ? -1 : 0;
}

/* ======================================================================
 * file generation, change summary, writing
 * ====================================================================== */
static int is_confd_include(const char *pat) {
    size_t n = strlen(pat);
    return n >= 13 && strcmp(pat + n - 13, "conf.d/*.conf") == 0;
}

static void build_conf_lines(lines_t *L, const items_t *its, const oldfile *own) {
    const char *group = NULL;
    for (int i = 0; i < its->n; i++) {
        const item_t *it = &its->v[i];
        if (!it->final) continue;
        if (!group || strcmp(group, it->group) != 0) {
            ln_group(L, it->group);
            group = it->group;
        }
        const char *comment = NULL;
        if (it->is_type) comment = it->label;
        else if (it->answered) comment = "from the answers file";
        else if (it->derived && same_str(it->final, it->rec)) comment = it->reason;
        ln_add(L, it->key, it->final, comment);
    }
    int extra = 0;
    for (int i = 0; i < own->pairs.n; i++) {
        if (it_find(its, own->pairs.v[i].key)) continue;
        if (!extra++) ln_group(L, "Other keys kept from the previous file");
        ln_add(L, own->pairs.v[i].key, own->pairs.v[i].val, NULL);
    }
    if (own->kept.n) {
        ln_group(L, "Lines the previous file had that this version cannot use");
        for (int i = 0; i < own->kept.n; i++) {
            char *c = xprintf("kept %s", own->kept.v[i]);
            ln_add(L, NULL, NULL, c);
            free(c);
        }
    }
    ln_add(L, NULL, NULL, NULL);
    for (int i = 0; i < own->incl.n; i++)
        if (!is_confd_include(own->incl.v[i])) ln_add(L, "include", own->incl.v[i], NULL);
    ln_add(L, "include", "conf.d/*.conf", NULL);
}

static void lines_kv(const lines_t *L, kvlist *out) {
    for (int i = 0; i < L->n; i++)
        if (L->v[i].key && strcmp(L->v[i].key, "include") != 0) kv_set(out, L->v[i].key, L->v[i].value, 0);
}

/* Prints the changed keys; returns the number of changes (a new file counts). */
static int print_changes(const char *label, const oldfile *old, const kvlist *neu, int list_all_new) {
    int changed = 0, added = 0, removed = 0;
    for (int i = 0; i < neu->n; i++) {
        const kv_t *o = kv_find(&old->pairs, neu->v[i].key);
        if (!o) added++;
        else if (strcmp(o->val, neu->v[i].val) != 0) changed++;
    }
    for (int i = 0; i < old->pairs.n; i++)
        if (!kv_find(neu, old->pairs.v[i].key)) removed++;
    for (int i = 0; i < old->kept.n; i++) printf("  %s: %s -> kept as a comment\n", label, old->kept.v[i]);
    if (!old->exists) {
        printf("  %s: new file, %d keys\n", label, neu->n);
        if (list_all_new)
            for (int i = 0; i < neu->n; i++) printf("    + %s = %s\n", neu->v[i].key, neu->v[i].val);
        return neu->n + 1;
    }
    if (!changed && !added && !removed) {
        printf("  %s: no changes\n", label);
        return 0;
    }
    printf("  %s: %d changed, %d added, %d removed\n", label, changed, added, removed);
    for (int i = 0; i < neu->n; i++) {
        const kv_t *o = kv_find(&old->pairs, neu->v[i].key);
        if (o && strcmp(o->val, neu->v[i].val) != 0)
            printf("    ~ %s: %s -> %s\n", neu->v[i].key, o->val, neu->v[i].val);
    }
    for (int i = 0; i < neu->n; i++)
        if (!kv_find(&old->pairs, neu->v[i].key)) printf("    + %s = %s\n", neu->v[i].key, neu->v[i].val);
    for (int i = 0; i < old->pairs.n; i++)
        if (!kv_find(neu, old->pairs.v[i].key))
            printf("    - %s (was %s)\n", old->pairs.v[i].key, old->pairs.v[i].val);
    return changed + added + removed;
}

static int write_with_backup(const char *path, const char *text, size_t len, const char *stamp) {
    char *old = NULL;
    size_t oldn = 0;
    if (tlm_read_file(path, &old, &oldn) == 0) {
        char bak[PATH_MAX + 64];
        snprintf(bak, sizeof bak, "%s.bak.%s", path, stamp);
        for (int k = 1; access(bak, F_OK) == 0 && k < 1000; k++) snprintf(bak, sizeof bak, "%s.bak.%s-%d", path, stamp, k);
        int rc = tlm_write_file_atomic(bak, old, oldn);
        free(old);
        if (rc != 0) return rc;
        printf("  backup  %s\n", bak);
    }
    int rc = tlm_write_file_atomic(path, text, len);
    if (rc == 0) printf("  wrote   %s\n", path);
    return rc;
}

/* ======================================================================
 * system suggestions (--apply-system writes snippets under <etc>/system)
 * ====================================================================== */
static const char k_sysctl_snippet[] =
    "# tierlm: enable io_uring for the expert reader (RHEL ships it disabled).\n"
    "# 0 = allowed for every user; 1 = only root and members of kernel.io_uring_group\n"
    "# (then also set kernel.io_uring_group to the gid of the group that runs tierlm).\n"
    "kernel.io_uring_disabled = 0\n";

static const char k_limits_snippet[] =
    "# tierlm: let the account that runs tierlm lock the expert slot pool in RAM.\n"
    "# Replace @tierlm with that user or group.\n"
    "@tierlm    soft    memlock    unlimited\n"
    "@tierlm    hard    memlock    unlimited\n";

static const char k_tuned_snippet[] =
    "# tierlm tuned profile: throughput-performance with transparent huge pages\n"
    "# in madvise mode (tierlm madvises and pre-faults its slot pool).\n"
    "[main]\n"
    "summary=tierlm: throughput-performance with THP madvise\n"
    "include=throughput-performance\n"
    "\n"
    "[vm]\n"
    "transparent_hugepages=madvise\n";

static void system_suggestions(const probe_t *P, const items_t *its, const tlm_cfg *eff, const opts_t *o, int write) {
    printf("\nSystem suggestions (%s):\n", o->apply_system && write ? "snippets go to <etc>/system/" : "not applied");
    const item_t *ib = it_find(its, "io.backend");
    const char *backend = ib && ib->final ? ib->final : eff_str(eff, "io.backend");
    int want_uring = (backend && strcmp(backend, "io_uring") == 0) || (P->ran && P->uring_lib);
    if (P->ran) {
        if (P->uring_sysctl > 0 && want_uring)
            printf("  [needed] io_uring is %s by kernel.io_uring_disabled = %d -> 90-tierlm-sysctl.conf\n",
                   P->uring_sysctl == 1 ? "restricted" : "disabled", P->uring_sysctl);
        else
            printf("  [ok]     io_uring sysctl (%s)\n", P->uring_sysctl < 0 ? "absent" : P->uring_sysctl == 0 ? "enabled" : "not needed: pread backend");
        uint64_t cache = 0;
        const item_t *ic = it_find(its, "memory.expert_cache");
        const char *cv = ic && ic->final ? ic->final : eff_str(eff, "memory.expert_cache");
        if (!cv || tlm_cfg_parse_size(cv, &cache) != 0) {
            int capped;
            uint64_t r = reserve_for(P->ram, &capped);
            cache = P->ram > r ? P->ram - r : 0;
        }
        char b1[32], b2[32];
        if (P->memlock != UINT64_MAX && P->memlock < cache)
            printf("  [needed] memlock limit %s < expert cache (~%s): the slot pool cannot be mlocked -> 90-tierlm-limits.conf\n",
                   gib(P->memlock, b1, sizeof b1), gib(cache, b2, sizeof b2));
        else
            printf("  [ok]     memlock limit\n");
        if (strcmp(P->thp, "never") == 0)
            printf("  [needed] transparent huge pages are 'never' -> tuned profile 'tierlm' (THP madvise)\n");
        else if (P->tuned[0] && strcmp(P->tuned, "throughput-performance") != 0 && strcmp(P->tuned, "tierlm") != 0)
            printf("  [suggested] tuned profile is '%s'; throughput-performance (or 'tierlm') is recommended\n", P->tuned);
        else
            printf("  [ok]     THP / tuned profile\n");
        if (P->ndrives && P->nvme_ps_latency > 0)
            printf("  [suggested] keep NVMe drives out of deep power states: kernel argument "
                   "nvme_core.default_ps_max_latency_us=0 (now %d)\n", P->nvme_ps_latency);
    }
    if (!o->apply_system) {
        printf("  run with --apply-system to write these as snippets under %s/system/\n", o->etc);
        return;
    }
    char dir[PATH_MAX + 16], path[PATH_MAX + 64];
    snprintf(dir, sizeof dir, "%s/system", o->etc);
    if (!write) {
        printf("  not written: would write %s/{90-tierlm-sysctl.conf,90-tierlm-limits.conf,tuned-tierlm.conf}\n", dir);
        return;
    }
    if (tlm_mkdirs(dir, 0755) != 0) {
        fprintf(stderr, "tierlm-setup: %s\n", tlm_errmsg());
        return;
    }
    static const struct { const char *name; const char *text; } snip[] = {
        {"90-tierlm-sysctl.conf", k_sysctl_snippet},
        {"90-tierlm-limits.conf", k_limits_snippet},
        {"tuned-tierlm.conf", k_tuned_snippet}};
    for (size_t i = 0; i < sizeof snip / sizeof snip[0]; i++) {
        snprintf(path, sizeof path, "%s/%s", dir, snip[i].name);
        if (tlm_write_file_atomic(path, snip[i].text, strlen(snip[i].text)) != 0)
            fprintf(stderr, "tierlm-setup: %s\n", tlm_errmsg());
        else
            printf("  wrote   %s\n", path);
    }
    printf("\n  Review the snippets, then install them as root (nothing outside %s was changed):\n", dir);
    printf("    install -m 0644 %s/90-tierlm-sysctl.conf /etc/sysctl.d/90-tierlm.conf && sysctl --system\n", dir);
    printf("    install -m 0644 %s/90-tierlm-limits.conf /etc/security/limits.d/90-tierlm.conf\n", dir);
    printf("    mkdir -p /etc/tuned/tierlm && install -m 0644 %s/tuned-tierlm.conf /etc/tuned/tierlm/tuned.conf"
           " && tuned-adm profile tierlm\n", dir);
    printf("    grubby --update-kernel=ALL --args=nvme_core.default_ps_max_latency_us=0   # NVMe power states, then reboot\n");
}

/* ======================================================================
 * --show
 * ====================================================================== */
static int cmd_show(const opts_t *o) {
    tlm_cfg *c = tlm_cfg_new();
    if (!c) oom();
    tlm_cfg_quiet(c, 1);
    int rc = tlm_cfg_load_etc(c, o->etc, o->strict);
    int crc = tlm_cfg_check(c);
    int errors = 0;
    for (int i = 0; i < tlm_cfg_ndiag(c); i++) {
        int is_err = 0;
        const char *d = tlm_cfg_diag(c, i, &is_err);
        fprintf(stderr, "tierlm-setup: %s: %s\n", is_err ? "error" : "warning", d);
        errors += is_err;
    }
    lines_t L = {0};
    int it = 0;
    const char *k, *v;
    while (tlm_cfg_next(c, &it, &k, &v)) {
        char org[PATH_MAX + 64];
        origin_label(c, k, o->etc, org, sizeof org);
        ln_add(&L, k, v, org);
    }
    for (const tlm_cfg_key *s = tlm_cfg_schema; s->key; s++) {
        if (strchr(s->key, '*') || tlm_cfg_str(c, s->key)) continue;
        char *cm = xprintf("%s is unset", s->key);
        ln_add(&L, NULL, NULL, cm);
        free(cm);
    }
    char *hdr = xprintf("effective configuration from %s\n"
                        "precedence (later wins): default < hardware.conf < tierlm.conf < conf.d/*.conf < TIERLM_* env\n"
                        "each line ends with the file:line (relative to that directory), env variable or 'default' that set it",
                        o->etc);
    int wrc = tlm_cfg_write(stdout, hdr, L.v, L.n);
    free(hdr);
    ln_free(&L);
    tlm_cfg_free(c);
    if (wrc != 0) fprintf(stderr, "tierlm-setup: %s\n", tlm_errmsg());
    return (rc != 0 || crc != 0 || errors || wrc != 0) ? 1 : 0;
}

/* ======================================================================
 * sections
 * ====================================================================== */
static const char k_content_note[] =
    "  Depth per content type: exclude | shallow | standard | deep. A child type\n"
    "  inherits its parent's depth; content.default covers every type not set.\n"
    "  Excluding a type never deletes experts directly: it only makes specialist\n"
    "  experts (specialisation focused/narrow, >= 80% of their label mass in\n"
    "  excluded types, < 5% in deep types) pruning candidates. REAP saliency ranks\n"
    "  them, prune.max_ratio and prune.min_keep_per_layer cap them, and the eval\n"
    "  gate decides what goes. Pruning needs compile.profile = balanced or\n"
    "  aggressive; with exact, depths only change calibration weighting and pinning.\n";

/* Returns 0, or -1 when interactive input ended. */
static int run_section(items_t *its, int section, const tlm_cfg *eff, const opts_t *o) {
    const char *group = NULL;
    int accept_rest = 0;
    if (section == SEC_CONTENT) {
        print_group_header("Experts and content");
        fputs(k_content_note, stdout);
    }
    for (int i = 0; i < its->n; i++) {
        item_t *it = &its->v[i];
        if (it->section != section) continue;
        if (!group || strcmp(group, it->group) != 0) {
            print_group_header(it->group);
            group = it->group;
        }
        if (it->done) {
            print_row(its, it, eff, o->etc);
            continue;
        }
        if (o->interactive && !accept_rest) {
            int rc = prompt_item(its, it, eff, o->etc);
            if (rc < 0) return -1;
            if (rc == 1) accept_rest = 1;
            continue;
        }
        set_str(&it->final, it->rec);
        it->done = 1;
        print_row(its, it, eff, o->etc);
    }
    return 0;
}

static void content_summary(const items_t *its, const tlm_cfg *eff) {
    static const char *levels[] = {"deep", "standard", "shallow", "exclude"};
    printf("\nContent summary:\n");
    const item_t *d = it_find(its, "content.default");
    printf("  default   %s\n", d && d->final ? d->final : "standard");
    int nexcl = 0;
    for (int l = 0; l < 4; l++) {
        int n = 0;
        for (int i = 0; i < its->n; i++) {
            const item_t *it = &its->v[i];
            if (!it->is_type || !it->final || strcmp(it->final, levels[l]) != 0) continue;
            if (n == 0) printf("  %-9s %s", levels[l], it->key + 8);
            else printf(", %s", it->key + 8);
            n++;
        }
        if (n) printf("\n");
        if (l == 3) nexcl = n;
    }
    const item_t *pr = it_find(its, "compile.profile");
    const char *profile = pr && pr->final ? pr->final : eff_str(eff, "compile.profile");
    const item_t *mr = it_find(its, "prune.max_ratio"), *mk = it_find(its, "prune.min_keep_per_layer");
    if (nexcl) {
        printf("  Exclusions only make specialist experts pruning candidates; REAP saliency and the\n"
               "  eval gate decide what goes, capped by prune.max_ratio = %s and\n"
               "  prune.min_keep_per_layer = %s.\n",
               mr && mr->final ? mr->final : "0.25", mk && mk->final ? mk->final : "0.5");
        if (profile && strcmp(profile, "exact") == 0)
            printf("  ! compile.profile = exact: exclusions only change calibration weighting and pinning;\n"
                   "    compile with --profile balanced or aggressive (or set compile.profile) to prune.\n");
    }
    const item_t *pt = it_find(its, "experts.protect"), *ex = it_find(its, "experts.exclude");
    const char *pv = pt && pt->final ? pt->final : "", *xv = ex && ex->final ? ex->final : "";
    if (*pv || *xv) {
        printf("  experts   protect: %s; exclude: %s\n", *pv ? pv : "none", *xv ? xv : "none");
        char *a = xstrdup(pv);
        for (char *tok = strtok(a, ", "); tok; tok = strtok(NULL, ", ")) {
            char pat[64];
            snprintf(pat, sizeof pat, "%s", tok);
            const char *p = strstr(xv, pat);
            size_t pl = strlen(pat);
            while (p) {
                int left = p == xv || p[-1] == ' ' || p[-1] == ',';
                int right = p[pl] == 0 || p[pl] == ',' || p[pl] == ' ';
                if (left && right) {
                    printf("  ! expert %s is both protected and excluded: protect wins\n", pat);
                    break;
                }
                p = strstr(p + 1, pat);
            }
        }
        free(a);
    }
}

/* ======================================================================
 * main
 * ====================================================================== */
static void print_diags(const tlm_cfg *c) {
    if (!tlm_cfg_ndiag(c)) return;
    printf("Existing configuration:\n");
    for (int i = 0; i < tlm_cfg_ndiag(c); i++) {
        int is_err = 0;
        const char *d = tlm_cfg_diag(c, i, &is_err);
        printf("  %s: %s\n", is_err ? "error" : "warning", d);
    }
}

int main(int argc, char **argv) {
    opts_t o;
    parse_args(argc, argv, &o);
    if (o.print_taxonomy) {
        fputs(k_default_taxonomy, stdout);
        return 0;
    }
    if (o.show) {
        int rc = cmd_show(&o);
        sl_free(&o.drives);
        return rc;
    }

    int note_tty = 0;
    if (!o.probe_only && !o.dry_run && !o.auto_yes && !o.answers) {
        if (isatty(STDIN_FILENO)) o.interactive = 1;
        else {
            o.dry_run = 1;
            note_tty = 1;
        }
    }
    const char *mode = o.probe_only ? "probe only" : o.interactive ? "interactive" : o.answers ? "answers file"
                     : o.auto_yes ? "accept all recommendations" : "dry run";
    printf("tierlm-setup %s -- %s (%s%s)\n", SETUP_VERSION, o.etc, mode,
           o.dry_run && !note_tty && strcmp(mode, "dry run") != 0 ? ", dry run" : "");
    if (note_tty)
        printf("note: stdin is not a terminal, so this is a dry run: recommendations are shown and nothing is\n"
               "      written. Use --auto to accept them or --answers FILE for an unattended install.\n");

    tlm_cfg *eff = tlm_cfg_new();
    if (!eff) oom();
    tlm_cfg_quiet(eff, 1);
    int lrc = tlm_cfg_load_etc(eff, o.etc, o.strict);
    print_diags(eff);
    if (lrc != 0 && o.strict) {
        fprintf(stderr, "tierlm-setup: %s\n", tlm_errmsg());
        tlm_cfg_free(eff);
        return 1;
    }
    if (!o.dry_run && access(o.etc, F_OK) == 0 && access(o.etc, W_OK) != 0)
        printf("note: %s is not writable by this user; review now, write with sudo (or use --etc DIR)\n", o.etc);

    char conf_path[PATH_MAX + 32], hw_path[PATH_MAX + 32], tax_path[PATH_MAX + 32];
    snprintf(conf_path, sizeof conf_path, "%s/tierlm.conf", o.etc);
    snprintf(hw_path, sizeof hw_path, "%s/hardware.conf", o.etc);
    snprintf(tax_path, sizeof tax_path, "%s/content-taxonomy.conf", o.etc);
    oldfile own, oldhw;
    read_old(conf_path, &own);
    read_old(hw_path, &oldhw);

    /* probe */
    probe_t P;
    memset(&P, 0, sizeof P);
    int do_probe = o.probe_only || (o.section & SEC_ENGINE);
    if (do_probe) {
        strlist drives = {0};
        if (o.drives.n) {
            for (int i = 0; i < o.drives.n; i++) sl_add(&drives, o.drives.v[i]);
        } else {
            const char *dr[64];
            int nd = tlm_cfg_list(eff, "io.drives", dr, 64);
            for (int i = 0; i < nd; i++)
                if (is_dir(dr[i])) sl_add(&drives, dr[i]);
        }
        run_probe(&P, &drives, o.seconds);
        sl_free(&drives);
        print_probe(&P);
    }

    /* taxonomy, items, recommendations */
    taxonomy_t tax;
    load_taxonomy(&tax, o.etc);
    for (int i = 0; i < tax.warnings.n; i++) printf("warning: taxonomy: %s\n", tax.warnings.v[i]);
    items_t its = {0};
    build_items(&its, &tax, &own);
    recommend(&its, &P, &own, &o);
    check_fits(&its, eff, &P);

    int exit_code = 0;
    lines_t hwL = {0};
    if (P.ran) build_hw_lines(&hwL, &P);
    char date[32], stamp[32];
    date_local(date, sizeof date);
    stamp_local(stamp, sizeof stamp);

    if (o.probe_only) {
        run_section(&its, SEC_ENGINE, eff, &o);
        int misfit = 0;
        for (int i = 0; i < its.n; i++) misfit += its.v[i].section == SEC_ENGINE && its.v[i].warn != NULL;
        printf("\nChanges:\n");
        kvlist hwkv = {0};
        lines_kv(&hwL, &hwkv);
        print_changes("hardware.conf", &oldhw, &hwkv, 0);
        kv_free(&hwkv);
        if (!o.dry_run) {
            char *hdr = xprintf("%s -- measured by tierlm-setup %s on %s\n"
                                "facts only: preferences live in tierlm.conf; re-run `tierlm-setup --probe`\n"
                                "after a hardware change", hw_path, SETUP_VERSION, P.date);
            size_t len = 0;
            char *text = render(hdr, &hwL, 0, &len);
            free(hdr);
            if (!text || tlm_mkdirs(o.etc, 0755) != 0 || write_with_backup(hw_path, text, len, stamp) != 0) {
                fprintf(stderr, "tierlm-setup: cannot write %s: %s\n", hw_path, tlm_errmsg());
                exit_code = 1;
            }
            free(text);
        } else {
            printf("dry run: nothing written\n");
        }
        if (misfit)
            printf("\n%d preference%s no longer fit%s this machine (marked !); run tierlm-setup to update tierlm.conf\n",
                   misfit, misfit == 1 ? "" : "s", misfit == 1 ? "s" : "");
        system_suggestions(&P, &its, eff, &o, !o.dry_run && exit_code == 0);
        goto done;
    }

    if (o.answers && apply_answers(&its, &o) != 0) {
        fprintf(stderr, "tierlm-setup: answers file rejected; nothing written\n");
        exit_code = 1;
        goto done;
    }
    items_group_sort(&its);

    /* select */
    if (o.interactive)
        printf("\nFor each key: Enter accepts the recommendation, '=' keeps the current value, "
               "'?' explains, '*' accepts the rest of the section.\n");
    for (int sec = SEC_ENGINE; sec <= SEC_CONTENT; sec <<= 1) {
        if (!(o.section & sec)) continue;
        if (run_section(&its, sec, eff, &o) != 0) {
            printf("\ninput closed; nothing written\n");
            exit_code = 1;
            goto done;
        }
    }
    for (int i = 0; i < its.n; i++) {                /* keys of sections not processed stay as they are */
        item_t *it = &its.v[i];
        if (o.section & it->section) continue;
        const kv_t *ov = kv_find(&own.pairs, it->key);
        set_str(&it->final, ov ? ov->val : NULL);
        it->done = 1;
    }
    if (o.section & SEC_CONTENT) content_summary(&its, eff);

    /* build, validate, show changes */
    lines_t L = {0};
    build_conf_lines(&L, &its, &own);
    char *hdr = xprintf("%s -- generated by tierlm-setup %s on %s\n"
                        "one key per line; '#' starts a comment; later lines override earlier ones\n"
                        "local overrides belong in conf.d/*.conf (read after this file, never rewritten)",
                        conf_path, SETUP_VERSION, date);
    size_t blen = 0, tlen = 0;
    char *body = render(hdr, &L, 1, &blen);
    char *text = body ? render(hdr, &L, 0, &tlen) : NULL;
    free(hdr);
    if (!body || !text) {
        fprintf(stderr, "tierlm-setup: cannot render tierlm.conf: %s\n", tlm_errmsg());
        exit_code = 1;
    } else {
        tlm_cfg *chk = tlm_cfg_new();
        if (!chk) oom();
        tlm_cfg_quiet(chk, 1);
        int vrc = tlm_cfg_load_string(chk, body, conf_path, 1);
        if (vrc == 0) vrc = tlm_cfg_check(chk);
        tlm_cfg_free(chk);
        if (vrc != 0) {
            fprintf(stderr, "tierlm-setup: the new tierlm.conf does not validate: %s\nnothing written\n", tlm_errmsg());
            exit_code = 1;
        }
    }
    if (exit_code == 0) {
        printf("\nChanges:\n");
        kvlist nkv = {0}, hwkv = {0};
        lines_kv(&L, &nkv);
        int nconf = print_changes("tierlm.conf", &own, &nkv, o.dry_run);
        int nhw = 0;
        if (P.ran) {
            lines_kv(&hwL, &hwkv);
            nhw = print_changes("hardware.conf", &oldhw, &hwkv, 0);
        }
        kv_free(&nkv);
        kv_free(&hwkv);
        int tax_install = (o.section & SEC_CONTENT) && !tax.from_etc;
        if (tax_install) printf("  content-taxonomy.conf: install from %s\n", tax.src);

        int go = !o.dry_run;
        if (go && o.interactive && (nconf || nhw || tax_install)) {
            char q[PATH_MAX + 64];
            snprintf(q, sizeof q, "\nWrite these files to %s?", o.etc);
            int yn = yes_no(q, 1);
            if (yn <= 0) {
                printf("nothing written\n");
                go = 0;
            }
        }
        if (go) {
            printf("\nWriting:\n");
            if (tlm_mkdirs(o.etc, 0755) != 0) {
                fprintf(stderr, "tierlm-setup: %s\n", tlm_errmsg());
                exit_code = 1;
            }
            if (!exit_code && P.ran && nhw) {
                char *h = xprintf("%s -- measured by tierlm-setup %s on %s\n"
                                  "facts only: preferences live in tierlm.conf; re-run `tierlm-setup --probe`\n"
                                  "after a hardware change", hw_path, SETUP_VERSION, P.date);
                size_t hl = 0;
                char *ht = render(h, &hwL, 0, &hl);
                free(h);
                if (!ht || write_with_backup(hw_path, ht, hl, stamp) != 0) {
                    fprintf(stderr, "tierlm-setup: cannot write %s: %s\n", hw_path, tlm_errmsg());
                    exit_code = 1;
                }
                free(ht);
            }
            if (!exit_code && tax_install) {
                if (tlm_write_file_atomic(tax_path, tax.text, strlen(tax.text)) != 0) {
                    fprintf(stderr, "tierlm-setup: cannot write %s: %s\n", tax_path, tlm_errmsg());
                    exit_code = 1;
                } else {
                    printf("  wrote   %s (from %s)\n", tax_path, tax.src);
                }
            }
            if (!exit_code && nconf && write_with_backup(conf_path, text, tlen, stamp) != 0) {
                fprintf(stderr, "tierlm-setup: cannot write %s: %s\n", conf_path, tlm_errmsg());
                exit_code = 1;
            }
            if (!exit_code && !nconf) printf("  %s unchanged\n", conf_path);
        } else if (o.dry_run) {
            printf("\ndry run: nothing written\n");
        }
        if (!exit_code) system_suggestions(&P, &its, eff, &o, go);
        if (!exit_code && go)
            printf("\nNext: `tierlm-setup --show` lists every effective value and its origin;\n"
                   "keep local overrides in %s/conf.d/ -- tierlm-setup never rewrites them.\n", o.etc);
    }
    free(body);
    free(text);
    ln_free(&L);

done:
    ln_free(&hwL);
    items_free(&its);
    taxonomy_free(&tax);
    old_free(&own);
    old_free(&oldhw);
    free(P.drives);
    tlm_cfg_free(eff);
    sl_free(&o.drives);
    return exit_code;
}
