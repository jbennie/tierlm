/* tierlm.c - runtime (stage 2 of research/03-two-stage-design.md).
 *
 *   tierlm run    MANIFEST [--prompt TEXT | --tokens 1,2,3] [-n 32] [--temp 0]
 *                 [--top-p P] [--min-p P] [--seed N] [--cache SIZE] [--threads N]
 *                 [--io pread|io_uring] [--direct yes|no] [--prefetch router|off]
 *                 [--pin-frac F] [--warm FILE] [--kernel scalar|avx2|avx512]
 *                 [--stats] [--etc DIR] [--print-tokens]
 *   tierlm bench  MANIFEST [--prompt-len N] [-n N] [runtime flags as for run]
 *   tierlm config show [--origin] [--etc DIR]
 *   tierlm config check [--etc DIR]
 *
 * Settings: built-in defaults < /etc/tierlm (or --etc / TIERLM_ETC) <
 * TIERLM_* environment < flags (tlm_cfg_set_origin, P-113). The runtime does
 * no layout or quantisation work: it validates the manifest, loads the
 * resident packets (verified, P-03/P-05), pins the compiler's hot set (P-19),
 * and streams the rest through the expert store (P-10..P-18, P-23, P-31,
 * P-33/P-34). A drive that is missing is replaced by /dev/null when every
 * packet on it has a replica, so reads fall back to the other copy (P-31).
 * Prompts are byte-level for synthetic models: token = byte mod vocab.
 */
#include "tlm_bpe.h"
#include "tlm_cfg.h"
#include "tlm_io.h"
#include "tlm_model.h"
#include "tlm_packet.h"
#include "tlm_platform.h"
#include "tlm_pool.h"
#include "tlm_quant.h"
#include "tlm_store.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#define TIERLM_VERSION "0.1"

static void die(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));
static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("tierlm: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

static const char *need(int argc, char **argv, int *i) {
    if (*i + 1 >= argc) die("%s needs a value", argv[*i]);
    return argv[++*i];
}

static long num_arg(const char *s, const char *flag, long lo, long hi) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || end == s || v < lo || v > hi)
        die("%s: '%s' is not an integer in [%ld, %ld]", flag, s, lo, hi);
    return v;
}

static double frac_arg(const char *s, const char *flag, double lo, double hi) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (!end || *end || end == s || !(v >= lo && v <= hi)) die("%s: '%s' is not a number in [%g, %g]", flag, s, lo, hi);
    return v;
}

static void fmt_bytes(uint64_t b, char *out, size_t cap) {
    if (b >= (1ull << 30)) snprintf(out, cap, "%.2f GiB", (double)b / (1 << 30));
    else if (b >= (1ull << 20)) snprintf(out, cap, "%.2f MiB", (double)b / (1 << 20));
    else snprintf(out, cap, "%.1f KiB", (double)b / 1024);
}

/* ---- options shared by run and bench ------------------------------------- */
typedef struct {
    const char *manifest, *prompt, *tokens, *etc, *warm, *kernel, *tokenizer;
    int         n, stats, print_tokens, prompt_len;
    int         have_temp, have_seed;
    double      temp, top_p, min_p;
    uint64_t    seed;
    tlm_cfg    *cfg;
} opts;

/* Flags that map to config keys are applied with origin SET (they win). */
static int runtime_flag(opts *o, int argc, char **argv, int *i) {
    static const struct { const char *flag, *key; } map[] = {
        {"--cache", "memory.expert_cache"}, {"--threads", "cpu.threads"}, {"--io", "io.backend"},
        {"--direct", "io.direct"},          {"--prefetch", "io.prefetch"}, {"--pin-frac", "memory.pin_frac"},
    };
    for (size_t k = 0; k < sizeof map / sizeof map[0]; k++)
        if (!strcmp(argv[*i], map[k].flag)) {
            const char *v = need(argc, argv, i);
            if (tlm_cfg_set_origin(o->cfg, map[k].key, v, TLM_ORIGIN_SET, map[k].flag))
                die("%s %s: %s", map[k].flag, v, tlm_errmsg());
            return 1;
        }
    return 0;
}

static void load_cfg(opts *o, int argc, char **argv) {
    /* --etc must be known before any flag is applied */
    for (int i = 2; i < argc; i++)
        if (!strcmp(argv[i], "--etc") && i + 1 < argc) o->etc = argv[i + 1];
    o->cfg = tlm_cfg_new();
    if (!o->cfg) die("out of memory");
    const char *etc = o->etc ? o->etc : getenv("TIERLM_ETC");
    if (tlm_cfg_load_etc(o->cfg, etc && *etc ? etc : TLM_ETC_DEFAULT, 0)) die("config: %s", tlm_errmsg());
    const char *lvl = tlm_cfg_str(o->cfg, "log.level");
    if (lvl) tlm_log_level(!strcmp(lvl, "error") ? TLM_LOG_ERROR : !strcmp(lvl, "info") ? TLM_LOG_INFO
                           : !strcmp(lvl, "debug") ? TLM_LOG_DEBUG : TLM_LOG_WARN);
}

static void parse_opts(opts *o, int argc, char **argv, int bench) {
    memset(o, 0, sizeof *o);
    o->n = bench ? 64 : 32;
    o->prompt_len = 64;
    o->top_p = 1.0;
    load_cfg(o, argc, argv);
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (runtime_flag(o, argc, argv, &i)) continue;
        if (!strcmp(a, "--etc")) { i++; continue; }
        if (!strcmp(a, "--prompt")) o->prompt = need(argc, argv, &i);
        else if (!strcmp(a, "--tokens")) o->tokens = need(argc, argv, &i);
        else if (!strcmp(a, "--tokenizer")) o->tokenizer = need(argc, argv, &i);
        else if (!strcmp(a, "-n") || !strcmp(a, "--max-tokens")) o->n = (int)num_arg(need(argc, argv, &i), a, 0, 1 << 24);
        else if (!strcmp(a, "--temp")) { o->temp = frac_arg(need(argc, argv, &i), a, 0, 100); o->have_temp = 1; }
        else if (!strcmp(a, "--top-p")) o->top_p = frac_arg(need(argc, argv, &i), a, 0, 1);
        else if (!strcmp(a, "--min-p")) o->min_p = frac_arg(need(argc, argv, &i), a, 0, 1);
        else if (!strcmp(a, "--seed")) { o->seed = (uint64_t)num_arg(need(argc, argv, &i), a, 0, LONG_MAX); o->have_seed = 1; }
        else if (!strcmp(a, "--stats")) o->stats = 1;
        else if (!strcmp(a, "--print-tokens")) o->print_tokens = 1;
        else if (!strcmp(a, "--warm")) o->warm = need(argc, argv, &i);
        else if (!strcmp(a, "--kernel")) o->kernel = need(argc, argv, &i);
        else if (bench && !strcmp(a, "--prompt-len")) o->prompt_len = (int)num_arg(need(argc, argv, &i), a, 1, 1 << 20);
        else if (a[0] == '-' && a[1]) die("unknown option %s", a);
        else if (!o->manifest) o->manifest = a;
        else die("unexpected argument %s", a);
    }
    if (!o->manifest) die("a MANIFEST is required");
    /* Settings the schema accepts but this runtime does not act on yet
     * (docs/STATUS.md milestones M2/M3): say so instead of ignoring them. */
    const char *spec = tlm_cfg_str(o->cfg, "spec.mode");
    if (spec && strcmp(spec, "off"))
        tlm_log(TLM_LOG_WARN, "spec.mode = %s is not implemented yet; decoding without speculation", spec);
    const char *kvd = tlm_cfg_str(o->cfg, "kv.device");
    if (kvd && strcmp(kvd, "cpu"))
        tlm_log(TLM_LOG_WARN, "kv.device = %s is not implemented yet; KV cache stays in RAM", kvd);
}

/* ---- runtime assembly -------------------------------------------------------- */
typedef struct {
    tlm_manifest m;
    tlm_io      *io;
    tlm_store   *st;
    tlm_pool    *pool;
    tlm_model   *mdl;
    uint64_t     budget, slot, expert_total;
    double       pin_frac;
    int          threads, lookahead, missing;
    double       load_s;
} runtime;

static int file_exists(const char *p) {
    struct stat s;
    return stat(p, &s) == 0 && S_ISREG(s.st_mode);
}

static void rt_open(runtime *r, opts *o) {
    memset(r, 0, sizeof *r);
    uint64_t t0 = tlm_now_ns();
    if (o->kernel) {   /* forced kernel tier (results are identical by contract, P-43) */
        tlm_cpu_force(o->kernel);
        tlm_quant_init();
    }
    if (tlm_manifest_load(&r->m, o->manifest)) die("%s", tlm_errmsg());
    const tlm_model_geom *g = &r->m.geom;
    if (strncmp(g->arch, "tlm-moe", 7) != 0)
        die("%s: architecture '%s' has no runtime here; rebuild with: tlmc compile SRC --out %s", o->manifest,
            g->arch, o->manifest);

    /* drives present? a missing one is tolerated only if fully replicated */
    char paths[TLM_MAX_DRIVES][1024];
    const char *pp[TLM_MAX_DRIVES];
    for (uint32_t d = 0; d < r->m.n_drives; d++) {
        tlm_manifest_drive_path(&r->m, (int)d, paths[d], sizeof paths[d]);
        pp[d] = paths[d];
        if (file_exists(paths[d])) continue;
        for (uint32_t i = 0; i < r->m.n_packets; i++) {
            const tlm_packet_loc *l = &r->m.packets[i];
            if (l->drive == d && !(l->flags & 2u) && l->replica_drive == 0xFF)
                die("drive %u (%s) is missing and packet type %u layer %u id %u has no replica", d, paths[d],
                    l->type, l->layer, l->id);
        }
        tlm_log(TLM_LOG_WARN, "drive %u (%s) is missing; reading its packets from replicas", d, paths[d]);
        pp[d] = "/dev/null";
        r->missing++;
    }

    /* threads, pinning (P-23, CCD grouping) */
    r->threads = tlm_cfg_is_auto(o->cfg, "cpu.threads") ? 0 : (int)tlm_cfg_int(o->cfg, "cpu.threads", 0);
    const char *groups = tlm_cfg_str(o->cfg, "cpu.thread_groups");
    int pin = tlm_cfg_bool(o->cfg, "cpu.pin", 1) && !(groups && !strcmp(groups, "none"));
    r->pool = tlm_pool_new(r->threads, pin);
    if (!r->pool) die("out of memory");
    r->threads = tlm_pool_size(r->pool);

    /* I/O (P-23 pread pool, P-24 io_uring opt-in, P-25 O_DIRECT switch) */
    tlm_io_opts io;
    memset(&io, 0, sizeof io);
    const char *be = tlm_cfg_str(o->cfg, "io.backend");
    io.backend = be && !strcmp(be, "io_uring") ? TLM_IO_URING : TLM_IO_PREAD;
    io.direct = tlm_cfg_bool(o->cfg, "io.direct", 0);
    io.threads_per_drive = (int)tlm_cfg_int(o->cfg, "io.threads_per_drive", 4);
    io.queue_depth = (int)tlm_cfg_int(o->cfg, "io.queue_depth", 32);
    r->io = tlm_io_open(pp, (int)r->m.n_drives, &io);
    if (!r->io) die("%s", tlm_errmsg());

    /* expert cache budget (P-10, P-20): auto = what RAM allows after the
     * reserve and the resident packets, never more than all experts */
    uint64_t dense = 0;
    int n_ex = 0;
    for (uint32_t i = 0; i < r->m.n_packets; i++) {
        const tlm_packet_loc *l = &r->m.packets[i];
        if (l->type == TLM_PK_EXPERT) {
            if (l->flags & 2u) continue;
            r->expert_total += l->total_len;
            if (l->total_len > r->slot) r->slot = l->total_len;
            n_ex++;
        } else {
            dense += l->total_len;
        }
    }
    uint64_t budget;
    if (tlm_cfg_is_auto(o->cfg, "memory.expert_cache")) {
        tlm_mem_info mi;
        uint64_t avail = tlm_mem_snapshot(&mi) == 0 ? mi.available_bytes : (8ull << 30);
        uint64_t reserve;
        if (tlm_cfg_is_auto(o->cfg, "memory.ram_reserve")) {
            reserve = mi.total_bytes / 10;
            if (reserve < (4ull << 30)) reserve = 4ull << 30;
            if (reserve > avail / 2) reserve = avail / 2;
        } else {
            reserve = tlm_cfg_size(o->cfg, "memory.ram_reserve", 4ull << 30);
        }
        budget = avail > reserve + dense ? avail - reserve - dense : 0;
        if (budget > r->expert_total) budget = r->expert_total;
    } else {
        budget = tlm_cfg_size(o->cfg, "memory.expert_cache", 0);
    }
    uint64_t need_min = (uint64_t)g->top_k * r->slot;
    if (n_ex > 0 && budget < need_min) {
        char b1[32], b2[32];
        fmt_bytes(budget, b1, sizeof b1);
        fmt_bytes(need_min, b2, sizeof b2);
        tlm_log(TLM_LOG_WARN, "expert cache %s is below top_k slots; using %s", b1, b2);
        budget = need_min;
    }
    r->budget = budget;
    /* pins never take the streaming slots a token needs (P-19 cap) */
    r->pin_frac = tlm_cfg_frac(o->cfg, "memory.pin_frac", 0.30);
    uint64_t ns = r->slot ? budget / r->slot : 0;
    if (ns > (uint64_t)n_ex) ns = (uint64_t)n_ex;
    if (ns > 0 && r->pin_frac > 0) {
        double maxf = (double)(ns > g->top_k ? ns - g->top_k : 0) / (double)ns;
        if (r->pin_frac > maxf) r->pin_frac = maxf;
    }
    tlm_store_opts so;
    memset(&so, 0, sizeof so);
    so.budget_bytes = budget;
    so.pin_frac = r->pin_frac;
    const char *pf = tlm_cfg_str(o->cfg, "io.prefetch");
    so.prefetch = !(pf && !strcmp(pf, "off"));
    const char *hp = tlm_cfg_str(o->cfg, "memory.hugepages");
    so.huge = !(hp && !strcmp(hp, "off"));
    so.warm_path = o->warm;
    r->st = tlm_store_open(&r->m, r->io, &so);
    if (!r->st) die("%s", tlm_errmsg());
    if (tlm_store_warm(r->st)) die("%s", tlm_errmsg());
    r->lookahead = so.prefetch ? 1 : 0;
    tlm_model_opts mo = {r->lookahead, 0};
    r->mdl = tlm_model_open(&r->m, r->io, r->st, r->pool, &mo);
    if (!r->mdl) die("%s", tlm_errmsg());
    r->load_s = (double)(tlm_now_ns() - t0) * 1e-9;
}

static void rt_close(runtime *r, const opts *o) {
    if (o->warm && r->st && tlm_store_save_warm(r->st, o->warm))
        tlm_log(TLM_LOG_WARN, "warm sidecar %s: %s", o->warm, tlm_errmsg());
    tlm_model_close(r->mdl);
    tlm_store_close(r->st);   /* before the io: waits for in-flight reads */
    tlm_io_close(r->io);
    tlm_pool_free(r->pool);
    tlm_manifest_free(&r->m);
    memset(r, 0, sizeof *r);
}

static void print_stats(runtime *r, FILE *f, int n_prompt, int n_gen, double prefill_s, double decode_s) {
    tlm_model_stats ms;
    tlm_model_get_stats(r->mdl, &ms);
    tlm_store_stats ss;
    tlm_store_get_stats(r->st, &ss);
    char b1[32], b2[32], b3[32];
    fprintf(f, "-- stats --\n");
    fprintf(f, "load        %.3f s (manifest, resident packets, pins); kernel %s, %d thread%s\n", r->load_s,
            tlm_quant_tier(), r->threads, r->threads == 1 ? "" : "s");
    fprintf(f, "tokens      prompt %d, generated %d; prefill %.1f tok/s, decode %.1f tok/s\n", n_prompt, n_gen,
            prefill_s > 0 ? n_prompt / prefill_s : 0.0, decode_s > 0 && n_gen > 1 ? (n_gen - 1) / decode_s : 0.0);
    double per = ms.tokens ? 1e-6 / (double)ms.tokens : 0;
    fprintf(f, "per token   attention %.3f ms, moe %.3f ms (expert wait %.3f ms), other %.3f ms\n",
            (double)ms.attn_ns * per, (double)ms.moe_ns * per, (double)ms.expert_wait_ns * per,
            (double)ms.other_ns * per);
    fmt_bytes(r->slot, b1, sizeof b1);
    fmt_bytes(r->budget, b2, sizeof b2);
    fmt_bytes(r->expert_total, b3, sizeof b3);
    fprintf(f, "cache       %d slots x %s (budget %s of %s experts), %d pinned (pin_frac %.2f)\n", ss.slots, b1, b2,
            b3, ss.pinned, r->pin_frac);
    uint64_t acc = ss.hits + ss.misses;
    fmt_bytes(ss.bytes_read, b1, sizeof b1);
    fprintf(f, "store       hits %llu, misses %llu (hit rate %.1f%%), evictions %llu, read %s, wait %.3f ms\n",
            (unsigned long long)ss.hits, (unsigned long long)ss.misses, acc ? 100.0 * ss.hits / acc : 100.0,
            (unsigned long long)ss.evictions, b1, ss.wait_ns * 1e-6);
    fprintf(f, "prefetch    %s: issued %llu, used %llu, dropped %llu\n", r->lookahead ? "router lookahead" : "off",
            (unsigned long long)ss.prefetch_issued, (unsigned long long)ss.prefetch_used,
            (unsigned long long)ss.prefetch_dropped);
    fprintf(f, "io          backend %s%s\n", tlm_io_backend_name(r->io), r->missing ? " (running from replicas)" : "");
    for (uint32_t d = 0; d < r->m.n_drives; d++) {
        tlm_io_file_stats is;
        tlm_io_stats(r->io, (int)d, &is);
        fmt_bytes(is.bytes, b1, sizeof b1);
        fprintf(f, "  drive %u   %llu reads, %s, busy %.3f ms\n", d, (unsigned long long)is.reads, b1,
                is.busy_ns * 1e-6);
    }
}

/* ---- prompt ------------------------------------------------------------------- */
static int *make_prompt(const opts *o, int V, const tlm_bpe *tok, int *n) {
    if (o->tokens) {
        int cap = 16, k = 0;
        int *t = (int *)malloc(sizeof(int) * (size_t)cap);
        char *copy = strdup(o->tokens);
        if (!t || !copy) die("out of memory");
        for (char *s = strtok(copy, ", "); s; s = strtok(NULL, ", ")) {
            long v = num_arg(s, "--tokens", 0, V - 1);
            if (k == cap) {
                cap *= 2;
                t = (int *)realloc(t, sizeof(int) * (size_t)cap);
                if (!t) die("out of memory");
            }
            t[k++] = (int)v;
        }
        free(copy);
        if (!k) die("--tokens: empty list");
        *n = k;
        return t;
    }
    const char *p = o->prompt ? o->prompt : "Hello";
    size_t L = strlen(p);
    if (!L) die("--prompt: empty");
    if (tok) {
        /* real tokenizer: bare BPE encode, added tokens off (P-99) */
        int cap = (int)(L * 4) + 16, k;
        int *t = (int *)malloc(sizeof(int) * (size_t)cap);
        if (!t) die("out of memory");
        for (;;) {
            k = tlm_bpe_encode(tok, p, L, 0, t, cap);
            if (k >= 0) break;
            cap *= 2;
            int *nt = (int *)realloc(t, sizeof(int) * (size_t)cap);
            if (!nt) die("out of memory");
            t = nt;
        }
        if (!k) die("--prompt encodes to zero tokens");
        *n = k;
        return t;
    }
    int *t = (int *)malloc(sizeof(int) * L);
    if (!t) die("out of memory");
    for (size_t i = 0; i < L; i++) t[i] = (unsigned char)p[i] % V;   /* byte-level, mod vocab */
    *n = (int)L;
    return t;
}

typedef struct { int print_tokens, count; tlm_bpe_dec *dec; } emit_ctx;
static int emit(void *ctx, int token) {
    emit_ctx *e = (emit_ctx *)ctx;
    if (e->print_tokens) {
        printf("%s%d", e->count ? " " : "", token);
    } else if (e->dec) {
        char buf[256];
        int k;
        for (;;) {
            k = tlm_bpe_dec_push(e->dec, token, buf, sizeof buf);
            if (k >= 0) break;
            fprintf(stderr, "emit: decode overflow\n");
            break;
        }
        if (k > 0) fwrite(buf, 1, (size_t)k, stdout);
    } else if (token >= 32 && token < 127) {
        putchar(token);
    } else if (token == '\n' || token == '\t') {
        putchar(token);
    } else {
        printf("<%d>", token);
    }
    e->count++;
    fflush(stdout);
    return 0;
}

static int cmd_run(int argc, char **argv) {
    opts o;
    parse_opts(&o, argc, argv, 0);
    tlm_bpe *tbpe = NULL;
    if (o.tokenizer) {
        char err[512] = "";
        tbpe = tlm_bpe_load(o.tokenizer, err, sizeof err);
        if (!tbpe) die("--tokenizer: %s", err);
    }
    runtime r;
    rt_open(&r, &o);
    const tlm_model_geom *g = tlm_model_geom_of(r.mdl);
    int np = 0;
    int *prompt = make_prompt(&o, (int)g->vocab, tbpe, &np);
    int ctx = np + o.n + 1;
    if (g->max_ctx && ctx > (int)g->max_ctx) ctx = (int)g->max_ctx;
    if (np >= ctx) die("prompt of %d tokens does not fit the context (%u)", np, g->max_ctx);
    tlm_kv *kv = tlm_kv_new(r.mdl, ctx);
    if (!kv) die("%s", tlm_errmsg());
    tlm_sampler s = {(float)o.temp, (float)o.top_p, (float)o.min_p, o.have_seed ? o.seed : 0};
    tlm_bpe_dec *dec = tbpe && !o.print_tokens ? tlm_bpe_dec_new(tbpe, 1) : NULL;
    emit_ctx ec = {o.print_tokens, 0, dec};
    /* prefill and decode timed separately for --stats */
    float *lg = (float *)malloc(sizeof(float) * g->vocab);
    if (!lg) die("out of memory");
    uint64_t t0 = tlm_now_ns();
    if (tlm_model_prefill(r.mdl, kv, prompt, np, lg)) die("%s", tlm_errmsg());
    uint64_t t1 = tlm_now_ns();
    uint64_t rng = 0;
    int gen = 0;
    while (gen < o.n) {
        int tok = tlm_sample(&s, &rng, lg, (int)g->vocab);
        emit(&ec, tok);
        gen++;
        if (gen >= o.n || tlm_kv_len(kv) >= tlm_kv_capacity(kv)) break;
        if (tlm_model_step(r.mdl, kv, tok, lg)) die("%s", tlm_errmsg());
    }
    uint64_t t2 = tlm_now_ns();
    if (dec) {
        char tail[256];
        int k = tlm_bpe_dec_flush(dec, tail, sizeof tail);
        if (k > 0) fwrite(tail, 1, (size_t)k, stdout);
    }
    putchar('\n');
    fflush(stdout);
    if (o.stats) print_stats(&r, stderr, np, gen, (double)(t1 - t0) * 1e-9, (double)(t2 - t1) * 1e-9);
    free(lg);
    free(prompt);
    tlm_kv_free(kv);
    tlm_bpe_dec_free(dec);
    tlm_bpe_free(tbpe);
    rt_close(&r, &o);
    tlm_cfg_free(o.cfg);
    return 0;
}
/* ---- oracle: teacher-forced per-position top-k dump (P-121) --------------- */
/* Fill ids/sc (descending: sc[0] is the argmax) with the top-k (id, score) of lg[0..V). */
static void topk_of(const float *lg, int V, int k, int *ids, float *sc) {
    if (k > V) k = V;
    for (int j = 0; j < k; j++) { sc[j] = -INFINITY; ids[j] = 0; }
    for (int v = 0; v < V; v++) {
        float s = lg[v];
        int j = 0;
        while (j < k && s <= sc[j]) j++;
        if (j == k) continue;
        for (int m = k - 1; m > j; m--) { sc[m] = sc[m - 1]; ids[m] = ids[m - 1]; }
        sc[j] = s; ids[j] = v;
    }
}

/* tierlm oracle: feed a fixed id sequence through the runtime one token at a
 * time and, per position, dump the vocab top-k to stdout. Position i is the
 * distribution over the token following t0..ti, the same index as transformers'
 * logits[i]. Argmax and the top-k id set are invariant to the raw-vs-logprob
 * scale, so this lines up with a HF reference run on the identical ids.
 * Emits one JSON object per line: {"pos": i, "topk": [[id, score], ...]}. */
static int cmd_oracle(int argc, char **argv) {
    opts o;
    parse_opts(&o, argc, argv, 0);
    if (!o.tokens) die("oracle: --tokens <id,id,...> is required (the fixed teacher-forced sequence)");
    enum { TOPK = 10 };
    runtime r;
    rt_open(&r, &o);
    const tlm_model_geom *g = tlm_model_geom_of(r.mdl);
    int np = 0;
    int *seq = make_prompt(&o, (int)g->vocab, NULL, &np);
    if (np < 2) die("oracle: need at least 2 ids to get per-position distributions");
    tlm_kv *kv = tlm_kv_new(r.mdl, np + 1);
    if (!kv) die("%s", tlm_errmsg());
    float *lg = (float *)malloc(sizeof(float) * (size_t)g->vocab);
    int *ids = (int *)malloc(sizeof(int) * TOPK);
    float *sc = (float *)malloc(sizeof(float) * TOPK);
    if (!lg || !ids || !sc) die("out of memory");
    for (int i = 0; i < np; i++) {
        if (tlm_model_step(r.mdl, kv, seq[i], lg)) die("%s", tlm_errmsg());
        topk_of(lg, (int)g->vocab, TOPK, ids, sc);
        int k = TOPK < (int)g->vocab ? TOPK : (int)g->vocab;
        fprintf(stdout, "{\"pos\": %d, \"topk\": [", i);
        for (int j = 0; j < k; j++) fprintf(stdout, "%s[%d, %.9g]", j ? ", " : "", ids[j], sc[j]);
        fprintf(stdout, "]}\n");
    }
    free(ids); free(sc); free(lg); free(seq);
    tlm_kv_free(kv);
    rt_close(&r, &o);
    tlm_cfg_free(o.cfg);
    return 0;
}

static int cmd_bench(int argc, char **argv) {
    opts o;
    parse_opts(&o, argc, argv, 1);
    runtime r;
    rt_open(&r, &o);
    const tlm_model_geom *g = tlm_model_geom_of(r.mdl);
    int np = o.prompt_len;
    int *prompt = (int *)malloc(sizeof(int) * (size_t)np);
    if (!prompt) die("out of memory");
    uint64_t x = o.have_seed ? o.seed : 42;
    for (int i = 0; i < np; i++) {   /* deterministic pseudo-random prompt */
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        prompt[i] = (int)(x % g->vocab);
    }
    int ctx = np + o.n + 1;
    if (g->max_ctx && ctx > (int)g->max_ctx) die("prompt-len + n exceeds the context (%u)", g->max_ctx);
    tlm_kv *kv = tlm_kv_new(r.mdl, ctx);
    float *lg = (float *)malloc(sizeof(float) * g->vocab);
    if (!kv || !lg) die("out of memory");
    tlm_model_reset_stats(r.mdl);
    uint64_t t0 = tlm_now_ns();
    if (tlm_model_prefill(r.mdl, kv, prompt, np, lg)) die("%s", tlm_errmsg());
    uint64_t t1 = tlm_now_ns();
    int gen = 0;
    for (; gen < o.n; gen++) {
        int tok = tlm_sample(NULL, NULL, lg, (int)g->vocab);
        if (gen + 1 < o.n && tlm_model_step(r.mdl, kv, tok, lg)) die("%s", tlm_errmsg());
    }
    uint64_t t2 = tlm_now_ns();
    double ps = (double)(t1 - t0) * 1e-9, ds = (double)(t2 - t1) * 1e-9;
    printf("bench %s: prompt %d tokens in %.3f s (%.1f tok/s), decode %d tokens in %.3f s (%.1f tok/s)\n",
           r.m.name, np, ps, ps > 0 ? np / ps : 0.0, gen, ds, ds > 0 && gen > 1 ? (gen - 1) / ds : 0.0);
    print_stats(&r, stdout, np, gen, ps, ds);
    free(lg);
    free(prompt);
    tlm_kv_free(kv);
    rt_close(&r, &o);
    tlm_cfg_free(o.cfg);
    return 0;
}

/* ---- config show / check ------------------------------------------------------ */
static int cmd_config(int argc, char **argv) {
    if (argc < 3) die("usage: tierlm config show|check [--origin] [--etc DIR]");
    const char *sub = argv[2], *etc = NULL;
    int origin = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--etc")) etc = need(argc, argv, &i);
        else if (!strcmp(argv[i], "--origin")) origin = 1;
        else die("config: unknown option %s", argv[i]);
    }
    if (!etc) etc = getenv("TIERLM_ETC");
    if (!etc || !*etc) etc = TLM_ETC_DEFAULT;
    tlm_cfg *c = tlm_cfg_new();
    if (!c) die("out of memory");
    if (!strcmp(sub, "show")) {
        if (tlm_cfg_load_etc(c, etc, 0)) die("config: %s", tlm_errmsg());
        printf("# effective tierlm settings (%s)\n", etc);
        int it = 0;
        const char *k, *v;
        while (tlm_cfg_next(c, &it, &k, &v)) {
            if (origin) {
                const char *where = NULL;
                tlm_cfg_origin og = tlm_cfg_origin_of(c, k, &where);
                int show_where = where && *where && og != TLM_ORIGIN_DEFAULT;
                printf("%-28s = %-24s # %s%s%s\n", k, v ? v : "", tlm_cfg_origin_name(og), show_where ? " " : "",
                       show_where ? where : "");
            } else {
                printf("%-28s = %s\n", k, v ? v : "");
            }
        }
        tlm_cfg_free(c);
        return 0;
    }
    if (!strcmp(sub, "check")) {
        tlm_cfg_quiet(c, 1);
        int rc = tlm_cfg_load_etc(c, etc, 1);
        if (rc == 0) rc = tlm_cfg_check(c);
        int errors = 0;
        for (int i = 0; i < tlm_cfg_ndiag(c); i++) {
            int is_err = 0;
            const char *d = tlm_cfg_diag(c, i, &is_err);
            fprintf(stderr, "%s: %s\n", is_err ? "error" : "warning", d);
            errors += is_err;
        }
        if (rc && !errors) fprintf(stderr, "error: %s\n", tlm_errmsg());
        printf("%s: %s\n", etc, rc ? "invalid" : "ok");
        tlm_cfg_free(c);
        return rc ? 1 : 0;
    }
    die("config: show or check");
}

static void usage(FILE *f) {
    fprintf(f,
            "tierlm %s - tiered MoE runtime\n"
            "  tierlm run   MANIFEST [--prompt TEXT | --tokens 1,2,3] [-n 32] [--temp 0] [--top-p P]\n"
            "               [--min-p P] [--seed N] [--cache SIZE] [--threads N] [--io pread|io_uring]\n"
            "               [--direct yes|no] [--prefetch router|off] [--pin-frac F] [--warm FILE]\n"
            "               [--kernel scalar|avx2|avx512] [--stats] [--etc DIR] [--print-tokens]\n"
            "  tierlm bench MANIFEST [--prompt-len N] [-n N] [runtime flags as for run]\n"
            "  tierlm oracle MANIFEST --tokens 1,2,3   (per-position top-k dump, P-121)\n"
            "  tierlm config show [--origin] [--etc DIR]\n"
            "  tierlm config check [--etc DIR]\n",
            TIERLM_VERSION);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    tlm_log_level(TLM_LOG_WARN);
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "help")) {
        usage(argc < 2 ? stderr : stdout);
        return argc < 2 ? 2 : 0;
    }
    if (!strcmp(argv[1], "run")) return cmd_run(argc, argv);
    if (!strcmp(argv[1], "bench")) return cmd_bench(argc, argv);
    if (!strcmp(argv[1], "oracle")) return cmd_oracle(argc, argv);
    if (!strcmp(argv[1], "config")) return cmd_config(argc, argv);
    usage(stderr);
    return 2;
}
