/* tlm_cfg_schema.c - the built-in configuration schema, value parsers and
 * per-key validation. See tlm_cfg.h for the conventions (prefix rules,
 * keywords such as "auto") and research/03-two-stage-design.md for the file
 * format. Pattern: F-CFG-01 (strict full-string parsing, ranges named in
 * every error).
 */
#include "tlm_cfg.h"
#include "tlm_platform.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define NR 1, 0                                   /* no numeric range */
#define KIB 1024.0
#define GIB (1024.0 * 1024.0 * 1024.0)
#define PIB (GIB * 1024.0 * 1024.0)
#define DEPTHS "exclude|shallow|standard|deep"

const tlm_cfg_key tlm_cfg_schema[] = {
    /* logging */
    {"log.level", TLM_CFG_ENUM, "warn", NR, "error|warn|info|debug", "log verbosity"},

    /* cpu */
    {"cpu.threads", TLM_CFG_INT, "auto", 1, 4096, "auto", "compute threads; auto = one per physical core"},
    {"cpu.thread_groups", TLM_CFG_ENUM, "ccd", NR, "ccd|none", "group workers by L3 domain (CCD) so an expert runs inside one CCD"},
    {"cpu.pin", TLM_CFG_BOOL, "yes", NR, NULL, "pin compute threads to physical cores"},

    /* memory */
    {"memory.ram_reserve", TLM_CFG_SIZE, "auto", 0, PIB, "auto", "RAM left for the OS and other processes; auto = max(4G, 10% of RAM)"},
    {"memory.expert_cache", TLM_CFG_SIZE, "auto", 0, PIB, "auto", "expert slot cache; auto = RAM - reserve - dense weights"},
    {"memory.pin_frac", TLM_CFG_FRAC, "0.30", 0, 1, NULL, "share of the expert cache kept for the pinned hot set"},
    {"memory.hugepages", TLM_CFG_ENUM, "thp", NR, "thp|off|explicit", "huge pages for the slot pool: transparent, off, or preallocated"},

    /* I/O */
    {"io.backend", TLM_CFG_ENUM, "pread", NR, "pread|io_uring", "read backend; io_uring needs a liburing build and an enabled kernel"},
    {"io.direct", TLM_CFG_BOOL, "no", NR, NULL, "O_DIRECT expert reads (bypass the page cache)"},
    {"io.queue_depth", TLM_CFG_INT, "32", 1, 4096, NULL, "outstanding reads per drive (io_uring)"},
    {"io.threads_per_drive", TLM_CFG_INT, "4", 1, 256, NULL, "pread workers per drive"},
    {"io.drives", TLM_CFG_LIST, "", NR, NULL, "expert store mount points, one per NVMe drive"},
    {"io.prefetch", TLM_CFG_ENUM, "router", NR, "router|off", "prefetch layer l+1 experts from its router run on layer l's output"},

    /* KV cache */
    {"kv.device", TLM_CFG_LIST, "cpu", NR, NULL, "KV cache placement: cpu, or GPUs such as hip:0, hip:1"},
    {"kv.vram_frac", TLM_CFG_FRAC, "0.05", 0, 0.10, NULL, "share of each GPU's VRAM for KV pages (hard cap 0.10)"},
    {"kv.dtype", TLM_CFG_ENUM, "f16", NR, "f16|f8_e4m3|f32", "KV element type; f8_e4m3 doubles the context that fits"},
    {"kv.page_tokens", TLM_CFG_INT, "64", 16, 1024, NULL, "tokens per KV page"},

    /* speculation */
    {"spec.mode", TLM_CFG_ENUM, "off", NR, "off|mtp", "speculative decoding with the model's own MTP head"},
    {"spec.shape", TLM_CFG_ENUM, "chain", NR, "chain", "draft shape"},
    {"spec.depth", TLM_CFG_INT, "auto", 0, 4, "auto", "draft length; auto = chosen per step by measured utility"},
    {"spec.union_guard", TLM_CFG_BOOL, "yes", NR, NULL, "trim drafts that would pull too many experts from disk"},
    {"spec.restrict_experts", TLM_CFG_BOOL, "no", NR, NULL, "lossy expert-restricted drafting (opt-in)"},

    /* quantisation and compile defaults */
    {"quant.experts", TLM_CFG_ENUM, "q6_k", NR, "q6_k|q5_k|q4_k|q8_0|native", "routed expert format; native keeps shipped FP4/FP8"},
    {"quant.dense", TLM_CFG_ENUM, "q8_0", NR, "q8_0|f16|bf16|native", "attention, shared experts, router, embeddings (never below q8_0)"},
    {"quant.mtp_head", TLM_CFG_ENUM, "q8_0", NR, "q8_0|f16|bf16|native", "MTP head format (never below q8_0)"},
    {"compile.profile", TLM_CFG_ENUM, "exact", NR, "exact|balanced|aggressive", "exact = lossless; balanced/aggressive allow mixed precision and pruning"},

    /* paths and server */
    {"paths.models", TLM_CFG_STR, "/srv/tlm/models", NR, NULL, "compiled model manifests"},
    {"paths.kv_store", TLM_CFG_STR, "/var/lib/tierlm/kv", NR, NULL, "saved KV sessions"},
    {"paths.log", TLM_CFG_STR, "/var/log/tierlm", NR, NULL, "log directory"},
    {"server.listen", TLM_CFG_STR, "127.0.0.1:8000", NR, NULL, "HTTP listen address host:port"},

    /* experts and content */
    {"experts.protect", TLM_CFG_LIST, "", NR, NULL, "layer:expert ids never pruned or demoted"},
    {"experts.exclude", TLM_CFG_LIST, "", NR, NULL, "layer:expert ids to remove (still eval-gated)"},
    {"prune.max_ratio", TLM_CFG_FRAC, "0.25", 0, 0.5, NULL, "cap on the share of routed experts removed"},
    {"prune.min_keep_per_layer", TLM_CFG_FRAC, "0.5", 0.5, 1, NULL, "share of experts per layer that always survive"},
    {"content.default", TLM_CFG_ENUM, "standard", NR, DEPTHS, "depth for content types not set explicitly"},
    {"content.*", TLM_CFG_ENUM, NULL, NR, DEPTHS, "depth for one content type (children inherit from parents)"},

    /* decision engine */
    {"decide.backend", TLM_CFG_ENUM, "rules", NR, "rules|self|encoder", "decision backend; rules always gate the result"},
    {"decide.encoder_model", TLM_CFG_STR, NULL, NR, NULL, "encoder decision model (.tlmd)"},
    {"decide.auto_threshold", TLM_CFG_FRAC, "0.90", 0.5, 1, NULL, "confidence at or above which decisions apply automatically"},
    {"decide.ask_threshold", TLM_CFG_FRAC, "0.60", 0, 1, NULL, "confidence below which the safe default is taken"},
    {"decide.log", TLM_CFG_STR, "/var/lib/tierlm/decisions", NR, NULL, "decision audit log directory (JSONL)"},

    /* probe output (hardware.conf) */
    {"hardware.*", TLM_CFG_STR, NULL, NR, NULL, "measured hardware fact written by tierlm-setup --probe"},

    {NULL, TLM_CFG_STR, NULL, 0, 0, NULL, NULL}
};

const tlm_cfg_key *tlm_cfg_schema_find(const char *key) {
    if (!key) return NULL;
    const tlm_cfg_key *best = NULL;
    size_t best_len = 0, klen = strlen(key);
    for (const tlm_cfg_key *k = tlm_cfg_schema; k->key; k++) {
        size_t n = strlen(k->key);
        if (n == klen && memcmp(k->key, key, n) == 0) return k;
        if (n >= 2 && k->key[n - 1] == '*' && k->key[n - 2] == '.') {
            size_t pl = n - 1;                       /* "content." */
            if (klen > pl && memcmp(k->key, key, pl) == 0 && pl > best_len) {
                best = k;
                best_len = pl;
            }
        }
    }
    return best;
}

/* ---- syntax ------------------------------------------------------------- */
int tlm_cfg_key_valid(const char *key) {
    if (!key || !*key) return 0;
    size_t n = strlen(key);
    if (n > 127) return 0;
    if (key[0] < 'a' || key[0] > 'z') return 0;
    if (key[n - 1] == '.') return 0;
    for (size_t i = 0; i < n; i++) {
        char c = key[i];
        if (c == '.') {
            if (key[i + 1] == '.') return 0;
            continue;
        }
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return 0;
    }
    return 1;
}

/* ---- parsers ------------------------------------------------------------ */
static int suffix_shift(char c) {
    switch (c) {
    case 'k': case 'K': return 10;
    case 'm': case 'M': return 20;
    case 'g': case 'G': return 30;
    case 't': case 'T': return 40;
    case 'p': case 'P': return 50;
    default: return -1;
    }
}

int tlm_cfg_parse_int(const char *s, int64_t *out) {
    if (!s) return TLM_EINVAL;
    const char *p = s;
    int neg = 0;
    if (*p == '+' || *p == '-') { neg = *p == '-'; p++; }
    if (*p < '0' || *p > '9') return TLM_EINVAL;
    uint64_t v = 0;
    while (*p >= '0' && *p <= '9') {
        unsigned d = (unsigned)(*p - '0');
        if (v > (UINT64_MAX - d) / 10) return TLM_ERANGE;
        v = v * 10 + d;
        p++;
    }
    if (*p) {
        int sh = suffix_shift(*p);
        if (sh < 0 || sh > 40 || p[1] != 0) return TLM_EINVAL;
        if (v > (UINT64_MAX >> sh)) return TLM_ERANGE;
        v <<= sh;
    }
    if (!neg && v > (uint64_t)INT64_MAX) return TLM_ERANGE;
    if (neg && v > (uint64_t)INT64_MAX + 1u) return TLM_ERANGE;
    if (out) *out = neg ? (int64_t)(0 - v) : (int64_t)v;
    return 0;
}

int tlm_cfg_parse_size(const char *s, uint64_t *out) {
    if (!s) return TLM_EINVAL;
    const char *p = s;
    if (*p < '0' || *p > '9') return TLM_EINVAL;
    uint64_t ip = 0;
    int overflow = 0;
    while (*p >= '0' && *p <= '9') {
        unsigned d = (unsigned)(*p - '0');
        if (ip > (UINT64_MAX - d) / 10) overflow = 1;
        else ip = ip * 10 + d;
        p++;
    }
    const char *frac = NULL;
    if (*p == '.') {
        p++;
        frac = p;
        if (*p < '0' || *p > '9') return TLM_EINVAL;
        while (*p >= '0' && *p <= '9') p++;
    }
    int sh = 0;
    if (*p && *p != 'b' && *p != 'B') {
        sh = suffix_shift(*p);
        if (sh < 0) return TLM_EINVAL;
        p++;
        if (*p == 'i' || *p == 'I') p++;
    }
    if (*p == 'b' || *p == 'B') p++;
    if (*p) return TLM_EINVAL;
    if (overflow) return TLM_ERANGE;
    if (!frac) {
        if (sh && ip > (UINT64_MAX >> sh)) return TLM_ERANGE;
        if (out) *out = ip << sh;
        return 0;
    }
    double f = strtod(frac - 1, NULL);                 /* ".5" */
    double v = ((double)ip + f) * ldexp(1.0, sh);
    if (!(v < 18446744073709551615.0)) return TLM_ERANGE;
    if (out) *out = (uint64_t)(v + 0.5);
    return 0;
}

int tlm_cfg_parse_frac(const char *s, double *out) {
    if (!s) return TLM_EINVAL;
    const char *p = s;
    if (*p == '+' || *p == '-') p++;
    int digits = 0;
    while (*p >= '0' && *p <= '9') { p++; digits++; }
    if (*p == '.') {
        p++;
        while (*p >= '0' && *p <= '9') { p++; digits++; }
    }
    if (!digits) return TLM_EINVAL;
    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '+' || *p == '-') p++;
        if (*p < '0' || *p > '9') return TLM_EINVAL;
        while (*p >= '0' && *p <= '9') p++;
    }
    int pct = 0;
    if (*p == '%') { pct = 1; p++; }
    if (*p) return TLM_EINVAL;
    double v = strtod(s, NULL);
    if (!isfinite(v)) return TLM_ERANGE;
    if (pct) v /= 100.0;
    if (out) *out = v;
    return 0;
}

int tlm_cfg_parse_bool(const char *s, int *out) {
    static const char *t[] = {"yes", "on", "true", "1"};
    static const char *f[] = {"no", "off", "false", "0"};
    if (!s) return TLM_EINVAL;
    for (int i = 0; i < 4; i++) {
        if (strcasecmp(s, t[i]) == 0) { if (out) *out = 1; return 0; }
        if (strcasecmp(s, f[i]) == 0) { if (out) *out = 0; return 0; }
    }
    return TLM_EINVAL;
}

void tlm_cfg_fmt_size(uint64_t b, char *out, size_t cap) {
    static const char suf[] = "TGMK";
    static const int sh[] = {40, 30, 20, 10};
    if (b) {
        for (int i = 0; i < 4; i++) {
            uint64_t unit = (uint64_t)1 << sh[i];
            if (b % unit == 0) {
                snprintf(out, cap, "%llu%c", (unsigned long long)(b / unit), suf[i]);
                return;
            }
        }
    }
    snprintf(out, cap, "%llu", (unsigned long long)b);
}

/* ---- describe and validate --------------------------------------------- */
static void fmt_num(const tlm_cfg_key *k, double v, char *out, size_t cap) {
    if (k->type == TLM_CFG_SIZE && v >= 0 && v < 1.8e19) tlm_cfg_fmt_size((uint64_t)v, out, cap);
    else snprintf(out, cap, "%g", v);
}

void tlm_cfg_describe(const tlm_cfg_key *k, char *out, size_t cap) {
    char lo[32], hi[32], range[80] = "";
    if (!k) { snprintf(out, cap, "unknown key"); return; }
    if (k->min <= k->max) {
        fmt_num(k, k->min, lo, sizeof lo);
        fmt_num(k, k->max, hi, sizeof hi);
        snprintf(range, sizeof range, " in [%s, %s]", lo, hi);
    }
    const char *kw = (k->type != TLM_CFG_ENUM && k->enum_values) ? k->enum_values : NULL;
    switch (k->type) {
    case TLM_CFG_STR: snprintf(out, cap, "a string"); return;
    case TLM_CFG_INT: snprintf(out, cap, "an integer%s%s%s", range, kw ? " or " : "", kw ? kw : ""); return;
    case TLM_CFG_SIZE: snprintf(out, cap, "a size such as 512M or 12G%s%s%s", range, kw ? " or " : "", kw ? kw : ""); return;
    case TLM_CFG_FRAC: snprintf(out, cap, "a number%s%s%s", range, kw ? " or " : "", kw ? kw : ""); return;
    case TLM_CFG_BOOL: snprintf(out, cap, "yes|no (also on|off, true|false)"); return;
    case TLM_CFG_LIST: snprintf(out, cap, "a comma-separated list"); return;
    case TLM_CFG_ENUM: snprintf(out, cap, "one of %s", k->enum_values ? k->enum_values : "(none)"); return;
    }
    snprintf(out, cap, "a value");
}

/* Find `v` (case-insensitive) in "a|b|c"; copy the schema spelling to out. */
static int match_word(const char *list, const char *v, char *out, size_t cap) {
    if (!list || !v) return 0;
    size_t vl = strlen(v);
    const char *p = list;
    while (*p) {
        const char *e = strchr(p, '|');
        size_t n = e ? (size_t)(e - p) : strlen(p);
        if (n == vl && strncasecmp(p, v, n) == 0) {
            if (out && cap) snprintf(out, cap, "%.*s", (int)n, p);
            return 1;
        }
        if (!e) break;
        p = e + 1;
    }
    return 0;
}

static void set_why(char *why, size_t cap, const char *fmt, const char *a, const char *b) {
    if (why && cap) snprintf(why, cap, fmt, a, b);
}

static int parse_uint_str(const char *s, size_t n, unsigned long max, unsigned long *out) {
    if (n == 0 || n > 10) return -1;
    unsigned long v = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] < '0' || s[i] > '9') return -1;
        v = v * 10 + (unsigned long)(s[i] - '0');
    }
    if (v > max) return -1;
    if (out) *out = v;
    return 0;
}

/* Item check for list keys with a defined item grammar; returns NULL if ok
 * or a description of the expected form. */
static const char *check_item(const char *key, const char *item) {
    if (strcmp(key, "experts.protect") == 0 || strcmp(key, "experts.exclude") == 0) {
        const char *colon = strchr(item, ':');
        if (!colon || parse_uint_str(item, (size_t)(colon - item), 100000, NULL) != 0 ||
            parse_uint_str(colon + 1, strlen(colon + 1), 1000000, NULL) != 0)
            return "layer:expert with non-negative integers, e.g. 12:178";
        return NULL;
    }
    if (strcmp(key, "kv.device") == 0) {
        if (strcmp(item, "cpu") == 0) return NULL;
        if (strncmp(item, "hip:", 4) == 0 && parse_uint_str(item + 4, strlen(item + 4), 63, NULL) == 0) return NULL;
        return "cpu or hip:N";
    }
    return NULL;
}

static const char *check_listen(const char *v) {
    const char *colon = strrchr(v, ':');
    if (!colon || colon == v) return "host:port, e.g. 127.0.0.1:8000";
    unsigned long port;
    if (parse_uint_str(colon + 1, strlen(colon + 1), 65535, &port) != 0 || port == 0)
        return "host:port with port 1..65535";
    if (v[0] == '[' && colon[-1] != ']') return "[ipv6]:port";
    return NULL;
}

int tlm_cfg_validate(const char *key, const char *value, char *norm, size_t normcap,
                     char *why, size_t whycap) {
    char desc[160];
    if (why && whycap) why[0] = 0;
    if (!key || !tlm_cfg_key_valid(key)) {
        set_why(why, whycap, "invalid key '%s': keys are lowercase dotted names [a-z0-9_.]%s", key ? key : "", "");
        return TLM_EINVAL;
    }
    const tlm_cfg_key *k = tlm_cfg_schema_find(key);
    if (!k) {
        set_why(why, whycap, "unknown key '%s'%s", key, "");
        return TLM_ENOTFOUND;
    }
    if (!value) value = "";
    if (strchr(value, '\n') || strchr(value, '\r')) {
        set_why(why, whycap, "%s: value contains a line break%s", key, "");
        return TLM_EINVAL;
    }
    tlm_cfg_describe(k, desc, sizeof desc);

    /* keywords such as "auto" */
    if (k->type != TLM_CFG_ENUM && k->enum_values && match_word(k->enum_values, value, norm, normcap))
        return 0;

    switch (k->type) {
    case TLM_CFG_STR: {
        if (strcmp(key, "server.listen") == 0) {
            const char *e = check_listen(value);
            if (e) { set_why(why, whycap, "server.listen: expected %s, got '%s'", e, value); return TLM_EINVAL; }
        }
        break;
    }
    case TLM_CFG_INT: {
        int64_t v;
        int rc = tlm_cfg_parse_int(value, &v);
        if (rc != 0) { set_why(why, whycap, "'%s' is not %s", value, desc); return TLM_EINVAL; }
        if (k->min <= k->max && ((double)v < k->min || (double)v > k->max)) {
            set_why(why, whycap, "%s is out of range: expected %s", value, desc);
            return TLM_ERANGE;
        }
        break;
    }
    case TLM_CFG_SIZE: {
        uint64_t v;
        int rc = tlm_cfg_parse_size(value, &v);
        if (rc != 0) { set_why(why, whycap, "'%s' is not %s", value, desc); return TLM_EINVAL; }
        if (k->min <= k->max && ((double)v < k->min || (double)v > k->max)) {
            set_why(why, whycap, "%s is out of range: expected %s", value, desc);
            return TLM_ERANGE;
        }
        break;
    }
    case TLM_CFG_FRAC: {
        double v;
        int rc = tlm_cfg_parse_frac(value, &v);
        if (rc != 0) { set_why(why, whycap, "'%s' is not %s", value, desc); return TLM_EINVAL; }
        if (k->min <= k->max && (v < k->min - 1e-12 || v > k->max + 1e-12)) {
            set_why(why, whycap, "%s is out of range: expected %s", value, desc);
            return TLM_ERANGE;
        }
        break;
    }
    case TLM_CFG_BOOL: {
        int b;
        if (tlm_cfg_parse_bool(value, &b) != 0) { set_why(why, whycap, "'%s' is not %s", value, desc); return TLM_EINVAL; }
        if (norm && normcap) {
            snprintf(norm, normcap, "%s", value);
            for (char *q = norm; *q; q++) *q = (char)tolower((unsigned char)*q);
        }
        return 0;
    }
    case TLM_CFG_ENUM: {
        if (!match_word(k->enum_values, value, norm, normcap)) {
            set_why(why, whycap, "'%s' is not allowed: expected %s", value, desc);
            return TLM_ERANGE;
        }
        return 0;
    }
    case TLM_CFG_LIST: {
        /* normalise to "a, b, c"; empty list allowed, empty items are not */
        size_t vl = strlen(value);
        char *buf = (char *)malloc(vl + 1);
        char *outb = (char *)malloc(vl * 2 + 2);
        if (!buf || !outb) { free(buf); free(outb); return TLM_ENOMEM; }
        memcpy(buf, value, vl + 1);
        size_t on = 0;
        outb[0] = 0;
        int nitems = 0, rc = 0;
        int all_blank = 1;
        for (const char *q = value; *q; q++) if (*q != ' ' && *q != '\t') { all_blank = 0; break; }
        if (all_blank && strcmp(key, "kv.device") == 0) {
            set_why(why, whycap, "kv.device needs a value: %s%s", "cpu, or GPUs such as hip:0, hip:1", "");
            rc = TLM_EINVAL;
        } else if (!all_blank) {
            char *save = buf;
            for (;;) {
                char *comma = strchr(save, ',');
                if (comma) *comma = 0;
                char *it = save;
                while (*it == ' ' || *it == '\t') it++;
                char *end = it + strlen(it);
                while (end > it && (end[-1] == ' ' || end[-1] == '\t')) *--end = 0;
                if (!*it) {
                    set_why(why, whycap, "%s: empty item in list '%s'", key, value);
                    rc = TLM_EINVAL;
                    break;
                }
                const char *e = check_item(key, it);
                if (e) {
                    set_why(why, whycap, "'%s' is not a valid item: expected %s", it, e);
                    rc = TLM_EINVAL;
                    break;
                }
                if (strcmp(key, "kv.device") == 0 && strcmp(it, "cpu") == 0 && (nitems > 0 || comma)) {
                    set_why(why, whycap, "kv.device: 'cpu' cannot be combined with GPUs ('%s')%s", value, "");
                    rc = TLM_EINVAL;
                    break;
                }
                on += (size_t)sprintf(outb + on, "%s%s", nitems ? ", " : "", it);
                nitems++;
                if (!comma) break;
                save = comma + 1;
            }
        }
        if (rc == 0 && norm && normcap) snprintf(norm, normcap, "%s", outb);
        free(buf);
        free(outb);
        return rc;
    }
    }
    if (norm && normcap) snprintf(norm, normcap, "%s", value);
    return 0;
}
