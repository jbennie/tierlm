/* tlm_cfg.c - key = value configuration: parsing, includes, precedence,
 * environment overrides, typed getters, origins and the writer.
 * Rules are documented in tlm_cfg.h. Pattern: F-CFG-01.
 */
#include "tlm_cfg.h"
#include "tlm_platform.h"

#include <ctype.h>
#include <errno.h>
#include <glob.h>
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

extern char **environ;

#define MAX_INCLUDE_DEPTH 4

typedef struct {
    char *key;
    char *value;
    tlm_cfg_origin origin;
    char *where;
    char *listbuf;            /* scratch for tlm_cfg_list */
} cfg_ent;

struct tlm_cfg {
    cfg_ent *ents;            /* sorted by key */
    int n, cap;
    char **diag;
    unsigned char *diag_err;
    int ndiag, diagcap;
    int quiet;
};

static char *xstrdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

const char *tlm_cfg_origin_name(tlm_cfg_origin o) {
    switch (o) {
    case TLM_ORIGIN_DEFAULT: return "default";
    case TLM_ORIGIN_FILE: return "file";
    case TLM_ORIGIN_ENV: return "env";
    case TLM_ORIGIN_SET: return "set";
    }
    return "?";
}

/* ---- diagnostics -------------------------------------------------------- */
static void diag_add(tlm_cfg *c, int is_err, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;
static void diag_add(tlm_cfg *c, int is_err, const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (!c->quiet) tlm_log(is_err ? TLM_LOG_ERROR : TLM_LOG_WARN, "config: %s", buf);
    if (c->ndiag == c->diagcap) {
        int nc = c->diagcap ? c->diagcap * 2 : 16;
        char **d = (char **)realloc(c->diag, (size_t)nc * sizeof *d);
        if (!d) return;
        c->diag = d;
        unsigned char *e = (unsigned char *)realloc(c->diag_err, (size_t)nc);
        if (!e) return;
        c->diag_err = e;
        c->diagcap = nc;
    }
    c->diag[c->ndiag] = xstrdup(buf);
    c->diag_err[c->ndiag] = (unsigned char)(is_err != 0);
    if (c->diag[c->ndiag]) c->ndiag++;
}

void tlm_cfg_quiet(tlm_cfg *c, int quiet) { if (c) c->quiet = quiet; }
int tlm_cfg_ndiag(const tlm_cfg *c) { return c ? c->ndiag : 0; }
const char *tlm_cfg_diag(const tlm_cfg *c, int i, int *is_error) {
    if (!c || i < 0 || i >= c->ndiag) return NULL;
    if (is_error) *is_error = c->diag_err[i];
    return c->diag[i];
}
void tlm_cfg_clear_diag(tlm_cfg *c) {
    if (!c) return;
    for (int i = 0; i < c->ndiag; i++) free(c->diag[i]);
    c->ndiag = 0;
}

/* ---- storage ------------------------------------------------------------ */
static int find_idx(const tlm_cfg *c, const char *key, int *ins) {
    int lo = 0, hi = c->n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int r = strcmp(c->ents[mid].key, key);
        if (r == 0) return mid;
        if (r < 0) lo = mid + 1; else hi = mid - 1;
    }
    if (ins) *ins = lo;
    return -1;
}

static const cfg_ent *get_ent(const tlm_cfg *c, const char *key) {
    if (!c || !key) return NULL;
    int i = find_idx(c, key, NULL);
    return i >= 0 ? &c->ents[i] : NULL;
}

/* Store a validated value; lower-ranked origins never replace higher ones.
 * Returns 1 if stored, 0 if shadowed, <0 on error. */
static int put(tlm_cfg *c, const char *key, const char *value, tlm_cfg_origin origin, const char *where) {
    int ins = 0;
    int i = find_idx(c, key, &ins);
    if (i >= 0) {
        cfg_ent *e = &c->ents[i];
        if (origin < e->origin) return 0;
        char *v = xstrdup(value), *w = xstrdup(where ? where : "");
        if (!v || !w) { free(v); free(w); return TLM_ENOMEM; }
        free(e->value); free(e->where); free(e->listbuf);
        e->value = v; e->where = w; e->listbuf = NULL; e->origin = origin;
        return 1;
    }
    if (c->n == c->cap) {
        int nc = c->cap ? c->cap * 2 : 64;
        cfg_ent *ne = (cfg_ent *)realloc(c->ents, (size_t)nc * sizeof *ne);
        if (!ne) return TLM_ENOMEM;
        c->ents = ne;
        c->cap = nc;
    }
    cfg_ent e = {xstrdup(key), xstrdup(value), origin, xstrdup(where ? where : ""), NULL};
    if (!e.key || !e.value || !e.where) { free(e.key); free(e.value); free(e.where); return TLM_ENOMEM; }
    memmove(&c->ents[ins + 1], &c->ents[ins], (size_t)(c->n - ins) * sizeof(cfg_ent));
    c->ents[ins] = e;
    c->n++;
    return 1;
}

tlm_cfg *tlm_cfg_new(void) {
    tlm_cfg *c = (tlm_cfg *)calloc(1, sizeof *c);
    if (!c) return NULL;
    for (const tlm_cfg_key *k = tlm_cfg_schema; k->key; k++) {
        if (!k->def || strchr(k->key, '*')) continue;
        if (put(c, k->key, k->def, TLM_ORIGIN_DEFAULT, "default") < 0) { tlm_cfg_free(c); return NULL; }
    }
    return c;
}

void tlm_cfg_free(tlm_cfg *c) {
    if (!c) return;
    for (int i = 0; i < c->n; i++) {
        free(c->ents[i].key); free(c->ents[i].value); free(c->ents[i].where); free(c->ents[i].listbuf);
    }
    free(c->ents);
    tlm_cfg_clear_diag(c);
    free(c->diag);
    free(c->diag_err);
    free(c);
}

int tlm_cfg_unset(tlm_cfg *c, const char *key) {
    if (!c || !key) return TLM_EINVAL;
    int i = find_idx(c, key, NULL);
    if (i < 0) return TLM_ENOTFOUND;
    free(c->ents[i].key); free(c->ents[i].value); free(c->ents[i].where); free(c->ents[i].listbuf);
    memmove(&c->ents[i], &c->ents[i + 1], (size_t)(c->n - i - 1) * sizeof(cfg_ent));
    c->n--;
    return 0;
}

/* validate + store; returns the tlm error code (0 also when shadowed) */
static int set_checked(tlm_cfg *c, const char *key, const char *value, tlm_cfg_origin origin,
                       const char *where, char *why, size_t whycap) {
    size_t cap = strlen(value ? value : "") * 2 + 64;
    char *norm = (char *)malloc(cap);
    if (!norm) return TLM_ENOMEM;
    int rc = tlm_cfg_validate(key, value, norm, cap, why, whycap);
    if (rc == 0) {
        rc = put(c, key, norm, origin, where);
        if (rc > 0) rc = 0;
    }
    free(norm);
    return rc;
}

int tlm_cfg_set_origin(tlm_cfg *c, const char *key, const char *value, tlm_cfg_origin origin, const char *where) {
    if (!c || !key) return TLM_EINVAL;
    char why[256];
    int rc = set_checked(c, key, value ? value : "", origin, where, why, sizeof why);
    if (rc != 0) return tlm_seterr(rc, "%s%s%s", where && *where ? where : "", where && *where ? ": " : "", why);
    return 0;
}

int tlm_cfg_set(tlm_cfg *c, const char *key, const char *value) {
    return tlm_cfg_set_origin(c, key, value, TLM_ORIGIN_SET, "set");
}

/* ---- line syntax -------------------------------------------------------- */
static void rtrim(char *s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r')) s[--n] = 0;
}
static char *ltrim(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

/* Cut a '#' comment outside double quotes (backslash escapes inside quotes). */
static void strip_comment(char *s) {
    int inq = 0;
    for (char *p = s; *p; p++) {
        if (inq && *p == '\\' && p[1]) { p++; continue; }
        if (*p == '"') inq = !inq;
        else if (*p == '#' && !inq) { *p = 0; return; }
    }
}

/* If v is "...", unescape in place. Returns 0 ok, -1 malformed. */
static int unquote(char *v) {
    if (v[0] != '"') return 0;
    char *o = v, *p = v + 1;
    while (*p && *p != '"') {
        if (*p == '\\' && (p[1] == '"' || p[1] == '\\')) p++;
        *o++ = *p++;
    }
    if (*p != '"') return -1;                 /* unterminated */
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p) return -1;                        /* text after closing quote */
    *o = 0;
    return 0;
}

typedef enum { LN_EMPTY, LN_KV, LN_INCLUDE, LN_ERROR } line_kind;

/* Parse one line in place. On LN_KV: *k, *v; LN_INCLUDE: *v = glob; LN_ERROR: *v = message */
static line_kind parse_line(char *line, char **k, char **v) {
    strip_comment(line);
    rtrim(line);
    char *s = ltrim(line);
    if (!*s) return LN_EMPTY;
    if (strncmp(s, "include", 7) == 0 && (s[7] == ' ' || s[7] == '\t')) {
        char *g = ltrim(s + 7);
        if (unquote(g) != 0) { *v = (char *)"malformed quoted include path"; return LN_ERROR; }
        if (!*g) { *v = (char *)"include needs a path or glob"; return LN_ERROR; }
        *v = g;
        return LN_INCLUDE;
    }
    char *eq = strchr(s, '=');
    if (!eq) { *v = (char *)"expected 'key = value'"; return LN_ERROR; }
    *eq = 0;
    rtrim(s);
    char *val = ltrim(eq + 1);
    if (!*s) { *v = (char *)"missing key before '='"; return LN_ERROR; }
    if (!tlm_cfg_key_valid(s)) { *v = (char *)"invalid key: keys are lowercase dotted names [a-z0-9_.]"; *k = s; return LN_ERROR; }
    if (unquote(val) != 0) { *v = (char *)"unterminated or malformed quoted value"; *k = s; return LN_ERROR; }
    *k = s;
    *v = val;
    return LN_KV;
}

int tlm_cfg_parse_text(const char *text, const char *name, tlm_cfg_line_fn fn, void *ctx) {
    if (!text) return TLM_EINVAL;
    if (!name) name = "<string>";
    char *buf = xstrdup(text);
    if (!buf) return TLM_ENOMEM;
    int lineno = 0, rc = 0;
    char *p = buf;
    while (p && rc == 0) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = 0;
        lineno++;
        char *k = NULL, *v = NULL;
        switch (parse_line(p, &k, &v)) {
        case LN_EMPTY: break;
        case LN_KV: rc = fn ? fn(ctx, k, v, lineno) : 0; break;
        case LN_INCLUDE: rc = fn ? fn(ctx, "include", v, lineno) : 0; break;
        case LN_ERROR: rc = tlm_seterr(TLM_EFORMAT, "%s:%d: %s", name, lineno, v); break;
        }
        p = nl ? nl + 1 : NULL;
    }
    free(buf);
    return rc;
}

/* ---- loading ------------------------------------------------------------ */
typedef struct {
    tlm_cfg *c;
    int strict;
    int suppress;             /* re-applying a file: no diagnostics */
    int first_err;
    char **seen; int nseen;   /* real paths loaded in this load_etc */
} lctx;

static void lfail(lctx *L, int code, int is_err, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 4, 5)))
#endif
    ;
/* Record a diagnostic; code != 0 makes it fatal (first one kept in errmsg). */
static void lfail(lctx *L, int code, int is_err, const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (!L->suppress) diag_add(L->c, is_err, "%s", buf);
    if (code != 0 && L->first_err == 0) {
        L->first_err = code;
        tlm_seterr(code, "%s", buf);
    }
}

static void seen_add(lctx *L, const char *path) {
    char rp[PATH_MAX];
    if (!realpath(path, rp)) return;
    char **n = (char **)realloc(L->seen, (size_t)(L->nseen + 1) * sizeof *n);
    if (!n) return;
    L->seen = n;
    L->seen[L->nseen] = xstrdup(rp);
    if (L->seen[L->nseen]) L->nseen++;
}
static int seen_has(const lctx *L, const char *path) {
    char rp[PATH_MAX];
    if (!realpath(path, rp)) return 0;
    for (int i = 0; i < L->nseen; i++) if (strcmp(L->seen[i], rp) == 0) return 1;
    return 0;
}

static int load_path(lctx *L, const char *path, int depth, int must_exist);

static int has_glob_chars(const char *s) { return strpbrk(s, "*?[") != NULL; }

static void do_include(lctx *L, const char *name, int lineno, const char *pattern, int depth) {
    if (depth + 1 > MAX_INCLUDE_DEPTH) {
        lfail(L, TLM_EFORMAT, 1, "%s:%d: include nesting deeper than %d", name, lineno, MAX_INCLUDE_DEPTH);
        return;
    }
    char full[PATH_MAX];
    if (pattern[0] == '/') {
        snprintf(full, sizeof full, "%s", pattern);
    } else {
        const char *slash = strrchr(name, '/');
        if (slash && name[0] != '<')
            snprintf(full, sizeof full, "%.*s/%s", (int)(slash - name), name, pattern);
        else
            snprintf(full, sizeof full, "%s", pattern);
    }
    glob_t g;
    memset(&g, 0, sizeof g);
    int gr = glob(full, 0, NULL, &g);
    if (gr == GLOB_NOMATCH) {
        if (!has_glob_chars(full))
            lfail(L, L->strict ? TLM_ENOTFOUND : 0, L->strict, "%s:%d: include %s: no such file", name, lineno, full);
    } else if (gr != 0) {
        lfail(L, TLM_EIO, 1, "%s:%d: include %s: glob failed", name, lineno, full);
    } else {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            struct stat st;
            if (stat(g.gl_pathv[i], &st) == 0 && S_ISDIR(st.st_mode)) continue;
            load_path(L, g.gl_pathv[i], depth + 1, 1);
        }
    }
    globfree(&g);
}

static void load_text(lctx *L, const char *text, const char *name, int depth) {
    char *buf = xstrdup(text);
    if (!buf) { lfail(L, TLM_ENOMEM, 1, "%s: out of memory", name); return; }
    char where[PATH_MAX + 32];
    int lineno = 0;
    char *p = buf;
    while (p) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = 0;
        lineno++;
        char *k = NULL, *v = NULL;
        line_kind kind = parse_line(p, &k, &v);
        if (kind == LN_ERROR) {
            if (k) lfail(L, TLM_EFORMAT, 1, "%s:%d: %s ('%s')", name, lineno, v, k);
            else lfail(L, TLM_EFORMAT, 1, "%s:%d: %s", name, lineno, v);
        } else if (kind == LN_INCLUDE) {
            do_include(L, name, lineno, v, depth);
        } else if (kind == LN_KV) {
            char why[512];
            snprintf(where, sizeof where, "%s:%d", name, lineno);
            int rc = set_checked(L->c, k, v, TLM_ORIGIN_FILE, where, why, sizeof why);
            if (rc == TLM_ENOTFOUND) {
                lfail(L, L->strict ? TLM_EINVAL : 0, L->strict, "%s:%d: unknown key '%s'%s", name, lineno, k,
                      L->strict ? "" : " (ignored)");
            } else if (rc == TLM_ENOMEM) {
                lfail(L, rc, 1, "%s:%d: out of memory", name, lineno);
            } else if (rc != 0) {
                lfail(L, L->strict ? rc : 0, 1, "%s:%d: %s: %s%s", name, lineno, k, why,
                      L->strict ? "" : " (value ignored)");
            }
        }
        p = nl ? nl + 1 : NULL;
    }
    free(buf);
}

static int load_path(lctx *L, const char *path, int depth, int must_exist) {
    char *text = NULL;
    size_t n = 0;
    int rc = tlm_read_file(path, &text, &n);
    if (rc != 0) {
        if (!must_exist && rc == TLM_ENOTFOUND) return 0;
        lfail(L, rc, 1, "%s: cannot read: %s", path, tlm_errmsg());
        return rc;
    }
    if (memchr(text, 0, n)) {
        lfail(L, TLM_EFORMAT, 1, "%s: contains NUL bytes (not a text file)", path);
        free(text);
        return TLM_EFORMAT;
    }
    seen_add(L, path);
    load_text(L, text, path, depth);
    free(text);
    return 0;
}

static void lctx_done(lctx *L) {
    for (int i = 0; i < L->nseen; i++) free(L->seen[i]);
    free(L->seen);
}

int tlm_cfg_load_string(tlm_cfg *c, const char *text, const char *name, int strict) {
    if (!c || !text) return TLM_EINVAL;
    lctx L = {c, strict, 0, 0, NULL, 0};
    load_text(&L, text, name ? name : "<string>", 0);
    lctx_done(&L);
    return L.first_err;
}

int tlm_cfg_load_file(tlm_cfg *c, const char *path, int strict) {
    if (!c || !path) return TLM_EINVAL;
    lctx L = {c, strict, 0, 0, NULL, 0};
    load_path(&L, path, 0, 1);
    lctx_done(&L);
    return L.first_err;
}

/* ---- environment -------------------------------------------------------- */
static void env_name(const char *key, char *out, size_t cap) {
    size_t n = (size_t)snprintf(out, cap, "TIERLM_");
    for (const char *p = key; *p && n + 1 < cap; p++)
        out[n++] = *p == '.' ? '_' : (char)toupper((unsigned char)*p);
    out[n < cap ? n : cap - 1] = 0;
}

typedef struct { char **names; int n; } nameset;
static int ns_has(const nameset *s, const char *nm) {
    for (int i = 0; i < s->n; i++) if (strcmp(s->names[i], nm) == 0) return 1;
    return 0;
}
static void ns_add(nameset *s, const char *nm) {
    char **n = (char **)realloc(s->names, (size_t)(s->n + 1) * sizeof *n);
    if (!n) return;
    s->names = n;
    s->names[s->n] = xstrdup(nm);
    if (s->names[s->n]) s->n++;
}

static int env_apply_one(tlm_cfg *c, const char *key, const char *envnm, const char *val, int *first) {
    char where[160], why[512];
    snprintf(where, sizeof where, "env %s", envnm);
    int rc = set_checked(c, key, val, TLM_ORIGIN_ENV, where, why, sizeof why);
    if (rc != 0) {
        diag_add(c, 1, "%s: %s: %s", where, key, why);
        if (*first == 0) {
            *first = rc == TLM_ENOTFOUND ? TLM_EINVAL : rc;
            tlm_seterr(*first, "%s: %s: %s", where, key, why);
        }
    }
    return rc;
}

int tlm_cfg_apply_env(tlm_cfg *c) {
    if (!c) return TLM_EINVAL;
    int first = 0;
    nameset done = {NULL, 0};
    char nm[256];
    /* exact schema keys */
    for (const tlm_cfg_key *k = tlm_cfg_schema; k->key; k++) {
        if (strchr(k->key, '*')) continue;
        env_name(k->key, nm, sizeof nm);
        const char *v = getenv(nm);
        if (!v) continue;
        ns_add(&done, nm);
        env_apply_one(c, k->key, nm, v, &first);
    }
    /* keys already present (content.*, hardware.* from files) */
    int n0 = c->n;
    char **present = (char **)calloc((size_t)(n0 ? n0 : 1), sizeof(char *));
    for (int i = 0; present && i < n0; i++) present[i] = xstrdup(c->ents[i].key);
    for (int i = 0; present && i < n0; i++) {
        if (!present[i]) continue;
        env_name(present[i], nm, sizeof nm);
        if (ns_has(&done, nm)) continue;
        const char *v = getenv(nm);
        if (!v) continue;
        ns_add(&done, nm);
        env_apply_one(c, present[i], nm, v, &first);
    }
    if (present) for (int i = 0; i < n0; i++) free(present[i]);
    free(present);
    /* remaining TIERLM_CONTENT_* / TIERLM_HARDWARE_*: '_' -> '.' best effort */
    for (char **e = environ; e && *e; e++) {
        const char *ent = *e;
        const char *eq = strchr(ent, '=');
        if (!eq || strncmp(ent, "TIERLM_", 7) != 0) continue;
        size_t nl = (size_t)(eq - ent);
        if (nl >= sizeof nm) continue;
        memcpy(nm, ent, nl);
        nm[nl] = 0;
        if (ns_has(&done, nm)) continue;
        if (strncmp(nm, "TIERLM_CONTENT_", 15) != 0 && strncmp(nm, "TIERLM_HARDWARE_", 16) != 0) continue;
        char key[256];
        size_t j = 0;
        for (const char *p = nm + 7; *p && j + 1 < sizeof key; p++)
            key[j++] = *p == '_' ? '.' : (char)tolower((unsigned char)*p);
        key[j] = 0;
        ns_add(&done, nm);
        env_apply_one(c, key, nm, eq + 1, &first);
    }
    for (int i = 0; i < done.n; i++) free(done.names[i]);
    free(done.names);
    return first;
}

/* ---- /etc loading ------------------------------------------------------- */
static int is_regular(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

int tlm_cfg_load_etc(tlm_cfg *c, const char *etc_dir, int strict) {
    if (!c) return TLM_EINVAL;
    if (!etc_dir || !*etc_dir) {
        etc_dir = getenv("TIERLM_ETC");
        if (!etc_dir || !*etc_dir) etc_dir = TLM_ETC_DEFAULT;
    }
    lctx L = {c, strict, 0, 0, NULL, 0};
    char path[PATH_MAX];
    snprintf(path, sizeof path, "%s/hardware.conf", etc_dir);
    if (is_regular(path)) load_path(&L, path, 0, 1);
    snprintf(path, sizeof path, "%s/tierlm.conf", etc_dir);
    if (is_regular(path)) load_path(&L, path, 0, 1);
    snprintf(path, sizeof path, "%s/conf.d/*.conf", etc_dir);
    glob_t g;
    memset(&g, 0, sizeof g);
    int gr = glob(path, 0, NULL, &g);
    if (gr == 0) {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            if (!is_regular(g.gl_pathv[i])) continue;
            int again = seen_has(&L, g.gl_pathv[i]);
            int save = L.suppress;
            if (again) L.suppress = 1;       /* already reported via include */
            load_path(&L, g.gl_pathv[i], 0, 1);
            L.suppress = save;
        }
    } else if (gr != GLOB_NOMATCH) {
        lfail(&L, TLM_EIO, 1, "%s: glob failed", path);
    }
    globfree(&g);
    lctx_done(&L);
    char saved[512] = "";
    if (L.first_err) snprintf(saved, sizeof saved, "%s", tlm_errmsg());
    int erc = tlm_cfg_apply_env(c);
    if (L.first_err) {
        /* keep the first file error as the message (apply_env may overwrite it) */
        return tlm_seterr(L.first_err, "%s", saved);
    }
    if (erc != 0 && strict) return erc;
    return 0;
}

/* ---- getters ------------------------------------------------------------ */
const char *tlm_cfg_str(const tlm_cfg *c, const char *key) {
    const cfg_ent *e = get_ent(c, key);
    return e ? e->value : NULL;
}

int tlm_cfg_is_auto(const tlm_cfg *c, const char *key) {
    const char *v = tlm_cfg_str(c, key);
    return v && strcasecmp(v, "auto") == 0;
}

int64_t tlm_cfg_int(const tlm_cfg *c, const char *key, int64_t dflt) {
    const char *v = tlm_cfg_str(c, key);
    int64_t x;
    if (!v || tlm_cfg_parse_int(v, &x) != 0) return dflt;
    return x;
}

uint64_t tlm_cfg_size(const tlm_cfg *c, const char *key, uint64_t dflt) {
    const char *v = tlm_cfg_str(c, key);
    uint64_t x;
    if (!v || tlm_cfg_parse_size(v, &x) != 0) return dflt;
    return x;
}

double tlm_cfg_frac(const tlm_cfg *c, const char *key, double dflt) {
    const char *v = tlm_cfg_str(c, key);
    double x;
    if (!v || tlm_cfg_parse_frac(v, &x) != 0) return dflt;
    return x;
}

int tlm_cfg_bool(const tlm_cfg *c, const char *key, int dflt) {
    const char *v = tlm_cfg_str(c, key);
    int x;
    if (!v || tlm_cfg_parse_bool(v, &x) != 0) return dflt;
    return x;
}

int tlm_cfg_list(const tlm_cfg *c, const char *key, const char **out, int max) {
    if (!c || !key) return 0;
    int i = find_idx(c, key, NULL);
    if (i < 0) return 0;
    cfg_ent *e = &c->ents[i];            /* entries are owned by c; listbuf is scratch */
    free(e->listbuf);
    e->listbuf = xstrdup(e->value);
    if (!e->listbuf) return 0;
    int n = 0;
    char *p = e->listbuf;
    while (*p && n < max) {
        char *comma = strchr(p, ',');
        if (comma) *comma = 0;
        char *it = ltrim(p);
        rtrim(it);
        if (*it && out) out[n++] = it;
        else if (*it) n++;
        if (!comma) break;
        p = comma + 1;
    }
    return n;
}

tlm_cfg_origin tlm_cfg_origin_of(const tlm_cfg *c, const char *key, const char **where) {
    const cfg_ent *e = get_ent(c, key);
    if (!e) {
        if (where) *where = "unset";
        return TLM_ORIGIN_DEFAULT;
    }
    if (where) *where = e->where;
    return e->origin;
}

int tlm_cfg_next(const tlm_cfg *c, int *iter, const char **key, const char **value) {
    if (!c || !iter || *iter < 0 || *iter >= c->n) return 0;
    const cfg_ent *e = &c->ents[(*iter)++];
    if (key) *key = e->key;
    if (value) *value = e->value;
    return 1;
}

/* ---- cross-key checks --------------------------------------------------- */
int tlm_cfg_check(tlm_cfg *c) {
    if (!c) return TLM_EINVAL;
    int rc = 0;
    double ask = tlm_cfg_frac(c, "decide.ask_threshold", 0.60);
    double aut = tlm_cfg_frac(c, "decide.auto_threshold", 0.90);
    if (ask > aut) {
        const char *w1 = NULL, *w2 = NULL;
        tlm_cfg_origin_of(c, "decide.ask_threshold", &w1);
        tlm_cfg_origin_of(c, "decide.auto_threshold", &w2);
        diag_add(c, 1, "decide.ask_threshold (%g, %s) must not exceed decide.auto_threshold (%g, %s)",
                 ask, w1, aut, w2);
        rc = tlm_seterr(TLM_EINVAL, "decide.ask_threshold (%g, %s) must not exceed decide.auto_threshold (%g, %s)",
                        ask, w1, aut, w2);
    }
    const char *kv = tlm_cfg_str(c, "kv.device");
    if (kv) {
        char why[256];
        if (tlm_cfg_validate("kv.device", kv, NULL, 0, why, sizeof why) != 0) {
            diag_add(c, 1, "%s", why);
            if (!rc) rc = tlm_seterr(TLM_EINVAL, "%s", why);
        }
    }
    return rc;
}

/* ---- writer ------------------------------------------------------------- */
static int needs_quotes(const char *v) {
    if (!*v) return 0;
    if (v[0] == ' ' || v[0] == '\t') return 1;
    size_t n = strlen(v);
    if (v[n - 1] == ' ' || v[n - 1] == '\t') return 1;
    return strpbrk(v, "#\"\\") != NULL;
}

static void write_value(FILE *out, const char *v) {
    if (!needs_quotes(v)) { fputs(v, out); return; }
    fputc('"', out);
    for (const char *p = v; *p; p++) {
        if (*p == '"' || *p == '\\') fputc('\\', out);
        fputc(*p, out);
    }
    fputc('"', out);
}

int tlm_cfg_write(FILE *out, const char *header, const tlm_cfg_line *lines, int n) {
    if (!out) return TLM_EINVAL;
    int width = 22;
    for (int i = 0; i < n; i++) {
        if (!lines[i].key || strcmp(lines[i].key, "include") == 0) continue;
        int l = (int)strlen(lines[i].key);
        if (l > width) width = l < 40 ? l : 40;
    }
    if (header && *header) {
        const char *p = header;
        while (*p) {
            const char *nl = strchr(p, '\n');
            int len = nl ? (int)(nl - p) : (int)strlen(p);
            if (len == 0) fputs("#\n", out);
            else if (p[0] == '#') fprintf(out, "%.*s\n", len, p);
            else fprintf(out, "# %.*s\n", len, p);
            if (!nl) break;
            p = nl + 1;
        }
        fputc('\n', out);
    }
    for (int i = 0; i < n; i++) {
        const tlm_cfg_line *L = &lines[i];
        if (!L->key) {
            if (L->comment && *L->comment) {
                if (L->comment[0] == '#') fprintf(out, "%s\n", L->comment);
                else fprintf(out, "# %s\n", L->comment);
            } else {
                fputc('\n', out);
            }
            continue;
        }
        const char *v = L->value ? L->value : "";
        if (strchr(v, '\n') || strchr(v, '\r'))
            return tlm_seterr(TLM_EINVAL, "config writer: value of %s contains a line break", L->key);
        if (strcmp(L->key, "include") == 0) {
            fputs("include ", out);
            write_value(out, v);
        } else {
            if (!tlm_cfg_key_valid(L->key))
                return tlm_seterr(TLM_EINVAL, "config writer: invalid key '%s'", L->key);
            if (!*v && !(L->comment && *L->comment)) {
                fprintf(out, "%-*s =\n", width, L->key);       /* empty value: no trailing space */
                continue;
            }
            fprintf(out, "%-*s = ", width, L->key);
            if (L->comment && *L->comment) {
                write_value(out, v);
                size_t used = strlen(v);
                if (needs_quotes(v)) {
                    used += 2;
                    for (const char *q = v; *q; q++) used += (*q == '"' || *q == '\\');
                }
                for (size_t pad = used; pad < 12; pad++) fputc(' ', out);
                fprintf(out, "  # %s", L->comment);
            } else {
                write_value(out, v);
            }
        }
        fputc('\n', out);
    }
    if (ferror(out)) return tlm_seterr(TLM_EIO, "config writer: write failed");
    return 0;
}
