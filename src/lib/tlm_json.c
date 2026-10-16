/* tlm_json.c - strict RFC 8259 JSON reader (arena DOM) and a small writer.
 *
 * Pattern: F-JSON-01 (support.md). Design points:
 *   - every node and string lives in an arena of linked blocks that are never
 *     reallocated, so pointers stay valid until tlm_json_free();
 *   - container children are collected on a scratch stack and copied into the
 *     arena once complete (no growth inside the arena);
 *   - input is bounded by `len` (it need not be NUL-terminated);
 *   - strings are validated as UTF-8, escapes decoded, surrogate pairs joined;
 *   - numbers are validated by hand, then converted with strtod over exactly
 *     that span (strtod alone accepts NaN, Infinity and hex).
 */
#include "tlm_json.h"
#include "tlm_platform.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

/* ---- arena ------------------------------------------------------------- */
#define ARENA_BLOCK (64u * 1024u)

typedef struct ablk {
    struct ablk *next;
    size_t used, cap;
    /* data follows, 16-byte aligned (header padded below) */
} ablk;
#define ABLK_HDR ((sizeof(ablk) + 15u) & ~(size_t)15u)

struct tlm_json {
    ablk     *head;
    tlm_jval *root;
};

static void *arena_alloc(tlm_json *j, size_t n) {
    n = (n + 15u) & ~(size_t)15u;
    if (n == 0) n = 16;
    ablk *b = j->head;
    if (!b || b->cap - b->used < n) {
        size_t cap = n > ARENA_BLOCK ? n : ARENA_BLOCK;
        if (cap > SIZE_MAX - ABLK_HDR) return NULL;
        b = (ablk *)malloc(ABLK_HDR + cap);
        if (!b) return NULL;
        b->used = 0;
        b->cap = cap;
        /* a dedicated oversized block goes behind the current one so the
         * current block keeps serving small allocations */
        if (j->head && n > ARENA_BLOCK) {
            b->next = j->head->next;
            j->head->next = b;
        } else {
            b->next = j->head;
            j->head = b;
        }
    }
    void *p = (unsigned char *)b + ABLK_HDR + b->used;
    b->used += n;
    return p;
}

void tlm_json_free(tlm_json *j) {
    if (!j) return;
    ablk *b = j->head;
    while (b) {
        ablk *n = b->next;
        free(b);
        b = n;
    }
    free(j);
}

const tlm_jval *tlm_json_root(const tlm_json *j) { return j ? j->root : NULL; }

/* ---- parser ------------------------------------------------------------ */
typedef struct {
    const unsigned char *start, *p, *end;
    tlm_json *j;
    char *err;
    size_t errcap;
    int failed;
    tlm_jval **vs; size_t vn, vcap;    /* scratch value stack */
    char **ks; size_t kn, kcap;        /* scratch key stack */
} jp;

static void jp_fail_at(jp *P, const unsigned char *at, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;
static void jp_fail_at(jp *P, const unsigned char *at, const char *fmt, ...) {
    if (P->failed) return;
    P->failed = 1;
    if (!P->err || P->errcap == 0) return;
    size_t off = (size_t)(at - P->start);
    size_t line = 1, col = 1;
    for (const unsigned char *q = P->start; q < at; q++) {
        if (*q == '\n') { line++; col = 1; } else col++;
    }
    char what[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(what, sizeof what, fmt, ap);
    va_end(ap);
    snprintf(P->err, P->errcap, "json: %s at offset %zu (line %zu, column %zu)", what, off, line, col);
}
#define FAIL(...) do { jp_fail_at(P, P->p, __VA_ARGS__); return NULL; } while (0)

static void skip_ws(jp *P) {
    while (P->p < P->end && (*P->p == ' ' || *P->p == '\t' || *P->p == '\n' || *P->p == '\r')) P->p++;
}

static tlm_jval *new_val(jp *P, tlm_jtype t) {
    tlm_jval *v = (tlm_jval *)arena_alloc(P->j, sizeof *v);
    if (!v) { jp_fail_at(P, P->p, "out of memory"); return NULL; }
    memset(v, 0, sizeof *v);
    v->type = t;
    return v;
}

static int vpush(jp *P, tlm_jval *v) {
    if (P->vn == P->vcap) {
        size_t nc = P->vcap ? P->vcap * 2 : 64;
        tlm_jval **n = (tlm_jval **)realloc(P->vs, nc * sizeof *n);
        if (!n) return -1;
        P->vs = n; P->vcap = nc;
    }
    P->vs[P->vn++] = v;
    return 0;
}
static int kpush(jp *P, char *k) {
    if (P->kn == P->kcap) {
        size_t nc = P->kcap ? P->kcap * 2 : 64;
        char **n = (char **)realloc(P->ks, nc * sizeof *n);
        if (!n) return -1;
        P->ks = n; P->kcap = nc;
    }
    P->ks[P->kn++] = k;
    return 0;
}

static int hexval(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int read_hex4(jp *P, const unsigned char *q, unsigned *out) {
    if (P->end - q < 4) return -1;
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        int h = hexval(q[i]);
        if (h < 0) return -1;
        v = (v << 4) | (unsigned)h;
    }
    *out = v;
    return 0;
}

static size_t put_utf8(unsigned char *o, unsigned cp) {
    if (cp < 0x80) { o[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800) { o[0] = (unsigned char)(0xC0 | (cp >> 6)); o[1] = (unsigned char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (unsigned char)(0xE0 | (cp >> 12));
        o[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (unsigned char)(0x80 | (cp & 0x3F));
        return 3;
    }
    o[0] = (unsigned char)(0xF0 | (cp >> 18));
    o[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    o[3] = (unsigned char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Length of a valid UTF-8 sequence starting at q (lead byte >= 0x80), or 0. */
static size_t utf8_seq(const unsigned char *q, const unsigned char *end) {
    unsigned char c = q[0];
    size_t n;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) n = 2;
    else if (c >= 0xE0 && c <= 0xEF) {
        n = 3;
        if (c == 0xE0) lo = 0xA0;
        if (c == 0xED) hi = 0x9F;           /* no UTF-16 surrogates */
    } else if (c >= 0xF0 && c <= 0xF4) {
        n = 4;
        if (c == 0xF0) lo = 0x90;
        if (c == 0xF4) hi = 0x8F;           /* <= U+10FFFF */
    } else return 0;
    if ((size_t)(end - q) < n) return 0;
    if (q[1] < lo || q[1] > hi) return 0;
    for (size_t i = 2; i < n; i++)
        if (q[i] < 0x80 || q[i] > 0xBF) return 0;
    return n;
}

/* Parses a string starting at the opening quote; returns arena copy. */
static char *parse_string(jp *P, size_t *out_len) {
    const unsigned char *q = P->p + 1;
    /* pass 1: find the closing quote */
    while (q < P->end && *q != '"') {
        if (*q == '\\') { q++; if (q >= P->end) break; }
        q++;
    }
    if (q >= P->end) { P->p = q; jp_fail_at(P, P->end, "unterminated string"); return NULL; }
    size_t raw = (size_t)(q - (P->p + 1));
    unsigned char *o = (unsigned char *)arena_alloc(P->j, raw + 1);
    if (!o) { jp_fail_at(P, P->p, "out of memory"); return NULL; }
    size_t n = 0;
    const unsigned char *s = P->p + 1;
    while (s < q) {
        unsigned char c = *s;
        if (c < 0x20) { P->p = s; jp_fail_at(P, s, "raw control character 0x%02x in string", c); return NULL; }
        if (c == '\\') {
            const unsigned char *esc = s;
            s++;
            switch (*s) {
            case '"': o[n++] = '"'; s++; break;
            case '\\': o[n++] = '\\'; s++; break;
            case '/': o[n++] = '/'; s++; break;
            case 'b': o[n++] = '\b'; s++; break;
            case 'f': o[n++] = '\f'; s++; break;
            case 'n': o[n++] = '\n'; s++; break;
            case 'r': o[n++] = '\r'; s++; break;
            case 't': o[n++] = '\t'; s++; break;
            case 'u': {
                unsigned cp;
                if (read_hex4(P, s + 1, &cp) != 0 || s + 5 > q) {
                    jp_fail_at(P, esc, "\\u escape needs 4 hex digits"); return NULL;
                }
                s += 5;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    unsigned lo;
                    if (s + 6 <= q && s[0] == '\\' && s[1] == 'u' && read_hex4(P, s + 2, &lo) == 0 &&
                        lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        s += 6;
                    } else {
                        jp_fail_at(P, esc, "lone high surrogate \\u%04X", cp); return NULL;
                    }
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    jp_fail_at(P, esc, "lone low surrogate \\u%04X", cp); return NULL;
                }
                n += put_utf8(o + n, cp);
                break;
            }
            default:
                jp_fail_at(P, esc, "invalid escape '\\%c'", (*s >= 0x20 && *s < 0x7f) ? *s : '?');
                return NULL;
            }
        } else if (c >= 0x80) {
            size_t k = utf8_seq(s, q);
            if (k == 0) { jp_fail_at(P, s, "invalid UTF-8 in string"); return NULL; }
            memcpy(o + n, s, k);
            n += k;
            s += k;
        } else {
            o[n++] = c;
            s++;
        }
    }
    o[n] = 0;
    P->p = q + 1;
    *out_len = n;
    return (char *)o;
}

static tlm_jval *parse_number(jp *P) {
    const unsigned char *s = P->p, *q = P->p;
    int has_frac = 0, has_exp = 0;
    if (q < P->end && *q == '-') q++;
    if (q >= P->end) FAIL("expected digit after '-'");
    if (*q == '0') {
        q++;
        if (q < P->end && *q >= '0' && *q <= '9') { P->p = q; FAIL("leading zero in number"); }
    } else if (*q >= '1' && *q <= '9') {
        while (q < P->end && *q >= '0' && *q <= '9') q++;
    } else { P->p = q; FAIL("expected digit"); }
    if (q < P->end && *q == '.') {
        has_frac = 1;
        q++;
        if (q >= P->end || *q < '0' || *q > '9') { P->p = q; FAIL("expected digit after '.'"); }
        while (q < P->end && *q >= '0' && *q <= '9') q++;
    }
    if (q < P->end && (*q == 'e' || *q == 'E')) {
        has_exp = 1;
        q++;
        if (q < P->end && (*q == '+' || *q == '-')) q++;
        if (q >= P->end || *q < '0' || *q > '9') { P->p = q; FAIL("expected digit in exponent"); }
        while (q < P->end && *q >= '0' && *q <= '9') q++;
    }
    size_t len = (size_t)(q - s);
    char stackbuf[128];
    char *buf = stackbuf;
    if (len + 1 > sizeof stackbuf) {
        buf = (char *)arena_alloc(P->j, len + 1);
        if (!buf) FAIL("out of memory");
    }
    memcpy(buf, s, len);
    buf[len] = 0;
    tlm_jval *v = new_val(P, TLM_J_NUM);
    if (!v) return NULL;
    char *e = NULL;
    double d = strtod(buf, &e);
    if (!isfinite(d)) FAIL("number out of range");
    v->u.num.d = d;
    if (!has_frac && !has_exp) {
        errno = 0;
        long long ll = strtoll(buf, &e, 10);
        if (errno == 0 && *e == 0) { v->u.num.i = (int64_t)ll; v->u.num.is_int = 1; }
    }
    if (!v->u.num.is_int && d == floor(d) && fabs(d) < 9223372036854775808.0) {
        v->u.num.i = (int64_t)d;
        v->u.num.is_int = 1;
    }
    P->p = q;
    return v;
}

static int match_lit(jp *P, const char *lit) {
    size_t n = strlen(lit);
    if ((size_t)(P->end - P->p) < n || memcmp(P->p, lit, n) != 0) return 0;
    P->p += n;
    return 1;
}

static tlm_jval *parse_value(jp *P, int depth);

static tlm_jval *parse_array(jp *P, int depth) {
    if (depth + 1 > TLM_JSON_MAX_DEPTH) FAIL("nesting deeper than %d", TLM_JSON_MAX_DEPTH);
    tlm_jval *v = new_val(P, TLM_J_ARR);
    if (!v) return NULL;
    P->p++;   /* '[' */
    size_t base = P->vn;
    skip_ws(P);
    if (P->p < P->end && *P->p == ']') { P->p++; return v; }
    for (;;) {
        tlm_jval *it = parse_value(P, depth + 1);
        if (!it) return NULL;
        if (vpush(P, it) != 0) FAIL("out of memory");
        skip_ws(P);
        if (P->p >= P->end) FAIL("unterminated array");
        if (*P->p == ',') { P->p++; continue; }
        if (*P->p == ']') { P->p++; break; }
        FAIL("expected ',' or ']' in array");
    }
    size_t n = P->vn - base;
    v->u.arr.items = (tlm_jval **)arena_alloc(P->j, n * sizeof(tlm_jval *));
    if (!v->u.arr.items) FAIL("out of memory");
    memcpy(v->u.arr.items, P->vs + base, n * sizeof(tlm_jval *));
    v->u.arr.n = n;
    P->vn = base;
    return v;
}

static tlm_jval *parse_object(jp *P, int depth) {
    if (depth + 1 > TLM_JSON_MAX_DEPTH) FAIL("nesting deeper than %d", TLM_JSON_MAX_DEPTH);
    tlm_jval *v = new_val(P, TLM_J_OBJ);
    if (!v) return NULL;
    P->p++;   /* '{' */
    size_t vbase = P->vn, kbase = P->kn;
    skip_ws(P);
    if (P->p < P->end && *P->p == '}') { P->p++; return v; }
    for (;;) {
        skip_ws(P);
        if (P->p >= P->end) FAIL("unterminated object");
        if (*P->p != '"') FAIL("expected string key in object");
        const unsigned char *kat = P->p;
        size_t klen = 0;
        char *k = parse_string(P, &klen);
        if (!k) return NULL;
        if (strlen(k) != klen) { jp_fail_at(P, kat, "object key contains U+0000"); return NULL; }
        skip_ws(P);
        if (P->p >= P->end || *P->p != ':') FAIL("expected ':' after object key");
        P->p++;
        tlm_jval *it = parse_value(P, depth + 1);
        if (!it) return NULL;
        if (kpush(P, k) != 0 || vpush(P, it) != 0) FAIL("out of memory");
        skip_ws(P);
        if (P->p >= P->end) FAIL("unterminated object");
        if (*P->p == ',') { P->p++; continue; }
        if (*P->p == '}') { P->p++; break; }
        FAIL("expected ',' or '}' in object");
    }
    size_t n = P->vn - vbase;
    v->u.obj.keys = (char **)arena_alloc(P->j, n * sizeof(char *));
    v->u.obj.vals = (tlm_jval **)arena_alloc(P->j, n * sizeof(tlm_jval *));
    if (!v->u.obj.keys || !v->u.obj.vals) FAIL("out of memory");
    memcpy(v->u.obj.keys, P->ks + kbase, n * sizeof(char *));
    memcpy(v->u.obj.vals, P->vs + vbase, n * sizeof(tlm_jval *));
    v->u.obj.n = n;
    P->vn = vbase;
    P->kn = kbase;
    return v;
}

static tlm_jval *parse_value(jp *P, int depth) {
    skip_ws(P);
    if (P->p >= P->end) FAIL("unexpected end of input");
    unsigned char c = *P->p;
    switch (c) {
    case '{': return parse_object(P, depth);
    case '[': return parse_array(P, depth);
    case '"': {
        tlm_jval *v = new_val(P, TLM_J_STR);
        if (!v) return NULL;
        size_t n = 0;
        char *s = parse_string(P, &n);
        if (!s) return NULL;
        v->u.str.s = s;
        v->u.str.len = n;
        return v;
    }
    case 't': case 'f': case 'n': {
        tlm_jval *v;
        if (match_lit(P, "true")) { v = new_val(P, TLM_J_BOOL); if (v) v->u.b = 1; return v; }
        if (match_lit(P, "false")) { v = new_val(P, TLM_J_BOOL); if (v) v->u.b = 0; return v; }
        if (match_lit(P, "null")) return new_val(P, TLM_J_NULL);
        FAIL("invalid literal");
    }
    default:
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(P);
        if (c >= 0x20 && c < 0x7f) FAIL("unexpected character '%c'", c);
        FAIL("unexpected byte 0x%02x", c);
    }
}

tlm_json *tlm_json_parse(const char *text, size_t len, char *err, size_t errcap) {
    if (err && errcap) err[0] = 0;
    if (!text) {
        if (err && errcap) snprintf(err, errcap, "json: no input");
        return NULL;
    }
    tlm_json *j = (tlm_json *)calloc(1, sizeof *j);
    if (!j) {
        if (err && errcap) snprintf(err, errcap, "json: out of memory");
        return NULL;
    }
    jp P;
    memset(&P, 0, sizeof P);
    P.start = P.p = (const unsigned char *)text;
    P.end = P.start + len;
    P.j = j;
    P.err = err;
    P.errcap = errcap;
    if (len >= 3 && P.p[0] == 0xEF && P.p[1] == 0xBB && P.p[2] == 0xBF) P.p += 3;   /* BOM */
    tlm_jval *root = parse_value(&P, 0);
    if (root) {
        skip_ws(&P);
        if (P.p != P.end) { jp_fail_at(&P, P.p, "trailing characters after JSON value"); root = NULL; }
    }
    free(P.vs);
    free(P.ks);
    if (!root) {
        if (err && errcap && !err[0]) snprintf(err, errcap, "json: parse error");
        tlm_json_free(j);
        return NULL;
    }
    j->root = root;
    return j;
}

tlm_json *tlm_json_parse_file(const char *path, char *err, size_t errcap) {
    char *buf = NULL;
    size_t n = 0;
    if (tlm_read_file(path, &buf, &n) != 0) {
        if (err && errcap) snprintf(err, errcap, "json: %s", tlm_errmsg());
        return NULL;
    }
    tlm_json *j = tlm_json_parse(buf, n, err, errcap);
    if (!j && err && errcap) {
        char tmp[512];
        snprintf(tmp, sizeof tmp, "%s: %s", path, err);
        snprintf(err, errcap, "%s", tmp);
    }
    free(buf);
    return j;
}

/* ---- getters ----------------------------------------------------------- */
const tlm_jval *tlm_jobj_get(const tlm_jval *obj, const char *key) {
    if (!obj || obj->type != TLM_J_OBJ || !key) return NULL;
    for (size_t i = obj->u.obj.n; i-- > 0;)          /* last duplicate wins */
        if (strcmp(obj->u.obj.keys[i], key) == 0) return obj->u.obj.vals[i];
    return NULL;
}

const char *tlm_jstr(const tlm_jval *v, const char *dflt) {
    return (v && v->type == TLM_J_STR) ? v->u.str.s : dflt;
}

double tlm_jnum(const tlm_jval *v, double dflt) {
    return (v && v->type == TLM_J_NUM) ? v->u.num.d : dflt;
}

int64_t tlm_jint(const tlm_jval *v, int64_t dflt) {
    return (v && v->type == TLM_J_NUM && v->u.num.is_int) ? v->u.num.i : dflt;
}

int tlm_jbool(const tlm_jval *v, int dflt) {
    return (v && v->type == TLM_J_BOOL) ? v->u.b : dflt;
}

size_t tlm_jlen(const tlm_jval *v) {
    if (!v) return 0;
    if (v->type == TLM_J_ARR) return v->u.arr.n;
    if (v->type == TLM_J_OBJ) return v->u.obj.n;
    return 0;
}

const tlm_jval *tlm_jat(const tlm_jval *v, size_t i) {
    if (!v) return NULL;
    if (v->type == TLM_J_ARR) return i < v->u.arr.n ? v->u.arr.items[i] : NULL;
    if (v->type == TLM_J_OBJ) return i < v->u.obj.n ? v->u.obj.vals[i] : NULL;
    return NULL;
}

/* The object key at index i (objects only; NULL for arrays/out of range). */
const char *tlm_jkey(const tlm_jval *obj, size_t i) {
    if (!obj || obj->type != TLM_J_OBJ || i >= obj->u.obj.n) return NULL;
    return obj->u.obj.keys[i];
}

const char *tlm_jtype_name(tlm_jtype t) {
    switch (t) {
    case TLM_J_NULL: return "null";
    case TLM_J_BOOL: return "bool";
    case TLM_J_NUM: return "number";
    case TLM_J_STR: return "string";
    case TLM_J_ARR: return "array";
    case TLM_J_OBJ: return "object";
    }
    return "?";
}

/* ---- writer ------------------------------------------------------------ */
size_t tlm_json_escape(char *dst, size_t cap, const char *s, size_t len) {
    static const char hex[] = "0123456789abcdef";
    size_t n = 0;
#define EMIT(ch) do { if (n + 1 < cap) dst[n] = (ch); n++; } while (0)
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"': EMIT('\\'); EMIT('"'); break;
        case '\\': EMIT('\\'); EMIT('\\'); break;
        case '\b': EMIT('\\'); EMIT('b'); break;
        case '\f': EMIT('\\'); EMIT('f'); break;
        case '\n': EMIT('\\'); EMIT('n'); break;
        case '\r': EMIT('\\'); EMIT('r'); break;
        case '\t': EMIT('\\'); EMIT('t'); break;
        default:
            if (c < 0x20 || c == 0x7f) {
                EMIT('\\'); EMIT('u'); EMIT('0'); EMIT('0');
                EMIT(hex[c >> 4]); EMIT(hex[c & 15]);
            } else {
                EMIT((char)c);
            }
        }
    }
#undef EMIT
    if (cap) dst[n < cap ? n : cap - 1] = 0;
    return n;
}

void tlm_json_write_str(FILE *f, const char *s, size_t len) {
    char small[256];
    if (!s) { fputs("null", f); return; }
    size_t need = tlm_json_escape(small, sizeof small, s, len);
    fputc('"', f);
    if (need < sizeof small) {
        fwrite(small, 1, need, f);
    } else {
        char *big = (char *)malloc(need + 1);
        if (big) {
            tlm_json_escape(big, need + 1, s, len);
            fwrite(big, 1, need, f);
            free(big);
        }
    }
    fputc('"', f);
}

void tlm_jw_init(tlm_jw *w, FILE *f) {
    memset(w, 0, sizeof *w);
    w->f = f;
}

static int jw_level(const tlm_jw *w) {
    int d = w->depth;
    if (d < 0) d = 0;
    if (d > 63) d = 63;
    return d;
}

static void jw_prefix(tlm_jw *w, const char *key) {
    int d = jw_level(w);
    if (w->need_comma[d]) fputc(',', w->f);
    w->need_comma[d] = 1;
    if (key) {
        tlm_json_write_str(w->f, key, strlen(key));
        fputc(':', w->f);
    }
}

static void jw_open(tlm_jw *w, const char *key, char ch) {
    jw_prefix(w, key);
    fputc(ch, w->f);
    w->depth++;
    w->need_comma[jw_level(w)] = 0;
}

static void jw_close(tlm_jw *w, char ch) {
    fputc(ch, w->f);
    if (w->depth > 0) w->depth--;
}

void tlm_jw_begin_obj(tlm_jw *w) { jw_open(w, NULL, '{'); }
void tlm_jw_end_obj(tlm_jw *w) { jw_close(w, '}'); }
void tlm_jw_begin_arr(tlm_jw *w, const char *key) { jw_open(w, key, '['); }
void tlm_jw_end_arr(tlm_jw *w) { jw_close(w, ']'); }
void tlm_jw_key_obj(tlm_jw *w, const char *key) { jw_open(w, key, '{'); }

void tlm_jw_str(tlm_jw *w, const char *key, const char *v) {
    jw_prefix(w, key);
    if (v) tlm_json_write_str(w->f, v, strlen(v));
    else fputs("null", w->f);
}

void tlm_jw_num(tlm_jw *w, const char *key, double v) {
    jw_prefix(w, key);
    if (!isfinite(v)) { fputs("null", w->f); return; }
    char buf[40];
    for (int prec = 15; prec <= 17; prec++) {      /* shortest of 15..17 digits that round-trips */
        snprintf(buf, sizeof buf, "%.*g", prec, v);
        if (strtod(buf, NULL) == v) break;
    }
    fputs(buf, w->f);
}

void tlm_jw_int(tlm_jw *w, const char *key, int64_t v) {
    jw_prefix(w, key);
    fprintf(w->f, "%" PRId64, v);
}

void tlm_jw_bool(tlm_jw *w, const char *key, int v) {
    jw_prefix(w, key);
    fputs(v ? "true" : "false", w->f);
}

void tlm_jw_null(tlm_jw *w, const char *key) {
    jw_prefix(w, key);
    fputs("null", w->f);
}
