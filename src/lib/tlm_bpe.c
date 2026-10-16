/* tlm_bpe.c - byte-level BPE tokenizer (merges-list) for the Qwen2/3 family.
 *
 * Pattern: F-TQ-01..F-TQ-04 (fragments/tokenizer_qwen3.md). The pre-tokeniser
 * is the reference regex rule set R1-R7 implemented imperatively over
 * codepoints (P-98: no regex library; the Unicode classes come from the
 * generated tables in tlm_uni.c). NFC comes from tlm_nfc.c.
 *
 * BPE per piece (F-TQ-03): start with one symbol per byte char; repeatedly
 * find the lowest-rank adjacent pair in the merges list (leftmost wins ties)
 * and merge it; stop when no adjacent pair has a rank. There is deliberately
 * no shortcut for a whole piece that is already a vocab token: the reference
 * never short-circuits, and the merge-list ranks can stop the loop early.
 * Merging one leftmost lowest-rank occurrence per iteration is equivalent to
 * merging all occurrences of the best pair at once (every pair touching a
 * freshly formed symbol has a strictly higher rank), so the loop below keeps
 * a min-heap of candidate pairs for O(n log n) on long letter runs (CJK).
 *
 * Decode (P-103): every token piece is turned into raw bytes when all of its
 * codepoints are GPT-2 byte chars, into raw UTF-8 otherwise; the pieces are
 * concatenated and the whole buffer is UTF-8 decoded lossily (U+FFFD per
 * invalid byte), once, at the end.
 */
#include "tlm_bpe.h"
#include "tlm_json.h"
#include "tlm_nfc.h"
#include "tlm_uni.h"
#include "tlm_platform.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define BPE_KEY_NUL 0x00u   /* merges key separator; byte chars never encode NUL */

/* ---- GPT-2 byte <-> codepoint map (F-TOK-01; public spec) ----------------
 * Printable bytes are 33..126, 161..172, 174..255; they map to themselves.
 * Every other byte b maps to 256 + order of b among the non-printables, in
 * ascending byte order. The inverse table is -1 where undefined.            */
typedef struct gpt2map {
    uint32_t b2cp[256];
    int16_t  cp2b[512];     /* 256 non-printables: 256 + 187 = 443 < 512 */
} gpt2map;

static const gpt2map *gpt2(void) {
    static gpt2map m;
    static int ready = 0;
    if (ready) return &m;
    int next = 256;
    for (int b = 0; b < 256; b++) {
        int p = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t cp = p ? (uint32_t)b : (uint32_t)next++;
        m.b2cp[b] = cp;
        m.cp2b[cp] = (int16_t)b;
    }
    for (int c = 0; c < 512; c++) m.cp2b[c] = -1;
    for (int b = 0; b < 256; b++) m.cp2b[m.b2cp[b]] = (int16_t)b;
    ready = 1;
    return &m;
}

/* ---- UTF-8 -------------------------------------------------------------- */
static int utf8_encode(uint32_t cp, char *o) {   /* returns bytes written */
    if (cp < 0x80u) { o[0] = (char)cp; return 1; }
    if (cp < 0x800u) {
        o[0] = (char)(0xC0u | (cp >> 6));
        o[1] = (char)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp < 0x10000u) {
        o[0] = (char)(0xE0u | (cp >> 12));
        o[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        o[2] = (char)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    o[0] = (char)(0xF0u | (cp >> 18));
    o[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
    o[2] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
    o[3] = (char)(0x80u | (cp & 0x3Fu));
    return 4;
}

/* Decode a bounded UTF-8 span. Invalid bytes (overlongs, surrogates, >
 * U+10FFFF, truncated tails, stray continuations) become U+FFFD, one per
 * invalid byte. Returns the number of codepoints written. */
static int utf8_decode(const char *s, size_t len, uint32_t *cps, int cap) {
    int n = 0;
    size_t i = 0;
    while (i < len) {
        uint8_t b0 = (uint8_t)s[i];
        uint32_t cp;
        int k;
        if (b0 < 0x80u) { cp = b0; k = 1; }
        else if ((b0 & 0xE0u) == 0xC0u) { cp = b0 & 0x1Fu; k = 2; }
        else if ((b0 & 0xF0u) == 0xE0u) { cp = b0 & 0x0Fu; k = 3; }
        else if ((b0 & 0xF8u) == 0xF0u) { cp = b0 & 0x07u; k = 4; }
        else { if (n < cap) cps[n++] = 0xFFFDu; i++; continue; }
        if (i + (size_t)k > len) {
            for (int j = 0; j < k; j++) if (i + j < len && n < cap) cps[n++] = 0xFFFDu;
            i = len;
            continue;
        }
        int ok = 1;
        for (int j = 1; j < k; j++) {
            uint8_t b = (uint8_t)s[i + j];
            if ((b & 0xC0u) != 0x80u) { ok = 0; break; }
            cp = (cp << 6) | (b & 0x3Fu);
        }
        static const uint32_t lo[5] = {0, 0, 0x80u, 0x800u, 0x10000u};
        if (ok && (cp < lo[k] || (k < 4 && cp >= 0xD800u && cp <= 0xDFFFu) || cp > 0x10FFFFu)) ok = 0;
        if (!ok) {
            for (int j = 0; j < k; j++) if (n < cap) cps[n++] = 0xFFFDu;
        } else if (n < cap) {
            cps[n++] = cp;
        }
        i += (size_t)k;
    }
    return n;
}

/* Lossy emit (Rust from_utf8_lossy semantics): valid complete sequences are
 * copied; each maximal invalid subsequence becomes one U+FFFD; an
 * incomplete tail at the end is held back into tailbuf (copied) and not
 * emitted. Returns bytes written to out, or -1 when they would not fit in
 * cap (out left untouched). */
static int utf8_emit(const uint8_t *buf, size_t n, char *out, size_t cap,
                     uint8_t *tailbuf, size_t *tail) {
    static const uint32_t lo[5] = {0, 0, 0x80u, 0x800u, 0x10000u};
    size_t i = 0, o = 0;
    while (i < n) {
        uint8_t b0 = buf[i];
        int k = b0 < 0x80u ? 1 : (b0 & 0xE0u) == 0xC0u ? 2
                : (b0 & 0xF0u) == 0xE0u ? 3 : (b0 & 0xF8u) == 0xF0u ? 4 : 0;
        if (k == 0) {                       /* stray continuation byte */
            if (o + 3 > cap) return -1;
            out[o++] = (char)0xEF; out[o++] = (char)0xBF; out[o++] = (char)0xBD;
            i++;
            continue;
        }
        if (i + (size_t)k > n) {            /* incomplete tail: hold back */
            if (o > cap) return -1;
            memcpy(tailbuf, buf + i, n - i);
            *tail = n - i;
            return (int)o;
        }
        uint32_t cp = k == 1 ? b0 : b0 & (0xFFu >> (k + 1));
        int ok = 1;
        for (int q = 1; q < k; q++) {
            uint8_t bq = buf[i + q];
            if ((bq & 0xC0u) != 0x80u) { ok = 0; break; }
            cp = (cp << 6) | (bq & 0x3Fu);
        }
        if (ok && (cp < lo[k] || cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu))) ok = 0;
        if (ok) {
            if (o + (size_t)k > cap) return -1;
            memcpy(out + o, buf + i, (size_t)k);
            o += (size_t)k;
        } else {
            if (o + 3 > cap) return -1;
            out[o++] = (char)0xEF; out[o++] = (char)0xBF; out[o++] = (char)0xBD;
        }
        i += (size_t)k;
    }
    *tail = 0;
    return (int)o;
}

/* ---- open-addressed string hash table (vocab: text->id, merges: pair->rank)
 * Linear probing, FNV-1a, capacity a power of two, load <= 1/2. Keys are
 * kept in one growing buffer; slots hold (offset, length, value). An empty
 * slot has val 0; stored values are id+1 / rank+1.                       */
typedef struct strtab {
    size_t   cap, n, kbuf, kbufcap;
    char    *keys;
    uint32_t *off;
    uint32_t *klen;
    uint32_t *val;
} strtab;

static uint64_t fnv1a(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ull; }
    return h;
}

static void strtab_init(strtab *t, size_t cap) {
    t->cap = 1;
    while (t->cap < cap) t->cap <<= 1;
    t->n = t->kbuf = t->kbufcap = 0;
    t->keys = NULL;
    t->off  = (uint32_t *)calloc(t->cap, sizeof(uint32_t));
    t->klen = (uint32_t *)calloc(t->cap, sizeof(uint32_t));
    t->val  = (uint32_t *)calloc(t->cap, sizeof(uint32_t));
}
static int strtab_get(const strtab *t, const char *key, size_t klen, uint32_t *val) {
    uint64_t h = fnv1a(key, klen), m = t->cap - 1;
    for (size_t s = h & m; ; s = (s + 1) & m) {
        if (!t->val[s]) return 0;
        if (t->klen[s] == klen && memcmp(t->keys + t->off[s], key, klen) == 0) {
            *val = t->val[s];
            return 1;
        }
    }
}

/* Replace-free insert (keys are unique in both tables we use); 0 on OOM. */
static int strtab_put(strtab *t, const char *key, size_t klen, uint32_t val) {
    if (t->kbuf + klen + 1 > t->kbufcap) {
        size_t nc = t->kbufcap ? t->kbufcap * 2 : 4096;
        while (nc < t->kbuf + klen + 1) nc *= 2;
        char *nk = (char *)realloc(t->keys, nc);
        if (!nk) return 0;
        t->keys = nk;
        t->kbufcap = nc;
    }
    uint64_t h = fnv1a(key, klen), m = t->cap - 1;
    size_t s;
    for (s = h & m; t->val[s]; s = (s + 1) & m)
        if (t->klen[s] == klen && memcmp(t->keys + t->off[s], key, klen) == 0)
            return 1;   /* already present; leave it */
    if (t->n * 2 >= t->cap) return 0;   /* unreachable at load 1/2 */
    memcpy(t->keys + t->kbuf, key, klen);
    t->keys[t->kbuf + klen] = 0;
    t->off[s] = (uint32_t)t->kbuf;
    t->kbuf += klen + 1;
    t->klen[s] = (uint32_t)klen;
    t->val[s] = val;
    t->n++;
    return 1;
}

static void strtab_free(strtab *t) {
    free(t->keys);
    free(t->off);
    free(t->klen);
    free(t->val);
}

/* ---- handle --------------------------------------------------------------- */
struct tlm_bpe {
    strtab   vocab;              /* byte-char text -> id+1 */
    strtab   merges;             /* left + NUL + right -> rank+1 */
    uint32_t *id2tok;            /* id -> offset into tokb (NUL-terminated UTF-8) */
    uint32_t *id2len;
    char     *tokb;
    int       ntok;
    int       max_id;
    int      *add_id;            /* added token ids, list order */
    int      *add_spec;          /* special flag */
    uint32_t *add_off;           /* content bytes into addb */
    uint32_t *add_len;
    char     *addb;
    int       nadd;
};

int tlm_bpe_vocab_size(const tlm_bpe *b) { return (int)b->vocab.n; }
int tlm_bpe_max_id(const tlm_bpe *b) { return b->max_id; }

static int err_copy(char *err, size_t errcap, const char *msg) {
    if (err && errcap) snprintf(err, errcap, "%s", msg);
    return 0;
}

tlm_bpe *tlm_bpe_load(const char *dir, char *err, size_t errcap) {
    char path[4096];
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    tlm_json *j = tlm_json_parse_file(path, NULL, 0);
    if (!j) {
        if (err && errcap)
            snprintf(err, errcap, "bpe: %s: %s", path, tlm_errmsg());
        return NULL;
    }
    const tlm_jval *root = tlm_json_root(j);
    const tlm_jval *model = tlm_jobj_get(root, "model");
    const tlm_jval *dec   = tlm_jobj_get(root, "decoder");
    const tlm_jval *norm  = tlm_jobj_get(root, "normalizer");
    const tlm_jval *vocab = model && model->type == TLM_J_OBJ ? tlm_jobj_get(model, "vocab") : NULL;
    const tlm_jval *mrg   = model && model->type == TLM_J_OBJ ? tlm_jobj_get(model, "merges") : NULL;
    const tlm_jval *added0 = tlm_jobj_get(root, "added_tokens");
    const tlm_jval *added = (added0 && added0->type == TLM_J_ARR) ? added0 : NULL;
    if (!model || model->type != TLM_J_OBJ || strcmp(tlm_jstr(tlm_jobj_get(model, "type"), ""), "BPE") != 0 ||
        !vocab || vocab->type != TLM_J_OBJ || tlm_jlen(vocab) == 0 ||
        !mrg || mrg->type != TLM_J_ARR || tlm_jlen(mrg) == 0 ||
        !dec || dec->type != TLM_J_OBJ ||
        strcmp(tlm_jstr(tlm_jobj_get(dec, "type"), ""), "ByteLevel") != 0 ||
        tlm_jbool(tlm_jobj_get(dec, "add_prefix_space"), 1) != 0 ||
        (norm && norm->type != TLM_J_OBJ && norm->type != TLM_J_NULL) ||
        (norm && norm->type == TLM_J_OBJ && strcmp(tlm_jstr(tlm_jobj_get(norm, "type"), ""), "NFC") != 0)) {
        char msg[256];
        snprintf(msg, sizeof msg,
                 "bpe: unsupported tokenizer.json (need model.type=BPE, vocab, non-empty merges, "
                 "decoder.type=ByteLevel with add_prefix_space=false, normaliser null or NFC)");
        err_copy(err, errcap, msg);
        tlm_json_free(j);
        return NULL;
    }

    size_t nv = tlm_jlen(vocab), nm = tlm_jlen(mrg), na = added ? tlm_jlen(added) : 0;
    tlm_bpe *b = (tlm_bpe *)calloc(1, sizeof *b);
    if (!b) { err_copy(err, errcap, "bpe: out of memory"); tlm_json_free(j); return NULL; }
    b->max_id = -1;
    strtab_init(&b->vocab, nv * 2 + 8);
    strtab_init(&b->merges, nm * 2 + 8);

    /* two vocab passes: size the id table, then fill it */
    int hi = -1;
    for (size_t i = 0; i < nv; i++) {
        int64_t id = tlm_jint(tlm_jat(vocab, i), -1);
        if (id < 0 || id > 0x7FFFFFFF) { err_copy(err, errcap, "bpe: vocab id out of range"); goto fail; }
        if (id > hi) hi = (int)id;
    }
    if (added) {
        for (size_t i = 0; i < na; i++) {
            const tlm_jval *a = tlm_jat(added, i);
            if (a->type != TLM_J_OBJ) {
                err_copy(err, errcap, "bpe: added_tokens entries must be objects");
                goto fail;
            }
            {
                const tlm_jval *ac = tlm_jobj_get(a, "content");
                const tlm_jval *ai = tlm_jobj_get(a, "id");
                if (!ac || ac->type != TLM_J_STR || !ai || ai->type != TLM_J_NUM) {
                    err_copy(err, errcap, "bpe: added_tokens entry needs string content and integer id");
                    goto fail;
                }
            }
            int64_t id = tlm_jint(tlm_jobj_get(a, "id"), -1);
            if (id < 0 || id > 0x7FFFFFFF || (int)id <= hi) {
                err_copy(err, errcap, "bpe: added token id must be a non-negative integer");
                goto fail;
            }
            for (size_t k = 0; k < i; k++)
                if (tlm_jint(tlm_jobj_get(tlm_jat(added, k), "id"), -2) == id) {
                    err_copy(err, errcap, "bpe: duplicate added token id");
                    goto fail;
                }
            if (id > hi) hi = (int)id;
        }
    }
    b->max_id = hi;
    b->ntok = (int)nv;
    b->id2tok = (uint32_t *)calloc((size_t)hi + 1, sizeof(uint32_t));
    b->id2len = (uint32_t *)calloc((size_t)hi + 1, sizeof(uint32_t));
    if (!b->id2tok || !b->id2len) { err_copy(err, errcap, "bpe: out of memory"); goto fail; }

    /* vocab is a JSON object: key = token text, value = id. tlm_jat gives
     * the value (id); tlm_jkey gives the token text. Concatenate every
     * entry's text once, indexed by id, then fill the string table.
     * Pass 1: lengths and total. Pass 2: offsets, text, table. */
    {
        size_t total = 64;
        for (size_t i = 0; i < nv; i++) {
            const char *s = tlm_jkey(vocab, i);
            if (!s) { err_copy(err, errcap, "bpe: vocab entry missing key"); goto fail; }
            int id = (int)tlm_jint(tlm_jat(vocab, i), -1);
            if (id < 0) { err_copy(err, errcap, "bpe: vocab id out of range"); goto fail; }
            b->id2len[(size_t)id] = (uint32_t)strlen(s);
            total += b->id2len[(size_t)id] + 1;
        }
        b->tokb = (char *)malloc(total);
        if (!b->tokb) { err_copy(err, errcap, "bpe: out of memory"); goto fail; }
        size_t off = 0;
        for (size_t i = 0; i < nv; i++) {
            const char *s = tlm_jkey(vocab, i);
            int id = (int)tlm_jint(tlm_jat(vocab, i), -1);
            size_t len = b->id2len[(size_t)id];
            b->id2tok[(size_t)id] = (uint32_t)off;
            memcpy(b->tokb + off, s, len + 1);
            off += len + 1;
            if (!strtab_put(&b->vocab, s, len, (uint32_t)id + 1)) {
                err_copy(err, errcap, "bpe: out of memory");
                goto fail;
            }
        }
    }

    /* merges: "left right" pairs in rank order; key = left + NUL + right.
     * The separator is safe: GPT-2 byte chars never encode a NUL byte in
     * UTF-8 (printable ASCII 33..126 has no 0, the rest are >= U+0101). */
    for (size_t i = 0; i < nm; i++) {
        const tlm_jval *m = tlm_jat(mrg, i);
        if (m->type != TLM_J_ARR || tlm_jlen(m) != 2 ||
            tlm_jat(m, 0)->type != TLM_J_STR || tlm_jat(m, 1)->type != TLM_J_STR) {
            err_copy(err, errcap, "bpe: merges entries must be [left, right] string pairs");
            goto fail;
        }
        const char *l = tlm_jstr(tlm_jat(m, 0), ""), *r = tlm_jstr(tlm_jat(m, 1), "");
        size_t ll = strlen(l), rl = strlen(r);
        if (ll == 0 || rl == 0 || ll + 1 + rl > 0xFFFFu) {
            err_copy(err, errcap, "bpe: empty or oversized merge side");
            goto fail;
        }
        char *key = (char *)malloc(ll + 1 + rl);
        if (!key) { err_copy(err, errcap, "bpe: out of memory"); goto fail; }
        memcpy(key, l, ll);
        key[ll] = (char)BPE_KEY_NUL;
        memcpy(key + ll + 1, r, rl);
        int oom = !strtab_put(&b->merges, key, ll + 1 + rl, (uint32_t)i + 1);
        free(key);
        if (oom) { err_copy(err, errcap, "bpe: out of memory"); goto fail; }
    }

    /* added tokens: content bytes, id, special flag (list order kept; the
     * duplicate-id check already ran in the sizing pass) */
    if (added) {
        b->add_id   = (int *)calloc(na, sizeof(int));
        b->add_spec = (int *)calloc(na, sizeof(int));
        b->add_off  = (uint32_t *)calloc(na, sizeof(uint32_t));
        b->add_len  = (uint32_t *)calloc(na, sizeof(uint32_t));
        size_t total = 64;
        for (size_t i = 0; i < na; i++) {
            const char *s = tlm_jstr(tlm_jobj_get(tlm_jat(added, i), "content"), "");
            size_t len = strlen(s);
            total += len + 1;
            b->add_len[i] = (uint32_t)len;
        }
        if (!b->add_id || !b->add_spec || !b->add_off || !b->add_len ||
            !(b->addb = (char *)malloc(total))) {
            err_copy(err, errcap, "bpe: out of memory");
            goto fail;
        }
        size_t off = 0;
        for (size_t i = 0; i < na; i++) {
            const tlm_jval *a = tlm_jat(added, i);
            const char *s = tlm_jstr(tlm_jobj_get(a, "content"), "");
            b->add_id[i] = (int)tlm_jint(tlm_jobj_get(a, "id"), 0);
            b->add_spec[i] = tlm_jbool(tlm_jobj_get(a, "special"), 0);
            b->add_off[i] = (uint32_t)off;
            memcpy(b->addb + off, s, b->add_len[i] + 1);
            off += b->add_len[i] + 1;
        }
        b->nadd = (int)na;
    }
    tlm_json_free(j);
    return b;

fail:
    strtab_free(&b->vocab);
    strtab_free(&b->merges);
    free(b->id2tok);
    free(b->id2len);
    free(b->tokb);
    free(b->add_id);
    free(b->add_spec);
    free(b->add_off);
    free(b->add_len);
    free(b->addb);
    free(b);
    tlm_json_free(j);
    return NULL;
}

void tlm_bpe_free(tlm_bpe *b) {
    if (!b) return;
    strtab_free(&b->vocab);
    strtab_free(&b->merges);
    free(b->id2tok);
    free(b->id2len);
    free(b->tokb);
    free(b->add_id);
    free(b->add_spec);
    free(b->add_off);
    free(b->add_len);
    free(b->addb);
    free(b);
}

int tlm_bpe_add_id(const tlm_bpe *b, const char *content, size_t len) {
    for (int i = 0; i < b->nadd; i++)
        if (b->add_len[i] == len && memcmp(b->addb + b->add_off[i], content, len) == 0)
            return b->add_id[i];
    return -1;
}

/* ---- pre-tokeniser (F-TQ-02) ---------------------------------------------
 * The reference rule set, as ordered alternatives per position over
 * codepoints (every position of this rule set matches some rule; the
 * scanner never stalls):
 *   R1  's | 't | 're | 've | 'm | 'll | 'd        (case-insensitive)
 *   R2  [^\r\n\pL\pN]?\pL+                          (prefix stays in the piece)
 *   R3  \pN                                          (single digit)
 *   R4   ?[^\s\pL\pN]+[\r\n]*                        (optional leading space)
 *   R5  \s*[\r\n]+
 *   R6  \s+(?!\S)                                    (backtracks: a run >= 2
 *        before a non-space loses its LAST codepoint, which re-enters the
 *        loop and usually glues onto the next word via R2)
 *   R7  \s+
 * Rule order is load-bearing: a space before a letter is a word piece (R2
 * before R4); spaces before a newline are swallowed into the newline piece
 * (R5 before R6/R7); a space before any other non-letter is part of the
 * punctuation piece (R4 before R6/R7). A bare apostrophe (no contraction)
 * is a R2 prefix before letters, else a one-char punct piece (R4). The
 * Unicode classes are the generated tables: \pL = tlm_uni_is_letter,
 * \pN = tlm_uni_is_number, \s = tlm_uni_is_space (White_Space minus U+2007
 * and U+200B). */
static const char *const CONTR[] = { "s", "t", "re", "ve", "m", "ll", "d" };

/* Match a contraction at an apostrophe: returns end index or -1. */
static int try_contraction(const uint32_t *cps, int n, int i) {
    for (size_t k = 0; k < sizeof CONTR / sizeof *CONTR; k++) {
        size_t l = strlen(CONTR[k]);
        if (i + (int)l > n) continue;
        int ok = 1;
        for (size_t j = 0; j < l; j++) {
            uint32_t c = cps[i + 1 + j];
            uint32_t lo = (c >= 'A' && c <= 'Z') ? c + 32u : c;
            if (lo != (uint32_t)CONTR[k][j]) { ok = 0; break; }
        }
        if (ok) return i + 1 + (int)l;
    }
    return -1;
}

/* Split cps[0..n) into pieces. Piece list: (start, len) codepoint spans.
 * Returns the number of pieces, or -1 on OOM. */
typedef struct { int start, len; } piece;

static int pretok(const uint32_t *cps, int n, piece **out) {
    piece *ps = (piece *)malloc(sizeof(piece) * (size_t)(n > 0 ? n : 1));
    if (!ps) return -1;
    int np = 0, i = 0;
    while (i < n) {
        uint32_t cp = cps[i];
        int a = i, b = -1;
        if (cp == 0x27u) {
            b = try_contraction(cps, n, i);
            if (b >= 0) { ps[np++] = (piece){a, b - a}; i = b; continue; }
            /* no contraction: fall through; R2 takes the apostrophe as a
             * word prefix before a letter, R4 as a one-char punct piece. */
        }
        if (tlm_uni_is_letter(cp)) {
            b = i;
            while (b < n && tlm_uni_is_letter(cps[b])) b++;
        } else if (cp != 0x0Du && cp != 0x0Au && !tlm_uni_is_number(cp) &&
                   i + 1 < n && tlm_uni_is_letter(cps[i + 1])) {
            b = i + 1;
            while (b < n && tlm_uni_is_letter(cps[b])) b++;
        } else if (tlm_uni_is_number(cp)) {
            b = i + 1;
        } else {
            int j = i;
            if (cp == 0x20u && i + 1 < n && !tlm_uni_is_space(cps[i + 1]) &&
                !tlm_uni_is_letter(cps[i + 1]) && !tlm_uni_is_number(cps[i + 1]))
                j = i + 1;
            int k = j;
            while (k < n && !tlm_uni_is_space(cps[k]) &&
                   !tlm_uni_is_letter(cps[k]) && !tlm_uni_is_number(cps[k]))
                k++;
            if (k > j) {
                while (k < n && (cps[k] == 0x0Du || cps[k] == 0x0Au)) k++;
                b = k;
            }
        }
        if (b < 0) {   /* R5/R6/R7 */
            int m = i;
            while (m < n && tlm_uni_is_space(cps[m])) m++;   /* space run [i,m) */
            int q = m - 1;
            while (q >= i && cps[q] != 0x0Du && cps[q] != 0x0Au) q--;
            if (q >= i) {
                while (q < m && (cps[q] == 0x0Du || cps[q] == 0x0Au)) q++;
                b = q;                                       /* R5: to last CRLF run */
            } else {
                if (m == n) b = m;                           /* R6: run ends string */
                else if (m - i >= 2) b = m - 1;              /* R6: drop last space */
                else b = m;                                   /* R7: single space */
            }
        }
        ps[np++] = (piece){a, b - a};
        i = b;
    }
    piece *np2 = (piece *)realloc(ps, sizeof(piece) * (size_t)(np > 0 ? np : 1));
    if (np2) ps = np2;
    *out = ps;
    return np;
}

/* ---- BPE per piece (F-TQ-03) ---------------------------------------------
 * Symbols are (offset, len) spans of the byte-char string, linked. A
 * min-heap of (rank, left id, ver_l, ver_r, left, right) candidate pairs
 * drives the loop; versions catch entries whose symbol content changed.
 * The order key (rank, left id) makes each iteration merge the leftmost
 * lowest-rank pair, matching the reference BPE: a tie between equal-rank
 * pairs must go to the leftmost, because the merge key "L\0R" is
 * order-sensitive (e.g. YYY = "YY"+"Y", not "Y"+"YY").                  */
typedef struct bsym {
    struct bsym *prev, *next;
    size_t  off, len;
    int     ver;
    int     alive;
    int     id;      /* initial index: order-preserving, smallest = leftmost */
} bsym;

typedef struct bheap {
    bsym **l, **r;
    uint32_t *rank, *vl, *vr;
    int n, cap;
} bheap;

static void bhp_swap(bheap *h, int i, int j) {
    bsym *tl = h->l[i]; h->l[i] = h->l[j]; h->l[j] = tl;
    bsym *tr = h->r[i]; h->r[i] = h->r[j]; h->r[j] = tr;
    uint32_t t;
    t = h->rank[i]; h->rank[i] = h->rank[j]; h->rank[j] = t;
    t = h->vl[i]; h->vl[i] = h->vl[j]; h->vl[j] = t;
    t = h->vr[i]; h->vr[i] = h->vr[j]; h->vr[j] = t;
}

/* Order key: (rank, left id). Rank ties break left-first (smallest left id),
 * matching the reference BPE, which merges the leftmost occurrence of the
 * best pair. The left id is stable under merges, so the minimum key always
 * names the leftmost lowest-rank pair regardless of heap shuffling. */
static int bhp_before(const bheap *h, int i, int j) {
    if (h->rank[i] != h->rank[j]) return h->rank[i] < h->rank[j];
    return h->l[i]->id < h->l[j]->id;
}

static int bhp_push(bheap *h, uint32_t rank, bsym *l, bsym *r) {
    if (h->n == h->cap) {
        int nc = h->cap ? h->cap * 2 : 64;
        bsym **nl = (bsym **)realloc(h->l, sizeof(bsym *) * (size_t)nc);
        if (!nl) return 0;
        h->l = nl;
        bsym **nr = (bsym **)realloc(h->r, sizeof(bsym *) * (size_t)nc);
        if (!nr) return 0;
        h->r = nr;
        uint32_t *nk = (uint32_t *)realloc(h->rank, sizeof(uint32_t) * (size_t)nc);
        if (!nk) return 0;
        h->rank = nk;
        uint32_t *nv = (uint32_t *)realloc(h->vl, sizeof(uint32_t) * (size_t)nc);
        if (!nv) return 0;
        h->vl = nv;
        uint32_t *nw = (uint32_t *)realloc(h->vr, sizeof(uint32_t) * (size_t)nc);
        if (!nw) return 0;
        h->vr = nw;
        h->cap = nc;
    }
    int i = h->n++;
    h->l[i] = l; h->r[i] = r; h->rank[i] = rank;
    h->vl[i] = l->ver; h->vr[i] = r->ver;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!bhp_before(h, i, p)) break;
        bhp_swap(h, i, p);
        i = p;
    }
    return 1;
}

/* Pop the lowest-(rank, left id) valid candidate; 0 when none (heap fully
 * stale/empty). Rank ties break to the smallest left id, i.e. the leftmost
 * pair, matching the reference BPE. See bhp_before.                      */
static int bhp_pop(bheap *h, bsym **lo, bsym **ro) {
    while (h->n > 0) {
        bsym *l = h->l[0], *r = h->r[0];
        if (l->alive && r->alive && l->next == r && l->ver == (int)h->vl[0] && r->ver == (int)h->vr[0]) {
            *lo = l; *ro = r;
            h->n--;
            if (h->n > 0) {
                h->l[0] = h->l[h->n]; h->r[0] = h->r[h->n];
                h->rank[0] = h->rank[h->n]; h->vl[0] = h->vl[h->n]; h->vr[0] = h->vr[h->n];
                int i = 0;
                for (;;) {
                    int s = i, c2 = i * 2 + 1, c3 = c2 + 1;
                    if (c2 < h->n && bhp_before(h, c2, s)) s = c2;
                    if (c3 < h->n && bhp_before(h, c3, s)) s = c3;
                    if (s == i) break;
                    bhp_swap(h, i, s);
                    i = s;
                }
            }
            return 1;
        }
        h->n--;
        if (h->n > 0) {
            h->l[0] = h->l[h->n]; h->r[0] = h->r[h->n];
            h->rank[0] = h->rank[h->n]; h->vl[0] = h->vl[h->n]; h->vr[0] = h->vr[h->n];
            int i = 0;
            for (;;) {
                int s = i, c2 = i * 2 + 1, c3 = c2 + 1;
                if (c2 < h->n && bhp_before(h, c2, s)) s = c2;
                if (c3 < h->n && bhp_before(h, c3, s)) s = c3;
                if (s == i) break;
                bhp_swap(h, i, s);
                i = s;
            }
        }
    }
    return 0;
}

static void bhp_free(bheap *h) {
    free(h->l); free(h->r); free(h->rank); free(h->vl); free(h->vr);
}

/* Pair rank of two adjacent symbols, with a reusable key scratch buffer.
 * Returns 1 with *rank set when the pair has a merge rank, 0 when it has no
 * rank (normal), -1 on OOM growing the scratch. */
static int pair_rank(const strtab *mt, const char *s, char **kbuf, size_t *kcap,
                     const bsym *l, const bsym *r, uint32_t *rank) {
    size_t kl = l->len + 1 + r->len;
    if (*kcap < kl) {
        size_t nc = *kcap ? *kcap * 2 : 64;
        while (nc < kl) nc *= 2;
        char *nk = (char *)realloc(*kbuf, nc);
        if (!nk) return -1;
        *kbuf = nk;
        *kcap = nc;
    }
    memcpy(*kbuf, s + l->off, l->len);
    (*kbuf)[l->len] = (char)BPE_KEY_NUL;
    memcpy(*kbuf + l->len + 1, s + r->off, r->len);
    return strtab_get(mt, *kbuf, kl, rank);
}

/* Push a candidate pair onto the heap. Returns 1 when pushed, 0 when the
 * pair has no merge rank (normal, not an error), -1 on OOM. */
static int push_pair(bheap *h, const strtab *mt, const char *s, char **kbuf,
                     size_t *kcap, bsym *l, bsym *r) {
    uint32_t rank;
    int pr = pair_rank(mt, s, kbuf, kcap, l, r, &rank);
    if (pr != 1) return pr;            /* 0 = no rank (ok), -1 = OOM */
    return bhp_push(h, rank, l, r) ? 1 : -1;
}

/* Merge one byte-char string into vocab ids. s is the byte-char UTF-8
 * encoding of the piece's bytes; n is its length in bytes. Writes at most
 * cap ids; returns the count, or -1 on OOM / a non-vocab final piece. */
static int bpe_piece(const tlm_bpe *b, const char *s, size_t n, int *ids, int cap) {
    bsym *syms = (bsym *)calloc(n + 1, sizeof(bsym));
    if (!syms) return -1;
    bheap h;
    h.l = h.r = NULL; h.rank = h.vl = h.vr = NULL; h.n = h.cap = 0;
    char *kbuf = NULL;
    size_t kcap = 0;
    int ok = 1, k = 0;

    /* one symbol per byte char (1-2 UTF-8 bytes each) */
    size_t i = 0;
    while (i < n && ok) {
        size_t cl = (uint8_t)s[i] < 0x80u ? 1 : 2;
        if (i + cl > n) { ok = 0; break; }
        syms[k].off = i;
        syms[k].len = cl;
        syms[k].ver = 0;
        syms[k].alive = 1;
        syms[k].id = k;
        syms[k].prev = k ? &syms[k - 1] : NULL;
        syms[k].next = NULL;
        i += cl;
        k++;
    }
    if (ok)
        for (int t = 0; t + 1 < k; t++) syms[t].next = &syms[t + 1];
    if (ok)
        for (int t = 0; t + 1 < k; t++) {
            int pr = push_pair(&h, &b->merges, s, &kbuf, &kcap, &syms[t], &syms[t + 1]);
            if (pr < 0) { ok = 0; break; }
        }
    while (ok) {
        bsym *l, *r;
        if (!bhp_pop(&h, &l, &r)) break;
        l->len += r->len;
        l->next = r->next;
        if (l->next) l->next->prev = l;
        r->alive = 0;
        l->ver++;
        if (l->next) {
            int pr = push_pair(&h, &b->merges, s, &kbuf, &kcap, l, l->next);
            if (pr < 0) { ok = 0; break; }
        }
        if (l->prev) {
            int pr = push_pair(&h, &b->merges, s, &kbuf, &kcap, l->prev, l);
            if (pr < 0) { ok = 0; break; }
        }
    }
    int cnt = 0;
    for (bsym *sy = syms; sy; sy = sy->next) {
        uint32_t v;
        if (cnt >= cap || !strtab_get(&b->vocab, s + sy->off, sy->len, &v)) { ok = 0; break; }
        ids[cnt++] = (int)v - 1;
    }
    free(syms);
    free(kbuf);
    bhp_free(&h);
    return ok ? cnt : -1;
}

/* ---- encode (F-TQ-02/03/04) ----------------------------------------------
 * The text is NFC-normalised and re-encoded to bytes. Added tokens are
 * matched on those bytes: leftmost position, longest content, ties to the
 * earlier added_tokens entry (both sides are valid UTF-8, so a byte prefix
 * match equals a codepoint prefix match). The unmatched spans go through
 * the pre-tokeniser and BPE. */
static int xrealloc_u8(uint8_t **p, size_t *cap, size_t need) {
    if (*cap >= need) return 1;
    size_t nc = *cap ? *cap : 256;
    while (nc < need) nc *= 2;
    uint8_t *np = (uint8_t *)realloc(*p, nc);
    if (!np) return 0;
    *p = np;
    *cap = nc;
    return 1;
}

/* Decode one codepoint from well-formed UTF-8 at s[j]. Returns the number
 * of bytes consumed (1-4), or 0 when it would run past sl. */
static int one_cp(const char *s, size_t j, size_t sl, uint32_t *cp) {
    uint8_t u0 = (uint8_t)s[j];
    int k = u0 < 0x80u ? 1 : (u0 & 0xE0u) == 0xC0u ? 2
            : (u0 & 0xF0u) == 0xE0u ? 3 : (u0 & 0xF8u) == 0xF0u ? 4 : 1;
    if (j + (size_t)k > sl) return 0;
    if (k == 1) *cp = u0;
    else {
        *cp = (uint32_t)(u0 & (0x1Fu >> (k - 2)));
        for (int q = 1; q < k; q++) *cp = (*cp << 6) | ((uint8_t)s[j + q] & 0x3Fu);
    }
    return k;
}

/* One token's content as raw bytes, appended to buf (grown as needed):
 * per cp its raw byte when every cp is a GPT-2 byte char, else the raw
 * UTF-8 of the content. 0 on OOM. */
static int append_tok_bytes(const gpt2map *gm, const char *s, size_t sl,
                            uint8_t **buf, size_t *n, size_t *cap) {
    size_t j = 0;
    int allbyte = 1;
    while (j < sl) {
        uint32_t cp;
        int k = one_cp(s, j, sl, &cp);
        if (k <= 0 || cp >= 512u || gm->cp2b[cp] < 0) { allbyte = 0; break; }
        j += (size_t)k;
    }
    if (xrealloc_u8(buf, cap, *n + sl)) {
        if (allbyte) {
            j = 0;
            while (j < sl) {
                uint32_t cp = 0;
                int k = one_cp(s, j, sl, &cp);
                if (k <= 0) break;   /* unreachable: allbyte verified above */
                (*buf)[(*n)++] = (uint8_t)gm->cp2b[cp];
                j += (size_t)k;
            }
        } else {
            memcpy(*buf + *n, s, sl);
            *n += sl;
        }
        return 1;
    }
    return 0;
}

/* Content and kind of one id: 0 when the id is unknown. */
static int tok_content(const tlm_bpe *b, int id, int *special, const char **s, size_t *len) {
    if (id < 0 || id > b->max_id) return 0;
    if (id < b->ntok) {
        *s = b->tokb + b->id2tok[(size_t)id];
        *len = b->id2len[(size_t)id];
        *special = 0;
        return 1;
    }
    for (int a = 0; a < b->nadd; a++)
        if (b->add_id[a] == id) {
            *s = b->addb + b->add_off[a];
            *len = b->add_len[a];
            *special = b->add_spec[a];
            return 1;
        }
    return 0;
}

/* Pre-tokenise and BPE one UTF-8 span; appends to ids (global position
 * *total, limit cap). Returns 0 ok, -1 on OOM / capacity / non-vocab. */
static int encode_span(const tlm_bpe *b, const char *sp, size_t sl,
                       char **pb, size_t *pbcap, char **ps, size_t *pscap,
                       int **tmpids, size_t *tmpcap, int *ids, int cap, int *total) {
    uint32_t *c2 = (uint32_t *)malloc(sizeof(uint32_t) * (sl ? sl : 1));
    if (!c2) return -1;
    int n2 = utf8_decode(sp, sl, c2, (int)sl);
    piece *pcs = NULL;
    int np = pretok(c2, n2, &pcs);
    if (np < 0) { free(c2); return -1; }
    int r = 0;
    const gpt2map *gm = gpt2();
    for (int p = 0; p < np && r == 0; p++) {
        size_t need = (size_t)pcs[p].len * 4 + 1;
        if (*pbcap < need) {
            size_t nc = *pbcap ? *pbcap : 64;
            while (nc < need) nc *= 2;
            char *npb = (char *)realloc(*pb, nc);
            if (!npb) { r = -1; break; }
            *pb = npb;
            *pbcap = nc;
        }
        size_t bl = 0;
        for (int c = pcs[p].start; c < pcs[p].start + pcs[p].len; c++)
            bl += (size_t)utf8_encode(c2[c], *pb + bl);
        if (*pscap < bl * 2 + 1) {
            size_t nc = *pscap ? *pscap : 64;
            while (nc < bl * 2 + 1) nc *= 2;
            char *nps = (char *)realloc(*ps, nc);
            if (!nps) { r = -1; break; }
            *ps = nps;
            *pscap = nc;
        }
        size_t cl = 0;
        for (size_t j = 0; j < bl; j++)
            cl += (size_t)utf8_encode(gm->b2cp[(uint8_t)(*pb)[j]], *ps + cl);
        if (*tmpcap < bl) {
            size_t nc = *tmpcap ? *tmpcap : 16;
            while (nc < bl) nc *= 2;
            int *nt = (int *)realloc(*tmpids, nc * sizeof(int));
            if (!nt) { r = -1; break; }
            *tmpids = nt;
            *tmpcap = nc;
        }
        int c3 = bpe_piece(b, *ps, cl, *tmpids, (int)bl);
        if (c3 < 0 || *total + c3 > cap) { r = -1; break; }
        memcpy(ids + *total, *tmpids, (size_t)c3 * sizeof(int));
        *total += c3;
    }
    free(pcs);
    free(c2);
    return r;
}

int tlm_bpe_encode(const tlm_bpe *b, const char *text, size_t len, int allow_added,
                   int *ids, int cap) {
    uint32_t *in = (uint32_t *)malloc(sizeof(uint32_t) * (len ? len : 1));
    if (!in) return -1;
    int nin = utf8_decode(text, len, in, (int)len);
    uint32_t *cps = NULL;
    int n = nin;
    if (nin > 0) {
        cps = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)(3 * nin + 8));
        if (!cps) { free(in); return -1; }
        n = tlm_nfc_normalize(in, nin, cps, 3 * nin + 8);
        if (n < 0) { free(in); free(cps); return -1; }
    }
    free(in);

    char *s = (char *)malloc((size_t)n * 4 + 1);
    if (!s) { free(cps); return -1; }
    size_t slen = 0;
    for (int i = 0; i < n; i++) slen += (size_t)utf8_encode(cps[i], s + slen);
    s[slen] = 0;
    free(cps);

    int total = 0, r = 0;
    char *pb = NULL, *ps = NULL;
    size_t pbcap = 0, pscap = 0, tmpcap = 0;
    int *tmpids = NULL;
    size_t i = 0, span0 = 0;
    while (i < slen && r == 0) {
        int aid = -1, alen = 0;
        if (allow_added)
            for (int a = 0; a < b->nadd; a++) {
                size_t al = b->add_len[a];
                if (al && al <= slen - i && al > (size_t)alen &&
                    memcmp(b->addb + b->add_off[a], s + i, al) == 0) {
                    aid = b->add_id[a];
                    alen = (int)al;
                }
            }
        if (aid >= 0) {
            if (i > span0)
                r = encode_span(b, s + span0, i - span0, &pb, &pbcap, &ps, &pscap,
                                &tmpids, &tmpcap, ids, cap, &total);
            if (r == 0 && total >= cap) r = -1;
            if (r == 0) ids[total++] = aid;
            i += (size_t)alen;
            span0 = i;
        } else {
            uint8_t u0 = (uint8_t)s[i];
            i += (size_t)(u0 < 0x80u ? 1 : (u0 & 0xE0u) == 0xC0u ? 2
                                     : (u0 & 0xF0u) == 0xE0u ? 3 : 4);
        }
    }
    if (r == 0 && slen > span0)
        r = encode_span(b, s + span0, slen - span0, &pb, &pbcap, &ps, &pscap,
                        &tmpids, &tmpcap, ids, cap, &total);
    free(s);
    free(pb);
    free(ps);
    free(tmpids);
    return r == 0 ? total : -1;
}

/* ---- decode (P-103) ------------------------------------------------------- */
int tlm_bpe_decode(const tlm_bpe *b, const int *ids, int n, int skip_special,
                   char *out, size_t cap) {
    const gpt2map *gm = gpt2();
    uint8_t *buf = NULL;
    size_t nbuf = 0, capbuf = 0;
    for (int i = 0; i < n; i++) {
        int special;
        const char *s;
        size_t sl;
        if (!tok_content(b, ids[i], &special, &s, &sl)) continue;
        if (skip_special && special) continue;
        if (!append_tok_bytes(gm, s, sl, &buf, &nbuf, &capbuf)) { free(buf); return -1; }
    }
    uint8_t tail[8];
    size_t tlen;
    int r = utf8_emit(buf, nbuf, out, cap, tail, &tlen);
    free(buf);
    return r;
}

/* ---- incremental decoder --------------------------------------------------- */
struct tlm_bpe_dec {
    const tlm_bpe *b;
    int    skip_special;
    uint8_t *buf;
    size_t  n, cap;
};

tlm_bpe_dec *tlm_bpe_dec_new(const tlm_bpe *b, int skip_special) {
    tlm_bpe_dec *d = (tlm_bpe_dec *)calloc(1, sizeof *d);
    if (d) { d->b = b; d->skip_special = skip_special; }
    return d;
}

void tlm_bpe_dec_free(tlm_bpe_dec *d) {
    if (d) { free(d->buf); free(d); }
}

void tlm_bpe_dec_reset(tlm_bpe_dec *d) {
    if (d) { free(d->buf); d->buf = NULL; d->n = d->cap = 0; }
}

int tlm_bpe_dec_push(tlm_bpe_dec *d, int id, char *out, size_t cap) {
    const gpt2map *gm = gpt2();
    int special;
    const char *s;
    size_t sl;
    if (tok_content(d->b, id, &special, &s, &sl)) {
        if (!(d->skip_special && special))
            if (!append_tok_bytes(gm, s, sl, &d->buf, &d->n, &d->cap)) return -1;
    }
    uint8_t tail[8];
    size_t tlen;
    int r = utf8_emit(d->buf, d->n, out, cap, tail, &tlen);
    if (r < 0) return -1;
    /* keep only the held-back tail for the next push */
    memmove(d->buf, d->buf + (size_t)r, tlen);
    d->n = tlen;
    return r;
}

int tlm_bpe_dec_flush(tlm_bpe_dec *d, char *out, size_t cap) {
    /* emit the rest lossily: U+FFFD per maximal invalid subsequence */
    static const uint32_t lo[5] = {0, 0, 0x80u, 0x800u, 0x10000u};
    size_t nc = d->n * 3 + 1;
    char *tmp = (char *)malloc(nc);
    if (!tmp) return -1;
    size_t t = 0;
    for (size_t i = 0; i < d->n; i++) {
        uint8_t u0 = d->buf[i];
        int k = u0 < 0x80u ? 1 : (u0 & 0xE0u) == 0xC0u ? 2
                : (u0 & 0xF0u) == 0xE0u ? 3 : (u0 & 0xF8u) == 0xF0u ? 4 : 0;
        if (k == 0) {
            tmp[t++] = (char)0xEF; tmp[t++] = (char)0xBF; tmp[t++] = (char)0xBD;
            continue;
        }
        if (i + (size_t)k > d->n) {   /* incomplete tail: one replacement */
            tmp[t++] = (char)0xEF; tmp[t++] = (char)0xBF; tmp[t++] = (char)0xBD;
            break;
        }
        uint32_t cp = k == 1 ? u0 : u0 & (0xFFu >> (k + 1));
        int ok = 1;
        for (int q = 1; q < k; q++) {
            if ((d->buf[i + q] & 0xC0u) != 0x80u) { ok = 0; break; }
            cp = (cp << 6) | (d->buf[i + q] & 0x3Fu);
        }
        if (ok && (cp < lo[k] || cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu))) ok = 0;
        if (ok) {
            memcpy(tmp + t, d->buf + i, (size_t)k);
            t += (size_t)k;
            i += (size_t)k - 1;
        } else {
            tmp[t++] = (char)0xEF; tmp[t++] = (char)0xBF; tmp[t++] = (char)0xBD;
        }
    }
    int r = t > cap ? -1 : (int)t;
    if (r >= 0) {
        memcpy(out, tmp, t);
        d->n = 0;
    }
    free(tmp);
    return r;
}
