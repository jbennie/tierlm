/* tlm_st.c - safetensors reader and writer (F32/F16/BF16/F8_E4M3 tensors).
 *
 * Patterns: P-104 hardened safetensors indexing (header cap, overflow-checked
 * shapes, byte-span cross-check, index.json for shards, lazy positioned
 * reads), P-03 validated container, P-30 full reads, P-08 atomic write.
 * Format (public spec, fragments/support.md F-FMT-01): u64 LE header length
 * N, N bytes of JSON, then the data buffer; data_offsets are relative to
 * the start of the data buffer, end exclusive.
 */
#include "tlm_st.h"
#include "tlm_json.h"
#include "tlm_platform.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define ST_HEADER_CAP   (256ull << 20)   /* refuse absurd headers */
#define ST_MAX_SHARDS   4096

struct tlm_st {
    int            n, cap;
    tlm_st_tensor *t;
    int            nfiles;
    tlm_file      *files;
    /* open-addressing name hash: index+1, 0 = empty */
    int           *hash;
    int            hcap;
};

static uint64_t name_hash(const char *s) {
    uint64_t h = 1469598103934665603ull;          /* FNV-1a */
    while (*s) { h ^= (uint8_t)*s++; h *= 1099511628211ull; }
    return h;
}

static int dtype_of(const char *s, tlm_dtype *dt, int *fp8, int *elsize) {
    *fp8 = 0;
    if (strcmp(s, "F32") == 0) { *dt = TLM_DT_F32; *elsize = 4; return 1; }
    if (strcmp(s, "F16") == 0) { *dt = TLM_DT_F16; *elsize = 2; return 1; }
    if (strcmp(s, "BF16") == 0) { *dt = TLM_DT_BF16; *elsize = 2; return 1; }
    if (strcmp(s, "F8_E4M3") == 0) { *dt = TLM_DT_FP8E4M3_B128; *fp8 = 1; *elsize = 1; return 1; }
    return 0;
}

static int find_index(const tlm_st *st, const char *name) {
    if (!st->hash) return -1;
    uint64_t h = name_hash(name);
    for (int i = 0; i < st->hcap; i++) {
        int slot = (int)((h + (uint64_t)i) & (uint64_t)(st->hcap - 1));
        int v = st->hash[slot];
        if (v == 0) return -1;
        if (strcmp(st->t[v - 1].name, name) == 0) return v - 1;
    }
    return -1;
}

static void hash_insert(int *hash, int hcap, const char *name, int idx) {
    uint64_t h = name_hash(name);
    for (int i = 0; i < hcap; i++) {
        int slot = (int)((h + (uint64_t)i) & (uint64_t)(hcap - 1));
        if (hash[slot] == 0) { hash[slot] = idx + 1; return; }
    }
}

/* Append a tensor and index its name (load factor kept <= 0.5). */
static int st_push(tlm_st *st, const tlm_st_tensor *t) {
    if (st->n == st->cap) {
        int nc = st->cap ? st->cap * 2 : 256;
        tlm_st_tensor *nt = (tlm_st_tensor *)realloc(st->t, (size_t)nc * sizeof *nt);
        if (!nt) return tlm_seterr(TLM_ENOMEM, "safetensors: out of memory");
        st->t = nt;
        st->cap = nc;
    }
    if (2 * (st->n + 1) > st->hcap) {
        int cap = st->hcap ? st->hcap * 2 : 512;
        int *nh = (int *)calloc((size_t)cap, sizeof(int));
        if (!nh) return tlm_seterr(TLM_ENOMEM, "safetensors: out of memory");
        for (int k = 0; k < st->n; k++) hash_insert(nh, cap, st->t[k].name, k);
        free(st->hash);
        st->hash = nh;
        st->hcap = cap;
    }
    st->t[st->n] = *t;
    hash_insert(st->hash, st->hcap, t->name, st->n);
    st->n++;
    return TLM_OK;
}

typedef struct { uint64_t b, e; int idx; } span;
static int span_cmp(const void *a, const void *b) {
    const span *x = (const span *)a, *y = (const span *)b;
    if (x->b != y->b) return x->b < y->b ? -1 : 1;
    if (x->e != y->e) return x->e < y->e ? -1 : 1;
    return 0;
}

/* Parse and validate one shard's header; append its tensors.
 * `wm` (may be NULL) is the index weight_map used to resolve duplicates. */
static int parse_shard(tlm_st *st, int fi, const char *fname, const tlm_jval *wm) {
    tlm_file *f = &st->files[fi];
    if (f->size < 8) return tlm_seterr(TLM_EFORMAT, "%s: too small for a safetensors file", f->path);
    uint8_t lenb[8];
    int rc = tlm_pread_full(f, lenb, 8, 0, 0);
    if (rc) return rc;
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; i--) hlen = (hlen << 8) | lenb[i];
    if (hlen < 2 || hlen > ST_HEADER_CAP || hlen > f->size - 8)
        return tlm_seterr(TLM_EFORMAT, "%s: bad header length %llu (file %llu bytes)", f->path,
                          (unsigned long long)hlen, (unsigned long long)f->size);
    char *hdr = (char *)malloc((size_t)hlen + 1);
    if (!hdr) return tlm_seterr(TLM_ENOMEM, "safetensors: out of memory");
    rc = tlm_pread_full(f, hdr, hlen, 8, 0);
    if (rc) { free(hdr); return rc; }
    hdr[hlen] = 0;
    char err[256];
    tlm_json *j = tlm_json_parse(hdr, (size_t)hlen, err, sizeof err);
    free(hdr);
    if (!j) return tlm_seterr(TLM_EFORMAT, "%s: header JSON: %s", f->path, err);
    const tlm_jval *root = tlm_json_root(j);
    rc = TLM_OK;
    if (!root || root->type != TLM_J_OBJ) {
        rc = tlm_seterr(TLM_EFORMAT, "%s: header is not a JSON object", f->path);
        goto done;
    }
    uint64_t data_start = 8 + hlen, data_len = f->size - data_start;
    size_t nobj = root->u.obj.n;
    span *spans = (span *)calloc(nobj ? nobj : 1, sizeof *spans);
    if (!spans) { rc = tlm_seterr(TLM_ENOMEM, "safetensors: out of memory"); goto done; }
    size_t nspans = 0;
    for (size_t k = 0; k < nobj && rc == TLM_OK; k++) {
        const char *name = root->u.obj.keys[k];
        const tlm_jval *d = root->u.obj.vals[k];
        if (strcmp(name, "__metadata__") == 0) continue;
        if (!d || d->type != TLM_J_OBJ) {
            rc = tlm_seterr(TLM_EFORMAT, "%s: tensor '%s' descriptor is not an object", f->path, name);
            break;
        }
        const tlm_jval *jdt = tlm_jobj_get(d, "dtype"), *jsh = tlm_jobj_get(d, "shape"),
                       *jof = tlm_jobj_get(d, "data_offsets");
        if (!jdt || jdt->type != TLM_J_STR || !jsh || jsh->type != TLM_J_ARR || !jof ||
            jof->type != TLM_J_ARR || tlm_jlen(jof) != 2) {
            rc = tlm_seterr(TLM_EFORMAT, "%s: tensor '%s' needs dtype, shape, data_offsets", f->path, name);
            break;
        }
        uint64_t off[2];
        for (int q = 0; q < 2; q++) {
            const tlm_jval *v = tlm_jat(jof, (size_t)q);
            if (!v || v->type != TLM_J_NUM || !v->u.num.is_int || v->u.num.i < 0) {
                rc = tlm_seterr(TLM_EFORMAT, "%s: tensor '%s' has a bad data offset", f->path, name);
                break;
            }
            off[q] = (uint64_t)v->u.num.i;
        }
        if (rc) break;
        if (off[0] > off[1] || off[1] > data_len) {
            rc = tlm_seterr(TLM_EFORMAT, "%s: tensor '%s' offsets [%llu, %llu) outside data (%llu bytes)",
                            f->path, name, (unsigned long long)off[0], (unsigned long long)off[1],
                            (unsigned long long)data_len);
            break;
        }
        if (off[1] > off[0]) {   /* empty tensors occupy no bytes and cannot overlap */
            spans[nspans].b = off[0];
            spans[nspans].e = off[1];
            spans[nspans].idx = (int)k;
            nspans++;
        }
        size_t nd = tlm_jlen(jsh);
        if (nd > 4) {
            rc = tlm_seterr(TLM_EFORMAT, "%s: tensor '%s' has rank %zu (max 4)", f->path, name, nd);
            break;
        }
        tlm_st_tensor t;
        memset(&t, 0, sizeof t);
        if (strlen(name) >= sizeof t.name) {
            rc = tlm_seterr(TLM_EFORMAT, "%s: tensor name too long: %.60s...", f->path, name);
            break;
        }
        snprintf(t.name, sizeof t.name, "%s", name);
        t.ndim = (int)nd;
        uint64_t numel = 1;
        for (size_t q = 0; q < nd; q++) {
            const tlm_jval *v = tlm_jat(jsh, q);
            if (!v || v->type != TLM_J_NUM || !v->u.num.is_int || v->u.num.i < 0) {
                rc = tlm_seterr(TLM_EFORMAT, "%s: tensor '%s' has a bad shape", f->path, name);
                break;
            }
            uint64_t dim = (uint64_t)v->u.num.i;
            if (dim != 0 && numel > UINT64_MAX / dim / 8) {
                rc = tlm_seterr(TLM_EFORMAT, "%s: tensor '%s' shape overflows", f->path, name);
                break;
            }
            numel *= dim;
            t.shape[q] = (int64_t)dim;
        }
        if (rc) break;
        int elsize = 0, fp8 = 0;
        tlm_dtype dt;
        if (!dtype_of(jdt->u.str.s, &dt, &fp8, &elsize)) {
            tlm_log(TLM_LOG_WARN, "%s: skipping tensor '%s' with unsupported dtype %s", f->path, name,
                    jdt->u.str.s);
            continue;
        }
        if (numel * (uint64_t)elsize != off[1] - off[0]) {
            rc = tlm_seterr(TLM_EFORMAT, "%s: tensor '%s' is %llu bytes, dtype x shape says %llu",
                            f->path, name, (unsigned long long)(off[1] - off[0]),
                            (unsigned long long)(numel * (uint64_t)elsize));
            break;
        }
        t.dtype = dt;
        t.is_fp8 = fp8;
        t.file = fi;
        t.offset = data_start + off[0];
        t.nbytes = off[1] - off[0];
        int prev = find_index(st, t.name);
        if (prev >= 0) {
            const char *owner = wm ? tlm_jstr(tlm_jobj_get(wm, t.name), NULL) : NULL;
            if (st->t[prev].file == fi) {
                rc = tlm_seterr(TLM_EFORMAT, "%s: duplicate tensor '%s'", f->path, name);
                break;
            }
            if (!owner) {
                rc = tlm_seterr(TLM_EFORMAT, "tensor '%s' appears in two shards and the index cannot decide", name);
                break;
            }
            if (strcmp(owner, fname) == 0) st->t[prev] = t;   /* this shard is authoritative */
            continue;
        }
        rc = st_push(st, &t);
    }
    if (rc == TLM_OK && nspans > 1) {
        qsort(spans, nspans, sizeof *spans, span_cmp);
        for (size_t k = 1; k < nspans; k++)
            if (spans[k].b < spans[k - 1].e) {
                rc = tlm_seterr(TLM_EFORMAT, "%s: tensors '%s' and '%s' overlap", f->path,
                                root->u.obj.keys[spans[k - 1].idx], root->u.obj.keys[spans[k].idx]);
                break;
            }
    }
    free(spans);
done:
    tlm_json_free(j);
    return rc;
}

static int is_dir(const char *p) {
    struct stat s;
    return stat(p, &s) == 0 && S_ISDIR(s.st_mode);
}
static int exists(const char *p) {
    struct stat s;
    return stat(p, &s) == 0;
}

static int str_cmp(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

tlm_st *tlm_st_open(const char *path) {
    tlm_st *st = (tlm_st *)calloc(1, sizeof *st);
    if (!st) { tlm_seterr(TLM_ENOMEM, "safetensors: out of memory"); return NULL; }
    char single[1024], index[1024], dir[1024];
    int rc = TLM_OK;
    tlm_json *ij = NULL;
    const tlm_jval *wm = NULL;
    const char **names = NULL;
    int nnames = 0;
    if (is_dir(path)) {
        snprintf(dir, sizeof dir, "%s", path);
        snprintf(single, sizeof single, "%s/model.safetensors", path);
        snprintf(index, sizeof index, "%s/model.safetensors.index.json", path);
    } else {
        snprintf(single, sizeof single, "%s", path);
        snprintf(dir, sizeof dir, "%s", path);
        char *slash = strrchr(dir, '/');
        if (slash) *slash = 0; else snprintf(dir, sizeof dir, ".");
        size_t L = strlen(path);
        if (L > 5 && strcmp(path + L - 5, ".json") == 0) snprintf(index, sizeof index, "%s", path);
        else index[0] = 0;
        if (index[0]) single[0] = 0;
    }
    if (index[0] && exists(index)) {
        char *text = NULL;
        size_t tl = 0;
        rc = tlm_read_file(index, &text, &tl);
        if (rc) goto fail;
        char err[256];
        ij = tlm_json_parse(text, tl, err, sizeof err);
        free(text);
        if (!ij) { rc = tlm_seterr(TLM_EFORMAT, "%s: %s", index, err); goto fail; }
        wm = tlm_jobj_get(tlm_json_root(ij), "weight_map");
        if (!wm || wm->type != TLM_J_OBJ) { rc = tlm_seterr(TLM_EFORMAT, "%s: no weight_map object", index); goto fail; }
        names = (const char **)calloc(wm->u.obj.n + 1, sizeof *names);
        if (!names) { rc = tlm_seterr(TLM_ENOMEM, "out of memory"); goto fail; }
        for (size_t k = 0; k < wm->u.obj.n; k++) {
            const char *fn = tlm_jstr(wm->u.obj.vals[k], NULL);
            if (!fn || strchr(fn, '/') || strcmp(fn, "..") == 0) {
                rc = tlm_seterr(TLM_EFORMAT, "%s: bad shard name for '%s'", index, wm->u.obj.keys[k]);
                goto fail;
            }
            int seen = 0;
            for (int q = 0; q < nnames; q++) if (strcmp(names[q], fn) == 0) { seen = 1; break; }
            if (!seen) {
                if (nnames >= ST_MAX_SHARDS) { rc = tlm_seterr(TLM_EFORMAT, "%s: too many shards", index); goto fail; }
                names[nnames++] = fn;
            }
        }
        qsort(names, (size_t)nnames, sizeof *names, str_cmp);
    } else if (single[0] && exists(single)) {
        names = (const char **)calloc(1, sizeof *names);
        if (!names) { rc = tlm_seterr(TLM_ENOMEM, "out of memory"); goto fail; }
        names[0] = single;
        nnames = 1;
        dir[0] = 0;
    } else {
        rc = tlm_seterr(TLM_ENOTFOUND, "%s: no model.safetensors or model.safetensors.index.json", path);
        goto fail;
    }
    if (nnames == 0) { rc = tlm_seterr(TLM_EFORMAT, "%s: no shards", path); goto fail; }
    st->files = (tlm_file *)calloc((size_t)nnames, sizeof *st->files);
    if (!st->files) { rc = tlm_seterr(TLM_ENOMEM, "out of memory"); goto fail; }
    for (int i = 0; i < nnames; i++) st->files[i].fd = st->files[i].fd_direct = -1;
    for (int i = 0; i < nnames; i++) {
        char fp[1400];
        if (dir[0]) snprintf(fp, sizeof fp, "%s/%s", dir, names[i]);
        else snprintf(fp, sizeof fp, "%s", names[i]);
        rc = tlm_file_open(&st->files[i], fp, TLM_F_READ);
        if (rc) {
            tlm_seterr(rc, "%s: cannot open shard (missing or partial download?): %s", fp, tlm_errmsg());
            goto fail;
        }
        st->nfiles = i + 1;
        rc = parse_shard(st, i, names[i], wm);
        if (rc) goto fail;
    }
    if (wm) {
        for (size_t k = 0; k < wm->u.obj.n; k++)
            if (find_index(st, wm->u.obj.keys[k]) < 0) {
                rc = tlm_seterr(TLM_EFORMAT, "tensor '%s' listed in the index is missing from shard %s",
                                wm->u.obj.keys[k], tlm_jstr(wm->u.obj.vals[k], "?"));
                goto fail;
            }
    }
    free(names);
    if (ij) tlm_json_free(ij);
    return st;
fail:
    free(names);
    if (ij) tlm_json_free(ij);
    tlm_st_close(st);
    (void)rc;
    return NULL;
}

void tlm_st_close(tlm_st *st) {
    if (!st) return;
    for (int i = 0; i < st->nfiles; i++) tlm_file_close(&st->files[i]);
    free(st->files);
    free(st->t);
    free(st->hash);
    free(st);
}

int tlm_st_count(const tlm_st *st) { return st ? st->n : 0; }
const tlm_st_tensor *tlm_st_at(const tlm_st *st, int i) {
    return (st && i >= 0 && i < st->n) ? &st->t[i] : NULL;
}
const tlm_st_tensor *tlm_st_find(const tlm_st *st, const char *name) {
    int i = st ? find_index(st, name) : -1;
    return i >= 0 ? &st->t[i] : NULL;
}

int64_t tlm_st_numel(const tlm_st_tensor *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

int tlm_st_read_raw(tlm_st *st, const tlm_st_tensor *t, void *dst) {
    if (!st || !t || t->file < 0 || t->file >= st->nfiles) return tlm_seterr(TLM_EINVAL, "bad tensor");
    if (t->nbytes == 0) return TLM_OK;
    return tlm_pread_full(&st->files[t->file], dst, t->nbytes, t->offset, 0);
}

/* F8_E4M3 is converted without block scales (raw values); callers that need
 * scaled weights apply the companion scale tensor themselves. */
int tlm_st_read_f32(tlm_st *st, const tlm_st_tensor *t, float *dst) {
    int64_t n = tlm_st_numel(t);
    if (t->dtype == TLM_DT_F32 && !t->is_fp8) return tlm_st_read_raw(st, t, dst);
    void *tmp = malloc(t->nbytes ? (size_t)t->nbytes : 1);
    if (!tmp) return tlm_seterr(TLM_ENOMEM, "out of memory reading %s", t->name);
    int rc = tlm_st_read_raw(st, t, tmp);
    if (rc == TLM_OK) {
        if (t->is_fp8) {
            const uint8_t *s = (const uint8_t *)tmp;
            for (int64_t i = 0; i < n; i++) dst[i] = tlm_e4m3_to_f32(s[i]);
        } else if (t->dtype == TLM_DT_F16 || t->dtype == TLM_DT_BF16) {
            const uint8_t *s = (const uint8_t *)tmp;
            for (int64_t i = 0; i < n; i++) {
                uint16_t h = (uint16_t)(s[2 * i] | (s[2 * i + 1] << 8));
                dst[i] = t->dtype == TLM_DT_F16 ? tlm_f16_to_f32(h) : tlm_bf16_to_f32(h);
            }
        } else {
            rc = tlm_seterr(TLM_EUNSUP, "%s: cannot convert dtype %d to f32", t->name, (int)t->dtype);
        }
    }
    free(tmp);
    return rc;
}

/* FP8 (E4M3) 2-D weight -> f32, decoded with its block-scale tensor
 * `<name>_scale_inv`: an F32 grid [ceil(rows/128)] x [ceil(cols/128)],
 * row-major over the 128x128 block grid. The checkpoints' "scale_inv" is
 * the multiplier (P-47). Refused: 1-D FP8 tensors, missing / non-F32 /
 * misshaped scales, and E4M3 NaN codes 0x7F/0xFF (fail loudly, never
 * guess). dst must hold rows x cols floats. */
int tlm_st_read_fp8_block(tlm_st *st, const tlm_st_tensor *t, float *dst) {
    if (!st || !t || !dst) return tlm_seterr(TLM_EINVAL, "read_fp8_block: bad argument");
    if (!t->is_fp8) return tlm_seterr(TLM_EFORMAT, "%s: not an F8_E4M3 tensor", t->name);
    if (t->ndim != 2)
        return tlm_seterr(TLM_EUNSUP, "%s: 1-D FP8 tensors are not supported (norms and biases must be F32)", t->name);
    int64_t rows = t->shape[0], cols = t->shape[1];
    char snm[192];
    snprintf(snm, sizeof snm, "%s_scale_inv", t->name);
    const tlm_st_tensor *s = tlm_st_find(st, snm);
    if (!s)
        return tlm_seterr(TLM_ENOTFOUND, "%s is FP8 (E4M3) but its 128x128 block-scale tensor '%s' is missing",
                          t->name, snm);
    if (s->dtype != TLM_DT_F32)
        return tlm_seterr(TLM_EFORMAT, "%s must be an F32 block-scale tensor (single-byte UE8M0 scales are not supported)",
                          snm);
    int64_t krb = (rows + 127) / 128, nkb = (cols + 127) / 128;
    if (s->ndim != 2 || s->shape[0] != krb || s->shape[1] != nkb)
        return tlm_seterr(TLM_EFORMAT, "%s has shape [%lld, %lld], expected [%lld, %lld] for 128x128 blocks of %s", snm,
                          (long long)s->shape[0], (long long)s->shape[1], (long long)krb, (long long)nkb, t->name);
    int64_t n = rows * cols;
    uint8_t *codes = (uint8_t *)malloc(n ? (size_t)n : 1);
    float *sc = (float *)malloc(sizeof(float) * (size_t)krb * (size_t)nkb);
    if (!codes || !sc) { free(codes); free(sc); return tlm_seterr(TLM_ENOMEM, "out of memory"); }
    int rc = tlm_st_read_raw(st, t, codes);
    if (!rc)
        for (int64_t i = 0; i < rows * cols; i++)
            if ((codes[i] & 0x7F) == 0x7F) {
                rc = tlm_seterr(TLM_EFORMAT, "%s contains an E4M3 NaN code 0x%02X at offset %lld; refusing to guess (P-47)",
                                t->name, codes[i], (long long)i);
                break;
            }
    if (!rc) rc = tlm_st_read_f32(st, s, sc);
    if (!rc)
        for (int64_t r = 0; r < rows && !rc; r++)
            rc = tlm_dequantize_fp8_row(codes + r * cols, dst + r * cols, cols, sc + (r / 128) * nkb);
    free(codes);
    free(sc);
    return rc;
}

/* ---- writer ------------------------------------------------------------ */
typedef struct {
    char        *name;
    const char  *dtype;          /* safetensors dtype string; must outlive the writer */
    uint8_t     *data;
    int          ndim;
    int64_t      shape[4];
    uint64_t     nbytes;
} st_wtensor;

struct tlm_st_writer {
    st_wtensor *t;
    int         n, cap;
};

tlm_st_writer *tlm_st_writer_new(void) {
    return (tlm_st_writer *)calloc(1, sizeof(tlm_st_writer));
}

static int dt_elsize(const char *dtype) {
    if (strcmp(dtype, "F32") == 0) return 4;
    if (strcmp(dtype, "F16") == 0) return 2;
    if (strcmp(dtype, "BF16") == 0) return 2;
    if (strcmp(dtype, "F8_E4M3") == 0) return 1;
    return 0;
}

int tlm_st_writer_add_raw(tlm_st_writer *w, const char *name, const char *dtype, const void *data,
                          uint64_t nbytes, int ndim, const int64_t *shape) {
    if (!w || !name || !dtype || ndim < 0 || ndim > 4) return tlm_seterr(TLM_EINVAL, "st writer: bad tensor");
    int es = dt_elsize(dtype);
    if (!es) return tlm_seterr(TLM_EINVAL, "st writer: unsupported dtype '%s'", dtype);
    int bad = name[0] == 0 || strlen(name) >= sizeof(((tlm_st_tensor *)0)->name) ||
              strcmp(name, "__metadata__") == 0;
    for (const char *c = name; *c && !bad; c++)   /* written into JSON unescaped */
        if (*c == '"' || *c == '\\' || (unsigned char)*c < 0x20) bad = 1;
    if (bad) return tlm_seterr(TLM_EINVAL, "st writer: unsupported tensor name '%s'", name);
    for (int i = 0; i < w->n; i++)
        if (strcmp(w->t[i].name, name) == 0) return tlm_seterr(TLM_EINVAL, "st writer: duplicate '%s'", name);
    uint64_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) return tlm_seterr(TLM_EINVAL, "st writer: negative dim");
        numel *= (uint64_t)shape[i];
    }
    if (nbytes != numel * (uint64_t)es)
        return tlm_seterr(TLM_EINVAL, "st writer: %s has %llu bytes, expected %llu for %s shape", name,
                          (unsigned long long)nbytes, (unsigned long long)(numel * (uint64_t)es), dtype);
    if (w->n == w->cap) {
        int nc = w->cap ? w->cap * 2 : 64;
        st_wtensor *nt = (st_wtensor *)realloc(w->t, (size_t)nc * sizeof *nt);
        if (!nt) return tlm_seterr(TLM_ENOMEM, "st writer: out of memory");
        w->t = nt;
        w->cap = nc;
    }
    st_wtensor *t = &w->t[w->n];
    memset(t, 0, sizeof *t);
    t->name = strdup(name);
    t->dtype = dtype;
    t->nbytes = nbytes;
    t->data = (uint8_t *)malloc(nbytes ? nbytes : 1);
    if (!t->name || !t->data) { free(t->name); free(t->data); return tlm_seterr(TLM_ENOMEM, "st writer: out of memory"); }
    if (nbytes) memcpy(t->data, data, nbytes);
    t->ndim = ndim;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    w->n++;
    return TLM_OK;
}

int tlm_st_writer_add(tlm_st_writer *w, const char *name, const float *data, int ndim,
                      const int64_t *shape) {
    uint64_t numel = 1;
    for (int i = 0; i < ndim; i++) numel *= (uint64_t)shape[i];
    return tlm_st_writer_add_raw(w, name, "F32", data, numel * 4, ndim, shape);
}

/* Growable text buffer for the header JSON. */
typedef struct { char *p; size_t len, cap; int oom; } sbuf;
static void sb_printf(sbuf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void sb_printf(sbuf *b, const char *fmt, ...) {
    if (b->oom) return;
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(b->p ? b->p + b->len : NULL, b->p ? b->cap - b->len : 0, fmt, ap);
        va_end(ap);
        if (n < 0) { b->oom = 1; return; }
        if (b->p && b->len + (size_t)n < b->cap) { b->len += (size_t)n; return; }
        size_t nc = b->cap ? b->cap * 2 : 1024;
        while (nc < b->len + (size_t)n + 16) nc *= 2;
        char *np = (char *)realloc(b->p, nc);
        if (!np) { b->oom = 1; return; }
        b->p = np;
        b->cap = nc;
    }
}

int tlm_st_writer_save(tlm_st_writer *w, const char *path) {
    if (!w || !path) return tlm_seterr(TLM_EINVAL, "st writer: null");
    /* header JSON: tensors in insertion order, data packed back to back */
    sbuf hb = {0};
    sb_printf(&hb, "{\"__metadata__\":{\"format\":\"pt\",\"writer\":\"tierlm\"}");
    uint64_t off = 0;
    for (int i = 0; i < w->n; i++) {
        st_wtensor *t = &w->t[i];
        sb_printf(&hb, ",\"%s\":{\"dtype\":\"%s\",\"shape\":[", t->name, t->dtype);
        for (int d = 0; d < t->ndim; d++) sb_printf(&hb, "%s%lld", d ? "," : "", (long long)t->shape[d]);
        sb_printf(&hb, "],\"data_offsets\":[%llu,%llu]}", (unsigned long long)off,
                  (unsigned long long)(off + t->nbytes));
        off += t->nbytes;
    }
    sb_printf(&hb, "}");
    /* pad with spaces (valid JSON whitespace) so the data buffer is 8-byte aligned */
    while (!hb.oom && (hb.len % 8) != 0) sb_printf(&hb, " ");
    if (hb.oom) { free(hb.p); return tlm_seterr(TLM_ENOMEM, "st writer: out of memory"); }
    char *h = hb.p;
    size_t len = hb.len;

    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp.%ld", path, (long)getpid());
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) { free(h); return tlm_seterr(TLM_EIO, "create %s: %s", tmp, strerror(errno)); }
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) lenb[i] = (uint8_t)((uint64_t)len >> (8 * i));
    int rc = tlm_write_full(fd, lenb, 8);
    if (!rc) rc = tlm_write_full(fd, h, len);
    for (int i = 0; i < w->n && !rc; i++) rc = tlm_write_full(fd, w->t[i].data, w->t[i].nbytes);
    if (!rc && fsync(fd) != 0) rc = tlm_seterr(TLM_EIO, "fsync %s: %s", tmp, strerror(errno));
    close(fd);
    free(h);
    if (!rc && rename(tmp, path) != 0) rc = tlm_seterr(TLM_EIO, "rename %s: %s", path, strerror(errno));
    if (rc) unlink(tmp);
    return rc;
}

void tlm_st_writer_free(tlm_st_writer *w) {
    if (!w) return;
    for (int i = 0; i < w->n; i++) { free(w->t[i].name); free(w->t[i].data); }
    free(w->t);
    free(w);
}
