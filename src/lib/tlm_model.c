/* tlm_model.c - config-driven MoE decoder over a compiled packet store.
 *
 * Patterns (fragments/forward_pass.md, fragments/CONSENSUS.md):
 *   P-79 / F-fwd-01  pre-norm decoder skeleton, f32 residual, final norm +
 *                    lm_head on the last row only
 *   P-80 / F-fwd-02  RMSNorm, sum of squares in double, eps inside the sqrt
 *   P-62 / F-fwd-06  per-head q/k RMSNorm after projection, before RoPE
 *                    (applied when "q_norm"/"k_norm" exist in the DENSE packet)
 *   P-81 / F-fwd-03  RoPE, NeoX half-split pairs, inv_freq cached per model
 *   P-61 / F-fwd-05  scalar causal GQA attention: scale after the dot,
 *                    max-subtracted softmax, consecutive q heads share a kv head
 *   P-71 / F-fwd-14  softmax-then-top-k router (renormalised or not)
 *   P-73 / F-fwd-16  sigmoid router, bias used for selection only, renormalise
 *                    (+1e-20) then routed scale
 *   P-75 / F-fwd-18  SwiGLU experts + shared expert added after the routed sum
 *   P-76 / C-13      routed outputs combined in ascending top-k rank order,
 *                    independent of which expert finished loading first
 *   P-42             each activation vector quantised once per consumer kind
 *   P-43             every output row is computed by exactly one worker, so
 *                    thread count never changes a float
 *   P-89 / F-fwd-25  position-addressed KV cache (head-major rows, f32)
 *   P-92 / F-fwd-30  prefill is the decode step repeated: identical numerics
 *   P-33 / P-34      router lookahead: layer L+1's norm + router on layer L's
 *                    post-attention state -> tlm_store_prefetch (hint only)
 *   P-94 / F-fwd-28  sampler: greedy (ties -> lowest id), temperature, top-p,
 *                    min-p in logit space, xorshift64* RNG seeded by splitmix64
 *
 * Pruned experts (manifest packet flags bit1) are masked out of routing
 * before softmax/top-k, so the survivors are renormalised exactly as the
 * compiler did during its evaluation gate.
 */
#include "tlm_model.h"
#include "tlm_platform.h"
#include "tlm_quant.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOPK     64
#define MAX_SEGS     (2 * MAX_TOPK + 8)
#define SERIAL_WORK  32768      /* rows*cols below which a matvec runs on the caller */
#define ROW_GRAIN    16

/* ---- weights ----------------------------------------------------------- */
typedef struct {
    tlm_dtype      dt;
    const uint8_t *data;
    size_t         row_bytes;
    int64_t        rows, cols;
} wmat;

typedef struct {
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm, *router_bias, *shared_gate_inp;
    wmat         q, k, v, o, router;
    wmat         sg, su, sd;
    wmat         dg, du, dd;      /* dense-FFN gate/up/down (has_dense_ffn) */
    int          has_shared;
    int          has_dense_ffn;   /* layer is a dense FFN, not a router + experts */
    int64_t      d_ffn;           /* dense FFN intermediate width */
    uint8_t     *buf;            /* packet buffer (owned) */
} layer_w;

typedef struct {
    uint32_t dtype;
    uint32_t off[3];
    uint64_t payload_len;
    uint8_t  present;
} expert_info;

typedef struct {               /* lazily quantised copies of one activation vector (P-42) */
    const float *x;
    int64_t      n;
    uint8_t     *q80, *q8k;
    int          v80, v8k;
} actc;

struct tlm_model {
    tlm_model_geom g;
    tlm_io        *io;
    tlm_store     *store;
    tlm_pool      *pool;
    tlm_model_opts o;
    layer_w       *L;
    const float   *out_norm;
    wmat           lm_head, embd;
    uint8_t       *meta_buf, *embed_buf;
    expert_info   *ex;           /* [n_layers * n_experts] */
    uint8_t       *mask;         /* 1 = pruned, [n_layers * n_experts] */
    float         *layer_heat;    /* [n_layers] given heat (mean manifest loc.heat), 0..1 */
    double        *instinct;      /* [n_layers] EMA of top-1 routed-mass share (learned) */
    int           *instinct_seen; /* [n_layers] real-route observations */
    float         *inv_freq;     /* [head_dim / 2] (YaRN-corrected if configured) */
    float          rope_attn_scale;   /* cos/sin scale (1.0 unless YaRN) */
    float        **owned;        /* converted vectors to free */
    int            n_owned, cap_owned;

    /* scratch (single sequence at a time) */
    float *h, *xb, *xb2, *q, *kx, *vx, *att, *ffn, *rl, *rp, *cs, *sn;
    actc   a_x, a_att, a_ffn, a_fin, a_la;
    float *eg[MAX_TOPK], *eh[MAX_TOPK], *eo[MAX_TOPK];
    actc   a_e[MAX_TOPK];
    float *sg_buf, *sh_buf, *s_out;
    actc   a_sh;
    float *dg_buf, *dh_buf;      /* dense-FFN gate/up (adjacent) and silu(g)*u */
    actc   a_dh;
    int64_t d_ffn_max;           /* widest dense FFN across layers (scratch sizing) */

    tlm_route_hook hook;
    void          *hook_ctx;
    tlm_model_stats st;
};

struct tlm_kv {
    const tlm_model *mdl;
    int    max_ctx, len;
    float *k, *v;                /* [layer][kv_head][max_ctx][head_dim] */
    float *scores;               /* [n_heads][max_ctx] */
};

/* ---- small helpers ----------------------------------------------------- */
static float *track(tlm_model *M, size_t n) {
    if (M->n_owned == M->cap_owned) {
        int nc = M->cap_owned ? M->cap_owned * 2 : 64;
        float **no = (float **)realloc(M->owned, (size_t)nc * sizeof *no);
        if (!no) return NULL;
        M->owned = no;
        M->cap_owned = nc;
    }
    float *p = (float *)tlm_alloc_aligned(n * sizeof(float) + 64, TLM_ALIGN_SIMD);
    if (p) M->owned[M->n_owned++] = p;
    return p;
}

static int actc_init(tlm_model *M, actc *a, int64_t n) {
    memset(a, 0, sizeof *a);
    a->n = n;
    if (n % 32 == 0) {
        a->q80 = (uint8_t *)track(M, tlm_qact_bytes(TLM_DT_Q8_0, n) / sizeof(float) + 16);
        if (!a->q80) return TLM_ENOMEM;
    }
    if (n % 256 == 0) {
        a->q8k = (uint8_t *)track(M, tlm_qact_bytes(TLM_DT_Q8_K, n) / sizeof(float) + 16);
        if (!a->q8k) return TLM_ENOMEM;
    }
    return TLM_OK;
}
static inline void actc_set(actc *a, const float *x) { a->x = x; a->v80 = a->v8k = 0; }
static const void *actc_get(actc *a, tlm_dtype wdt) {
    tlm_dtype kind = tlm_act_kind_for(wdt);
    if (kind == TLM_DT_Q8_0 && a->q80) {
        if (!a->v80) { tlm_quantize_act(TLM_DT_Q8_0, a->x, a->n, a->q80); a->v80 = 1; }
        return a->q80;
    }
    if (kind == TLM_DT_Q8_K && a->q8k) {
        if (!a->v8k) { tlm_quantize_act(TLM_DT_Q8_K, a->x, a->n, a->q8k); a->v8k = 1; }
        return a->q8k;
    }
    return a->x;   /* float weights take f32 activations */
}

/* RMSNorm (P-80): sum of squares in double, eps added to the mean. */
static void rmsnorm(float *out, const float *x, const float *w, int64_t n, float eps) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float r = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (int64_t i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ---- parallel matvec over several matrices (one pool call) ------------- */
typedef struct {
    tlm_dtype      dt;
    const uint8_t *W;
    size_t         row_bytes;
    const void    *act;
    float         *y;
    int64_t        n, rows;
} mv_seg;
typedef struct {
    mv_seg  seg[MAX_SEGS];
    int64_t start[MAX_SEGS + 1];
    int     nseg;
    int64_t work;
} mv_job;

static void mv_add(mv_job *j, const wmat *w, const void *act, float *y) {
    if (j->nseg == 0) j->start[0] = 0;
    mv_seg *s = &j->seg[j->nseg];
    s->dt = w->dt;
    s->W = w->data;
    s->row_bytes = w->row_bytes;
    s->act = act;
    s->y = y;
    s->n = w->cols;
    s->rows = w->rows;
    j->start[j->nseg + 1] = j->start[j->nseg] + w->rows;
    j->work += w->rows * w->cols;
    j->nseg++;
}

static void mv_run(void *ctx, int64_t b, int64_t e, int worker) {
    (void)worker;
    const mv_job *j = (const mv_job *)ctx;
    for (int s = 0; s < j->nseg; s++) {
        int64_t s0 = j->start[s], s1 = j->start[s + 1];
        if (e <= s0) break;
        if (b >= s1) continue;
        int64_t r0 = (b > s0 ? b : s0) - s0, r1 = (e < s1 ? e : s1) - s0;
        const mv_seg *g = &j->seg[s];
        tlm_matvec_rows(g->dt, g->W, g->row_bytes, g->act, g->y, g->n, r0, r1, NULL);
    }
}

static void mv_exec(tlm_model *M, mv_job *j) {
    if (j->nseg == 0) return;
    int64_t rows = j->start[j->nseg];
    tlm_pool_for(j->work < SERIAL_WORK ? NULL : M->pool, rows, ROW_GRAIN, mv_run, j);
}

/* ---- packet loading ---------------------------------------------------- */
static int read_packet(tlm_model *M, const tlm_packet_loc *loc, uint8_t **buf_out,
                       const uint8_t **payload, uint64_t *plen) {
    *buf_out = NULL;
    if (loc->total_len < sizeof(tlm_packet_hdr) || loc->total_len > (1ull << 40) ||
        loc->total_len % TLM_ALIGN_IO)
        return tlm_seterr(TLM_EFORMAT, "packet (type %u layer %u id %u) has bad length %llu",
                          loc->type, loc->layer, loc->id, (unsigned long long)loc->total_len);
    uint8_t *buf = (uint8_t *)tlm_alloc_aligned((size_t)loc->total_len, TLM_ALIGN_IO);
    if (!buf) return tlm_seterr(TLM_ENOMEM, "out of memory loading a %llu-byte packet",
                                (unsigned long long)loc->total_len);
    /* header + CRC checked against the manifest; falls back to the replica (P-05, P-31) */
    const void *pl = NULL;
    int rc = tlm_packet_read(M->io, loc, buf, &pl, plen);
    if (rc == TLM_OK && loc->type != TLM_PK_EXPERT) rc = tlm_dir_validate(pl, *plen);
    if (rc) { tlm_free_aligned(buf); return rc; }
    *buf_out = buf;
    *payload = (const uint8_t *)pl;
    return TLM_OK;
}

/* Matrix from a tensor directory: rows x cols, any row-quantised dtype. */
static int dir_mat(const uint8_t *pl, uint64_t plen, const char *name, int64_t rows, int64_t cols,
                   int required, wmat *out, int *found) {
    memset(out, 0, sizeof *out);
    if (found) *found = 0;
    const tlm_tensor_ref *t = tlm_dir_find(pl, plen, name);
    if (!t) return required ? tlm_seterr(TLM_ENOTFOUND, "tensor '%s' missing from packet", name) : TLM_OK;
    tlm_dtype dt = (tlm_dtype)t->dtype;
    int blk = tlm_dtype_block(dt);
    if (blk <= 0 || dt == TLM_DT_FP8E4M3_B128 || dt == TLM_DT_Q8_K)
        return tlm_seterr(TLM_EUNSUP, "tensor '%s': dtype %u not usable as a matrix", name, t->dtype);
    if ((int64_t)t->shape[0] != rows || (int64_t)t->shape[1] != cols || cols % blk != 0)
        return tlm_seterr(TLM_EFORMAT, "tensor '%s': shape [%llu, %llu], expected [%lld, %lld]", name,
                          (unsigned long long)t->shape[0], (unsigned long long)t->shape[1],
                          (long long)rows, (long long)cols);
    size_t rb = tlm_row_bytes(dt, cols);
    if (t->nbytes != (uint64_t)rows * rb || t->offset > plen || t->nbytes > plen - t->offset)
        return tlm_seterr(TLM_EFORMAT, "tensor '%s': %llu bytes at %llu do not fit", name,
                          (unsigned long long)t->nbytes, (unsigned long long)t->offset);
    out->dt = dt;
    out->data = pl + t->offset;
    out->row_bytes = rb;
    out->rows = rows;
    out->cols = cols;
    if (found) *found = 1;
    return TLM_OK;
}

/* f32 vector from a directory (converted and owned when stored otherwise). */
static int dir_vec(tlm_model *M, const uint8_t *pl, uint64_t plen, const char *name, int64_t n,
                   int required, const float **out) {
    *out = NULL;
    wmat w;
    int found = 0;
    int rc = dir_mat(pl, plen, name, 1, n, required, &w, &found);
    if (rc || !found) return rc;
    if (w.dt == TLM_DT_F32 && ((uintptr_t)w.data % sizeof(float)) == 0) {
        *out = (const float *)w.data;
        return TLM_OK;
    }
    float *v = track(M, (size_t)n);
    if (!v) return tlm_seterr(TLM_ENOMEM, "out of memory");
    if (w.dt == TLM_DT_F32) memcpy(v, w.data, (size_t)n * sizeof(float));
    else if (tlm_dequantize_row(w.dt, w.data, v, n) != 0)
        return tlm_seterr(TLM_EUNSUP, "tensor '%s': cannot convert to f32", name);
    *out = v;
    return TLM_OK;
}

static int load_layer(tlm_model *M, const tlm_manifest *m, int l) {
    const tlm_model_geom *g = &M->g;
    const tlm_packet_loc *loc = tlm_manifest_find(m, TLM_PK_DENSE, (uint16_t)l, 0);
    if (!loc) return tlm_seterr(TLM_ENOTFOUND, "manifest has no DENSE packet for layer %d", l);
    layer_w *L = &M->L[l];
    const uint8_t *pl;
    uint64_t plen;
    int rc = read_packet(M, loc, &L->buf, &pl, &plen);
    if (rc) return rc;
    int64_t D = g->d_model, hd = g->head_dim, H = g->n_heads, KV = g->n_kv_heads, E = g->n_experts;
    if ((rc = dir_vec(M, pl, plen, "attn_norm", D, 1, &L->attn_norm))) return rc;
    if ((rc = dir_vec(M, pl, plen, "ffn_norm", D, 1, &L->ffn_norm))) return rc;
    if ((rc = dir_vec(M, pl, plen, "q_norm", hd, 0, &L->q_norm))) return rc;
    if ((rc = dir_vec(M, pl, plen, "k_norm", hd, 0, &L->k_norm))) return rc;
    if ((rc = dir_mat(pl, plen, "attn_q", H * hd, D, 1, &L->q, NULL))) return rc;
    if ((rc = dir_mat(pl, plen, "attn_k", KV * hd, D, 1, &L->k, NULL))) return rc;
    if ((rc = dir_mat(pl, plen, "attn_v", KV * hd, D, 1, &L->v, NULL))) return rc;
    if ((rc = dir_mat(pl, plen, "attn_o", D, H * hd, 1, &L->o, NULL))) return rc;
    const tlm_tensor_ref *fg = (E > 0) ? tlm_dir_find(pl, plen, "ffn_gate") : NULL;
    if (fg) {
        /* Dense FFN layer (first_k_dense_replace / mlp_only_layers): no router
         * and no experts, one resident SwiGLU block (F-fwd-01, F-fwd-18). */
        int64_t dffn = (int64_t)fg->shape[0];
        if ((int64_t)fg->shape[1] != D)
            return tlm_seterr(TLM_EFORMAT, "layer %d: ffn_gate shape [%llu,%llu] != [?,%lld]", l,
                              (unsigned long long)fg->shape[0], (unsigned long long)fg->shape[1], (long long)D);
        if ((rc = dir_mat(pl, plen, "ffn_gate", dffn, D, 1, &L->dg, NULL))) return rc;
        if ((rc = dir_mat(pl, plen, "ffn_up", dffn, D, 1, &L->du, NULL))) return rc;
        if ((rc = dir_mat(pl, plen, "ffn_down", D, dffn, 1, &L->dd, NULL))) return rc;
        L->has_dense_ffn = 1;
        L->d_ffn = dffn;
        if (dffn > M->d_ffn_max) M->d_ffn_max = dffn;
    } else if (E > 0) {
        if ((rc = dir_mat(pl, plen, "router", E, D, 1, &L->router, NULL))) return rc;
        if ((rc = dir_vec(M, pl, plen, "router_bias", E, 0, &L->router_bias))) return rc;
        if (g->router == 2 && !L->router_bias)
            tlm_log(TLM_LOG_DEBUG, "layer %d: sigmoid router without router_bias (bias = 0)", l);
        if (g->n_shared > 0 && g->d_shared > 0) {
            int64_t S = g->d_shared;
            if ((rc = dir_mat(pl, plen, "shared_gate", S, D, 1, &L->sg, NULL))) return rc;
            if ((rc = dir_mat(pl, plen, "shared_up", S, D, 1, &L->su, NULL))) return rc;
            if ((rc = dir_mat(pl, plen, "shared_down", D, S, 1, &L->sd, NULL))) return rc;
            if ((rc = dir_vec(M, pl, plen, "shared_gate_inp", D, 0, &L->shared_gate_inp))) return rc;
            L->has_shared = 1;
        }
    }
    return TLM_OK;
}

static int load_experts(tlm_model *M, const tlm_manifest *m) {
    const tlm_model_geom *g = &M->g;
    int64_t D = g->d_model, de = g->d_expert;
    uint8_t *hb = (uint8_t *)tlm_alloc_aligned(TLM_ALIGN_IO, TLM_ALIGN_IO);
    if (!hb) return tlm_seterr(TLM_ENOMEM, "out of memory");
    int rc = TLM_OK;
    for (uint32_t i = 0; i < m->n_packets && rc == TLM_OK; i++) {
        const tlm_packet_loc *loc = &m->packets[i];
        if (loc->type != TLM_PK_EXPERT) continue;
        if (loc->layer >= g->n_layers || loc->id >= g->n_experts) {
            rc = tlm_seterr(TLM_EFORMAT, "EXPERT packet for layer %u expert %u is out of range",
                            loc->layer, loc->id);
            break;
        }
        size_t ix = (size_t)loc->layer * g->n_experts + loc->id;
        /* Pruned (flag bit1): the store does not serve it, so it is not
         * present and stays masked whatever mask is set later. */
        if (loc->flags & 2u) continue;
        if (loc->total_len < TLM_ALIGN_IO || loc->total_len % TLM_ALIGN_IO) {
            rc = tlm_seterr(TLM_EFORMAT, "EXPERT packet %u:%u has bad length", loc->layer, loc->id);
            break;
        }
        /* The packet header gives dtype and gate/up/down offsets, so experts
         * may differ in precision (pass 4). Only the first 4 KiB is read; the
         * payload itself is CRC-checked by the store on every load. */
        uint64_t len = TLM_ALIGN_IO;
        rc = tlm_io_read_sync(M->io, loc->drive, loc->offset, hb, len);
        if (rc && loc->replica_drive != 0xFF)
            rc = tlm_io_read_sync(M->io, loc->replica_drive, loc->replica_offset, hb, len);
        if (rc) break;
        const tlm_packet_hdr *h = (const tlm_packet_hdr *)hb;
        if (h->magic != TLM_PACKET_MAGIC || h->version != TLM_STORE_VERSION ||
            h->type != TLM_PK_EXPERT || h->layer != loc->layer || h->id != loc->id ||
            h->crc32c != loc->crc32c || h->payload_len > loc->total_len - sizeof *h) {
            rc = tlm_seterr(TLM_EFORMAT, "EXPERT packet header mismatch (layer %u expert %u)",
                            loc->layer, loc->id);
            break;
        }
        tlm_dtype dt = (tlm_dtype)h->dtype;
        int blk = tlm_dtype_block(dt);
        if (blk <= 0 || dt == TLM_DT_FP8E4M3_B128 || dt == TLM_DT_Q8_K || D % blk || de % blk) {
            rc = tlm_seterr(TLM_EUNSUP, "expert %u:%u has dtype %u, unusable for %lldx%lld",
                            loc->layer, loc->id, h->dtype, (long long)de, (long long)D);
            break;
        }
        uint64_t rb_up = tlm_row_bytes(dt, D), rb_dn = tlm_row_bytes(dt, de);
        if ((uint64_t)h->item_off[0] + de * rb_up > h->payload_len ||
            (uint64_t)h->item_off[1] + de * rb_up > h->payload_len ||
            (uint64_t)h->item_off[2] + D * rb_dn > h->payload_len) {
            rc = tlm_seterr(TLM_EFORMAT, "expert %u:%u payload layout does not fit", loc->layer, loc->id);
            break;
        }
        expert_info *e = &M->ex[ix];
        e->dtype = h->dtype;
        e->off[0] = h->item_off[0];
        e->off[1] = h->item_off[1];
        e->off[2] = h->item_off[2];
        e->payload_len = h->payload_len;
        e->present = 1;
    }
    tlm_free_aligned(hb);
    if (rc) return rc;
    for (uint32_t l = 0; l < g->n_layers; l++) {
        if (M->L[l].has_dense_ffn) continue;   /* no experts to route */
        int live = 0;
        for (uint32_t e = 0; e < g->n_experts; e++) {
            size_t ix = (size_t)l * g->n_experts + e;
            if (!M->ex[ix].present) M->mask[ix] = 1;   /* no packet: never routable */
            if (!M->mask[ix]) live++;
        }
        if (live < (int)g->top_k)
            return tlm_seterr(TLM_EFORMAT, "layer %u has %d routable experts, needs top_k=%u", l, live, g->top_k);
    }
    return TLM_OK;
}

static int alloc_scratch(tlm_model *M) {
    const tlm_model_geom *g = &M->g;
    int64_t D = g->d_model, Q = (int64_t)g->n_heads * g->head_dim, KVd = (int64_t)g->n_kv_heads * g->head_dim;
    int64_t E = g->n_experts > 0 ? g->n_experts : 1, de = g->d_expert, ds = g->d_shared;
    struct { float **p; int64_t n; } bufs[] = {
        {&M->h, D}, {&M->xb, D}, {&M->xb2, D}, {&M->q, Q}, {&M->kx, KVd}, {&M->vx, KVd},
        {&M->att, Q}, {&M->ffn, D}, {&M->rl, E}, {&M->rp, E}, {&M->cs, g->head_dim}, {&M->sn, g->head_dim},
    };
    for (size_t i = 0; i < sizeof bufs / sizeof bufs[0]; i++)
        if (!(*bufs[i].p = track(M, (size_t)bufs[i].n))) return TLM_ENOMEM;
    if (actc_init(M, &M->a_x, D) || actc_init(M, &M->a_att, Q) || actc_init(M, &M->a_ffn, D) ||
        actc_init(M, &M->a_fin, D) || actc_init(M, &M->a_la, D))
        return TLM_ENOMEM;
    for (uint32_t i = 0; i < g->top_k; i++) {
        if (!(M->eg[i] = track(M, (size_t)(2 * de))) || !(M->eh[i] = track(M, (size_t)de)) ||
            !(M->eo[i] = track(M, (size_t)D)))
            return TLM_ENOMEM;
        if (actc_init(M, &M->a_e[i], de)) return TLM_ENOMEM;
    }
    if (g->n_shared > 0 && ds > 0) {
        if (!(M->sg_buf = track(M, (size_t)(2 * ds))) || !(M->sh_buf = track(M, (size_t)ds)) ||
            !(M->s_out = track(M, (size_t)D)))
            return TLM_ENOMEM;
        if (actc_init(M, &M->a_sh, ds)) return TLM_ENOMEM;
    }
    if (M->d_ffn_max > 0) {
        if (!(M->dg_buf = track(M, (size_t)(2 * M->d_ffn_max))) ||
            !(M->dh_buf = track(M, (size_t)M->d_ffn_max)))
            return TLM_ENOMEM;
        if (actc_init(M, &M->a_dh, (int64_t)M->d_ffn_max)) return TLM_ENOMEM;
    }
    return TLM_OK;
}

/* HF YaRN (transformers' modeling_rope_utils._compute_yarn_parameters,
 * float32 path): rewrites the plain inv_freq table in place and returns the
 * cos/sin attention scale. The correction range is computed in double
 * (floor/ceil when `truncate`), the ramp and the extrapolate/interpolate
 * mix in f32, mirroring HF's torch float32 arithmetic. */
#define TLM_TWO_PI 6.283185307179586
int tlm_yarn_transform(float *inv_freq, int half, uint32_t head_dim, double theta,
                       double factor, double beta_fast, double beta_slow,
                       uint32_t orig_len, int truncate, float *attn_factor) {
    if (!inv_freq || !attn_factor || head_dim < 2 || (head_dim & 1) ||
        half != (int)(head_dim / 2) || !(theta > 0.0) || !(factor > 0.0) ||
        !(beta_fast > 0.0) || beta_fast < beta_slow || orig_len == 0)
        return tlm_seterr(TLM_EINVAL, "yarn: invalid rope scaling parameters");
    double low = (double)head_dim * log((double)orig_len / (beta_fast * TLM_TWO_PI)) / (2.0 * log(theta));
    double high = (double)head_dim * log((double)orig_len / (beta_slow * TLM_TWO_PI)) / (2.0 * log(theta));
    if (truncate) { low = floor(low); high = ceil(high); }
    if (low < 0.0) low = 0.0;
    if (high > (double)head_dim - 1.0) high = (double)head_dim - 1.0;
    if (low == high) high += 0.001;
    double af = factor > 1.0 ? 0.1 * log(factor) + 1.0 : 1.0;
    for (int j = 0; j < half; j++) {
        float extrap = inv_freq[j];
        float pos_freq = 1.0f / extrap;
        float interp = 1.0f / ((float)factor * pos_freq);
        float ramp = ((float)j - (float)low) / (float)(high - low);
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;
        inv_freq[j] = interp * ramp + extrap * (1.0f - ramp);
    }
    *attn_factor = (float)af;
    return TLM_OK;
}

tlm_model *tlm_model_open(const tlm_manifest *m, tlm_io *io, tlm_store *store, tlm_pool *pool,
                          const tlm_model_opts *o) {
    if (!m || !io) { tlm_seterr(TLM_EINVAL, "model: manifest and io are required"); return NULL; }
    const tlm_model_geom *g = &m->geom;
    if (g->n_layers == 0 || g->d_model == 0 || g->n_heads == 0 || g->n_kv_heads == 0 ||
        g->head_dim == 0 || (g->head_dim & 1) || g->n_heads % g->n_kv_heads || g->vocab == 0 ||
        g->top_k > MAX_TOPK || g->top_k > g->n_experts || (g->n_experts > 0 && g->top_k == 0) ||
        (g->n_experts > 0 && g->d_expert == 0) || g->router > 2 || g->n_experts > 65536 ||
        g->n_layers > 4096 || g->rope_scale > 2) {
        tlm_seterr(TLM_EFORMAT, "model: unsupported geometry (layers %u, d_model %u, heads %u/%u, "
                   "head_dim %u, experts %u, top_k %u, router %u)", g->n_layers, g->d_model,
                   g->n_heads, g->n_kv_heads, g->head_dim, g->n_experts, g->top_k, g->router);
        return NULL;
    }
    if (strncmp(g->arch, "tlm-moe", 7) != 0)
        tlm_log(TLM_LOG_WARN, "model: arch '%s' is run as the generic tlm-moe decoder", g->arch);
    if (g->n_experts > 0 && !store) { tlm_seterr(TLM_EINVAL, "model: an expert store is required"); return NULL; }

    tlm_model *M = (tlm_model *)calloc(1, sizeof *M);
    if (!M) { tlm_seterr(TLM_ENOMEM, "out of memory"); return NULL; }
    M->g = *g;
    if (!(M->g.routed_scale > 0.0f)) M->g.routed_scale = 1.0f;
    if (!(M->g.norm_eps > 0.0f)) M->g.norm_eps = 1e-6f;
    if (!(M->g.rope_theta > 0.0f)) M->g.rope_theta = 10000.0f;
    M->io = io;
    M->store = store;
    M->pool = pool;
    M->o.lookahead = o ? o->lookahead : 1;
    M->o.lookahead_k = o && o->lookahead_k > 0 ? o->lookahead_k : (int)g->top_k;
    if (M->o.lookahead < 0) M->o.lookahead = 0;
    if (M->o.lookahead_k > (int)g->n_experts) M->o.lookahead_k = (int)g->n_experts;
    M->o.interest = o ? o->interest : 0.0;
    if (M->o.interest < 0.0) M->o.interest = 0.0;
    if (M->o.interest > 1.0) M->o.interest = 1.0;

    int rc = TLM_OK;
    size_t nx = (size_t)g->n_layers * (g->n_experts ? g->n_experts : 1);
    M->L = (layer_w *)calloc(g->n_layers, sizeof *M->L);
    M->ex = (expert_info *)calloc(nx, sizeof *M->ex);
    M->mask = (uint8_t *)calloc(nx, 1);
    M->layer_heat = (float *)calloc(g->n_layers, sizeof *M->layer_heat);
    M->instinct = (double *)calloc(g->n_layers, sizeof *M->instinct);
    M->instinct_seen = (int *)calloc(g->n_layers, sizeof *M->instinct_seen);
    M->inv_freq = track(M, g->head_dim / 2);
    if (!M->L || !M->ex || !M->mask || !M->layer_heat || !M->instinct || !M->instinct_seen || !M->inv_freq) {
        rc = tlm_seterr(TLM_ENOMEM, "out of memory");
        goto fail;
    }
    for (uint32_t j = 0; j < g->head_dim / 2; j++)   /* F-fwd-03: built once, in double */
        M->inv_freq[j] = (float)pow((double)M->g.rope_theta, -2.0 * (double)j / (double)g->head_dim);
    M->rope_attn_scale = 1.0f;
    if (M->g.rope_scale != 0) {
        float af;
        if ((rc = tlm_yarn_transform(M->inv_freq, (int)(g->head_dim / 2), g->head_dim,
                                     (double)M->g.rope_theta, (double)M->g.rope_factor,
                                     (double)M->g.rope_beta_fast, (double)M->g.rope_beta_slow,
                                     g->rope_orig_len, M->g.rope_scale == 1, &af)))
            goto fail;
        M->rope_attn_scale = af;
    }
    {   /* given heat per layer ("my training tells me"): mean calibration
         * selection frequency of the layer's experts, straight from the manifest */
        double *sum = (double *)calloc(g->n_layers, sizeof *sum);
        int *cnt = (int *)calloc(g->n_layers, sizeof *cnt);
        if (sum && cnt) {
            for (uint32_t i = 0; i < m->n_packets; i++) {
                const tlm_packet_loc *p = &m->packets[i];
                if (p->type == TLM_PK_EXPERT && p->layer < g->n_layers) {
                    sum[p->layer] += p->heat;
                    cnt[p->layer]++;
                }
            }
            for (uint32_t l = 0; l < g->n_layers; l++)
                M->layer_heat[l] = cnt[l] ? (float)(sum[l] / (double)cnt[l]) : 0.0f;
        }
        free(sum);
        free(cnt);
    }

    for (uint32_t l = 0; l < g->n_layers && rc == TLM_OK; l++) rc = load_layer(M, m, (int)l);
    if (rc) goto fail;

    {   /* META: output_norm + lm_head */
        const tlm_packet_loc *loc = tlm_manifest_find(m, TLM_PK_META, TLM_LAYER_NONE, 0);
        if (!loc) { rc = tlm_seterr(TLM_ENOTFOUND, "manifest has no META packet"); goto fail; }
        const uint8_t *pl;
        uint64_t plen;
        if ((rc = read_packet(M, loc, &M->meta_buf, &pl, &plen))) goto fail;
        if ((rc = dir_vec(M, pl, plen, "output_norm", g->d_model, 1, &M->out_norm))) goto fail;
        if ((rc = dir_mat(pl, plen, "lm_head", g->vocab, g->d_model, 1, &M->lm_head, NULL))) goto fail;
    }
    {   /* EMBED: tok_embd */
        const tlm_packet_loc *loc = tlm_manifest_find(m, TLM_PK_EMBED, TLM_LAYER_NONE, 0);
        if (!loc) { rc = tlm_seterr(TLM_ENOTFOUND, "manifest has no EMBED packet"); goto fail; }
        const uint8_t *pl;
        uint64_t plen;
        if ((rc = read_packet(M, loc, &M->embed_buf, &pl, &plen))) goto fail;
        if ((rc = dir_mat(pl, plen, "tok_embd", g->vocab, g->d_model, 1, &M->embd, NULL))) goto fail;
    }
    if (g->n_experts > 0 && (rc = load_experts(M, m))) goto fail;
    if ((rc = alloc_scratch(M))) { tlm_seterr(TLM_ENOMEM, "out of memory"); goto fail; }
    return M;
fail:
    tlm_model_close(M);
    (void)rc;
    return NULL;
}

void tlm_model_close(tlm_model *M) {
    if (!M) return;
    if (M->L)
        for (uint32_t l = 0; l < M->g.n_layers; l++) tlm_free_aligned(M->L[l].buf);
    tlm_free_aligned(M->meta_buf);
    tlm_free_aligned(M->embed_buf);
    for (int i = 0; i < M->n_owned; i++) tlm_free_aligned(M->owned[i]);
    free(M->owned);
    free(M->L);
    free(M->ex);
    free(M->mask);
    free(M->layer_heat);
    free(M->instinct);
    free(M->instinct_seen);
    free(M);
}

const tlm_model_geom *tlm_model_geom_of(const tlm_model *M) { return M ? &M->g : NULL; }

void tlm_model_set_route_hook(tlm_model *M, tlm_route_hook fn, void *ctx) {
    M->hook = fn;
    M->hook_ctx = ctx;
}

int tlm_model_set_expert_mask(tlm_model *M, const uint8_t *pruned) {
    const tlm_model_geom *g = &M->g;
    for (uint32_t l = 0; l < g->n_layers; l++) {
        if (M->L[l].has_dense_ffn) continue;   /* no experts to mask */
        int live = 0;
        for (uint32_t e = 0; e < g->n_experts; e++) {
            size_t ix = (size_t)l * g->n_experts + e;
            if (M->ex[ix].present && !(pruned && pruned[ix])) live++;
        }
        if (live < (int)g->top_k)
            return tlm_seterr(TLM_EINVAL, "mask leaves layer %u with %d experts (top_k %u)", l, live, g->top_k);
    }
    for (uint32_t l = 0; l < g->n_layers; l++) {
        if (M->L[l].has_dense_ffn) continue;
        for (uint32_t e = 0; e < g->n_experts; e++) {
            size_t ix = (size_t)l * g->n_experts + e;
            M->mask[ix] = (uint8_t)(!M->ex[ix].present || (pruned && pruned[ix]));
        }
    }
    return TLM_OK;
}

const uint8_t *tlm_model_expert_mask(const tlm_model *M) { return M->mask; }

void tlm_model_get_stats(const tlm_model *M, tlm_model_stats *out) { *out = M->st; }
void tlm_model_reset_stats(tlm_model *M) { memset(&M->st, 0, sizeof M->st); }

/* ---- KV cache ---------------------------------------------------------- */
tlm_kv *tlm_kv_new(const tlm_model *M, int max_ctx) {
    const tlm_model_geom *g = &M->g;
    if (max_ctx <= 0) max_ctx = g->max_ctx > 0 ? (int)g->max_ctx : 4096;
    if (g->max_ctx > 0 && max_ctx > (int)g->max_ctx) max_ctx = (int)g->max_ctx;
    tlm_kv *kv = (tlm_kv *)calloc(1, sizeof *kv);
    if (!kv) { tlm_seterr(TLM_ENOMEM, "out of memory"); return NULL; }
    size_t per = (size_t)g->n_layers * g->n_kv_heads * (size_t)max_ctx * g->head_dim;
    kv->mdl = M;
    kv->max_ctx = max_ctx;
    kv->k = (float *)tlm_alloc_aligned(per * sizeof(float), TLM_ALIGN_SIMD);
    kv->v = (float *)tlm_alloc_aligned(per * sizeof(float), TLM_ALIGN_SIMD);
    kv->scores = (float *)tlm_alloc_aligned((size_t)g->n_heads * max_ctx * sizeof(float), TLM_ALIGN_SIMD);
    if (!kv->k || !kv->v || !kv->scores) {
        tlm_kv_free(kv);
        tlm_seterr(TLM_ENOMEM, "out of memory for a %d-token KV cache", max_ctx);
        return NULL;
    }
    return kv;
}
void tlm_kv_free(tlm_kv *kv) {
    if (!kv) return;
    tlm_free_aligned(kv->k);
    tlm_free_aligned(kv->v);
    tlm_free_aligned(kv->scores);
    free(kv);
}
void tlm_kv_reset(tlm_kv *kv) { kv->len = 0; }
int  tlm_kv_len(const tlm_kv *kv) { return kv->len; }
int  tlm_kv_capacity(const tlm_kv *kv) { return kv->max_ctx; }

/* ---- attention (P-61) -------------------------------------------------- */
typedef struct {
    const tlm_model *M;
    tlm_kv *kv;
    int     layer, pos;
} attn_job;

static void attn_heads(void *ctx, int64_t b, int64_t e, int worker) {
    (void)worker;
    const attn_job *j = (const attn_job *)ctx;
    const tlm_model *M = j->M;
    const tlm_model_geom *g = &M->g;
    int hd = (int)g->head_dim, grp = (int)(g->n_heads / g->n_kv_heads), T = j->pos + 1;
    float scale = 1.0f / sqrtf((float)hd);
    for (int64_t h = b; h < e; h++) {
        int kvh = (int)h / grp;
        size_t base = (((size_t)j->layer * g->n_kv_heads + (size_t)kvh) * (size_t)j->kv->max_ctx) * hd;
        const float *K = j->kv->k + base, *V = j->kv->v + base, *q = M->q + h * hd;
        float *s = j->kv->scores + (size_t)h * j->kv->max_ctx;
        float mx = -INFINITY;
        for (int t = 0; t < T; t++) {
            const float *kt = K + (size_t)t * hd;
            float d = 0.0f;
            for (int i = 0; i < hd; i++) d += q[i] * kt[i];
            d *= scale;                       /* scale after the dot, before max-subtract */
            s[t] = d;
            if (d > mx) mx = d;
        }
        double sum = 0.0;
        for (int t = 0; t < T; t++) {
            float p = expf(s[t] - mx);
            s[t] = p;
            sum += (double)p;
        }
        float inv = (float)(1.0 / sum);
        float *out = M->att + h * hd;
        for (int i = 0; i < hd; i++) out[i] = 0.0f;
        for (int t = 0; t < T; t++) {
            const float *vt = V + (size_t)t * hd;
            float p = s[t] * inv;
            for (int i = 0; i < hd; i++) out[i] += p * vt[i];
        }
    }
}

static void rope(float *v, const float *cs, const float *sn, int hd) {
    int half = hd / 2;                     /* NeoX / HF rotate_half pairs (j, j + hd/2) */
    for (int j = 0; j < half; j++) {
        float a = v[j], b = v[j + half];
        v[j] = a * cs[j] - b * sn[j];
        v[j + half] = b * cs[j] + a * sn[j];
    }
}

static void attention(tlm_model *M, tlm_kv *kv, int l, int pos) {
    const tlm_model_geom *g = &M->g;
    layer_w *L = &M->L[l];
    int hd = (int)g->head_dim, H = (int)g->n_heads, KV = (int)g->n_kv_heads;
    rmsnorm(M->xb, M->h, L->attn_norm, g->d_model, M->g.norm_eps);
    actc_set(&M->a_x, M->xb);
    mv_job j;
    j.nseg = 0;
    j.work = 0;
    mv_add(&j, &L->q, actc_get(&M->a_x, L->q.dt), M->q);
    mv_add(&j, &L->k, actc_get(&M->a_x, L->k.dt), M->kx);
    mv_add(&j, &L->v, actc_get(&M->a_x, L->v.dt), M->vx);
    mv_exec(M, &j);
    if (L->q_norm)
        for (int h = 0; h < H; h++) rmsnorm(M->q + h * hd, M->q + h * hd, L->q_norm, hd, M->g.norm_eps);
    if (L->k_norm)
        for (int h = 0; h < KV; h++) rmsnorm(M->kx + h * hd, M->kx + h * hd, L->k_norm, hd, M->g.norm_eps);
    for (int h = 0; h < H; h++) rope(M->q + h * hd, M->cs, M->sn, hd);
    for (int h = 0; h < KV; h++) {
        rope(M->kx + h * hd, M->cs, M->sn, hd);
        size_t off = ((((size_t)l * KV + (size_t)h) * (size_t)kv->max_ctx) + (size_t)pos) * hd;
        memcpy(kv->k + off, M->kx + h * hd, (size_t)hd * sizeof(float));
        memcpy(kv->v + off, M->vx + h * hd, (size_t)hd * sizeof(float));
    }
    attn_job aj = {M, kv, l, pos};
    int64_t work = (int64_t)H * (pos + 1) * hd * 2;
    tlm_pool_for(work < SERIAL_WORK ? NULL : M->pool, H, 1, attn_heads, &aj);
    actc_set(&M->a_att, M->att);
    j.nseg = 0;
    j.work = 0;
    mv_add(&j, &L->o, actc_get(&M->a_att, L->o.dt), M->xb2);
    mv_exec(M, &j);
    for (uint32_t i = 0; i < g->d_model; i++) M->h[i] += M->xb2[i];
}

/* ---- routing ----------------------------------------------------------- */
/* Router logits for layer l on the normalised input held by `a` (the
 * compiler keeps router weights in f32; other dtypes work through `a`). */
static void router_logits(tlm_model *M, int l, actc *a, float *logits) {
    layer_w *L = &M->L[l];
    tlm_matvec_rows(L->router.dt, L->router.data, L->router.row_bytes, actc_get(a, L->router.dt),
                    logits, L->router.cols, 0, L->router.rows, NULL);
}

/* Top-k by value among unmasked entries; ties -> lowest index; NaN skipped
 * (F-fwd-14). Deterministic fallback when nothing finite is left. */
static void topk(const float *v, const uint8_t *mask, int E, int k, int *ids) {
    for (int r = 0; r < k; r++) {
        int best = -1;
        for (int e = 0; e < E; e++) {
            if (mask[e] || v[e] != v[e]) continue;
            int taken = 0;
            for (int q = 0; q < r; q++) if (ids[q] == e) { taken = 1; break; }
            if (taken) continue;
            if (best < 0 || v[e] > v[best]) best = e;
        }
        if (best < 0)   /* everything NaN: lowest unmasked unused index */
            for (int e = 0; e < E && best < 0; e++) {
                if (mask[e]) continue;
                int taken = 0;
                for (int q = 0; q < r; q++) if (ids[q] == e) { taken = 1; break; }
                if (!taken) best = e;
            }
        ids[r] = best < 0 ? 0 : best;
    }
}

/* Router variants (P-71 / P-73). `logits` is overwritten with scratch values. */
static void route(tlm_model *M, int l, float *logits, int k, int *ids, float *w) {
    const tlm_model_geom *g = &M->g;
    int E = (int)g->n_experts;
    const uint8_t *mask = M->mask + (size_t)l * g->n_experts;
    float *p = M->rp;
    if (g->router == 2) {
        const float *bias = M->L[l].router_bias;
        for (int e = 0; e < E; e++) {
            p[e] = sigmoidf_(logits[e]);                     /* unbiased score = weight */
            logits[e] = p[e] + (bias ? bias[e] : 0.0f);      /* biased value = selection only */
        }
        topk(logits, mask, E, k, ids);
        double sum = 0.0;
        for (int r = 0; r < k; r++) sum += (double)p[ids[r]];
        float inv = (float)(1.0 / (sum + 1e-20));
        for (int r = 0; r < k; r++) w[r] = p[ids[r]] * inv * g->routed_scale;   /* scale after renorm */
        return;
    }
    /* softmax over the routable experts (pruned ones are excluded), then top-k */
    float mx = -INFINITY;
    for (int e = 0; e < E; e++)
        if (!mask[e] && logits[e] > mx) mx = logits[e];
    double sum = 0.0;
    for (int e = 0; e < E; e++) {
        p[e] = (mask[e] || !(logits[e] == logits[e]) || mx == -INFINITY) ? 0.0f : expf(logits[e] - mx);
        sum += (double)p[e];
    }
    float inv = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;
    for (int e = 0; e < E; e++) p[e] *= inv;
    topk(p, mask, E, k, ids);
    for (int r = 0; r < k; r++) w[r] = p[ids[r]];
    if (g->router == 0) {
        double s = 0.0;
        for (int r = 0; r < k; r++) s += (double)w[r];
        if (s > 0.0) {
            float is = (float)(1.0 / s);
            for (int r = 0; r < k; r++) w[r] *= is;
        }
    }
    if (g->routed_scale != 1.0f)
        for (int r = 0; r < k; r++) w[r] *= g->routed_scale;
}

/* Router lookahead (P-33): predict layer `l`'s experts from the current
 * residual stream and hint the store. Uses its own scratch; never changes
 * the forward results. */
static void lookahead(tlm_model *M, int l) {
    const tlm_model_geom *g = &M->g;
    if (!M->store || l >= (int)g->n_layers || M->o.lookahead <= 0 || g->n_experts == 0) return;
    if (M->L[l].has_dense_ffn) return;   /* no router to predict */
    int k = M->o.lookahead_k, ids[MAX_TOPK * 4];
    float w[MAX_TOPK * 4];
    if (k > MAX_TOPK * 4) k = MAX_TOPK * 4;
    int live = 0;
    for (uint32_t e = 0; e < g->n_experts; e++) live += !M->mask[(size_t)l * g->n_experts + e];
    if (k > live) k = live;
    rmsnorm(M->xb2, M->h, M->L[l].ffn_norm, g->d_model, g->norm_eps);
    actc_set(&M->a_la, M->xb2);
    router_logits(M, l, &M->a_la, M->rl);
    route(M, l, M->rl, k, ids, w);
    /* Three-term navigation weight (docs/design/navigation.md):
     * W = alpha * (1 - I) * (1 - H). alpha = "this question needs"
     * (nav.interest_aggressiveness); I = instinct vector (learned + current
     * peakedness); H = model heat (given). alpha == 0 is the legacy fixed-
     * breadth path. Prefetch breadth never changes output (P-34). */
    if (M->o.interest > 0.0 && k > (int)g->top_k) {
        double s = 0.0;
        for (int r = 0; r < k; r++) s += (double)w[r];
        double now = s > 0.0 ? (double)w[0] / s : 0.0;
        double I = (M->instinct_seen && M->instinct_seen[l]) ? 0.5 * (now + M->instinct[l]) : now;
        double H = M->layer_heat ? (double)M->layer_heat[l] : 0.0;
        double W = M->o.interest * (1.0 - I) * (1.0 - H);
        if (W < 0.0) W = 0.0;
        if (W > 1.0) W = 1.0;
        int span = k - (int)g->top_k;
        k = (int)g->top_k + (int)(W * (double)span + 0.5);
        if (k > (int)g->n_experts) k = (int)g->n_experts;
    }
    tlm_store_prefetch(M->store, l, ids, k);
}

/* ---- MoE block --------------------------------------------------------- */
static void expert_mats(const tlm_model *M, int l, int e, const tlm_expert_ref *ref, wmat *gw,
                        wmat *uw, wmat *dw) {
    const tlm_model_geom *g = &M->g;
    const expert_info *x = &M->ex[(size_t)l * g->n_experts + e];
    tlm_dtype dt = (tlm_dtype)x->dtype;
    const uint8_t *pl = (const uint8_t *)ref->payload;
    gw->dt = uw->dt = dw->dt = dt;
    gw->rows = uw->rows = g->d_expert;
    gw->cols = uw->cols = g->d_model;
    gw->row_bytes = uw->row_bytes = tlm_row_bytes(dt, g->d_model);
    dw->rows = g->d_model;
    dw->cols = g->d_expert;
    dw->row_bytes = tlm_row_bytes(dt, g->d_expert);
    gw->data = pl + x->off[0];
    uw->data = pl + x->off[1];
    dw->data = pl + x->off[2];
}

/* Compute experts `which[0..n)` (rank indices) and, if `shared`, the shared
 * expert: gate/up for all in one pool call, then down for all in another. */
static void run_batch(tlm_model *M, int l, const int *ids, const tlm_expert_ref *refs,
                      const int *which, int n, int shared) {
    const tlm_model_geom *g = &M->g;
    layer_w *L = &M->L[l];
    int64_t de = g->d_expert, ds = g->d_shared;
    wmat gw[MAX_TOPK], uw[MAX_TOPK], dw[MAX_TOPK];
    mv_job j;
    j.nseg = 0;
    j.work = 0;
    for (int q = 0; q < n; q++) {
        int r = which[q];
        expert_mats(M, l, ids[r], &refs[r], &gw[q], &uw[q], &dw[q]);
        const void *a = actc_get(&M->a_ffn, gw[q].dt);
        mv_add(&j, &gw[q], a, M->eg[r]);
        mv_add(&j, &uw[q], a, M->eg[r] + de);
    }
    if (shared) {
        const void *a = actc_get(&M->a_ffn, L->sg.dt);
        mv_add(&j, &L->sg, a, M->sg_buf);
        mv_add(&j, &L->su, a, M->sg_buf + ds);
    }
    mv_exec(M, &j);
    j.nseg = 0;
    j.work = 0;
    for (int q = 0; q < n; q++) {
        int r = which[q];
        float *gg = M->eg[r], *uu = M->eg[r] + de, *hh = M->eh[r];
        for (int64_t i = 0; i < de; i++) hh[i] = silu(gg[i]) * uu[i];
        actc_set(&M->a_e[r], hh);
        mv_add(&j, &dw[q], actc_get(&M->a_e[r], dw[q].dt), M->eo[r]);
    }
    if (shared) {
        float *gg = M->sg_buf, *uu = M->sg_buf + ds;
        for (int64_t i = 0; i < ds; i++) M->sh_buf[i] = silu(gg[i]) * uu[i];
        actc_set(&M->a_sh, M->sh_buf);
        mv_add(&j, &L->sd, actc_get(&M->a_sh, L->sd.dt), M->s_out);
    }
    mv_exec(M, &j);
}

static int moe(tlm_model *M, int l, int pos, int token) {
    const tlm_model_geom *g = &M->g;
    layer_w *L = &M->L[l];
    int D = (int)g->d_model, k = (int)g->top_k;
    rmsnorm(M->xb, M->h, L->ffn_norm, D, g->norm_eps);
    actc_set(&M->a_ffn, M->xb);
    if (L->has_dense_ffn) {
        /* Resident SwiGLU block: gate/up in one pool pass, down in another,
         * added to the residual (F-fwd-18, the expert FFN with one expert). */
        int64_t dffn = L->d_ffn;
        mv_job j;
        j.nseg = 0;
        j.work = 0;
        mv_add(&j, &L->dg, actc_get(&M->a_ffn, L->dg.dt), M->dg_buf);
        mv_add(&j, &L->du, actc_get(&M->a_ffn, L->du.dt), M->dg_buf + dffn);
        mv_exec(M, &j);
        float *gg = M->dg_buf, *uu = M->dg_buf + dffn;
        for (int64_t i = 0; i < dffn; i++) M->dh_buf[i] = silu(gg[i]) * uu[i];
        actc_set(&M->a_dh, M->dh_buf);
        j.nseg = 0;
        j.work = 0;
        mv_add(&j, &L->dd, actc_get(&M->a_dh, L->dd.dt), M->ffn);
        mv_exec(M, &j);
        for (int i = 0; i < D; i++) M->h[i] += M->ffn[i];
        return TLM_OK;
    }
    int ids[MAX_TOPK];
    float w[MAX_TOPK], onorm[MAX_TOPK];
    tlm_expert_ref refs[MAX_TOPK];
    int rc = TLM_OK;

    if (g->n_experts > 0) {
        router_logits(M, l, &M->a_ffn, M->rl);
        route(M, l, M->rl, k, ids, w);
        if (M->instinct_seen) {
            double s = 0.0;
            for (int r = 0; r < k; r++) s += (double)w[r];
            double share = s > 0.0 ? (double)w[0] / s : 0.0;
            M->instinct[l] += (share - M->instinct[l]) * 0.05;
            M->instinct_seen[l]++;
        }
        memset(refs, 0, sizeof refs);
        for (int r = 0; r < k; r++) refs[r].slot = -1;
        rc = tlm_store_acquire(M->store, l, ids, k, refs);
        if (rc) {   /* whatever was referenced must still be released */
            for (int r = 0; r < k; r++) tlm_store_release(M->store, &refs[r]);
            return rc;
        }
        /* P-33 router lookahead for the next layer(s), issued only now that
         * this layer's experts are referenced, so a hint can never take
         * their slots (P-34). Uses the post-attention residual of layer l. */
        for (int d = 1; d <= M->o.lookahead; d++) lookahead(M, l + d);
    }
    /* Compute whatever is resident first (and the shared expert), then wait
     * for the rest in rank order; the combine below is always in rank order. */
    int done[MAX_TOPK] = {0}, ndone = 0, first = 1;
    while (rc == TLM_OK && (ndone < (g->n_experts > 0 ? k : 0) || first)) {
        int which[MAX_TOPK], n = 0;
        for (int r = 0; r < k && g->n_experts > 0; r++)
            if (!done[r] && tlm_store_is_ready(M->store, &refs[r])) which[n++] = r;
        if (n == 0 && !(first && L->has_shared) && ndone < k) {
            int r = 0;
            while (done[r]) r++;
            uint64_t t0 = tlm_now_ns();
            rc = tlm_store_wait(M->store, &refs[r]);
            M->st.expert_wait_ns += tlm_now_ns() - t0;
            if (rc) break;
            which[n++] = r;
            for (int q = r + 1; q < k; q++)
                if (!done[q] && tlm_store_is_ready(M->store, &refs[q])) which[n++] = q;
        }
        for (int q = 0; q < n && rc == TLM_OK; q++) {
            tlm_expert_ref *ref = &refs[which[q]];
            /* a miss that completed is READY but its ref carries the payload
             * pointer only after tlm_store_wait (immediate here) */
            if (!ref->payload) rc = tlm_store_wait(M->store, ref);
            if (rc) break;
            const expert_info *x = &M->ex[(size_t)l * g->n_experts + ids[which[q]]];
            if (!ref->payload || ref->payload_len < x->payload_len)
                rc = tlm_seterr(TLM_EFORMAT, "expert %d:%d payload is %llu bytes, expected %llu", l,
                                ids[which[q]], (unsigned long long)ref->payload_len,
                                (unsigned long long)x->payload_len);
        }
        if (rc) break;
        run_batch(M, l, ids, refs, which, n, first && L->has_shared);
        for (int q = 0; q < n; q++) done[which[q]] = 1;
        ndone += n;
        first = 0;
    }
    if (rc == TLM_OK) {
        for (int i = 0; i < D; i++) M->ffn[i] = 0.0f;
        for (int r = 0; r < k && g->n_experts > 0; r++) {   /* ascending rank order (C-13) */
            const float *eo = M->eo[r];
            double nn = 0.0;
            for (int i = 0; i < D; i++) {
                float c = w[r] * eo[i];
                M->ffn[i] += c;
                nn += (double)c * (double)c;
            }
            onorm[r] = (float)sqrt(nn);
        }
        if (L->has_shared) {
            float sgate = 1.0f;
            if (L->shared_gate_inp) {
                float d = 0.0f;
                for (int i = 0; i < D; i++) d += L->shared_gate_inp[i] * M->xb[i];
                sgate = sigmoidf_(d);
            }
            for (int i = 0; i < D; i++) M->ffn[i] += sgate * M->s_out[i];
        }
        for (int i = 0; i < D; i++) M->h[i] += M->ffn[i];
    }
    for (int r = 0; r < k && g->n_experts > 0; r++) tlm_store_release(M->store, &refs[r]);
    if (rc == TLM_OK && M->hook && g->n_experts > 0) M->hook(M->hook_ctx, l, pos, token, ids, w, onorm, k);
    return rc;
}

/* ---- step / prefill ---------------------------------------------------- */
int tlm_model_step(tlm_model *M, tlm_kv *kv, int token, float *logits) {
    const tlm_model_geom *g = &M->g;
    if (token < 0 || (uint32_t)token >= g->vocab) return tlm_seterr(TLM_EINVAL, "token %d out of range", token);
    if (kv->len >= kv->max_ctx) return tlm_seterr(TLM_ERANGE, "context full (%d tokens)", kv->max_ctx);
    int pos = kv->len, D = (int)g->d_model, hd = (int)g->head_dim;
    uint64_t t0 = tlm_now_ns(), t_attn = 0, t_moe = 0;

    const uint8_t *row = M->embd.data + (size_t)token * M->embd.row_bytes;   /* F-fwd-01: plain gather */
    if (M->embd.dt == TLM_DT_F32) memcpy(M->h, row, (size_t)D * sizeof(float));
    else if (tlm_dequantize_row(M->embd.dt, row, M->h, D) != 0)
        return tlm_seterr(TLM_EUNSUP, "cannot dequantise embedding dtype %d", (int)M->embd.dt);
    for (int j = 0; j < hd / 2; j++) {
        double a = (double)pos * (double)M->inv_freq[j];
        M->cs[j] = (float)cos(a) * M->rope_attn_scale;
        M->sn[j] = (float)sin(a) * M->rope_attn_scale;
    }
    if (M->o.lookahead > 0) lookahead(M, 0);
    int rc = TLM_OK;
    for (int l = 0; l < (int)g->n_layers && rc == TLM_OK; l++) {
        uint64_t a0 = tlm_now_ns();
        attention(M, kv, l, pos);
        uint64_t a1 = tlm_now_ns();
        t_attn += a1 - a0;
        uint64_t m0 = tlm_now_ns();
        rc = moe(M, l, pos, token);
        t_moe += tlm_now_ns() - m0;
    }
    if (rc) return rc;
    kv->len = pos + 1;
    if (logits) {
        rmsnorm(M->xb, M->h, M->out_norm, D, g->norm_eps);
        actc_set(&M->a_fin, M->xb);
        mv_job j;
        j.nseg = 0;
        j.work = 0;
        mv_add(&j, &M->lm_head, actc_get(&M->a_fin, M->lm_head.dt), logits);
        mv_exec(M, &j);
    }
    if (M->store) tlm_store_tick(M->store);
    uint64_t tot = tlm_now_ns() - t0;
    M->st.tokens++;
    M->st.attn_ns += t_attn;
    M->st.moe_ns += t_moe;
    M->st.other_ns += tot > t_attn + t_moe ? tot - t_attn - t_moe : 0;
    return TLM_OK;
}

int tlm_model_prefill(tlm_model *M, tlm_kv *kv, const int *tokens, int n, float *logits) {
    if (n <= 0) return tlm_seterr(TLM_EINVAL, "prefill: no tokens");
    /* P-92: prefill is the decode step repeated, so prefill == step-by-step bit for bit */
    for (int i = 0; i < n; i++) {
        int rc = tlm_model_step(M, kv, tokens[i], i == n - 1 ? logits : NULL);
        if (rc) return rc;
    }
    return TLM_OK;
}

/* ---- sampler (P-94) ---------------------------------------------------- */
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static uint64_t xorshift64s(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static int argmax(const float *l, int n) {
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (l[i] != l[i]) continue;                 /* NaN skipped */
        if (best < 0 || l[i] > l[best]) best = i;   /* first max wins */
    }
    return best < 0 ? 0 : best;
}

typedef struct { double p; int id; } pid;
static int pid_cmp(const void *a, const void *b) {
    const pid *x = (const pid *)a, *y = (const pid *)b;
    if (x->p != y->p) return x->p > y->p ? -1 : 1;
    return x->id - y->id;
}

static int pid_id_cmp(const void *a, const void *b) {
    return ((const pid *)a)->id - ((const pid *)b)->id;
}

int tlm_sample(const tlm_sampler *s, uint64_t *rng, const float *logits, int vocab) {
    if (vocab <= 0) return 0;
    if (!s || !(s->temperature > 0.0f)) return argmax(logits, vocab);
    if (*rng == 0) {
        uint64_t seed = s->seed;
        *rng = splitmix64(&seed) | 1u;
    }
    float T = s->temperature, m = -INFINITY;
    for (int i = 0; i < vocab; i++)
        if (isfinite(logits[i]) && logits[i] > m) m = logits[i];
    if (m == -INFINITY) return argmax(logits, vocab);
    float floor_l = -INFINITY;
    if (s->min_p > 0.0f && s->min_p < 1.0f) floor_l = m + T * logf(s->min_p);   /* min-p in logit space */
    pid *v = (pid *)malloc((size_t)vocab * sizeof *v);
    if (!v) return argmax(logits, vocab);
    double sum = 0.0;
    int n = 0;
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(logits[i]) || logits[i] < floor_l) continue;
        double p = exp((double)(logits[i] - m) / (double)T);
        v[n].p = p;
        v[n].id = i;
        sum += p;
        n++;
    }
    if (!(sum > 0.0) || !isfinite(sum)) {
        free(v);
        static int warned;
        if (!warned) { warned = 1; tlm_log(TLM_LOG_WARN, "sampler: degenerate distribution, using argmax"); }
        return argmax(logits, vocab);
    }
    if (s->top_p > 0.0f && s->top_p < 1.0f) {
        qsort(v, (size_t)n, sizeof *v, pid_cmp);
        double cum = 0.0, lim = (double)s->top_p * sum;
        int keep = 0;
        while (keep < n) {
            cum += v[keep].p;
            keep++;
            if (cum >= lim) break;
        }
        n = keep;
        sum = cum;
        qsort(v, (size_t)n, sizeof *v, pid_id_cmp);   /* walk in id order, as without top-p */
    }
    double u = (double)(xorshift64s(rng) >> 11) * (1.0 / 9007199254740992.0) * sum;
    double c = 0.0;
    int pick = v[n - 1].id;
    for (int i = 0; i < n; i++) {
        c += v[i].p;
        if (u < c) { pick = v[i].id; break; }
    }
    free(v);
    return pick;
}

int tlm_generate(tlm_model *M, tlm_kv *kv, const int *prompt, int n_prompt, int max_new,
                 const tlm_sampler *s, const int *stop, int n_stop, tlm_token_fn on_token, void *ctx) {
    int V = (int)M->g.vocab;
    float *logits = (float *)tlm_alloc_aligned((size_t)V * sizeof(float), TLM_ALIGN_SIMD);
    if (!logits) return tlm_seterr(TLM_ENOMEM, "out of memory");
    uint64_t rng = 0;
    int count = 0, rc = TLM_OK;
    if (n_prompt > 0) rc = tlm_model_prefill(M, kv, prompt, n_prompt, logits);
    else rc = tlm_seterr(TLM_EINVAL, "generate: empty prompt");
    while (rc == TLM_OK && count < max_new) {
        int tok = tlm_sample(s, &rng, logits, V);
        count++;
        int halt = on_token ? on_token(ctx, tok) : 0;
        for (int i = 0; i < n_stop; i++) if (stop[i] == tok) halt = 1;
        if (halt || count >= max_new) break;
        if (tlm_kv_len(kv) >= kv->max_ctx) break;
        rc = tlm_model_step(M, kv, tok, logits);
    }
    tlm_free_aligned(logits);
    return rc ? rc : count;
}
