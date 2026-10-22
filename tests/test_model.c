/* test_model.c - the generic MoE decoder over a compiled packet store.
 *
 * - An all-F32 tiny model is checked against an independent naive forward
 *   pass written here (RMSNorm, per-head qk-norm, NeoX RoPE, GQA, the three
 *   routers, SwiGLU experts + shared expert): P-79/P-80/P-81/P-61/P-71/P-73/P-75.
 * - A quantised model (Q8_0 dense, mixed Q6_K/Q8_0/Q4_K experts, 3 drives,
 *   replicas) must give bit-identical logits and greedy tokens across thread
 *   counts, cache sizes (minimal vs fully resident), prefetch on/off and the
 *   scalar/AVX2/AVX-512 kernel tiers (P-43, P-34, C-13).
 * - prefill == step-by-step (P-92), route hook, expert mask / pruned
 *   packets, sampler (P-94), generate.
 */
#include "tlm_io.h"
#include "tlm_model.h"
#include "tlm_packet.h"
#include "tlm_platform.h"
#include "tlm_pool.h"
#include "tlm_quant.h"
#include "tlm_store.h"
#include "tlm_test.h"

#include <ftw.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

static char g_root[160];

static int rm_cb(const char *p, const struct stat *sb, int flag, struct FTW *ftw) {
    (void)sb; (void)flag; (void)ftw;
    return remove(p);
}
static void rm_rf(const char *p) { nftw(p, rm_cb, 16, FTW_DEPTH | FTW_PHYS); }

/* ---- synthetic model -------------------------------------------------------- */
typedef struct {
    int L, D, H, KV, hd, V, E, K, de, ds, router, qk_norm, n_drives, replicas;
    int n_dense, d_ffn;        /* first n_dense layers are dense-FFN (width d_ffn) */
    float routed_scale;
    tlm_dtype dense_dt;
    tlm_dtype expert_dt[3];   /* expert e uses expert_dt[e % 3] */
    uint32_t prune_mask;      /* bit e: expert e of layer 0 flagged pruned in the manifest */
    unsigned seed;
    uint32_t rope_scale;      /* 0 = plain, 1 = YaRN truncate, 2 = YaRN no-truncate */
    float rope_factor, rope_beta_fast, rope_beta_slow;
    uint32_t rope_orig_len;
} spec;

typedef struct {   /* f32 copies (dequantised) for the reference */
    float *attn_norm, *ffn_norm, *q_norm, *k_norm, *q, *k, *v, *o, *router, *rbias;
    float *sg, *su, *sd;
    float *fg, *fu, *fd;      /* dense FFN: [d_ffn*D], [d_ffn*D], [D*d_ffn] */
    float *eg, *eu, *ed;     /* [E][de*D], [E][de*D], [E][D*de] */
} ref_layer;

typedef struct {
    spec s;
    ref_layer *Lw;
    float *out_norm, *lm_head, *embd;
    char mpath[300];
    char name[32];
    uint64_t max_expert_total;
    int n_expert_packets;
} synth;

static float frand(float a) { return tlm_test_randf(-a, a); }
static float *rnd(size_t n, float a, float base) {
    float *p = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) p[i] = base + frand(a);
    return p;
}

/* Quantise rows (rows x cols) into dt; replace f32 copy by its dequantised
 * value so the reference sees exactly what the engine sees. */
static uint8_t *quant_mat(tlm_dtype dt, float *w, int64_t rows, int64_t cols, uint64_t *nbytes) {
    size_t rb = tlm_row_bytes(dt, cols);
    uint8_t *q = (uint8_t *)calloc((size_t)rows, rb);
    for (int64_t r = 0; r < rows; r++) {
        tlm_quantize_row(dt, w + r * cols, q + (size_t)r * rb, cols);
        if (dt != TLM_DT_F32) tlm_dequantize_row(dt, q + (size_t)r * rb, w + r * cols, cols);
    }
    *nbytes = (uint64_t)rows * rb;
    return q;
}

typedef struct { tlm_tensor_in t[24]; uint8_t *own[24]; int n; } dirb;
static void dir_add(dirb *d, const char *name, tlm_dtype dt, float *w, int64_t rows, int64_t cols) {
    uint64_t nb;
    uint8_t *q = quant_mat(dt, w, rows, cols, &nb);
    d->t[d->n] = (tlm_tensor_in){name, dt, (uint64_t)rows, (uint64_t)cols, q, nb};
    d->own[d->n] = q;
    d->n++;
}

static int build(synth *S, const spec *sp, const char *name) {
    memset(S, 0, sizeof *S);
    S->s = *sp;
    tlm_test_rng = 0x9E3779B97F4A7C15ull ^ ((unsigned long long)sp->seed * 0x2545F4914F6CDD1Dull);
    snprintf(S->name, sizeof S->name, "%s", name);
    const int L = sp->L, D = sp->D, H = sp->H, KV = sp->KV, hd = sp->hd, V = sp->V, E = sp->E,
              de = sp->de, ds = sp->ds;
    char dirs[TLM_MAX_DRIVES][256];
    const char *dp[TLM_MAX_DRIVES];
    for (int i = 0; i < sp->n_drives; i++) {
        snprintf(dirs[i], sizeof dirs[i], "%s/%s-d%d", g_root, name, i);
        dp[i] = dirs[i];
    }
    tlm_store_writer *w = tlm_writer_open(name, dp, sp->n_drives);
    if (!w) return -1;
    tlm_packet_loc *locs = (tlm_packet_loc *)calloc((size_t)(L * (E + 1) + 2), sizeof *locs);
    int nl = 0, rc = 0;
    S->Lw = (ref_layer *)calloc((size_t)L, sizeof *S->Lw);
    float sd_in = 1.0f / sqrtf((float)D);
    for (int l = 0; l < L && !rc; l++) {
        ref_layer *R = &S->Lw[l];
        R->attn_norm = rnd((size_t)D, 0.2f, 1.0f);
        R->ffn_norm = rnd((size_t)D, 0.2f, 1.0f);
        R->q = rnd((size_t)H * hd * D, 1.7f * sd_in, 0);
        R->k = rnd((size_t)KV * hd * D, 1.7f * sd_in, 0);
        R->v = rnd((size_t)KV * hd * D, 1.7f * sd_in, 0);
        R->o = rnd((size_t)D * H * hd, 0.8f / sqrtf((float)(H * hd)), 0);
        R->router = rnd((size_t)E * D, 3.0f * sd_in, 0);
        dirb d = {0};
        dir_add(&d, "attn_norm", TLM_DT_F32, R->attn_norm, 1, D);
        dir_add(&d, "ffn_norm", TLM_DT_F32, R->ffn_norm, 1, D);
        if (sp->qk_norm) {
            R->q_norm = rnd((size_t)hd, 0.3f, 1.0f);
            R->k_norm = rnd((size_t)hd, 0.3f, 1.0f);
            dir_add(&d, "q_norm", TLM_DT_F32, R->q_norm, 1, hd);
            dir_add(&d, "k_norm", TLM_DT_F32, R->k_norm, 1, hd);
        }
        dir_add(&d, "attn_q", sp->dense_dt, R->q, H * hd, D);
        dir_add(&d, "attn_k", sp->dense_dt, R->k, KV * hd, D);
        dir_add(&d, "attn_v", sp->dense_dt, R->v, KV * hd, D);
        dir_add(&d, "attn_o", sp->dense_dt, R->o, D, H * hd);
        if (l < sp->n_dense) {
            /* Dense-FFN layer: resident SwiGLU block, no router or experts */
            R->fg = rnd((size_t)sp->d_ffn * D, 1.7f * sd_in, 0);
            R->fu = rnd((size_t)sp->d_ffn * D, 1.7f * sd_in, 0);
            R->fd = rnd((size_t)D * sp->d_ffn, 0.8f / sqrtf((float)sp->d_ffn), 0);
            dir_add(&d, "ffn_gate", sp->dense_dt, R->fg, sp->d_ffn, D);
            dir_add(&d, "ffn_up", sp->dense_dt, R->fu, sp->d_ffn, D);
            dir_add(&d, "ffn_down", sp->dense_dt, R->fd, D, sp->d_ffn);
        } else {
            dir_add(&d, "router", TLM_DT_F32, R->router, E, D);
            if (sp->router == 2) {
                R->rbias = rnd((size_t)E, 0.3f, 0);
                dir_add(&d, "router_bias", TLM_DT_F32, R->rbias, 1, E);
            }
            if (ds > 0) {
                R->sg = rnd((size_t)ds * D, 1.7f * sd_in, 0);
                R->su = rnd((size_t)ds * D, 1.7f * sd_in, 0);
                R->sd = rnd((size_t)D * ds, 0.8f / sqrtf((float)ds), 0);
                dir_add(&d, "shared_gate", sp->dense_dt, R->sg, ds, D);
                dir_add(&d, "shared_up", sp->dense_dt, R->su, ds, D);
                dir_add(&d, "shared_down", sp->dense_dt, R->sd, D, ds);
            }
        }
        void *pl;
        uint64_t plen;
        rc = tlm_dir_build(d.t, d.n, &pl, &plen);
        for (int i = 0; i < d.n; i++) free(d.own[i]);
        if (rc) break;
        rc = tlm_writer_add(w, TLM_PK_DENSE, (uint16_t)l, 0, sp->dense_dt, pl, plen, NULL,
                            l % sp->n_drives, -1, &locs[nl++]);
        tlm_free_aligned(pl);
        if (l < sp->n_dense) continue;   /* dense layer: DENSE packet only */
        /* experts: gate | up | down, 64-byte aligned sub-offsets */
        R->eg = (float *)malloc(sizeof(float) * (size_t)E * de * D);
        R->eu = (float *)malloc(sizeof(float) * (size_t)E * de * D);
        R->ed = (float *)malloc(sizeof(float) * (size_t)E * D * de);
        for (int e = 0; e < E && !rc; e++) {
            tlm_dtype dt = sp->expert_dt[e % 3];
            float *g = R->eg + (size_t)e * de * D, *u = R->eu + (size_t)e * de * D,
                  *dn = R->ed + (size_t)e * D * de;
            for (size_t i = 0; i < (size_t)de * D; i++) g[i] = frand(1.7f * sd_in);
            for (size_t i = 0; i < (size_t)de * D; i++) u[i] = frand(1.7f * sd_in);
            for (size_t i = 0; i < (size_t)D * de; i++) dn[i] = frand(1.0f / sqrtf((float)de));
            uint64_t nbg, nbu, nbd;
            uint8_t *qg = quant_mat(dt, g, de, D, &nbg), *qu = quant_mat(dt, u, de, D, &nbu),
                    *qd = quant_mat(dt, dn, D, de, &nbd);
            uint32_t off[3];
            off[0] = 0;
            off[1] = (uint32_t)TLM_ALIGN_UP(nbg, 64);
            off[2] = (uint32_t)TLM_ALIGN_UP(off[1] + nbu, 64);
            uint64_t len = off[2] + nbd;
            uint8_t *pp = (uint8_t *)calloc(1, (size_t)len);
            memcpy(pp + off[0], qg, nbg);
            memcpy(pp + off[1], qu, nbu);
            memcpy(pp + off[2], qd, nbd);
            int drive = (l * E + e) % sp->n_drives;
            int rep = (sp->replicas && sp->n_drives > 1 && e % 2 == 0) ? (drive + 1) % sp->n_drives : -1;
            tlm_packet_loc *lc = &locs[nl++];
            rc = tlm_writer_add(w, TLM_PK_EXPERT, (uint16_t)l, (uint32_t)e, dt, pp, len, off, drive, rep, lc);
            lc->heat = (float)(E - e) / (float)E;
            if (l == 0 && (sp->prune_mask >> e) & 1u) lc->flags |= 2u;
            if (e == 1) lc->flags |= 1u;   /* compiler pin hint */
            if (lc->total_len > S->max_expert_total) S->max_expert_total = lc->total_len;
            S->n_expert_packets++;
            free(pp);
            free(qg);
            free(qu);
            free(qd);
        }
    }
    if (!rc) {   /* META + EMBED */
        S->out_norm = rnd((size_t)D, 0.2f, 1.0f);
        S->lm_head = rnd((size_t)V * D, 2.0f * sd_in, 0);
        S->embd = rnd((size_t)V * D, 1.0f, 0);
        dirb d = {0};
        dir_add(&d, "output_norm", TLM_DT_F32, S->out_norm, 1, D);
        dir_add(&d, "lm_head", sp->dense_dt, S->lm_head, V, D);
        void *pl;
        uint64_t plen;
        rc = tlm_dir_build(d.t, d.n, &pl, &plen);
        for (int i = 0; i < d.n; i++) free(d.own[i]);
        if (!rc) rc = tlm_writer_add(w, TLM_PK_META, TLM_LAYER_NONE, 0, sp->dense_dt, pl, plen, NULL, 0, -1,
                                     &locs[nl++]);
        tlm_free_aligned(pl);
        dirb d2 = {0};
        dir_add(&d2, "tok_embd", sp->dense_dt, S->embd, V, D);
        if (!rc) rc = tlm_dir_build(d2.t, d2.n, &pl, &plen);
        for (int i = 0; i < d2.n; i++) free(d2.own[i]);
        if (!rc) rc = tlm_writer_add(w, TLM_PK_EMBED, TLM_LAYER_NONE, 0, sp->dense_dt, pl, plen, NULL,
                                     sp->n_drives - 1, -1, &locs[nl++]);
        if (!rc) tlm_free_aligned(pl);
    }
    if (!rc) rc = tlm_writer_finish(w, 1);
    tlm_writer_free(w);
    if (rc) { free(locs); return rc; }
    tlm_manifest m;
    memset(&m, 0, sizeof m);
    m.magic = TLM_MANIFEST_MAGIC;
    m.version = TLM_STORE_VERSION;
    snprintf(m.name, sizeof m.name, "%s", name);
    snprintf(m.profile, sizeof m.profile, "exact");
    tlm_model_geom *g = &m.geom;
    snprintf(g->arch, sizeof g->arch, "tlm-moe");
    g->n_layers = (uint32_t)L; g->d_model = (uint32_t)D; g->n_heads = (uint32_t)H;
    g->n_kv_heads = (uint32_t)KV; g->head_dim = (uint32_t)hd; g->vocab = (uint32_t)V;
    g->n_experts = (uint32_t)E; g->top_k = (uint32_t)sp->K; g->d_expert = (uint32_t)de;
    g->n_shared = ds > 0 ? 1 : 0; g->d_shared = (uint32_t)ds; g->max_ctx = 256;
    g->router = (uint32_t)sp->router; g->rope_theta = 10000.0f; g->norm_eps = 1e-6f;
    g->routed_scale = sp->routed_scale;
    g->rope_scale = sp->rope_scale; g->rope_factor = sp->rope_factor;
    g->rope_beta_fast = sp->rope_beta_fast; g->rope_beta_slow = sp->rope_beta_slow;
    g->rope_orig_len = sp->rope_orig_len;
    m.n_drives = (uint32_t)sp->n_drives;
    for (int i = 0; i < sp->n_drives; i++) snprintf(m.drives[i], sizeof m.drives[i], "%s", dirs[i]);
    m.n_packets = (uint32_t)nl;
    m.packets = locs;
    for (int i = 0; i < nl; i++) if (locs[i].flags & 2u) m.n_pruned++;
    snprintf(S->mpath, sizeof S->mpath, "%s/%s.tlmm", g_root, name);
    rc = tlm_manifest_save(&m, S->mpath);
    free(locs);
    return rc;
}

static void synth_free(synth *S) {
    for (int l = 0; l < S->s.L && S->Lw; l++) {
        ref_layer *R = &S->Lw[l];
        free(R->attn_norm); free(R->ffn_norm); free(R->q_norm); free(R->k_norm);
        free(R->q); free(R->k); free(R->v); free(R->o); free(R->router); free(R->rbias);
        free(R->sg); free(R->su); free(R->sd); free(R->fg); free(R->fu); free(R->fd);
        free(R->eg); free(R->eu); free(R->ed);
    }
    free(S->Lw);
    free(S->out_norm); free(S->lm_head); free(S->embd);
    memset(S, 0, sizeof *S);
}

/* ---- naive reference forward (double where cheap) ------------------------ */
typedef struct {
    const synth *S;
    double *K, *Vc;      /* [L][pos][KV*hd] */
    int pos;
    double inv[32];      /* RoPE inverse frequencies (YaRN-corrected when set) */
    double af;           /* attention scale (1.0 unless YaRN) */
} ref_state;

static void r_rms(double *out, const double *x, const float *w, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    double r = 1.0 / sqrt(ss / n + 1e-6);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}
static void r_mv(double *y, const float *W, const double *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        double s = 0;
        for (int c = 0; c < cols; c++) s += (double)W[(size_t)r * cols + c] * x[c];
        y[r] = s;
    }
}
static void r_rope(double *v, int pos, int hd, const double *inv, double af) {
    for (int j = 0; j < hd / 2; j++) {
        double a = pos * inv[j], c = cos(a) * af, s = sin(a) * af;
        double x0 = v[j], x1 = v[j + hd / 2];
        v[j] = x0 * c - x1 * s;
        v[j + hd / 2] = x1 * c + x0 * s;
    }
}
/* double port of HF _compute_yarn_parameters (same formulas as the C f32 path). */
static void ref_yarn(double *inv, int half, int hd, double theta, double factor,
                     double bf, double bs, uint32_t orig, int trunc, double *af) {
    double low = (double)hd * log((double)orig / (bf * 6.283185307179586)) / (2.0 * log(theta));
    double high = (double)hd * log((double)orig / (bs * 6.283185307179586)) / (2.0 * log(theta));
    if (trunc) { low = floor(low); high = ceil(high); }
    if (low < 0.0) low = 0.0;
    if (high > hd - 1.0) high = (double)hd - 1.0;
    if (low == high) high += 0.001;
    *af = factor > 1.0 ? 0.1 * log(factor) + 1.0 : 1.0;
    for (int j = 0; j < half; j++) {
        double extrap = inv[j], pos_freq = 1.0 / extrap;
        double interp = 1.0 / (factor * pos_freq);
        double ramp = ((double)j - low) / (high - low);
        if (ramp < 0.0) ramp = 0.0;
        if (ramp > 1.0) ramp = 1.0;
        inv[j] = interp * ramp + extrap * (1.0 - ramp);
    }
}
static double r_silu(double x) { return x / (1.0 + exp(-x)); }

static void ref_step(ref_state *st, int token, float *logits, const uint8_t *mask) {
    const synth *S = st->S;
    const spec *sp = &S->s;
    int D = sp->D, H = sp->H, KV = sp->KV, hd = sp->hd, E = sp->E, K = sp->K, de = sp->de, ds = sp->ds;
    int pos = st->pos;
    for (int j = 0; j < hd / 2; j++) st->inv[j] = pow(10000.0, -2.0 * j / hd);
    st->af = 1.0;
    if (sp->rope_scale)
        ref_yarn(st->inv, hd / 2, hd, 10000.0, sp->rope_factor, sp->rope_beta_fast,
                 sp->rope_beta_slow, sp->rope_orig_len, sp->rope_scale == 1, &st->af);
    double *h = calloc((size_t)D, sizeof(double)), *x = calloc((size_t)D, sizeof(double));
    double *q = calloc((size_t)H * hd, sizeof(double)), *k = calloc((size_t)KV * hd, sizeof(double));
    double *v = calloc((size_t)KV * hd, sizeof(double)), *att = calloc((size_t)H * hd, sizeof(double));
    double *o = calloc((size_t)D, sizeof(double)), *lg = calloc((size_t)E, sizeof(double));
    int gsz = de > ds ? de : ds;
    if (sp->d_ffn > gsz) gsz = sp->d_ffn;
    double *gg = calloc((size_t)gsz, sizeof(double)), *uu = calloc((size_t)gsz, sizeof(double));
    double *sc = calloc((size_t)pos + 1, sizeof(double));
    for (int i = 0; i < D; i++) h[i] = S->embd[(size_t)token * D + i];
    for (int l = 0; l < sp->L; l++) {
        const ref_layer *R = &S->Lw[l];
        r_rms(x, h, R->attn_norm, D);
        r_mv(q, R->q, x, H * hd, D);
        r_mv(k, R->k, x, KV * hd, D);
        r_mv(v, R->v, x, KV * hd, D);
        for (int hh = 0; hh < H && R->q_norm; hh++) r_rms(q + hh * hd, q + hh * hd, R->q_norm, hd);
        for (int hh = 0; hh < KV && R->k_norm; hh++) r_rms(k + hh * hd, k + hh * hd, R->k_norm, hd);
        for (int hh = 0; hh < H; hh++) r_rope(q + hh * hd, pos, hd, st->inv, st->af);
        for (int hh = 0; hh < KV; hh++) r_rope(k + hh * hd, pos, hd, st->inv, st->af);
        double *Kl = st->K + ((size_t)l * 256) * KV * hd, *Vl = st->Vc + ((size_t)l * 256) * KV * hd;
        memcpy(Kl + (size_t)pos * KV * hd, k, sizeof(double) * KV * hd);
        memcpy(Vl + (size_t)pos * KV * hd, v, sizeof(double) * KV * hd);
        for (int hh = 0; hh < H; hh++) {
            int kvh = hh / (H / KV);
            double mx = -1e300, sum = 0;
            for (int t = 0; t <= pos; t++) {
                double d = 0;
                for (int i = 0; i < hd; i++) d += q[hh * hd + i] * Kl[(size_t)t * KV * hd + kvh * hd + i];
                sc[t] = d / sqrt((double)hd);
                if (sc[t] > mx) mx = sc[t];
            }
            for (int t = 0; t <= pos; t++) { sc[t] = exp(sc[t] - mx); sum += sc[t]; }
            for (int i = 0; i < hd; i++) {
                double a = 0;
                for (int t = 0; t <= pos; t++) a += sc[t] / sum * Vl[(size_t)t * KV * hd + kvh * hd + i];
                att[hh * hd + i] = a;
            }
        }
        r_mv(o, R->o, att, D, H * hd);
        for (int i = 0; i < D; i++) h[i] += o[i];
        r_rms(x, h, R->ffn_norm, D);
        if (l < sp->n_dense) {
            /* Dense-FFN layer: one resident SwiGLU block, no routing */
            int df = sp->d_ffn;
            r_mv(gg, R->fg, x, df, D);
            r_mv(uu, R->fu, x, df, D);
            for (int i = 0; i < df; i++) gg[i] = r_silu(gg[i]) * uu[i];
            for (int i = 0; i < D; i++) {
                double s = 0;
                for (int j = 0; j < df; j++) s += (double)R->fd[(size_t)i * df + j] * gg[j];
                o[i] = s;
            }
            for (int i = 0; i < D; i++) h[i] += o[i];
            continue;
        }
        r_mv(lg, R->router, x, E, D);
        /* routing */
        const uint8_t *mk = mask ? mask + (size_t)l * E : NULL;
        double p[64], sel[64];
        int ids[8];
        double w[8];
        if (sp->router == 2) {
            for (int e = 0; e < E; e++) { p[e] = 1.0 / (1.0 + exp(-lg[e])); sel[e] = p[e] + R->rbias[e]; }
        } else {
            double mx = -1e300, sum = 0;
            for (int e = 0; e < E; e++) if (!(mk && mk[e]) && lg[e] > mx) mx = lg[e];
            for (int e = 0; e < E; e++) { p[e] = (mk && mk[e]) ? 0 : exp(lg[e] - mx); sum += p[e]; }
            for (int e = 0; e < E; e++) { p[e] /= sum; sel[e] = p[e]; }
        }
        for (int r = 0; r < K; r++) {
            int best = -1;
            for (int e = 0; e < E; e++) {
                if (mk && mk[e]) continue;
                int used = 0;
                for (int z = 0; z < r; z++) used |= ids[z] == e;
                if (!used && (best < 0 || sel[e] > sel[best])) best = e;
            }
            ids[r] = best;
            w[r] = p[best];
        }
        if (sp->router != 1) {
            double s = 0;
            for (int r = 0; r < K; r++) s += w[r];
            for (int r = 0; r < K; r++) w[r] /= s;
        }
        for (int r = 0; r < K; r++) w[r] *= sp->routed_scale;
        for (int i = 0; i < D; i++) o[i] = 0;
        for (int r = 0; r < K; r++) {
            int e = ids[r];
            r_mv(gg, R->eg + (size_t)e * de * D, x, de, D);
            r_mv(uu, R->eu + (size_t)e * de * D, x, de, D);
            for (int i = 0; i < de; i++) gg[i] = r_silu(gg[i]) * uu[i];
            for (int i = 0; i < D; i++) {
                double s = 0;
                for (int j = 0; j < de; j++) s += (double)R->ed[(size_t)e * D * de + (size_t)i * de + j] * gg[j];
                o[i] += w[r] * s;
            }
        }
        if (ds > 0) {
            r_mv(gg, R->sg, x, ds, D);
            r_mv(uu, R->su, x, ds, D);
            for (int i = 0; i < ds; i++) gg[i] = r_silu(gg[i]) * uu[i];
            for (int i = 0; i < D; i++) {
                double s = 0;
                for (int j = 0; j < ds; j++) s += (double)R->sd[(size_t)i * ds + j] * gg[j];
                o[i] += s;
            }
        }
        for (int i = 0; i < D; i++) h[i] += o[i];
    }
    r_rms(x, h, S->out_norm, D);
    for (int t = 0; t < sp->V; t++) {
        double s = 0;
        for (int i = 0; i < D; i++) s += (double)S->lm_head[(size_t)t * D + i] * x[i];
        logits[t] = (float)s;
    }
    st->pos++;
    free(h); free(x); free(q); free(k); free(v); free(att); free(o); free(lg); free(gg); free(uu); free(sc);
}

/* ---- runtime harness ------------------------------------------------------ */
typedef struct {
    tlm_manifest m;
    tlm_io      *io;
    tlm_store   *store;
    tlm_pool    *pool;
    tlm_model   *mdl;
} rt;

typedef struct { int threads; uint64_t budget_slots; int prefetch; double pin_frac; int lookahead_k; } rt_cfg;

static int rt_open(rt *r, const synth *S, const rt_cfg *c) {
    memset(r, 0, sizeof *r);
    if (tlm_manifest_load(&r->m, S->mpath)) return -1;
    char paths[TLM_MAX_DRIVES][512];
    const char *pp[TLM_MAX_DRIVES];
    for (uint32_t i = 0; i < r->m.n_drives; i++) {
        tlm_manifest_drive_path(&r->m, (int)i, paths[i], sizeof paths[i]);
        pp[i] = paths[i];
    }
    tlm_io_opts io = {TLM_IO_PREAD, 0, 2, 32};
    r->io = tlm_io_open(pp, (int)r->m.n_drives, &io);
    if (!r->io) return -1;
    tlm_store_opts so;
    memset(&so, 0, sizeof so);
    so.budget_bytes = c->budget_slots * S->max_expert_total;
    so.pin_frac = c->pin_frac;
    so.prefetch = c->prefetch;
    r->store = tlm_store_open(&r->m, r->io, &so);
    if (!r->store) return -1;
    if (c->pin_frac > 0 && tlm_store_warm(r->store)) return -1;
    r->pool = c->threads > 1 ? tlm_pool_new(c->threads, 0) : NULL;
    tlm_model_opts mo = {c->prefetch ? 1 : 0, c->lookahead_k};
    r->mdl = tlm_model_open(&r->m, r->io, r->store, r->pool, &mo);
    return r->mdl ? 0 : -1;
}

static void rt_close(rt *r) {
    tlm_model_close(r->mdl);
    tlm_pool_free(r->pool);
    tlm_store_close(r->store);
    tlm_io_close(r->io);
    tlm_manifest_free(&r->m);
    memset(r, 0, sizeof *r);
}

#define NSTEP 12
typedef struct { int tokens[NSTEP]; float *logits; } trace;

/* Feed a 4-token prompt, then greedy-decode; record every step's logits. */
static int run_trace(rt *r, int V, trace *t) {
    tlm_kv *kv = tlm_kv_new(r->mdl, 64);
    if (!kv) return -1;
    int prompt[4] = {3, 17, 5, 3};
    t->logits = (float *)calloc((size_t)NSTEP * V, sizeof(float));
    int rc = tlm_model_prefill(r->mdl, kv, prompt, 4, t->logits);
    for (int i = 0; i < NSTEP && rc == 0; i++) {
        int tok = tlm_sample(NULL, NULL, t->logits + (size_t)i * V, V);
        t->tokens[i] = tok;
        if (i + 1 < NSTEP) rc = tlm_model_step(r->mdl, kv, tok, t->logits + (size_t)(i + 1) * V);
    }
    tlm_kv_free(kv);
    return rc;
}

static int same_trace(const trace *a, const trace *b, int V) {
    return memcmp(a->tokens, b->tokens, sizeof a->tokens) == 0 &&
           memcmp(a->logits, b->logits, sizeof(float) * (size_t)NSTEP * V) == 0;
}

static spec quant_spec(void) {
    spec s;
    memset(&s, 0, sizeof s);
    s.L = 3; s.D = 256; s.H = 4; s.KV = 2; s.hd = 64; s.V = 300; s.E = 8; s.K = 2; s.de = 256;
    s.ds = 256; s.router = 0; s.qk_norm = 1; s.n_drives = 3; s.replicas = 1; s.routed_scale = 1.0f;
    s.dense_dt = TLM_DT_Q8_0;
    s.expert_dt[0] = TLM_DT_Q6_K; s.expert_dt[1] = TLM_DT_Q8_0; s.expert_dt[2] = TLM_DT_Q4_K;
    s.seed = 7;
    return s;
}

/* ---- tests ------------------------------------------------------------------ */
static void check_reference(int router, float routed_scale, int qk_norm, int ds) {
    spec s;
    memset(&s, 0, sizeof s);
    s.L = 2; s.D = 64; s.H = 4; s.KV = 2; s.hd = 16; s.V = 50; s.E = 6; s.K = 2; s.de = 32;
    s.ds = ds; s.router = router; s.qk_norm = qk_norm; s.n_drives = 2; s.routed_scale = routed_scale;
    s.dense_dt = TLM_DT_F32;
    s.expert_dt[0] = s.expert_dt[1] = s.expert_dt[2] = TLM_DT_F32;
    s.seed = 100 + (unsigned)router;
    synth S;
    char name[32];
    snprintf(name, sizeof name, "ref%d", router);
    CHECK_EQ_INT(build(&S, &s, name), 0);
    rt r;
    rt_cfg c = {2, 4, 1, 0.0, 0};
    if (rt_open(&r, &S, &c)) { fprintf(stderr, "  open: %s\n", tlm_errmsg()); CHECK(0); synth_free(&S); return; }
    tlm_kv *kv = tlm_kv_new(r.mdl, 32);
    ref_state st = (ref_state){ .S = &S,
                                .K = calloc((size_t)s.L * 256 * s.KV * s.hd, sizeof(double)),
                                .Vc = calloc((size_t)s.L * 256 * s.KV * s.hd, sizeof(double)) };
    float lg[64], lr[64];
    const int toks[8] = {1, 7, 49, 0, 22, 7, 7, 3};
    double worst = 0;
    for (int i = 0; i < 8; i++) {
        CHECK_EQ_INT(tlm_model_step(r.mdl, kv, toks[i], lg), 0);
        ref_step(&st, toks[i], lr, NULL);
        for (int t = 0; t < s.V; t++) {
            double d = fabs((double)lg[t] - (double)lr[t]) / (1.0 + fabs((double)lr[t]));
            if (d > worst) worst = d;
        }
    }
    if (worst >= 1e-4) fprintf(stderr, "  router %d: worst relative logit error %g\n", router, worst);
    CHECK(worst < 1e-4);
    free(st.K);
    free(st.Vc);
    tlm_kv_free(kv);
    rt_close(&r);
    synth_free(&S);
}

static void test_reference_f32(void) {
    check_reference(0, 1.0f, 1, 48);    /* softmax top-k renorm, qk-norm, shared expert */
    check_reference(1, 1.0f, 0, 0);     /* softmax top-k, no renorm, no qk-norm, no shared */
    check_reference(2, 2.5f, 1, 32);    /* sigmoid + selection bias, renorm, routed scale */
}
/* fixtures from an independent f32 port of HF _compute_yarn_parameters
 * (scratch/yarn_ref.py, written against transformers/modeling_rope_utils.py) */
static const float plain64[32] = {
    1.0f, 0.749894202f, 0.562341332f, 0.421696514f, 0.316227764f, 0.237137377f, 0.177827939f, 0.133352146f,
    0.100000001f, 0.0749894232f, 0.0562341325f, 0.0421696492f, 0.0316227749f, 0.0237137377f, 0.0177827943f, 0.013335214f,
    0.00999999978f, 0.00749894232f, 0.00562341325f, 0.00421696482f, 0.00316227763f, 0.00237137382f, 0.00177827943f, 0.00133352145f,
    0.00100000005f, 0.000749894185f, 0.000562341302f, 0.000421696517f, 0.000316227757f, 0.00023713737f, 0.00017782794f, 0.00013335215f,
};
static const float yarn_a[32] = {  /* factor 4, truncate, orig 512 */
    1.0f, 0.749894202f, 0.562341332f, 0.421696514f, 0.297983855f, 0.209775373f, 0.147050023f, 0.102578573f,
    0.0711538419f, 0.0490315482f, 0.0335241929f, 0.0227067322f, 0.0152032562f, 0.0100327339f, 0.00649755914f, 0.0041031423f,
    0.00249999994f, 0.00187473558f, 0.00140585331f, 0.00105424121f, 0.000790569349f, 0.000592843455f, 0.000444569858f, 0.000333380362f,
    0.000250000012f, 0.000187473546f, 0.000140585325f, 0.000105424137f, 7.90569393e-05f, 5.92843462e-05f, 4.44569851e-05f, 3.33380376e-05f,
};
static const float yarn_b[32] = {  /* factor 4, no truncate, orig 512 */
    1.0f, 0.749894202f, 0.562341332f, 0.421696514f, 0.301406473f, 0.211252615f, 0.147340879f, 0.102184072f,
    0.0703986362f, 0.0481207296f, 0.0325828493f, 0.0218071025f, 0.014383357f, 0.00930895936f, 0.00587311294f, 0.00357361417f,
    0.00249999994f, 0.00187473558f, 0.00140585331f, 0.00105424121f, 0.000790569349f, 0.000592843455f, 0.000444569858f, 0.000333380362f,
    0.000250000012f, 0.000187473546f, 0.000140585325f, 0.000105424137f, 7.90569393e-05f, 5.92843462e-05f, 4.44569851e-05f, 3.33380376e-05f,
};
static const float yarn_c[32] = {  /* factor 1 (no-op except f32 rounding), truncate */
    1.0f, 0.749894202f, 0.562341332f, 0.421696514f, 0.316227764f, 0.237137377f, 0.177827939f, 0.133352146f,
    0.099999994f, 0.0749894232f, 0.0562341288f, 0.0421696492f, 0.0316227749f, 0.0237137359f, 0.0177827943f, 0.0133352131f,
    0.00999999978f, 0.00749894232f, 0.00562341325f, 0.00421696482f, 0.0031622774f, 0.00237137382f, 0.00177827943f, 0.00133352145f,
    0.00100000005f, 0.000749894185f, 0.000562341302f, 0.000421696546f, 0.000316227757f, 0.000237137385f, 0.00017782794f, 0.00013335215f,
};

static void test_yarn_table(void) {
    tlm_log_level(TLM_LOG_ERROR);
    float in[32], af = 0.0f;
    memcpy(in, plain64, sizeof in);
    CHECK_EQ_INT(tlm_yarn_transform(in, 32, 64, 10000.0, 4.0, 32.0, 1.0, 512, 1, &af), 0);
    if (memcmp(in, yarn_a, sizeof in)) fprintf(stderr, "  yarn truncate fixture mismatch\n");
    CHECK(memcmp(in, yarn_a, sizeof in) == 0);
    CHECK(af == 1.13862944f);
    memcpy(in, plain64, sizeof in);
    CHECK_EQ_INT(tlm_yarn_transform(in, 32, 64, 10000.0, 4.0, 32.0, 1.0, 512, 0, &af), 0);
    CHECK(memcmp(in, yarn_b, sizeof in) == 0);
    CHECK(af == 1.13862944f);
    memcpy(in, plain64, sizeof in);
    CHECK_EQ_INT(tlm_yarn_transform(in, 32, 64, 10000.0, 1.0, 32.0, 1.0, 512, 1, &af), 0);
    CHECK(memcmp(in, yarn_c, sizeof in) == 0);
    CHECK(af == 1.0f);
    float t2[2] = {1.0f, 0.5f};
    CHECK(tlm_yarn_transform(t2, 2, 2, 0.0, 4.0, 32.0, 1.0, 512, 1, &af) == TLM_EINVAL);  /* theta */
    CHECK(tlm_yarn_transform(t2, 2, 2, 10000.0, 0.0, 32.0, 1.0, 512, 1, &af) == TLM_EINVAL); /* factor */
    CHECK(tlm_yarn_transform(t2, 2, 2, 10000.0, 4.0, 1.0, 32.0, 512, 1, &af) == TLM_EINVAL); /* beta_fast < beta_slow */
    CHECK(tlm_yarn_transform(t2, 2, 2, 10000.0, 4.0, 32.0, 1.0, 0, 1, &af) == TLM_EINVAL);  /* orig_len */
    CHECK(tlm_yarn_transform(t2, 3, 2, 10000.0, 4.0, 32.0, 1.0, 512, 1, &af) == TLM_EINVAL); /* half mismatch */
    tlm_log_level(TLM_LOG_WARN);
}
/* f32 model with YaRN vs the double reference (P-23): exercises the whole
 * forward through the corrected frequencies and the attention scale. */
static void check_yarn_reference(int scale, float factor, float bf, float bs, uint32_t orig) {
    spec s;
    memset(&s, 0, sizeof s);
    s.L = 2; s.D = 256; s.H = 4; s.KV = 2; s.hd = 64; s.V = 50; s.E = 6; s.K = 2; s.de = 32;
    s.ds = 48; s.router = 0; s.qk_norm = 1; s.n_drives = 2; s.routed_scale = 1.0f;
    s.dense_dt = TLM_DT_F32;
    s.expert_dt[0] = s.expert_dt[1] = s.expert_dt[2] = TLM_DT_F32;
    s.seed = 400 + (unsigned)scale;
    s.rope_scale = (uint32_t)scale; s.rope_factor = factor;
    s.rope_beta_fast = bf; s.rope_beta_slow = bs; s.rope_orig_len = orig;
    synth S;
    char name[16];
    snprintf(name, sizeof name, "yarn%d", scale);
    CHECK_EQ_INT(build(&S, &s, name), 0);
    rt r;
    rt_cfg c = {2, 4, 1, 0.0, 0};
    if (rt_open(&r, &S, &c)) { fprintf(stderr, "  open: %s\n", tlm_errmsg()); CHECK(0); synth_free(&S); return; }
    tlm_kv *kv = tlm_kv_new(r.mdl, 32);
    ref_state st = (ref_state){ .S = &S,
                                .K = calloc((size_t)s.L * 256 * s.KV * s.hd, sizeof(double)),
                                .Vc = calloc((size_t)s.L * 256 * s.KV * s.hd, sizeof(double)) };
    float lg[64], lr[64];
    const int toks[8] = {1, 7, 49, 0, 22, 7, 7, 3};
    double worst = 0;
    for (int i = 0; i < 8; i++) {
        CHECK_EQ_INT(tlm_model_step(r.mdl, kv, toks[i], lg), 0);
        ref_step(&st, toks[i], lr, NULL);
        for (int t = 0; t < s.V; t++) {
            double d = fabs((double)lg[t] - (double)lr[t]) / (1.0 + fabs((double)lr[t]));
            if (d > worst) worst = d;
        }
    }
    if (worst >= 1e-4) fprintf(stderr, "  yarn scale %d: worst relative logit error %g\n", scale, worst);
    CHECK(worst < 1e-4);
    free(st.K);
    free(st.Vc);
    tlm_kv_free(kv);
    rt_close(&r);
    synth_free(&S);
}

static void test_yarn_reference(void) {
    /* hd=64, orig=512, factor=4: low=3, high=16 (truncate) so j>=8 extrapolates
     * fully and the middle ramps; af = 1 + 0.1*ln 4. */
    check_yarn_reference(1, 4.0f, 32.0f, 1.0f, 512);
    /* no-truncate: fractional low/high, full extrapolation weight */
    check_yarn_reference(2, 2.0f, 32.0f, 1.0f, 512);
}

/* thread / cache / prefetch / pin / tier configs must not change output bits;
 * base is the 1-thread no-prefetch trace */
static void inv_matrix(const synth *S, const spec *s, rt *r, const trace *base, int V) {
    rt_cfg cfgs[] = {
        {2, 64, 0, 0.0, 0},              /* threads */
        {4, 64, 0, 0.0, 0},
        {3, 64, 1, 0.5, 0},              /* resident + pinned + prefetch */
        {1, (uint64_t)s->K, 0, 0.0, 0},  /* minimal cache: top_k slots, constant eviction */
        {4, (uint64_t)s->K, 1, 0.0, 0},  /* minimal cache with prefetch (hints mostly dropped) */
        {2, (uint64_t)s->K + 2, 1, 0.0, 6}, /* small cache, aggressive lookahead */
        {2, 9, 1, 0.3, 0},               /* partial cache with pins */
    };
    for (size_t i = 0; i < sizeof cfgs / sizeof cfgs[0]; i++) {
        trace t;
        memset(&t, 0, sizeof t);
        if (rt_open(r, S, &cfgs[i])) {
            fprintf(stderr, "  cfg %zu open: %s\n", i, tlm_errmsg());
            CHECK(0);
            rt_close(r);
            continue;
        }
        CHECK_EQ_INT(run_trace(r, V, &t), 0);
        if (!same_trace(base, &t, V)) fprintf(stderr, "  cfg %zu differs\n", i);
        CHECK(same_trace(base, &t, V));
        tlm_store_stats ss;
        tlm_store_get_stats(r->store, &ss);
        if (cfgs[i].budget_slots == (uint64_t)s->K) CHECK(ss.evictions > 0 || ss.misses > (uint64_t)s->K * s->L);
        CHECK_EQ_INT(ss.in_use, 0);   /* every acquire was released */
        rt_close(r);
        free(t.logits);
    }

    /* kernel tiers: every tier must reproduce the scalar bits (P-43) */
    const char *tiers[] = {"scalar", "avx2", "avx512"};
    for (int i = 0; i < 3; i++) {
        tlm_cpu_force(tiers[i]);
        tlm_quant_init();
        if (strcmp(tlm_quant_tier(), tiers[i]) != 0) {
            printf("  (tier %s not available here, got %s)\n", tiers[i], tlm_quant_tier());
            continue;
        }
        trace t;
        memset(&t, 0, sizeof t);
        rt_cfg c = {2, 4, 1, 0.0, 0};
        if (rt_open(r, S, &c)) { CHECK(0); rt_close(r); continue; }
        CHECK_EQ_INT(run_trace(r, V, &t), 0);
        if (!same_trace(base, &t, V)) fprintf(stderr, "  tier %s differs\n", tiers[i]);
        CHECK(same_trace(base, &t, V));
        rt_close(r);
        free(t.logits);
    }
    tlm_cpu_force("auto");
    tlm_quant_init();
}
static void test_invariance(void) {
    spec s = quant_spec();
    synth S;
    CHECK_EQ_INT(build(&S, &s, "inv"), 0);
    int V = s.V;
    rt r;
    trace base;
    memset(&base, 0, sizeof base);
    rt_cfg c0 = {1, 64, 0, 0.0, 0};   /* 1 thread, everything fits, no prefetch */
    if (rt_open(&r, &S, &c0)) { fprintf(stderr, "  open: %s\n", tlm_errmsg()); CHECK(0); synth_free(&S); return; }
    CHECK_EQ_INT(run_trace(&r, V, &base), 0);
    rt_close(&r);
    int n_finite = 0;
    for (int i = 0; i < NSTEP * V; i++) n_finite += isfinite(base.logits[i]) != 0;
    CHECK_EQ_INT(n_finite, NSTEP * V);

    inv_matrix(&S, &s, &r, &base, V);

    /* YaRN variant (F-23): the same matrix must stay bit-identical */
    s.rope_scale = 1; s.rope_factor = 4.0f; s.rope_beta_fast = 32.0f;
    s.rope_beta_slow = 1.0f; s.rope_orig_len = 512; s.seed = 8;
    synth Sy;
    CHECK_EQ_INT(build(&Sy, &s, "invy"), 0);
    trace by;
    memset(&by, 0, sizeof by);
    if (rt_open(&r, &Sy, &c0)) {
        fprintf(stderr, "  yarn open: %s\n", tlm_errmsg());
        CHECK(0);
        synth_free(&S);
        synth_free(&Sy);
        return;
    }
    CHECK_EQ_INT(run_trace(&r, V, &by), 0);
    rt_close(&r);
    inv_matrix(&Sy, &s, &r, &by, V);
    free(by.logits);
    synth_free(&Sy);
    free(base.logits);
    synth_free(&S);
}

static void check_dense_reference(int n_dense, int d_ffn) {
    spec s;
    memset(&s, 0, sizeof s);
    s.L = 3; s.D = 64; s.H = 4; s.KV = 2; s.hd = 16; s.V = 50; s.E = 6; s.K = 2; s.de = 32;
    s.ds = 32; s.router = 0; s.qk_norm = 1; s.n_drives = 2; s.routed_scale = 1.0f;
    s.n_dense = n_dense; s.d_ffn = d_ffn;
    s.dense_dt = TLM_DT_F32;
    s.expert_dt[0] = s.expert_dt[1] = s.expert_dt[2] = TLM_DT_F32;
    s.seed = 300u + (unsigned)n_dense * 17u + (unsigned)d_ffn;
    synth S;
    char name[32];
    snprintf(name, sizeof name, "denseref%d_%d", n_dense, d_ffn);
    CHECK_EQ_INT(build(&S, &s, name), 0);
    rt r;
    rt_cfg c = {2, 4, 1, 0.0, 0};
    if (rt_open(&r, &S, &c)) { fprintf(stderr, "  open: %s\n", tlm_errmsg()); CHECK(0); synth_free(&S); return; }
    tlm_kv *kv = tlm_kv_new(r.mdl, 32);
    ref_state st = {&S, calloc((size_t)s.L * 256 * s.KV * s.hd, sizeof(double)),
                    calloc((size_t)s.L * 256 * s.KV * s.hd, sizeof(double)), 0};
    float lg[64], lr[64];
    const int toks[8] = {1, 7, 49, 0, 22, 7, 7, 3};
    double worst = 0;
    for (int i = 0; i < 8; i++) {
        CHECK_EQ_INT(tlm_model_step(r.mdl, kv, toks[i], lg), 0);
        ref_step(&st, toks[i], lr, NULL);
        for (int t = 0; t < s.V; t++) {
            double d = fabs((double)lg[t] - (double)lr[t]) / (1.0 + fabs((double)lr[t]));
            if (d > worst) worst = d;
        }
    }
    if (worst >= 1e-4) fprintf(stderr, "  dense %d/%d: worst relative logit error %g\n", n_dense, d_ffn, worst);
    CHECK(worst < 1e-4);
    free(st.K);
    free(st.Vc);
    tlm_kv_free(kv);
    rt_close(&r);
    synth_free(&S);
}

static void test_dense_prefix(void) {
    /* a dense prefix must match the naive reference (F-fwd-01, F-fwd-18) */
    check_dense_reference(1, 32);
    check_dense_reference(2, 48);
    /* a quantised dense prefix must be bit-identical across cache sizes,
     * thread counts and kernel tiers (P-43/P-34), like the all-MoE model */
    spec s = quant_spec();
    s.n_dense = 1;
    s.d_ffn = 128;
    s.seed = 2024;
    synth S;
    CHECK_EQ_INT(build(&S, &s, "densinv"), 0);
    int V = s.V;
    rt r;
    trace base;
    memset(&base, 0, sizeof base);
    rt_cfg c0 = {1, 64, 0, 0.0, 0};
    if (rt_open(&r, &S, &c0)) { fprintf(stderr, "  open: %s\n", tlm_errmsg()); CHECK(0); synth_free(&S); return; }
    CHECK_EQ_INT(run_trace(&r, V, &base), 0);
    rt_close(&r);
    int n_finite = 0;
    for (int i = 0; i < NSTEP * V; i++) n_finite += isfinite(base.logits[i]) != 0;
    CHECK_EQ_INT(n_finite, NSTEP * V);
    const rt_cfg cfgs[] = {
        {2, 64, 0, 0.0, 0},              /* threads */
        {4, 64, 0, 0.0, 0},
        {3, 64, 1, 0.5, 0},              /* resident + pinned + prefetch */
        {1, (uint64_t)s.K, 0, 0.0, 0},   /* minimal cache: top_k slots */
        {4, (uint64_t)s.K, 1, 0.0, 0},   /* minimal cache with prefetch */
        {2, (uint64_t)s.K + 2, 1, 0.0, 6}, /* small cache, aggressive lookahead */
        {2, 9, 1, 0.3, 0},               /* partial cache with pins */
    };
    for (size_t i = 0; i < sizeof cfgs / sizeof cfgs[0]; i++) {
        trace t;
        memset(&t, 0, sizeof t);
        if (rt_open(&r, &S, &cfgs[i])) {
            fprintf(stderr, "  dense cfg %zu open: %s\n", i, tlm_errmsg());
            CHECK(0);
            rt_close(&r);
            continue;
        }
        CHECK_EQ_INT(run_trace(&r, V, &t), 0);
        if (!same_trace(&base, &t, V)) fprintf(stderr, "  dense cfg %zu differs\n", i);
        CHECK(same_trace(&base, &t, V));
        tlm_store_stats ss;
        tlm_store_get_stats(r.store, &ss);
        CHECK_EQ_INT(ss.in_use, 0);   /* every acquire was released */
        rt_close(&r);
        free(t.logits);
    }
    const char *tiers[] = {"scalar", "avx2", "avx512"};
    for (int i = 0; i < 3; i++) {
        tlm_cpu_force(tiers[i]);
        tlm_quant_init();
        if (strcmp(tlm_quant_tier(), tiers[i]) != 0) {
            printf("  (tier %s not available here, got %s)\n", tiers[i], tlm_quant_tier());
            continue;
        }
        trace t;
        memset(&t, 0, sizeof t);
        rt_cfg c = {2, 4, 1, 0.0, 0};
        if (rt_open(&r, &S, &c)) { CHECK(0); rt_close(&r); continue; }
        CHECK_EQ_INT(run_trace(&r, V, &t), 0);
        if (!same_trace(&base, &t, V)) fprintf(stderr, "  dense tier %s differs\n", tiers[i]);
        CHECK(same_trace(&base, &t, V));
        rt_close(&r);
        free(t.logits);
    }
    tlm_cpu_force("auto");
    tlm_quant_init();
    free(base.logits);
    synth_free(&S);
}

static void test_prefill_equals_steps(void) {
    spec s = quant_spec();
    s.seed = 11;
    s.n_drives = 1;
    s.replicas = 0;
    synth S;
    CHECK_EQ_INT(build(&S, &s, "pre"), 0);
    rt r;
    rt_cfg c = {2, 6, 1, 0.0, 0};
    if (rt_open(&r, &S, &c)) { CHECK(0); synth_free(&S); return; }
    int V = s.V;
    const int toks[9] = {1, 2, 3, 250, 299, 0, 42, 42, 7};
    float *a = calloc((size_t)V, sizeof(float)), *b = calloc((size_t)V, sizeof(float));
    tlm_kv *k1 = tlm_kv_new(r.mdl, 32), *k2 = tlm_kv_new(r.mdl, 32);
    CHECK_EQ_INT(tlm_model_prefill(r.mdl, k1, toks, 9, a), 0);
    for (int i = 0; i < 9; i++) CHECK_EQ_INT(tlm_model_step(r.mdl, k2, toks[i], b), 0);
    CHECK(memcmp(a, b, sizeof(float) * (size_t)V) == 0);
    CHECK_EQ_INT(tlm_kv_len(k1), 9);
    CHECK_EQ_INT(tlm_kv_capacity(k1), 32);
    /* chunked prefill (2 chunks) == single shot */
    tlm_kv_reset(k2);
    CHECK_EQ_INT(tlm_kv_len(k2), 0);
    CHECK_EQ_INT(tlm_model_prefill(r.mdl, k2, toks, 4, NULL), 0);
    CHECK_EQ_INT(tlm_model_prefill(r.mdl, k2, toks + 4, 5, b), 0);
    CHECK(memcmp(a, b, sizeof(float) * (size_t)V) == 0);
    /* continuing from either cache gives the same next logits */
    CHECK_EQ_INT(tlm_model_step(r.mdl, k1, 5, a), 0);
    CHECK_EQ_INT(tlm_model_step(r.mdl, k2, 5, b), 0);
    CHECK(memcmp(a, b, sizeof(float) * (size_t)V) == 0);
    /* errors */
    tlm_log_level(TLM_LOG_ERROR);
    CHECK(tlm_model_step(r.mdl, k1, V, a) != 0);
    CHECK(tlm_model_step(r.mdl, k1, -1, a) != 0);
    CHECK(tlm_model_prefill(r.mdl, k1, toks, 0, a) != 0);
    tlm_kv *k3 = tlm_kv_new(r.mdl, 2);
    CHECK_EQ_INT(tlm_model_step(r.mdl, k3, 1, NULL), 0);
    CHECK_EQ_INT(tlm_model_step(r.mdl, k3, 1, NULL), 0);
    CHECK(tlm_model_step(r.mdl, k3, 1, NULL) != 0);   /* context full */
    tlm_log_level(TLM_LOG_WARN);
    tlm_model_stats ms;
    tlm_model_get_stats(r.mdl, &ms);
    CHECK(ms.tokens >= 20);
    tlm_model_reset_stats(r.mdl);
    tlm_model_get_stats(r.mdl, &ms);
    CHECK_EQ_INT(ms.tokens, 0);
    tlm_kv_free(k1);
    tlm_kv_free(k2);
    tlm_kv_free(k3);
    free(a);
    free(b);
    rt_close(&r);
    synth_free(&S);
}

typedef struct {
    int calls, bad, L, E, K, router;
    const uint8_t *mask;
    long sel[64];
} hook_ctx;

static void hook_fn(void *ctx, int layer, int pos, int token, const int *ids, const float *w,
                    const float *onorm, int k) {
    (void)pos; (void)token;
    hook_ctx *h = (hook_ctx *)ctx;
    h->calls++;
    if (layer < 0 || layer >= h->L || k != h->K) { h->bad++; return; }
    double s = 0;
    for (int i = 0; i < k; i++) {
        if (ids[i] < 0 || ids[i] >= h->E) { h->bad++; return; }
        for (int j = 0; j < i; j++) if (ids[j] == ids[i]) h->bad++;
        if (h->mask && h->mask[layer * h->E + ids[i]]) h->bad++;
        if (!(onorm[i] >= 0) || !(w[i] >= 0)) h->bad++;
        if (i > 0 && w[i] > w[i - 1] && h->router != 2) h->bad++;   /* rank order */
        s += w[i];
        if (layer == 0) h->sel[ids[i]]++;
    }
    if (h->router == 0 && fabs(s - 1.0) > 1e-5) h->bad++;
}

static void test_hook_and_mask(void) {
    spec s = quant_spec();
    s.seed = 21;
    s.L = 2;
    s.E = 12;
    s.n_drives = 2;
    s.prune_mask = (1u << 3) | (1u << 4);   /* layer 0 experts 3, 4 flagged pruned */
    synth S;
    CHECK_EQ_INT(build(&S, &s, "hook"), 0);
    rt r;
    rt_cfg c = {2, 32, 1, 0.0, 0};
    if (rt_open(&r, &S, &c)) { fprintf(stderr, "  %s\n", tlm_errmsg()); CHECK(0); synth_free(&S); return; }
    const uint8_t *m0 = tlm_model_expert_mask(r.mdl);
    CHECK(m0[3] && m0[4] && !m0[0] && !m0[12 + 3]);
    hook_ctx h;
    memset(&h, 0, sizeof h);
    h.L = s.L; h.E = s.E; h.K = s.K; h.router = 0;
    tlm_model_set_route_hook(r.mdl, hook_fn, &h);
    tlm_kv *kv = tlm_kv_new(r.mdl, 64);
    for (int i = 0; i < 40; i++) CHECK_EQ_INT(tlm_model_step(r.mdl, kv, (i * 37) % s.V, NULL), 0);
    CHECK_EQ_INT(h.calls, 40 * s.L);
    CHECK_EQ_INT(h.bad, 0);
    CHECK(h.sel[3] == 0 && h.sel[4] == 0);   /* pruned in the manifest: never routed */

    /* a compiler-style mask on top: experts 0..5 of layer 1 and 7 of layer 0 */
    uint8_t *pm = calloc((size_t)s.L * s.E, 1);
    for (int e = 0; e < 6; e++) pm[s.E + e] = 1;
    pm[7] = 1;
    CHECK_EQ_INT(tlm_model_set_expert_mask(r.mdl, pm), 0);
    const uint8_t *m1 = tlm_model_expert_mask(r.mdl);
    CHECK(m1[7] && m1[3] && m1[4] && m1[s.E + 5] && !m1[s.E + 6]);
    memset(&h, 0, sizeof h);
    h.L = s.L; h.E = s.E; h.K = s.K; h.mask = m1;
    tlm_kv_reset(kv);
    for (int i = 0; i < 30; i++) CHECK_EQ_INT(tlm_model_step(r.mdl, kv, (i * 11) % s.V, NULL), 0);
    CHECK_EQ_INT(h.bad, 0);
    CHECK_EQ_INT(h.calls, 30 * s.L);

    /* a mask leaving fewer than top_k experts in a layer is refused, mask unchanged */
    uint8_t *all = calloc((size_t)s.L * s.E, 1);
    for (int e = 0; e < s.E - 1; e++) all[e] = 1;
    tlm_log_level(TLM_LOG_ERROR);
    CHECK(tlm_model_set_expert_mask(r.mdl, all) != 0);
    tlm_log_level(TLM_LOG_WARN);
    CHECK(tlm_model_expert_mask(r.mdl)[s.E + 5] == 1 && tlm_model_expert_mask(r.mdl)[8] == 0);
    /* NULL restores all present experts; manifest-pruned stay masked */
    CHECK_EQ_INT(tlm_model_set_expert_mask(r.mdl, NULL), 0);
    const uint8_t *m2 = tlm_model_expert_mask(r.mdl);
    CHECK(m2[3] && m2[4] && !m2[7] && !m2[s.E + 5]);

    /* same mask -> same tokens as a fresh model with that mask (no state leaks) */
    tlm_model_set_route_hook(r.mdl, NULL, NULL);
    tlm_kv_free(kv);
    free(pm);
    free(all);
    rt_close(&r);
    synth_free(&S);
}

static void test_sampler(void) {
    float l[8] = {1, 5, 5, -2, NAN, 0, 4.9f, 5};
    CHECK_EQ_INT(tlm_sample(NULL, NULL, l, 8), 1);           /* ties -> lowest id, NaN skipped */
    tlm_sampler g = {0.0f, 1.0f, 0.0f, 1};
    uint64_t rng = 0;
    CHECK_EQ_INT(tlm_sample(&g, &rng, l, 8), 1);
    float nanv[3] = {NAN, NAN, NAN};
    CHECK_EQ_INT(tlm_sample(NULL, NULL, nanv, 3), 0);
    float neg[3] = {-INFINITY, -3, -INFINITY};
    CHECK_EQ_INT(tlm_sample(NULL, NULL, neg, 3), 1);

    /* seeded temperature sampling is reproducible and follows softmax */
    float lg[4] = {0.0f, 1.0f, 2.0f, -1.0f};
    tlm_sampler t1 = {1.0f, 1.0f, 0.0f, 1234};
    int cnt[4] = {0}, seq1[50], seq2[50];
    rng = 0;
    for (int i = 0; i < 20000; i++) cnt[tlm_sample(&t1, &rng, lg, 4)]++;
    double z = exp(0) + exp(1) + exp(2) + exp(-1);
    for (int i = 0; i < 4; i++) CHECK_NEAR(cnt[i] / 20000.0, exp(lg[i]) / z, 0.02);
    rng = 0;
    for (int i = 0; i < 50; i++) seq1[i] = tlm_sample(&t1, &rng, lg, 4);
    rng = 0;
    for (int i = 0; i < 50; i++) seq2[i] = tlm_sample(&t1, &rng, lg, 4);
    CHECK(memcmp(seq1, seq2, sizeof seq1) == 0);
    tlm_sampler t2 = t1;
    t2.seed = 99;
    rng = 0;
    int diff = 0;
    for (int i = 0; i < 50; i++) diff += tlm_sample(&t2, &rng, lg, 4) != seq1[i];
    CHECK(diff > 0);

    /* top-p small keeps only the argmax; min-p near 1 keeps only the max */
    tlm_sampler tp = {1.0f, 0.05f, 0.0f, 5};
    rng = 0;
    for (int i = 0; i < 200; i++) CHECK_EQ_INT(tlm_sample(&tp, &rng, lg, 4), 2);
    tlm_sampler mp = {1.0f, 1.0f, 0.99f, 5};
    rng = 0;
    for (int i = 0; i < 200; i++) CHECK_EQ_INT(tlm_sample(&mp, &rng, lg, 4), 2);
    /* top-p 0.7: ids 2 (0.64) and 1 (0.24) only */
    tlm_sampler tp2 = {1.0f, 0.7f, 0.0f, 6};
    rng = 0;
    int seen[4] = {0};
    for (int i = 0; i < 2000; i++) seen[tlm_sample(&tp2, &rng, lg, 4)]++;
    CHECK(seen[0] == 0 && seen[3] == 0 && seen[1] > 0 && seen[2] > seen[1]);
    /* very low temperature behaves like greedy */
    tlm_sampler cold = {1e-4f, 1.0f, 0.0f, 3};
    rng = 0;
    for (int i = 0; i < 50; i++) CHECK_EQ_INT(tlm_sample(&cold, &rng, lg, 4), 2);
}

typedef struct { int n; int toks[64]; } gen_ctx;
static int on_tok(void *ctx, int token) {
    gen_ctx *g = (gen_ctx *)ctx;
    if (g->n < 64) g->toks[g->n] = token;
    g->n++;
    return 0;
}

static void test_generate(void) {
    spec s = quant_spec();
    s.seed = 31;
    s.L = 2;
    s.n_drives = 2;
    s.router = 2;
    s.routed_scale = 2.0f;
    synth S;
    CHECK_EQ_INT(build(&S, &s, "gen"), 0);
    rt r;
    rt_cfg c = {2, 3, 1, 0.0, 0};
    if (rt_open(&r, &S, &c)) { fprintf(stderr, "  %s\n", tlm_errmsg()); CHECK(0); synth_free(&S); return; }
    tlm_kv *kv = tlm_kv_new(r.mdl, 64);
    int prompt[3] = {10, 20, 30};
    gen_ctx g1 = {0}, g2 = {0};
    tlm_sampler gs = {0.0f, 1.0f, 0.0f, 0};
    CHECK_EQ_INT(tlm_generate(r.mdl, kv, prompt, 3, 10, &gs, NULL, 0, on_tok, &g1), 10);
    CHECK_EQ_INT(g1.n, 10);
    CHECK_EQ_INT(tlm_kv_len(kv), 3 + 9);   /* the last sampled token is not fed */
    tlm_kv_reset(kv);
    CHECK_EQ_INT(tlm_generate(r.mdl, kv, prompt, 3, 10, &gs, NULL, 0, on_tok, &g2), 10);
    CHECK(memcmp(g1.toks, g2.toks, sizeof(int) * 10) == 0);
    /* stop token: generation ends at its first occurrence (inclusive) */
    gen_ctx g3 = {0};
    int stop = g1.toks[3];
    int first = 0;
    while (g1.toks[first] != stop) first++;
    tlm_kv_reset(kv);
    CHECK_EQ_INT(tlm_generate(r.mdl, kv, prompt, 3, 10, &gs, &stop, 1, on_tok, &g3), first + 1);
    /* sampled generation is reproducible for a seed */
    tlm_sampler ts = {0.9f, 0.95f, 0.05f, 77};
    gen_ctx g4 = {0}, g5 = {0};
    tlm_kv_reset(kv);
    CHECK_EQ_INT(tlm_generate(r.mdl, kv, prompt, 3, 12, &ts, NULL, 0, on_tok, &g4), 12);
    tlm_kv_reset(kv);
    CHECK_EQ_INT(tlm_generate(r.mdl, kv, prompt, 3, 12, &ts, NULL, 0, on_tok, &g5), 12);
    CHECK(memcmp(g4.toks, g5.toks, sizeof(int) * 12) == 0);
    /* context limit stops generation instead of failing */
    tlm_kv *small = tlm_kv_new(r.mdl, 5);
    gen_ctx g6 = {0};
    int n = tlm_generate(r.mdl, small, prompt, 3, 10, &gs, NULL, 0, on_tok, &g6);
    CHECK(n >= 1 && n <= 3);
    tlm_log_level(TLM_LOG_ERROR);
    CHECK(tlm_generate(r.mdl, kv, prompt, 0, 4, &gs, NULL, 0, NULL, NULL) < 0);
    tlm_log_level(TLM_LOG_WARN);
    tlm_kv_free(small);
    tlm_kv_free(kv);
    rt_close(&r);
    synth_free(&S);
}

/* A corrupted expert packet is detected (CRC) and reported, never used. */
static void test_corruption(void) {
    spec s = quant_spec();
    s.seed = 41;
    s.L = 1;
    s.n_drives = 1;
    s.replicas = 0;
    synth S;
    CHECK_EQ_INT(build(&S, &s, "bad"), 0);
    tlm_manifest m;
    CHECK_EQ_INT(tlm_manifest_load(&m, S.mpath), 0);
    char path[512];
    tlm_manifest_drive_path(&m, 0, path, sizeof path);
    /* flip one payload byte of every expert packet */
    FILE *f = fopen(path, "r+b");
    CHECK(f != NULL);
    for (uint32_t i = 0; f && i < m.n_packets; i++)
        if (m.packets[i].type == TLM_PK_EXPERT) {
            fseek(f, (long)(m.packets[i].offset + 64 + 100), SEEK_SET);
            int ch = fgetc(f);
            fseek(f, (long)(m.packets[i].offset + 64 + 100), SEEK_SET);
            fputc(ch ^ 0x5A, f);
        }
    if (f) fclose(f);
    tlm_manifest_free(&m);
    rt r;
    rt_cfg c = {1, 4, 0, 0.0, 0};
    tlm_log_level(TLM_LOG_ERROR);
    CHECK_EQ_INT(rt_open(&r, &S, &c), 0);
    if (r.mdl) {
        tlm_kv *kv = tlm_kv_new(r.mdl, 8);
        int rc = tlm_model_step(r.mdl, kv, 1, NULL);
        CHECK(rc == TLM_EFORMAT);
        tlm_store_stats ss;
        tlm_store_get_stats(r.store, &ss);
        CHECK_EQ_INT(ss.in_use, 0);
        tlm_kv_free(kv);
    }
    rt_close(&r);   /* the store logs the CRC failure from an I/O thread */
    tlm_log_level(TLM_LOG_WARN);
    synth_free(&S);
}

int main(void) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || !tmp[0]) tmp = "/tmp";
    snprintf(g_root, sizeof g_root, "%s/tlm-test-model-%ld", tmp, (long)getpid());
    rm_rf(g_root);
    CHECK_EQ_INT(tlm_mkdirs(g_root, 0700), 0);
    tlm_log_level(TLM_LOG_WARN);
    RUN(test_reference_f32);
    RUN(test_yarn_table);
    RUN(test_yarn_reference);
    RUN(test_invariance);
    RUN(test_dense_prefix);
    RUN(test_prefill_equals_steps);
    RUN(test_hook_and_mask);
    RUN(test_sampler);
    RUN(test_generate);
    RUN(test_corruption);
    rm_rf(g_root);
    TEST_MAIN_END();
}
