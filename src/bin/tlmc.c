/* tlmc.c - tierlm model compiler (stage 1 of research/03-two-stage-design.md).
 *
 *   tlmc synth   --out DIR [--layers N --d-model N --heads N --kv-heads N
 *                --head-dim N --experts N --top-k N --d-expert N --shared N
 *                --vocab N --seed N --router softmax|sigmoid --qk-norm yes|no]
 *   tlmc compile SRC --out MANIFEST --drives d1,d2,... [--experts q6_k]
 *                [--dense q8_0] [--profile exact|balanced|aggressive]
 *                [--replicas auto|none] [--corpus FILE] [--eval FILE]
 *                [--etc DIR] [--report] [--threads N] [--calib-tokens N]
 *                [--eval-tokens N] [--fresh]
 *   tlmc report  MANIFEST [--json]
 *   tlmc atlas   MANIFEST [--layer N] [--sort heat|saliency|id] [--class C] [--top N]
 *
 * Compiler passes (03-two-stage-design.md "Compiler passes"):
 *   1 ingest: config.json + safetensors (tlm_st, P-104), SHA-256 of every
 *     source file, tensor map validated against the geometry (P-03, P-05).
 *   2 reference quantise into a scratch store in the work dir (P-01 one
 *     expert = one packet, P-40 GGUF block formats, P-50 kept packed; router,
 *     norms and biases stay f32 per F-fwd-01/16). Reused when its stamp
 *     matches (P-04 resumable).
 *   3 calibrate by running the runtime itself with the route hook: per
 *     (layer, expert) selection count, mean gate, REAP saliency
 *     (mean of gate * ||expert out||), rank-0 count, co-activation within a
 *     layer, input byte-class histogram for the atlas (expert.specialisation).
 *   4 precision plan: exact = unchanged; balanced/aggressive ask
 *     expert.precision_tier and may demote cold Q6_K experts to Q5_K (P-52),
 *     accepted only if the eval gate still passes.
 *   5 prune (balanced/aggressive): rank by REAP saliency combined with
 *     P(not needed) from expert.needed_for_usecase (research/04 "How an
 *     exclusion becomes pruning"); hard rules: experts.protect, deep content,
 *     prune.max_ratio, prune.min_keep_per_layer, never below top_k*4 per
 *     layer; eval gate on top-1 agreement (97% balanced, 93% aggressive) with
 *     5-point back-off; the runtime masks pruned experts before softmax/top-k
 *     exactly as during the gate (tlm_model_set_expert_mask).
 *   6/7 pre-build + segment: layout 0 (row-major), one packet per expert.
 *   8 place: packets layer-major per drive; experts spread by expected read
 *     load (heat x size, P-110) with a co-activation penalty; compiler pins
 *     for the hottest (P-19); replicas of the warm middle on the least
 *     loaded other drive (P-31); every CRC re-read after writing (P-05).
 *   9 validate through the runtime: exact must be token- and bit-identical
 *     to pass 2 (P-121, P-123); the report JSON is sealed into the manifest
 *     (P-08 atomic save).
 * Decisions go through tlm_decide (rules backend) with a JSONL audit log in
 * the work dir (research/04-decision-engine.md).
 */
#include "tlm_cfg.h"
#include "tlm_decide.h"
#include "tlm_io.h"
#include "tlm_json.h"
#include "tlm_model.h"
#include "tlm_packet.h"
#include "tlm_platform.h"
#include "tlm_pool.h"
#include "tlm_quant.h"
#include "tlm_st.h"
#include "tlm_store.h"

#include <dirent.h>
#include <stddef.h>
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

#define TLMC_VERSION   "0.1"
#define NCLS           6
#define SEQ_LEN        128
#define MAX_E          4096
#define MIN_CALIB_WARN 200000

static const char *const CLS_NAME[NCLS] = {"lower", "upper", "digit", "space", "punct", "other"};

/* ---- small utilities ------------------------------------------------------ */
static void die(const char *fmt, ...) __attribute__((format(printf, 1, 2), noreturn));
static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("tlmc: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

static double now_s(void) { return (double)tlm_now_ns() * 1e-9; }

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

static double round6(double v) {   /* compact numbers in the report */
    if (!isfinite(v)) return v;
    char b[32];
    snprintf(b, sizeof b, "%.6g", v);
    return strtod(b, NULL);
}

static int byte_class(int token) {
    int b = token & 0xFF;
    if (b >= 'a' && b <= 'z') return 0;
    if (b >= 'A' && b <= 'Z') return 1;
    if (b >= '0' && b <= '9') return 2;
    if (b == ' ' || b == '\t' || b == '\n' || b == '\r') return 3;
    if (b > 32 && b < 127) return 4;
    return 5;
}

/* ---- SHA-256 (FIPS 180-4); tlm_packet.h exports none, so a local copy ---- */
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t fill; } sha256_ctx;
static const uint32_t SK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha_block(uint32_t h[8], const uint8_t *p) {
    uint32_t w[64], a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = k + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + SK[i] + w[i];
        uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}
static void sha_init(sha256_ctx *s) {
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(s->h, iv, sizeof iv);
    s->len = 0;
    s->fill = 0;
}
static void sha_update(sha256_ctx *s, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    s->len += n;
    while (n) {
        size_t take = 64 - s->fill < n ? 64 - s->fill : n;
        memcpy(s->buf + s->fill, p, take);
        s->fill += take;
        p += take;
        n -= take;
        if (s->fill == 64) { sha_block(s->h, s->buf); s->fill = 0; }
    }
}
static void sha_final(sha256_ctx *s, uint8_t out[32]) {
    uint64_t bits = s->len * 8;
    uint8_t one = 0x80, zero = 0;
    sha_update(s, &one, 1);
    while (s->fill != 56) sha_update(s, &zero, 1);
    uint8_t lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha_update(s, lb, 8);
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)(s->h[i] >> 24);
        out[4 * i + 1] = (uint8_t)(s->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(s->h[i] >> 8);
        out[4 * i + 3] = (uint8_t)s->h[i];
    }
}
static void hex32(const uint8_t *d, char *out) {
    for (int i = 0; i < 32; i++) snprintf(out + 2 * i, 3, "%02x", d[i]);
}
static int sha_file(const char *path, sha256_ctx *s) {
    FILE *f = fopen(path, "rb");
    if (!f) return tlm_seterr(TLM_EIO, "%s: %s", path, strerror(errno));
    static uint8_t buf[1 << 20];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) sha_update(s, buf, n);
    int bad = ferror(f);
    fclose(f);
    return bad ? tlm_seterr(TLM_EIO, "%s: read error", path) : TLM_OK;
}

/* ---- deterministic RNG (splitmix64) --------------------------------------- */
typedef struct { uint64_t s; } rng_t;
static uint64_t rng_next(rng_t *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double rng_unit(rng_t *r) { return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0); }
static float rng_normal(rng_t *r) {
    double u = 1.0 - rng_unit(r), v = rng_unit(r);
    return (float)(sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v));
}

/* ======================================================================
 * tlmc synth: a tiny Qwen3-MoE-like HF checkpoint with skewed routing
 * ==================================================================== */
static float *randn(rng_t *r, size_t n, float sd, float mean) {
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) die("out of memory");
    for (size_t i = 0; i < n; i++) p[i] = mean + sd * rng_normal(r);
    return p;
}

static void st_put(tlm_st_writer *w, const char *name, float *data, int ndim, int64_t a, int64_t b) {
    int64_t sh[2] = {a, b};
    if (tlm_st_writer_add(w, name, data, ndim, sh)) die("%s", tlm_errmsg());
    free(data);
}

static int cmd_synth(int argc, char **argv) {
    long L = 4, D = 256, H = 4, KV = 2, hd = 64, E = 16, K = 2, de = 256, shared = 1, V = 512;
    long seed = 1, qk = 1, router = 0;
    const char *out = NULL;
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--out")) out = need(argc, argv, &i);
        else if (!strcmp(a, "--layers")) L = num_arg(need(argc, argv, &i), a, 1, 512);
        else if (!strcmp(a, "--d-model")) D = num_arg(need(argc, argv, &i), a, 32, 65536);
        else if (!strcmp(a, "--heads")) H = num_arg(need(argc, argv, &i), a, 1, 512);
        else if (!strcmp(a, "--kv-heads")) KV = num_arg(need(argc, argv, &i), a, 1, 512);
        else if (!strcmp(a, "--head-dim")) hd = num_arg(need(argc, argv, &i), a, 2, 1024);
        else if (!strcmp(a, "--experts")) E = num_arg(need(argc, argv, &i), a, 1, MAX_E);
        else if (!strcmp(a, "--top-k")) K = num_arg(need(argc, argv, &i), a, 1, 64);
        else if (!strcmp(a, "--d-expert")) de = num_arg(need(argc, argv, &i), a, 32, 65536);
        else if (!strcmp(a, "--shared")) shared = num_arg(need(argc, argv, &i), a, 0, 64);
        else if (!strcmp(a, "--vocab")) V = num_arg(need(argc, argv, &i), a, 2, 1 << 22);
        else if (!strcmp(a, "--seed")) seed = num_arg(need(argc, argv, &i), a, 0, LONG_MAX);
        else if (!strcmp(a, "--router")) {
            const char *v = need(argc, argv, &i);
            if (!strcmp(v, "softmax")) router = 0;
            else if (!strcmp(v, "sigmoid")) router = 2;
            else die("--router: softmax or sigmoid");
        } else if (!strcmp(a, "--qk-norm")) {
            const char *v = need(argc, argv, &i);
            int b;
            if (tlm_cfg_parse_bool(v, &b)) die("--qk-norm: yes or no");
            qk = b;
        } else die("synth: unknown option %s", a);
    }
    if (!out) die("synth: --out DIR is required");
    if (H % KV) die("synth: --heads must be a multiple of --kv-heads");
    if (hd % 2) die("synth: --head-dim must be even");
    if (K > E) die("synth: --top-k exceeds --experts");
    if (D % 32 || de % 32) die("synth: --d-model and --d-expert must be multiples of 32");
    if (tlm_mkdirs(out, 0755)) die("%s", tlm_errmsg());

    rng_t r = {(uint64_t)seed * 0x9E3779B97F4A7C15ull + 12345};
    /* A shared direction u: every embedding row carries +u, and each router
     * row carries a_e * u with a_e graded over a per-layer permutation, so
     * routing is skewed: a few hot experts, a warm middle and rarely used
     * cold experts (what calibration, pinning and pruning need to see). */
    float *u = randn(&r, (size_t)D, 1.0f, 0.0f);
    double un = 0;
    for (long i = 0; i < D; i++) un += (double)u[i] * u[i];
    for (long i = 0; i < D; i++) u[i] = (float)(u[i] / sqrt(un));

    tlm_st_writer *w = tlm_st_writer_new();
    if (!w) die("out of memory");
    char nm[200];
    float sd_in = 1.0f / sqrtf((float)D);
    float *emb = randn(&r, (size_t)V * D, 1.0f, 0.0f);
    for (long t = 0; t < V; t++)
        for (long i = 0; i < D; i++) emb[t * D + i] += sqrtf((float)D) * u[i];
    st_put(w, "model.embed_tokens.weight", emb, 2, V, D);
    long S = shared * de;
    for (long l = 0; l < L; l++) {
#define NM(fmt) (snprintf(nm, sizeof nm, "model.layers.%ld." fmt, l), nm)
        st_put(w, NM("input_layernorm.weight"), randn(&r, (size_t)D, 0.1f, 1.0f), 1, D, 0);
        st_put(w, NM("self_attn.q_proj.weight"), randn(&r, (size_t)H * hd * D, sd_in, 0), 2, H * hd, D);
        st_put(w, NM("self_attn.k_proj.weight"), randn(&r, (size_t)KV * hd * D, sd_in, 0), 2, KV * hd, D);
        st_put(w, NM("self_attn.v_proj.weight"), randn(&r, (size_t)KV * hd * D, sd_in, 0), 2, KV * hd, D);
        st_put(w, NM("self_attn.o_proj.weight"), randn(&r, (size_t)D * H * hd, 0.5f / sqrtf((float)(H * hd)), 0),
               2, D, H * hd);
        if (qk) {
            st_put(w, NM("self_attn.q_norm.weight"), randn(&r, (size_t)hd, 0.1f, 1.0f), 1, hd, 0);
            st_put(w, NM("self_attn.k_norm.weight"), randn(&r, (size_t)hd, 0.1f, 1.0f), 1, hd, 0);
        }
        st_put(w, NM("post_attention_layernorm.weight"), randn(&r, (size_t)D, 0.1f, 1.0f), 1, D, 0);
        int *perm = (int *)malloc(sizeof(int) * (size_t)E);
        for (long e = 0; e < E; e++) perm[e] = (int)e;
        for (long e = E - 1; e > 0; e--) {
            long j = (long)(rng_next(&r) % (uint64_t)(e + 1));
            int t = perm[e];
            perm[e] = perm[j];
            perm[j] = t;
        }
        float *gate = randn(&r, (size_t)E * D, sd_in, 0);
        for (long e = 0; e < E; e++) {
            float a = E > 1 ? 1.0f - 2.0f * (float)perm[e] / (float)(E - 1) : 0.0f;
            /* x = rmsnorm(h) has x.u ~ 0.7 sqrt(D), so this adds ~1.1a to
             * the logit against unit-variance noise from the random part */
            for (long i = 0; i < D; i++) gate[e * D + i] += 1.6f / sqrtf((float)D) * a * u[i];
        }
        free(perm);
        st_put(w, NM("mlp.gate.weight"), gate, 2, E, D);
        if (router == 2) st_put(w, NM("mlp.gate.e_score_correction_bias"), randn(&r, (size_t)E, 0.05f, 0), 1, E, 0);
        for (long e = 0; e < E; e++) {
            char en[200];
            snprintf(en, sizeof en, "model.layers.%ld.mlp.experts.%ld.gate_proj.weight", l, e);
            st_put(w, en, randn(&r, (size_t)de * D, sd_in, 0), 2, de, D);
            snprintf(en, sizeof en, "model.layers.%ld.mlp.experts.%ld.up_proj.weight", l, e);
            st_put(w, en, randn(&r, (size_t)de * D, sd_in, 0), 2, de, D);
            snprintf(en, sizeof en, "model.layers.%ld.mlp.experts.%ld.down_proj.weight", l, e);
            st_put(w, en, randn(&r, (size_t)D * de, 0.5f / sqrtf((float)de), 0), 2, D, de);
        }
        if (S > 0) {
            st_put(w, NM("mlp.shared_expert.gate_proj.weight"), randn(&r, (size_t)S * D, sd_in, 0), 2, S, D);
            st_put(w, NM("mlp.shared_expert.up_proj.weight"), randn(&r, (size_t)S * D, sd_in, 0), 2, S, D);
            st_put(w, NM("mlp.shared_expert.down_proj.weight"), randn(&r, (size_t)D * S, 0.5f / sqrtf((float)S), 0),
                   2, D, S);
        }
#undef NM
    }
    st_put(w, "model.norm.weight", randn(&r, (size_t)D, 0.1f, 1.0f), 1, D, 0);
    st_put(w, "lm_head.weight", randn(&r, (size_t)V * D, 3.0f * sd_in, 0), 2, V, D);
    free(u);
    char path[1200];
    snprintf(path, sizeof path, "%s/model.safetensors", out);
    if (tlm_st_writer_save(w, path)) die("%s", tlm_errmsg());
    tlm_st_writer_free(w);

    snprintf(path, sizeof path, "%s/config.json", out);
    FILE *f = fopen(path, "w");
    if (!f) die("%s: %s", path, strerror(errno));
    fprintf(f,
            "{\n  \"architectures\": [\"Qwen3MoeForCausalLM\"],\n  \"model_type\": \"qwen3_moe\",\n"
            "  \"vocab_size\": %ld,\n  \"hidden_size\": %ld,\n  \"num_hidden_layers\": %ld,\n"
            "  \"num_attention_heads\": %ld,\n  \"num_key_value_heads\": %ld,\n  \"head_dim\": %ld,\n"
            "  \"num_experts\": %ld,\n  \"num_experts_per_tok\": %ld,\n  \"moe_intermediate_size\": %ld,\n"
            "  \"shared_expert_intermediate_size\": %ld,\n  \"n_shared_experts\": %ld,\n"
            "  \"norm_topk_prob\": true,\n  \"scoring_func\": \"%s\",\n  \"routed_scaling_factor\": %s,\n"
            "  \"rms_norm_eps\": 1e-06,\n  \"rope_theta\": 1000000.0,\n  \"max_position_embeddings\": 4096,\n"
            "  \"decoder_sparse_step\": 1,\n  \"mlp_only_layers\": [],\n  \"tie_word_embeddings\": false,\n"
            "  \"torch_dtype\": \"float32\",\n  \"tierlm_synthetic\": {\"seed\": %ld, \"generator\": \"tlmc %s\"}\n}\n",
            V, D, L, H, KV, hd, E, K, de, S, shared, router == 2 ? "sigmoid" : "softmax",
            router == 2 ? "2.5" : "1.0", seed, TLMC_VERSION);
    if (fclose(f)) die("%s: write error", path);
    printf("tlmc: synthetic model in %s (%ld layers, d_model %ld, %ld experts top-%ld, d_expert %ld, "
           "shared %ld, vocab %ld, %s router)\n",
           out, L, D, E, K, de, S, V, router == 2 ? "sigmoid" : "softmax");
    return 0;
}

/* ======================================================================
 * shared compile/report structures
 * ==================================================================== */
typedef struct { int *tok; int n; int *start; int nseq; } tokset;   /* seq i = [start[i], start[i+1]) */

static void tokset_free(tokset *t) { free(t->tok); free(t->start); memset(t, 0, sizeof *t); }

static void tokset_from_bytes(tokset *t, const uint8_t *b, size_t n, int V) {
    memset(t, 0, sizeof *t);
    t->tok = (int *)malloc(sizeof(int) * (n ? n : 1));
    t->start = (int *)malloc(sizeof(int) * (n / SEQ_LEN + 2));
    if (!t->tok || !t->start) die("out of memory");
    for (size_t i = 0; i < n; i++) t->tok[i] = b[i] % V;   /* byte-level tokens mod vocab */
    t->n = (int)n;
    for (int s = 0; s < t->n; s += SEQ_LEN) {
        if (t->n - s < 2 && t->nseq > 0) break;   /* drop a 1-token tail */
        t->start[t->nseq++] = s;
    }
    t->start[t->nseq] = t->n;
}

/* Pseudo-text so byte classes vary like real prompts (words, numbers,
 * punctuation, newlines). Different seeds never overlap in practice. */
static uint8_t *synth_text(uint64_t seed, size_t n) {
    uint8_t *b = (uint8_t *)malloc(n ? n : 1);
    if (!b) die("out of memory");
    rng_t r = {seed};
    size_t i = 0;
    while (i < n) {
        double k = rng_unit(&r);
        int len = 1 + (int)(rng_next(&r) % 9);
        for (int j = 0; j < len && i < n; j++) {
            if (k < 0.70) b[i++] = (uint8_t)('a' + rng_next(&r) % 26);
            else if (k < 0.80) b[i++] = (uint8_t)(j == 0 ? 'A' + rng_next(&r) % 26 : 'a' + rng_next(&r) % 26);
            else if (k < 0.92) b[i++] = (uint8_t)('0' + rng_next(&r) % 10);
            else b[i++] = (uint8_t)("{}()[];:,.=+-*/\"'<>"[rng_next(&r) % 19]);
        }
        if (i < n) b[i++] = rng_unit(&r) < 0.08 ? '\n' : ' ';
    }
    return b;
}

/* ---- runtime harness used by passes 3, 4, 5 and 9 ------------------------- */
typedef struct {
    tlm_manifest m;
    tlm_io      *io;
    tlm_store   *st;
    tlm_model   *mdl;
} rtm;

static void rtm_close(rtm *r) {
    tlm_model_close(r->mdl);
    tlm_store_close(r->st);
    tlm_io_close(r->io);
    tlm_manifest_free(&r->m);
    memset(r, 0, sizeof *r);
}

/* Every unpruned expert resident when RAM allows (calibration and eval read
 * each expert many times); otherwise as much as half the free RAM holds. */
static int rtm_open(rtm *r, const char *mpath, tlm_pool *pool) {
    memset(r, 0, sizeof *r);
    int rc = tlm_manifest_load(&r->m, mpath);
    if (rc) return rc;
    char paths[TLM_MAX_DRIVES][1024];
    const char *pp[TLM_MAX_DRIVES];
    for (uint32_t i = 0; i < r->m.n_drives; i++) {
        tlm_manifest_drive_path(&r->m, (int)i, paths[i], sizeof paths[i]);
        pp[i] = paths[i];
    }
    tlm_io_opts io = {TLM_IO_PREAD, 0, 2, 32};
    r->io = tlm_io_open(pp, (int)r->m.n_drives, &io);
    if (!r->io) { rc = TLM_EIO; goto fail; }
    uint64_t total = 0, slot = 0;
    for (uint32_t i = 0; i < r->m.n_packets; i++) {
        const tlm_packet_loc *l = &r->m.packets[i];
        if (l->type != TLM_PK_EXPERT || (l->flags & 2u)) continue;
        total += l->total_len;
        if (l->total_len > slot) slot = l->total_len;
    }
    tlm_mem_info mi;
    uint64_t cap = (tlm_mem_snapshot(&mi) == 0 && mi.available_bytes) ? mi.available_bytes / 2 : total;
    uint64_t floor_b = (uint64_t)(r->m.geom.top_k + 2) * slot;
    tlm_store_opts so;
    memset(&so, 0, sizeof so);
    so.budget_bytes = total < cap ? total : cap;
    if (so.budget_bytes < floor_b) so.budget_bytes = floor_b;
    so.prefetch = 1;
    r->st = tlm_store_open(&r->m, r->io, &so);
    if (!r->st) { rc = TLM_ENOMEM; goto fail; }
    tlm_model_opts mo = {1, 0};
    r->mdl = tlm_model_open(&r->m, r->io, r->st, pool, &mo);
    if (!r->mdl) { rc = TLM_EFORMAT; goto fail; }
    return TLM_OK;
fail:
    {
        char msg[512];
        snprintf(msg, sizeof msg, "%s", tlm_errmsg());
        rtm_close(r);
        return tlm_seterr(rc, "%s", msg);
    }
}

/* ---- evaluation: teacher forcing over a token set ------------------------- */
typedef struct { int npos, V; int *argmax; float *logp; } evalres;   /* logp may be NULL (too big) */

static void evalres_free(evalres *e) { free(e->argmax); free(e->logp); memset(e, 0, sizeof *e); }

typedef struct { double top1, kl; int identical; int npos; } evalcmp;

/* Run every sequence from a fresh KV; position s predicts token s+1. When
 * `ref` is given, compare against it; when `keep` is given, store results. */
static int eval_run(tlm_model *mdl, const tokset *ts, const evalres *ref, evalres *keep, evalcmp *cmp) {
    const tlm_model_geom *g = tlm_model_geom_of(mdl);
    int V = (int)g->vocab;
    float *lg = (float *)malloc(sizeof(float) * (size_t)V);
    double *lp = (double *)malloc(sizeof(double) * (size_t)V);
    if (!lg || !lp) die("out of memory");
    int store_lp = (uint64_t)ts->n * (uint64_t)V <= (128ull << 20);   /* <= 512 MB of f32 */
    if (keep) {
        memset(keep, 0, sizeof *keep);
        keep->V = V;
        keep->argmax = (int *)malloc(sizeof(int) * (size_t)(ts->n ? ts->n : 1));
        keep->logp = store_lp ? (float *)malloc(sizeof(float) * (size_t)ts->n * V) : NULL;
        if (!keep->argmax || (store_lp && !keep->logp)) die("out of memory");
    }
    tlm_kv *kv = tlm_kv_new(mdl, SEQ_LEN);
    if (!kv) die("%s", tlm_errmsg());
    int rc = TLM_OK, pos = 0, agree = 0, identical = 1;
    double kl = 0;
    for (int s = 0; s < ts->nseq && rc == TLM_OK; s++) {
        tlm_kv_reset(kv);
        for (int i = ts->start[s]; i < ts->start[s + 1] && rc == TLM_OK; i++) {
            rc = tlm_model_step(mdl, kv, ts->tok[i], lg);
            if (rc) break;
            int am = tlm_sample(NULL, NULL, lg, V);
            double mx = -INFINITY, sum = 0;
            for (int v = 0; v < V; v++) if (lg[v] > mx) mx = lg[v];
            for (int v = 0; v < V; v++) sum += exp((double)lg[v] - mx);
            double lse = mx + log(sum);
            for (int v = 0; v < V; v++) lp[v] = (double)lg[v] - lse;
            if (keep) {
                keep->argmax[pos] = am;
                if (keep->logp) for (int v = 0; v < V; v++) keep->logp[(size_t)pos * V + v] = (float)lp[v];
            }
            if (ref) {
                agree += ref->argmax[pos] == am;
                if (ref->argmax[pos] != am) identical = 0;
                if (ref->logp) {
                    const float *rp = ref->logp + (size_t)pos * V;
                    double k = 0;
                    for (int v = 0; v < V; v++) {   /* both sides rounded to f32 alike */
                        float cv = (float)lp[v];
                        if (cv != rp[v]) identical = 0;
                        double p = exp((double)rp[v]);
                        k += p * ((double)rp[v] - (double)cv);
                    }
                    kl += k > 0 ? k : 0;
                }
            }
            pos++;
        }
    }
    if (keep) keep->npos = pos;
    if (cmp) {
        cmp->npos = pos;
        cmp->top1 = pos ? (double)agree / pos : 1.0;
        cmp->kl = (pos && ref && ref->logp) ? kl / pos : NAN;
        cmp->identical = ref ? identical : 0;
    }
    tlm_kv_free(kv);
    free(lg);
    free(lp);
    return rc;
}

/* ---- calibration statistics (pass 3) ------------------------------------- */
typedef struct {
    uint64_t count, top1, cls[NCLS];
    double   sum_w, sum_n;
} ex_stat;

typedef struct {
    int       L, E;
    uint64_t  ntok, prior[NCLS];
    ex_stat  *s;        /* [L * E] */
    uint32_t *co;       /* [L * E * E], symmetric pair counts */
} calib;

static void calib_init(calib *c, int L, int E) {
    memset(c, 0, sizeof *c);
    c->L = L;
    c->E = E;
    c->s = (ex_stat *)calloc((size_t)L * E, sizeof *c->s);
    c->co = (uint32_t *)calloc((size_t)L * E * E, sizeof *c->co);
    if (!c->s || !c->co) die("out of memory for calibration statistics");
}
static void calib_free(calib *c) { free(c->s); free(c->co); memset(c, 0, sizeof *c); }

static void calib_hook(void *ctx, int layer, int pos, int token, const int *ids, const float *w,
                       const float *onorm, int k) {
    (void)pos;
    calib *c = (calib *)ctx;
    int cl = byte_class(token);
    if (layer == 0) { c->ntok++; c->prior[cl]++; }
    for (int r = 0; r < k; r++) {
        ex_stat *x = &c->s[(size_t)layer * c->E + ids[r]];
        x->count++;
        x->top1 += r == 0;
        x->cls[cl]++;
        x->sum_w += w[r];
        x->sum_n += onorm[r];   /* ||w * expert(x)|| : REAP's gate x output norm */
        for (int q = r + 1; q < k; q++) {
            int a = ids[r], b = ids[q];
            c->co[((size_t)layer * c->E + a) * c->E + b]++;
            c->co[((size_t)layer * c->E + b) * c->E + a]++;
        }
    }
}

static double heat_of(const calib *c, int l, int e) {
    return c->ntok ? (double)c->s[(size_t)l * c->E + e].count / (double)c->ntok : 0.0;
}
static double saliency_of(const calib *c, int l, int e) {
    const ex_stat *x = &c->s[(size_t)l * c->E + e];
    return x->count ? x->sum_n / (double)x->count : 0.0;
}
static double gate_of(const calib *c, int l, int e) {
    const ex_stat *x = &c->s[(size_t)l * c->E + e];
    return x->count ? x->sum_w / (double)x->count : 0.0;
}
/* Normalised entropy of the input byte-class distribution, enriched against
 * the corpus prior (so a corpus that is mostly letters does not make every
 * expert look like a letter specialist). -1 when there is too little data. */
static double class_entropy(const calib *c, int l, int e, int *top) {
    const ex_stat *x = &c->s[(size_t)l * c->E + e];
    double p[NCLS], z = 0;
    *top = NCLS - 1;
    if (x->count < 8) return -1.0;
    for (int k = 0; k < NCLS; k++) {
        p[k] = c->prior[k] ? (double)x->cls[k] / (double)c->prior[k] : 0.0;
        z += p[k];
    }
    if (!(z > 0)) return -1.0;
    double h = 0, best = -1;
    int nz = 0;
    for (int k = 0; k < NCLS; k++) nz += c->prior[k] > 0;
    for (int k = 0; k < NCLS; k++) {
        p[k] /= z;
        if (p[k] > best) { best = p[k]; *top = k; }
        if (p[k] > 0) h -= p[k] * log(p[k]);
    }
    return nz > 1 ? h / log((double)nz) : 0.0;
}

/* routing.bin: resumable pass-3 output (P-04). */
#define ROUTING_MAGIC 0x52434C54u   /* "TLCR" */
static int calib_save(const calib *c, const char *path, const uint8_t ref_sha[32], const uint8_t corp_sha[32]) {
    char tmp[1400];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return tlm_seterr(TLM_EIO, "%s: %s", tmp, strerror(errno));
    uint32_t hdr[4] = {ROUTING_MAGIC, 1, (uint32_t)c->L, (uint32_t)c->E};
    int ok = fwrite(hdr, sizeof hdr, 1, f) == 1 && fwrite(ref_sha, 32, 1, f) == 1 &&
             fwrite(corp_sha, 32, 1, f) == 1 && fwrite(&c->ntok, 8, 1, f) == 1 &&
             fwrite(c->prior, sizeof c->prior, 1, f) == 1 &&
             fwrite(c->s, sizeof *c->s, (size_t)c->L * c->E, f) == (size_t)c->L * c->E &&
             fwrite(c->co, sizeof *c->co, (size_t)c->L * c->E * c->E, f) == (size_t)c->L * c->E * c->E;
    ok = (fclose(f) == 0) && ok;
    if (!ok || rename(tmp, path) != 0) { unlink(tmp); return tlm_seterr(TLM_EIO, "%s: write failed", path); }
    return TLM_OK;
}
static int calib_load(calib *c, const char *path, const uint8_t ref_sha[32], const uint8_t corp_sha[32]) {
    FILE *f = fopen(path, "rb");
    if (!f) return TLM_ENOTFOUND;
    uint32_t hdr[4];
    uint8_t a[32], b[32];
    int ok = fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == ROUTING_MAGIC && hdr[1] == 1 &&
             hdr[2] == (uint32_t)c->L && hdr[3] == (uint32_t)c->E && fread(a, 32, 1, f) == 1 &&
             fread(b, 32, 1, f) == 1 && !memcmp(a, ref_sha, 32) && !memcmp(b, corp_sha, 32) &&
             fread(&c->ntok, 8, 1, f) == 1 && fread(c->prior, sizeof c->prior, 1, f) == 1 &&
             fread(c->s, sizeof *c->s, (size_t)c->L * c->E, f) == (size_t)c->L * c->E &&
             fread(c->co, sizeof *c->co, (size_t)c->L * c->E * c->E, f) == (size_t)c->L * c->E * c->E;
    fclose(f);
    return ok ? TLM_OK : TLM_EFORMAT;
}

/* ======================================================================
 * tlmc compile
 * ==================================================================== */
enum { PROF_EXACT = 0, PROF_BALANCED = 1, PROF_AGGRESSIVE = 2 };
static const char *const PROF_NAME[3] = {"exact", "balanced", "aggressive"};
static const double PROF_CAP[3] = {0.0, 0.25, 0.50};
static const double PROF_GATE[3] = {1.0, 0.97, 0.93};
enum { DEPTH_EXCLUDE = 0, DEPTH_SHALLOW, DEPTH_STANDARD, DEPTH_DEEP };
static const char *const DEPTH_NAME[4] = {"exclude", "shallow", "standard", "deep"};

typedef struct { int drive, replica; uint16_t flags; float heat; } place_t;

typedef struct {
    /* options */
    const char *src, *out, *corpus_path, *eval_path, *etc;
    int         n_drives;
    char        drive[TLM_MAX_DRIVES][256];
    double      drive_gbs[TLM_MAX_DRIVES];   /* probed read GB/s (mean of known if unknown) */
    int         drive_probed[TLM_MAX_DRIVES];
    const char *exp_name, *dense_name;
    int         profile, replicas, print_report, fresh, threads;
    long        calib_tokens, eval_tokens;
    /* derived */
    char        name[64], work[1100], refdir[1200], candir[1200], refman[1300], canman[1300], tmpman[1300];
    char        srcdir[1024];
    tlm_cfg    *cfg;
    tlm_decider *dec;
    tlm_model_geom g;
    tlm_st     *st;
    int         tie, has_qn, has_kn, has_bias, has_shared, has_sgate;
    const char *shared_prefix;   /* "shared_expert" or "shared_experts" */
    uint8_t     src_sha[32], corp_sha[32], ref_sha[32];
    int         n_src_files;
    tlm_dtype   dense_dt, exp_dt;
    tlm_dtype  *ex_dt;           /* [L*E] planned dtype per expert */
    int         n_demoted, prec_applied;
    evalcmp     prec_cmp;
    tokset      cal, ev;
    int         cal_synthetic, ev_synthetic;
    calib       C;
    evalres     R, A;            /* reference eval; accepted (pruned/demoted) eval */
    int         have_A;
    tlm_pool   *pool;
    uint8_t    *pruned, *protect, *exclude;
    int         depth;
    int         n_pruned;
    double      prune_ratio, prune_cap, gate;
    char       *attempts_json;   /* serialised attempt list */
    size_t      attempts_len;
    FILE       *attempts;
    int         n_attempts;
    int        *spec;            /* [L*E] specialisation index or -1 */
    float      *p_needed;        /* [L*E] */
    int         n_decisions, n_applied, n_defaulted;
    place_t    *pl;              /* [L*E] */
    int         n_pinned, n_replicated;
    evalcmp     val;
    int         free_run_identical;
    double      t[10];
    int         reused_ref, reused_cal;
} cc;

static int depth_of(const char *v) {
    for (int i = 0; i < 4; i++) if (v && !strcasecmp(v, DEPTH_NAME[i])) return i;
    return DEPTH_STANDARD;
}

/* "12:178, 17:44" -> flags[l*E+e] = 1 */
static void parse_ids(cc *c, const char *key, uint8_t *flags) {
    const char *items[1024];
    int n = tlm_cfg_list(c->cfg, key, items, 1024);
    for (int i = 0; i < n; i++) {
        int l, e;
        char extra;
        if (sscanf(items[i], " %d:%d %c", &l, &e, &extra) != 2 || l < 0 || e < 0 ||
            (uint32_t)l >= c->g.n_layers || (uint32_t)e >= c->g.n_experts) {
            fprintf(stderr, "tlmc: warning: %s: ignoring '%s' (want layer:expert)\n", key, items[i]);
            continue;
        }
        flags[(size_t)l * c->g.n_experts + e] = 1;
    }
}

/* ---- pass 1: ingest -------------------------------------------------------- */
static const tlm_jval *jget(const tlm_jval *o, const char *k) { return tlm_jobj_get(o, k); }

static int file_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static void pass_ingest(cc *c) {
    double t0 = now_s();
    struct stat sst;
    if (stat(c->src, &sst) != 0) die("ingest: %s: %s", c->src, strerror(errno));
    if (S_ISDIR(sst.st_mode)) snprintf(c->srcdir, sizeof c->srcdir, "%s", c->src);
    else {
        snprintf(c->srcdir, sizeof c->srcdir, "%s", c->src);
        char *sl = strrchr(c->srcdir, '/');
        if (sl == c->srcdir) sl[1] = 0;   /* "/model.safetensors" -> "/" */
        else if (sl) *sl = 0;
        else snprintf(c->srcdir, sizeof c->srcdir, ".");
    }
    char path[1200], err[256];
    snprintf(path, sizeof path, "%s/config.json", c->srcdir);
    tlm_json *j = tlm_json_parse_file(path, err, sizeof err);
    if (!j) die("ingest: %s: %s", path, err);
    const tlm_jval *r = tlm_json_root(j);
    tlm_model_geom *g = &c->g;
    memset(g, 0, sizeof *g);
    snprintf(g->arch, sizeof g->arch, "tlm-moe");
    g->n_layers = (uint32_t)tlm_jint(jget(r, "num_hidden_layers"), 0);
    g->d_model = (uint32_t)tlm_jint(jget(r, "hidden_size"), 0);
    g->n_heads = (uint32_t)tlm_jint(jget(r, "num_attention_heads"), 0);
    g->n_kv_heads = (uint32_t)tlm_jint(jget(r, "num_key_value_heads"), g->n_heads);
    g->head_dim = (uint32_t)tlm_jint(jget(r, "head_dim"), g->n_heads ? g->d_model / g->n_heads : 0);
    g->vocab = (uint32_t)tlm_jint(jget(r, "vocab_size"), 0);
    int64_t ne = tlm_jint(jget(r, "num_experts"), -1);
    if (ne < 0) ne = tlm_jint(jget(r, "n_routed_experts"), -1);
    if (ne < 0) ne = tlm_jint(jget(r, "num_local_experts"), 0);
    g->n_experts = (uint32_t)ne;
    g->top_k = (uint32_t)tlm_jint(jget(r, "num_experts_per_tok"), 0);
    g->d_expert = (uint32_t)tlm_jint(jget(r, "moe_intermediate_size"), 0);
    int64_t nsh = tlm_jint(jget(r, "n_shared_experts"), 0);
    int64_t dsh = tlm_jint(jget(r, "shared_expert_intermediate_size"), nsh * (int64_t)g->d_expert);
    g->d_shared = (uint32_t)(dsh > 0 ? dsh : 0);
    g->n_shared = g->d_shared ? (uint32_t)(nsh > 0 ? nsh : 1) : 0;
    int64_t mpe = tlm_jint(jget(r, "max_position_embeddings"), 4096);
    g->max_ctx = (uint32_t)(mpe > 0 && mpe < (1 << 24) ? mpe : 4096);
    g->rope_theta = (float)tlm_jnum(jget(r, "rope_theta"), 10000.0);
    g->norm_eps = (float)tlm_jnum(jget(r, "rms_norm_eps"), 1e-6);
    g->routed_scale = (float)tlm_jnum(jget(r, "routed_scaling_factor"), 1.0);
    const char *sf = tlm_jstr(jget(r, "scoring_func"), "softmax");
    if (!strcmp(sf, "sigmoid")) g->router = 2;
    else if (!strcmp(sf, "softmax")) g->router = tlm_jbool(jget(r, "norm_topk_prob"), 0) ? 0 : 1;
    else die("ingest: scoring_func '%s' is not supported", sf);
    c->tie = tlm_jbool(jget(r, "tie_word_embeddings"), 0);
    if (tlm_jint(jget(r, "decoder_sparse_step"), 1) != 1 || tlm_jlen(jget(r, "mlp_only_layers")) > 0 ||
        tlm_jint(jget(r, "first_k_dense_replace"), 0) > 0)
        die("ingest: dense FFN layers are not supported by the tlm-moe decoder yet");
    if (jget(r, "rope_scaling") && jget(r, "rope_scaling")->type == TLM_J_OBJ)
        fprintf(stderr, "tlmc: warning: rope_scaling is ignored (plain RoPE, F-fwd-03)\n");
    tlm_json_free(j);
    if (!g->n_layers || !g->d_model || !g->n_heads || !g->n_kv_heads || !g->head_dim || !g->vocab ||
        !g->n_experts || !g->top_k || !g->d_expert || g->top_k > g->n_experts || g->n_experts > MAX_E ||
        g->n_heads % g->n_kv_heads || g->head_dim % 2 || g->top_k > 64)
        die("ingest: %s: unsupported or incomplete geometry", path);

    c->st = tlm_st_open(c->src);
    if (!c->st) die("ingest: %s", tlm_errmsg());

    /* SHA-256 over config.json, the index and every shard, in name order */
    DIR *d = opendir(c->srcdir);
    if (!d) die("ingest: %s: %s", c->srcdir, strerror(errno));
    char **files = NULL;
    int nf = 0, capf = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        size_t L = strlen(de->d_name);
        int want = !strcmp(de->d_name, "config.json") || !strcmp(de->d_name, "model.safetensors.index.json") ||
                   (L > 12 && !strcmp(de->d_name + L - 12, ".safetensors"));
        if (!want) continue;
        if (nf == capf) {
            capf = capf ? capf * 2 : 16;
            files = (char **)realloc(files, sizeof(char *) * (size_t)capf);
            if (!files) die("out of memory");
        }
        files[nf++] = strdup(de->d_name);
    }
    closedir(d);
    qsort(files, (size_t)nf, sizeof *files, file_cmp);
    sha256_ctx all;
    sha_init(&all);
    char fp[1300], hx[65];
    snprintf(path, sizeof path, "%s/ingest.json", c->work);
    FILE *ij = fopen(path, "w");
    if (!ij) die("%s: %s", path, strerror(errno));
    fprintf(ij, "{\"source\":\"%s\",\"files\":[", c->src);
    for (int i = 0; i < nf; i++) {
        snprintf(fp, sizeof fp, "%s/%s", c->srcdir, files[i]);
        sha256_ctx one;
        sha_init(&one);
        if (sha_file(fp, &one)) die("ingest: %s", tlm_errmsg());
        uint8_t dg[32];
        sha_final(&one, dg);
        hex32(dg, hx);
        sha_update(&all, files[i], strlen(files[i]) + 1);
        sha_update(&all, dg, 32);
        fprintf(ij, "%s{\"name\":\"%s\",\"sha256\":\"%s\"}", i ? "," : "", files[i], hx);
        free(files[i]);
    }
    free(files);
    c->n_src_files = nf;
    sha_final(&all, c->src_sha);
    hex32(c->src_sha, hx);

    /* tensor map (HF Qwen3-MoE / Qwen2-MoE / DeepSeek naming) */
    char nm[256];
    const tlm_st_tensor *t;
    t = tlm_st_find(c->st, "model.layers.0.self_attn.q_proj.bias");
    if (t) die("ingest: attention biases are not supported");
    c->has_qn = tlm_st_find(c->st, "model.layers.0.self_attn.q_norm.weight") != NULL;
    c->has_kn = tlm_st_find(c->st, "model.layers.0.self_attn.k_norm.weight") != NULL;
    c->has_bias = tlm_st_find(c->st, "model.layers.0.mlp.gate.e_score_correction_bias") != NULL;
    c->shared_prefix = tlm_st_find(c->st, "model.layers.0.mlp.shared_experts.gate_proj.weight") ? "shared_experts"
                                                                                                 : "shared_expert";
    snprintf(nm, sizeof nm, "model.layers.0.mlp.%s.gate_proj.weight", c->shared_prefix);
    c->has_shared = tlm_st_find(c->st, nm) != NULL;
    c->has_sgate = tlm_st_find(c->st, "model.layers.0.mlp.shared_expert_gate.weight") != NULL;
    if (c->has_shared && !g->d_shared) {
        t = tlm_st_find(c->st, nm);
        g->d_shared = (uint32_t)t->shape[0];
        g->n_shared = 1;
    }
    if (!c->has_shared) { g->d_shared = 0; g->n_shared = 0; }
    if (g->router == 2 && !c->has_bias)
        fprintf(stderr, "tlmc: note: sigmoid router without e_score_correction_bias (bias = 0)\n");
    fprintf(ij, "],\"source_sha256\":\"%s\",\"tensors\":%d,\"geometry\":{\"layers\":%u,\"d_model\":%u,"
            "\"heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,\"vocab\":%u,\"experts\":%u,\"top_k\":%u,"
            "\"d_expert\":%u,\"d_shared\":%u,\"router\":%u,\"qk_norm\":%s,\"tie_embeddings\":%s}}\n",
            hx, tlm_st_count(c->st), g->n_layers, g->d_model, g->n_heads, g->n_kv_heads, g->head_dim,
            g->vocab, g->n_experts, g->top_k, g->d_expert, g->d_shared, g->router,
            c->has_qn ? "true" : "false", c->tie ? "true" : "false");
    fclose(ij);
    c->t[1] = now_s() - t0;
}

/* Read a source tensor as f32 after checking its shape is [rows, cols]
 * (cols = 0 for 1-D). */
static float *src_tensor(cc *c, const char *name, int64_t rows, int64_t cols) {
    const tlm_st_tensor *t = tlm_st_find(c->st, name);
    if (!t) die("ingest: tensor %s is missing", name);
    int ok = cols ? (t->ndim == 2 && t->shape[0] == rows && t->shape[1] == cols)
                  : ((t->ndim == 1 && t->shape[0] == rows) || (t->ndim == 2 && t->shape[0] == 1 && t->shape[1] == rows));
    if (!ok) die("ingest: tensor %s has an unexpected shape", name);
    if (t->is_fp8) die("ingest: %s is FP8; FP8 checkpoints need their block scales (not supported here)", name);
    float *p = (float *)malloc(sizeof(float) * (size_t)(rows * (cols ? cols : 1)));
    if (!p) die("out of memory");
    if (tlm_st_read_f32(c->st, t, p)) die("ingest: %s", tlm_errmsg());
    for (int64_t i = 0; i < rows * (cols ? cols : 1); i++)
        if (!isfinite(p[i])) die("ingest: %s contains NaN/Inf (P-05)", name);
    return p;
}

/* A dtype usable for a matrix with `cols` columns; dense never below Q8_0. */
static tlm_dtype fit_dtype(tlm_dtype want, int64_t cols) {
    if (cols % tlm_dtype_block(want) == 0) return want;
    if (cols % 32 == 0 && want != TLM_DT_F16 && want != TLM_DT_BF16) return TLM_DT_Q8_0;
    return TLM_DT_F32;
}

typedef struct { tlm_tensor_in t[16]; void *own[16]; int n; } dirb;

static void dir_put(dirb *d, const char *name, tlm_dtype dt, float *w, int64_t rows, int64_t cols) {
    size_t rb = tlm_row_bytes(dt, cols);
    uint8_t *q = (uint8_t *)malloc(rb * (size_t)rows);
    if (!q || !rb) die("out of memory / bad row size for %s", name);
    for (int64_t r = 0; r < rows; r++)
        if (tlm_quantize_row(dt, w + r * cols, q + (size_t)r * rb, cols)) die("quantise %s: %s", name, tlm_errmsg());
    free(w);
    d->t[d->n] = (tlm_tensor_in){name, dt, (uint64_t)rows, (uint64_t)cols, q, (uint64_t)rb * rows};
    d->own[d->n] = q;
    d->n++;
}

static void dir_done(dirb *d, void **pl, uint64_t *len) {
    if (tlm_dir_build(d->t, d->n, pl, len)) die("%s", tlm_errmsg());
    for (int i = 0; i < d->n; i++) free(d->own[i]);
    d->n = 0;
}

static void build_dense(cc *c, int l, void **pl, uint64_t *len) {
    const tlm_model_geom *g = &c->g;
    int64_t D = g->d_model, Q = (int64_t)g->n_heads * g->head_dim, KVd = (int64_t)g->n_kv_heads * g->head_dim;
    char n[24][160];
    int k = 0;
#define TN(fmt) (snprintf(n[k], sizeof n[k], "model.layers.%d." fmt, l), n[k++])
    dirb d = {0};
    dir_put(&d, "attn_norm", TLM_DT_F32, src_tensor(c, TN("input_layernorm.weight"), D, 0), 1, D);
    dir_put(&d, "ffn_norm", TLM_DT_F32, src_tensor(c, TN("post_attention_layernorm.weight"), D, 0), 1, D);
    if (c->has_qn) dir_put(&d, "q_norm", TLM_DT_F32, src_tensor(c, TN("self_attn.q_norm.weight"), g->head_dim, 0), 1, g->head_dim);
    if (c->has_kn) dir_put(&d, "k_norm", TLM_DT_F32, src_tensor(c, TN("self_attn.k_norm.weight"), g->head_dim, 0), 1, g->head_dim);
    dir_put(&d, "attn_q", fit_dtype(c->dense_dt, D), src_tensor(c, TN("self_attn.q_proj.weight"), Q, D), Q, D);
    dir_put(&d, "attn_k", fit_dtype(c->dense_dt, D), src_tensor(c, TN("self_attn.k_proj.weight"), KVd, D), KVd, D);
    dir_put(&d, "attn_v", fit_dtype(c->dense_dt, D), src_tensor(c, TN("self_attn.v_proj.weight"), KVd, D), KVd, D);
    dir_put(&d, "attn_o", fit_dtype(c->dense_dt, Q), src_tensor(c, TN("self_attn.o_proj.weight"), D, Q), D, Q);
    /* router and its bias stay f32: small and sensitive (F-fwd-01, F-fwd-16) */
    dir_put(&d, "router", TLM_DT_F32, src_tensor(c, TN("mlp.gate.weight"), g->n_experts, D), g->n_experts, D);
    if (c->has_bias)
        dir_put(&d, "router_bias", TLM_DT_F32, src_tensor(c, TN("mlp.gate.e_score_correction_bias"), g->n_experts, 0),
                1, g->n_experts);
    if (c->has_shared) {
        int64_t S = g->d_shared;
        snprintf(n[k], sizeof n[k], "model.layers.%d.mlp.%s.gate_proj.weight", l, c->shared_prefix);
        dir_put(&d, "shared_gate", fit_dtype(c->dense_dt, D), src_tensor(c, n[k++], S, D), S, D);
        snprintf(n[k], sizeof n[k], "model.layers.%d.mlp.%s.up_proj.weight", l, c->shared_prefix);
        dir_put(&d, "shared_up", fit_dtype(c->dense_dt, D), src_tensor(c, n[k++], S, D), S, D);
        snprintf(n[k], sizeof n[k], "model.layers.%d.mlp.%s.down_proj.weight", l, c->shared_prefix);
        dir_put(&d, "shared_down", fit_dtype(c->dense_dt, S), src_tensor(c, n[k++], D, S), D, S);
        if (c->has_sgate)
            dir_put(&d, "shared_gate_inp", TLM_DT_F32, src_tensor(c, TN("mlp.shared_expert_gate.weight"), D, 0), 1, D);
    }
#undef TN
    dir_done(&d, pl, len);
}

static void build_meta(cc *c, void **pl, uint64_t *len) {
    const tlm_model_geom *g = &c->g;
    dirb d = {0};
    dir_put(&d, "output_norm", TLM_DT_F32, src_tensor(c, "model.norm.weight", g->d_model, 0), 1, g->d_model);
    const char *head = c->tie ? "model.embed_tokens.weight" : "lm_head.weight";
    dir_put(&d, "lm_head", fit_dtype(c->dense_dt, g->d_model), src_tensor(c, head, g->vocab, g->d_model), g->vocab,
            g->d_model);
    dir_done(&d, pl, len);
}

static void build_embed(cc *c, void **pl, uint64_t *len) {
    const tlm_model_geom *g = &c->g;
    dirb d = {0};
    dir_put(&d, "tok_embd", fit_dtype(c->dense_dt, g->d_model),
            src_tensor(c, "model.embed_tokens.weight", g->vocab, g->d_model), g->vocab, g->d_model);
    dir_done(&d, pl, len);
}

/* EXPERT payload: gate rows | up rows | down rows, 64-byte aligned items. */
static void build_expert(cc *c, int l, int e, tlm_dtype dt, uint8_t **pl, uint64_t *len, uint32_t off[3]) {
    const tlm_model_geom *g = &c->g;
    int64_t D = g->d_model, de = g->d_expert;
    static const char *const part[3] = {"gate_proj", "up_proj", "down_proj"};
    size_t rb[3] = {tlm_row_bytes(dt, D), tlm_row_bytes(dt, D), tlm_row_bytes(dt, de)};
    int64_t rows[3] = {de, de, D}, cols[3] = {D, D, de};
    uint64_t o = 0;
    for (int i = 0; i < 3; i++) {
        off[i] = (uint32_t)o;
        o = TLM_ALIGN_UP(o + rb[i] * (uint64_t)rows[i], 64);
    }
    *len = off[2] + rb[2] * (uint64_t)rows[2];
    uint8_t *p = (uint8_t *)calloc(1, (size_t)*len);
    if (!p) die("out of memory");
    for (int i = 0; i < 3; i++) {
        char nm[200];
        snprintf(nm, sizeof nm, "model.layers.%d.mlp.experts.%d.%s.weight", l, e, part[i]);
        float *w = src_tensor(c, nm, rows[i], cols[i]);
        for (int64_t r = 0; r < rows[i]; r++)
            if (tlm_quantize_row(dt, w + r * cols[i], p + off[i] + (size_t)r * rb[i], cols[i]))
                die("quantise %s: %s", nm, tlm_errmsg());
        free(w);
    }
    *pl = p;
}

/* Write a one-drive store from the source with the given per-expert plan. */
static void store_build(cc *c, const char *dir, const char *mpath, const tlm_dtype *plan) {
    const tlm_model_geom *g = &c->g;
    const char *dirs[1] = {dir};
    tlm_store_writer *w = tlm_writer_open(c->name, dirs, 1);
    if (!w) die("%s", tlm_errmsg());
    size_t np = 2 + (size_t)g->n_layers * (1 + g->n_experts), n = 0;
    tlm_packet_loc *locs = (tlm_packet_loc *)calloc(np, sizeof *locs);
    if (!locs) die("out of memory");
    void *pl;
    uint64_t len;
    build_meta(c, &pl, &len);
    if (tlm_writer_add(w, TLM_PK_META, TLM_LAYER_NONE, 0, c->dense_dt, pl, len, NULL, 0, -1, &locs[n++]))
        die("%s", tlm_errmsg());
    tlm_free_aligned(pl);
    build_embed(c, &pl, &len);
    if (tlm_writer_add(w, TLM_PK_EMBED, TLM_LAYER_NONE, 0, c->dense_dt, pl, len, NULL, 0, -1, &locs[n++]))
        die("%s", tlm_errmsg());
    tlm_free_aligned(pl);
    for (uint32_t l = 0; l < g->n_layers; l++) {
        build_dense(c, (int)l, &pl, &len);
        if (tlm_writer_add(w, TLM_PK_DENSE, (uint16_t)l, 0, c->dense_dt, pl, len, NULL, 0, -1, &locs[n++]))
            die("%s", tlm_errmsg());
        tlm_free_aligned(pl);
        for (uint32_t e = 0; e < g->n_experts; e++) {
            uint8_t *ep;
            uint32_t off[3];
            tlm_dtype dt = plan[(size_t)l * g->n_experts + e];
            build_expert(c, (int)l, (int)e, dt, &ep, &len, off);
            if (tlm_writer_add(w, TLM_PK_EXPERT, (uint16_t)l, e, dt, ep, len, off, 0, -1, &locs[n++]))
                die("%s", tlm_errmsg());
            free(ep);
        }
    }
    if (tlm_writer_finish(w, 1)) die("%s", tlm_errmsg());   /* P-05: re-read every CRC */
    tlm_writer_free(w);
    tlm_manifest m;
    memset(&m, 0, sizeof m);
    m.magic = TLM_MANIFEST_MAGIC;
    m.version = TLM_STORE_VERSION;
    m.geom = c->g;
    snprintf(m.name, sizeof m.name, "%s", c->name);
    snprintf(m.profile, sizeof m.profile, "exact");
    m.n_drives = 1;
    snprintf(m.drives[0], sizeof m.drives[0], "%s", dir);
    memcpy(m.source_sha256, c->src_sha, 32);
    m.n_packets = (uint32_t)n;
    m.packets = locs;
    if (tlm_manifest_save(&m, mpath)) die("%s", tlm_errmsg());
    free(locs);
}

/* ---- pass 2 ------------------------------------------------------------------ */
static void pass_reference(cc *c) {
    double t0 = now_s();
    const tlm_model_geom *g = &c->g;
    /* dense dtype: never below Q8_0 (research/04 hard rules) */
    if (!strcasecmp(c->dense_name, "native")) c->dense_dt = TLM_DT_F32;
    else if (tlm_dtype_from_name(c->dense_name, &c->dense_dt)) die("--dense: %s", tlm_errmsg());
    if (c->dense_dt != TLM_DT_Q8_0 && c->dense_dt != TLM_DT_F16 && c->dense_dt != TLM_DT_BF16 &&
        c->dense_dt != TLM_DT_F32)
        die("--dense: %s is below Q8_0; dense weights allow q8_0, f16, bf16, f32", c->dense_name);
    tlm_dtype want;
    if (!strcasecmp(c->exp_name, "native")) {
        const tlm_st_tensor *t = tlm_st_find(c->st, "model.layers.0.mlp.experts.0.gate_proj.weight");
        want = t ? t->dtype : TLM_DT_F32;
        if (t && t->is_fp8) die("--experts native: FP8 experts are not supported yet");
    } else if (tlm_dtype_from_name(c->exp_name, &want)) {
        die("--experts: %s", tlm_errmsg());
    }
    if (want == TLM_DT_Q8_K || want == TLM_DT_FP8E4M3_B128) die("--experts: %s is not a weight format here", c->exp_name);
    c->exp_dt = want;
    int blk = tlm_dtype_block(want);
    if (g->d_model % blk || g->d_expert % blk) {
        c->exp_dt = (g->d_model % 32 == 0 && g->d_expert % 32 == 0) ? TLM_DT_Q8_0 : TLM_DT_F32;
        fprintf(stderr, "tlmc: warning: %s needs dimensions divisible by %d; experts use %s\n",
                tlm_dtype_name(want), blk, tlm_dtype_name(c->exp_dt));
    }
    size_t nx = (size_t)g->n_layers * g->n_experts;
    c->ex_dt = (tlm_dtype *)malloc(sizeof(tlm_dtype) * nx);
    if (!c->ex_dt) die("out of memory");
    for (size_t i = 0; i < nx; i++) c->ex_dt[i] = c->exp_dt;

    /* resumable (P-04): reuse the scratch store if its stamp matches */
    char stamp[1300], want_stamp[400], hx[65];
    hex32(c->src_sha, hx);
    snprintf(want_stamp, sizeof want_stamp, "tlmc %s src %s dense %s experts %s\n", TLMC_VERSION, hx,
             tlm_dtype_name(c->dense_dt), tlm_dtype_name(c->exp_dt));
    snprintf(stamp, sizeof stamp, "%s/ref.stamp", c->work);
    char *have = NULL;
    size_t hl = 0;
    tlm_manifest m;
    if (!c->fresh && tlm_read_file(stamp, &have, &hl) == 0 && !strcmp(have, want_stamp) &&
        tlm_manifest_load(&m, c->refman) == 0) {
        memcpy(c->ref_sha, m.table_sha256, 32);
        tlm_manifest_free(&m);
        c->reused_ref = 1;
    } else {
        unlink(stamp);
        if (tlm_mkdirs(c->refdir, 0755)) die("%s", tlm_errmsg());
        store_build(c, c->refdir, c->refman, c->ex_dt);
        if (tlm_manifest_load(&m, c->refman)) die("%s", tlm_errmsg());
        memcpy(c->ref_sha, m.table_sha256, 32);
        tlm_manifest_free(&m);
        if (tlm_write_file_atomic(stamp, want_stamp, strlen(want_stamp))) die("%s", tlm_errmsg());
    }
    free(have);
    char shp[1300];
    hex32(c->ref_sha, hx);
    snprintf(shp, sizeof shp, "%s/ref.sha256", c->work);
    char line[100];
    snprintf(line, sizeof line, "%s\n", hx);
    tlm_write_file_atomic(shp, line, strlen(line));
    c->t[2] = now_s() - t0;
}

/* ---- token sets ---------------------------------------------------------------- */
static void load_tokens(cc *c) {
    int V = (int)c->g.vocab;
    sha256_ctx s;
    sha_init(&s);
    uint8_t *cb = NULL, *eb = NULL;
    size_t cn = 0, en = 0;
    if (c->corpus_path) {
        char *t = NULL;
        if (tlm_read_file(c->corpus_path, &t, &cn)) die("--corpus: %s", tlm_errmsg());
        cb = (uint8_t *)t;
        sha_update(&s, "corpus", 7);
        sha_update(&s, cb, cn);
    }
    if (c->eval_path) {
        char *t = NULL;
        if (tlm_read_file(c->eval_path, &t, &en)) die("--eval: %s", tlm_errmsg());
        eb = (uint8_t *)t;
    }
    if (cb && eb && cn == en && !memcmp(cb, eb, cn))   /* research/03: sets never overlap */
        die("--corpus and --eval are the same text; the evaluation set must be held out");
    if (cb && !eb) {   /* hold out the last 20% for evaluation: never overlaps */
        size_t cut = cn - cn / 5;
        en = cn - cut;
        eb = (uint8_t *)malloc(en ? en : 1);
        if (!eb) die("out of memory");
        memcpy(eb, cb + cut, en);
        cn = cut;
    }
    if (!cb) {
        size_t n = c->calib_tokens > 0 ? (size_t)c->calib_tokens : 2048;
        cb = synth_text(0xC0FFEEull, n);
        cn = n;
        c->cal_synthetic = 1;
        char tag[64];
        snprintf(tag, sizeof tag, "synthetic calib %zu", n);
        sha_update(&s, tag, strlen(tag));
    }
    if (!eb) {
        size_t n = c->eval_tokens > 0 ? (size_t)c->eval_tokens : 512;
        eb = synth_text(0xE7A1ull, n);
        en = n;
        c->ev_synthetic = 1;
    }
    if (c->calib_tokens > 0 && cn > (size_t)c->calib_tokens) cn = (size_t)c->calib_tokens;
    if (c->eval_tokens > 0 && en > (size_t)c->eval_tokens) en = (size_t)c->eval_tokens;
    if (cn < 2 || en < 2) die("calibration and evaluation sets need at least 2 tokens each");
    sha_final(&s, c->corp_sha);
    tokset_from_bytes(&c->cal, cb, cn, V);
    tokset_from_bytes(&c->ev, eb, en, V);
    free(cb);
    free(eb);
}

/* ---- decisions ------------------------------------------------------------------ */
static const char *SPEC_OPTS[4] = {   /* tlm_question takes const char ** */"generalist", "broad", "focused", "narrow"};
static const char *SPEC_CRIT[4] = {
    "inputs spread over every byte class", "inputs over most byte classes",
    "inputs concentrated on a few byte classes", "inputs almost all from one byte class"};
static const char *PREC_OPTS[3] = {"q8_0", "q6_k", "q5_k"};
static const char *PREC_CRIT[3] = {"8-bit, for hot or deep-content experts", "6-bit K-quant, the default",
                                         "5-bit K-quant, for cold experts of shallow content"};

static void decide_one(cc *c, const tlm_question *q, tlm_feature *f, int nf, tlm_answer *a) {
    tlm_state st = {f, nf, NULL};
    if (tlm_decide(c->dec, &st, q, 1, a)) die("decide %s: %s", q->id, tlm_errmsg());
    c->n_decisions++;
    if (a->outcome == TLM_POLICY_APPLY) c->n_applied++;
    else c->n_defaulted++;
}

/* expert.specialisation for the atlas (all profiles; changes nothing in exact). */
static void label_experts(cc *c) {
    const tlm_model_geom *g = &c->g;
    size_t nx = (size_t)g->n_layers * g->n_experts;
    c->spec = (int *)malloc(sizeof(int) * nx);
    if (!c->spec) die("out of memory");
    tlm_question q = {"expert.specialisation", TLM_Q_SCORE,
                      "How specialised is this expert's calibration input (byte classes)?",
                      SPEC_OPTS, SPEC_CRIT, 4, 0};
    for (uint32_t l = 0; l < g->n_layers; l++)
        for (uint32_t e = 0; e < g->n_experts; e++) {
            int top;
            double h = class_entropy(&c->C, (int)l, (int)e, &top);
            tlm_feature f[3] = {{"label_entropy", h}, {"layer", l}, {"expert", e}};
            tlm_answer a;
            /* no label_entropy (too few selections) -> rule passes -> safe default */
            decide_one(c, &q, h >= 0 ? f : f + 1, h >= 0 ? 3 : 2, &a);
            c->spec[(size_t)l * g->n_experts + e] = a.final_choice;
        }
}

/* ---- pass 3 ------------------------------------------------------------------------ */
static void pass_calibrate(cc *c, rtm *ref) {
    double t0 = now_s();
    calib_init(&c->C, (int)c->g.n_layers, (int)c->g.n_experts);
    char rpath[1300];
    snprintf(rpath, sizeof rpath, "%s/routing.bin", c->work);
    if (!c->fresh && calib_load(&c->C, rpath, c->ref_sha, c->corp_sha) == TLM_OK) {
        c->reused_cal = 1;
    } else {
        memset(c->C.s, 0, sizeof *c->C.s * (size_t)c->C.L * c->C.E);
        memset(c->C.co, 0, sizeof *c->C.co * (size_t)c->C.L * c->C.E * c->C.E);
        c->C.ntok = 0;
        memset(c->C.prior, 0, sizeof c->C.prior);
        tlm_model_set_route_hook(ref->mdl, calib_hook, &c->C);
        tlm_kv *kv = tlm_kv_new(ref->mdl, SEQ_LEN);
        if (!kv) die("%s", tlm_errmsg());
        for (int s = 0; s < c->cal.nseq; s++) {
            tlm_kv_reset(kv);
            int n = c->cal.start[s + 1] - c->cal.start[s];
            if (tlm_model_prefill(ref->mdl, kv, c->cal.tok + c->cal.start[s], n, NULL))
                die("calibrate: %s", tlm_errmsg());
        }
        tlm_kv_free(kv);
        tlm_model_set_route_hook(ref->mdl, NULL, NULL);
        if (calib_save(&c->C, rpath, c->ref_sha, c->corp_sha)) die("%s", tlm_errmsg());
    }
    if (c->C.ntok < MIN_CALIB_WARN)
        fprintf(stderr, "tlmc: warning: %llu calibration tokens; routing statistics from fewer than %d "
                "tokens overfit (research/03 'Compiler inputs')\n", (unsigned long long)c->C.ntok, MIN_CALIB_WARN);
    /* reference evaluation: the 'exact' model every later pass is measured against */
    evalcmp dummy;
    if (eval_run(ref->mdl, &c->ev, NULL, &c->R, &dummy)) die("eval: %s", tlm_errmsg());
    label_experts(c);
    c->t[3] = now_s() - t0;
}

/* ---- pass 4 ------------------------------------------------------------------------ */
static void pass_precision(cc *c, int *use_cand) {
    double t0 = now_s();
    const tlm_model_geom *g = &c->g;
    *use_cand = 0;
    c->n_demoted = 0;
    if (c->profile != PROF_EXACT && c->exp_dt == TLM_DT_Q6_K) {
        tlm_question q = {"expert.precision_tier", TLM_Q_CHOICE, "Which precision tier should this expert use?",
                          PREC_OPTS, PREC_CRIT, 3, 0};
        for (uint32_t l = 0; l < g->n_layers; l++) {
            double hmax = 0, smax = 0;
            for (uint32_t e = 0; e < g->n_experts; e++) {
                if (heat_of(&c->C, (int)l, (int)e) > hmax) hmax = heat_of(&c->C, (int)l, (int)e);
                if (saliency_of(&c->C, (int)l, (int)e) > smax) smax = saliency_of(&c->C, (int)l, (int)e);
            }
            for (uint32_t e = 0; e < g->n_experts; e++) {
                size_t ix = (size_t)l * g->n_experts + e;
                tlm_feature f[3] = {{"heat", hmax > 0 ? heat_of(&c->C, (int)l, (int)e) / hmax : 0},
                                    {"saliency", smax > 0 ? saliency_of(&c->C, (int)l, (int)e) / smax : 0},
                                    {"depth", (double)c->depth}};
                tlm_answer a;
                decide_one(c, &q, f, 3, &a);
                /* this slice only demotes (never promotes beyond --experts) */
                if (a.final_choice == 2 && !c->protect[ix]) {
                    c->ex_dt[ix] = TLM_DT_Q5_K;
                    c->n_demoted++;
                }
            }
        }
    }
    char path[1300];
    snprintf(path, sizeof path, "%s/precision.json", c->work);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "{\"profile\":\"%s\",\"dense\":\"%s\",\"experts\":\"%s\",\"router\":\"f32\",\"norms\":\"f32\","
                "\"demoted_to_q5_k\":[", PROF_NAME[c->profile], tlm_dtype_name(c->dense_dt), tlm_dtype_name(c->exp_dt));
        int first = 1;
        for (size_t i = 0; i < (size_t)g->n_layers * g->n_experts; i++)
            if (c->ex_dt[i] != c->exp_dt) {
                fprintf(f, "%s\"%zu:%zu\"", first ? "" : ",", i / g->n_experts, i % g->n_experts);
                first = 0;
            }
        fprintf(f, "]}\n");
        fclose(f);
    }
    if (c->n_demoted) {
        /* candidate store with the plan, then the same gate as pruning */
        if (tlm_mkdirs(c->candir, 0755)) die("%s", tlm_errmsg());
        store_build(c, c->candir, c->canman, c->ex_dt);
        rtm cand;
        if (rtm_open(&cand, c->canman, c->pool)) die("candidate store: %s", tlm_errmsg());
        if (eval_run(cand.mdl, &c->ev, &c->R, NULL, &c->prec_cmp)) die("eval: %s", tlm_errmsg());
        rtm_close(&cand);
        if (c->prec_cmp.top1 >= c->gate) {
            *use_cand = 1;
            c->prec_applied = 1;
        } else {
            fprintf(stderr, "tlmc: precision plan failed the gate (top-1 %.2f%% < %.0f%%); rolled back\n",
                    100 * c->prec_cmp.top1, 100 * c->gate);
            for (size_t i = 0; i < (size_t)g->n_layers * g->n_experts; i++) c->ex_dt[i] = c->exp_dt;
        }
    }
    c->t[4] = now_s() - t0;
}

/* ---- pass 5 ------------------------------------------------------------------------ */
typedef struct { double score; int l, e; } cand_t;
static int cand_cmp(const void *a, const void *b) {
    const cand_t *x = (const cand_t *)a, *y = (const cand_t *)b;
    if (x->score != y->score) return x->score > y->score ? -1 : 1;
    if (x->l != y->l) return x->l - y->l;
    return x->e - y->e;
}

static void pass_prune(cc *c, rtm *base) {
    double t0 = now_s();
    const tlm_model_geom *g = &c->g;
    int L = (int)g->n_layers, E = (int)g->n_experts;
    size_t nx = (size_t)L * E;
    c->pruned = (uint8_t *)calloc(nx, 1);
    c->p_needed = (float *)malloc(sizeof(float) * nx);
    if (!c->pruned || !c->p_needed) die("out of memory");
    for (size_t i = 0; i < nx; i++) c->p_needed[i] = 1.0f;
    c->attempts = open_memstream(&c->attempts_json, &c->attempts_len);
    if (!c->attempts) die("out of memory");
    fputc('[', c->attempts);
    if (c->profile == PROF_EXACT) {
        fputc(']', c->attempts);
        fclose(c->attempts);
        c->attempts = NULL;
        c->t[5] = now_s() - t0;
        return;
    }
    double cap = PROF_CAP[c->profile];
    double ratio = cap;
    if (tlm_cfg_origin_of(c->cfg, "prune.max_ratio", NULL) != TLM_ORIGIN_DEFAULT) {
        double r = tlm_cfg_frac(c->cfg, "prune.max_ratio", cap);
        if (r < ratio) ratio = r;
    }
    double min_keep = tlm_cfg_frac(c->cfg, "prune.min_keep_per_layer", 0.5);
    int floor_l = (int)ceil(min_keep * E - 1e-9);
    if (floor_l < (int)g->top_k * 4) floor_l = (int)g->top_k * 4;   /* hard rule: never below top_k x 4 */
    if (floor_l > E) floor_l = E;
    c->prune_cap = ratio;

    /* expert.needed_for_usecase (NOUL). Without content labels every expert's
     * mass sits in content.default; exclusion only boosts focused/narrow
     * experts (research/04 hard rules); deep content is protected. */
    tlm_question q = {"expert.needed_for_usecase", TLM_Q_NOUL,
                      "This expert is needed for the selected content types", NULL, NULL, 0, 1};
    cand_t *cands = (cand_t *)malloc(sizeof(cand_t) * nx);
    if (!cands) die("out of memory");
    int nc = 0;
    for (int l = 0; l < L; l++) {
        double smax = 0;
        for (int e = 0; e < E; e++) if (saliency_of(&c->C, l, e) > smax) smax = saliency_of(&c->C, l, e);
        for (int e = 0; e < E; e++) {
            size_t ix = (size_t)l * E + e;
            double sn = smax > 0 ? saliency_of(&c->C, l, e) / smax : 0.0;
            int prot = c->protect[ix] || c->depth == DEPTH_DEEP;
            tlm_feature f[4] = {{"saliency_kept", sn},
                                {"mass_excluded", c->depth == DEPTH_EXCLUDE ? 1.0 : 0.0},
                                {"mass_deep", c->depth == DEPTH_DEEP ? 1.0 : 0.0},
                                {"protected", prot ? 1.0 : 0.0}};
            tlm_answer a;
            decide_one(c, &q, f, 4, &a);
            c->p_needed[ix] = a.prob[0];
            if (prot) continue;   /* hard rule: protect and deep always win */
            double p_not = 0.0;
            int specialist = c->spec[ix] >= 2;   /* focused or narrow */
            if (a.final_choice == 0 && a.outcome == TLM_POLICY_APPLY && specialist) p_not = 1.0 - a.prob[0];
            double score = (1.0 - sn) * (0.5 + 0.5 * p_not);
            if (c->exclude[ix]) score += 2.0;   /* explicit removals first, still gated */
            cands[nc++] = (cand_t){score, l, e};
        }
    }
    qsort(cands, (size_t)nc, sizeof *cands, cand_cmp);

    int total = L * E, accepted = 0, prev_got = -1;
    int pct = (int)floor(ratio * 100.0 + 1e-9);
    uint8_t *mask = (uint8_t *)calloc(nx, 1);
    int *per = (int *)calloc((size_t)L, sizeof(int));
    if (!mask || !per) die("out of memory");
    for (; pct > 0 && !accepted; pct -= 5) {   /* 5-point back-off */
        int target = (int)floor(pct / 100.0 * total + 1e-9), got = 0;
        memset(mask, 0, nx);
        memset(per, 0, sizeof(int) * (size_t)L);
        for (int i = 0; i < nc && got < target; i++) {
            if (E - per[cands[i].l] - 1 < floor_l) continue;
            mask[(size_t)cands[i].l * E + cands[i].e] = 1;
            per[cands[i].l]++;
            got++;
        }
        if (got == prev_got) continue;   /* floors bind: same selection as the failed attempt */
        prev_got = got;
        if (got == 0) {
            fprintf(c->attempts, "%s{\"ratio\":%.2f,\"pruned\":0,\"note\":\"per-layer floor %d leaves nothing to prune\"}",
                    c->n_attempts++ ? "," : "", pct / 100.0, floor_l);
            break;
        }
        if (tlm_model_set_expert_mask(base->mdl, mask)) die("prune: %s", tlm_errmsg());
        evalcmp ec;
        evalres keep;
        if (eval_run(base->mdl, &c->ev, &c->R, &keep, &ec)) die("eval: %s", tlm_errmsg());
        int pass = ec.top1 >= c->gate;
        fprintf(c->attempts, "%s{\"ratio\":%.2f,\"pruned\":%d,\"top1\":%.6f,\"mean_kl\":%.6g,\"pass\":%s}",
                c->n_attempts++ ? "," : "", pct / 100.0, got, ec.top1, isfinite(ec.kl) ? ec.kl : -1.0,
                pass ? "true" : "false");
        printf("tlmc: prune attempt %d%%: %d experts, top-1 agreement %.2f%% (gate %.0f%%) -> %s\n", pct, got,
               100 * ec.top1, 100 * c->gate, pass ? "pass" : "back off 5 points");
        if (pass) {
            accepted = 1;
            memcpy(c->pruned, mask, nx);
            c->n_pruned = got;
            c->prune_ratio = pct / 100.0;
            evalres_free(&c->A);
            c->A = keep;
            c->have_A = 1;
        } else {
            evalres_free(&keep);
        }
    }
    tlm_model_set_expert_mask(base->mdl, NULL);
    fputc(']', c->attempts);
    fclose(c->attempts);
    c->attempts = NULL;
    free(mask);
    free(per);
    free(cands);
    char path[1300];
    snprintf(path, sizeof path, "%s/prune.json", c->work);
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "{\"profile\":\"%s\",\"cap\":%.2f,\"floor_per_layer\":%d,\"gate_top1\":%.2f,\"pruned\":[",
                PROF_NAME[c->profile], ratio, floor_l, c->gate);
        int first = 1;
        for (size_t i = 0; i < nx; i++)
            if (c->pruned[i]) { fprintf(f, "%s\"%zu:%zu\"", first ? "" : ",", i / E, i % E); first = 0; }
        fprintf(f, "],\"attempts\":%s}\n", c->attempts_json);
        fclose(f);
    }
    c->t[5] = now_s() - t0;
}

/* ---- pass 8 ------------------------------------------------------------------------ */
/* Measured read bandwidth of each compile drive from hardware.conf
 * (tierlm-setup --probe: hardware.drive.<i>.path / direct_gbs /
 * buffered_gbs); the longest probed path that is a prefix of the drive dir
 * wins. Unknown drives get 0 (treated as the mean of the known ones). */
static void drive_bandwidth(cc *c) {
    int nprobe = (int)tlm_cfg_int(c->cfg, "hardware.drive.count", 0);
    for (int d = 0; d < c->n_drives; d++) {
        size_t best = 0;
        c->drive_gbs[d] = 0;
        for (int i = 0; i < nprobe && i < 256; i++) {
            char k[96];
            snprintf(k, sizeof k, "hardware.drive.%d.path", i);
            const char *pth = tlm_cfg_str(c->cfg, k);
            if (!pth || !*pth) continue;
            char rp[PATH_MAX];
            const char *pp = realpath(pth, rp) ? rp : pth;
            size_t pl = strlen(pp);
            while (pl > 1 && pp[pl - 1] == '/') pl--;
            if (strncmp(c->drive[d], pp, pl) || (c->drive[d][pl] && c->drive[d][pl] != '/' && pl > 1)) continue;
            if (pl < best) continue;
            snprintf(k, sizeof k, "hardware.drive.%d.direct_gbs", i);
            const char *v = tlm_cfg_str(c->cfg, k);
            if (!v) {
                snprintf(k, sizeof k, "hardware.drive.%d.buffered_gbs", i);
                v = tlm_cfg_str(c->cfg, k);
            }
            double gbs = v ? strtod(v, NULL) : 0;
            if (gbs > 0) { c->drive_gbs[d] = gbs; c->drive_probed[d] = 1; best = pl; }
        }
    }
    double sum = 0;
    int n = 0;
    for (int d = 0; d < c->n_drives; d++) if (c->drive_gbs[d] > 0) { sum += c->drive_gbs[d]; n++; }
    for (int d = 0; d < c->n_drives; d++)
        if (!(c->drive_gbs[d] > 0)) c->drive_gbs[d] = n ? sum / n : 1.0;
}

typedef struct { int e; double heat; } heat_ix;
static int heat_desc(const void *a, const void *b) {
    const heat_ix *x = (const heat_ix *)a, *y = (const heat_ix *)b;
    if (x->heat != y->heat) return x->heat > y->heat ? -1 : 1;
    return x->e - y->e;
}

static void pass_place(cc *c, const char *src_manifest, tlm_manifest *out_m) {
    double t0 = now_s();
    const tlm_model_geom *g = &c->g;
    int L = (int)g->n_layers, E = (int)g->n_experts, ND = c->n_drives;
    size_t nx = (size_t)L * E;
    tlm_manifest sm;
    if (tlm_manifest_load(&sm, src_manifest)) die("%s", tlm_errmsg());
    drive_bandwidth(c);
    c->pl = (place_t *)calloc(nx, sizeof *c->pl);
    if (!c->pl) die("out of memory");

    /* compiler pins: the hottest ~10% of surviving experts (P-19); replicas:
     * the next ~30% by heat, the warm middle (P-31, research/03 rule 4) */
    heat_ix *hx = (heat_ix *)malloc(sizeof(heat_ix) * (nx ? nx : 1));
    if (!hx) die("out of memory");
    int nh = 0;
    for (size_t i = 0; i < nx; i++) {
        double h = heat_of(&c->C, (int)(i / E), (int)(i % E));
        c->pl[i].heat = (float)h;
        c->pl[i].replica = -1;
        if (c->pruned[i]) c->pl[i].flags |= 2u;
        else hx[nh++] = (heat_ix){(int)i, h};
    }
    qsort(hx, (size_t)nh, sizeof *hx, heat_desc);
    int n_pin = (nh + 9) / 10, n_rep = (ND > 1 && c->replicas) ? (nh * 3 + 9) / 10 : 0;
    c->n_pinned = 0;
    for (int i = 0; i < n_pin && i < nh; i++)
        if (hx[i].heat > 0) { c->pl[hx[i].e].flags |= 1u; c->n_pinned++; }

    /* drive assignment: expected read load = heat x bytes for streamed
     * experts; within a layer, co-activated experts are pushed apart */
    double load[TLM_MAX_DRIVES] = {0}, bytes[TLM_MAX_DRIVES] = {0};
    for (int l = 0; l < L; l++) {
        heat_ix *lx = (heat_ix *)malloc(sizeof(heat_ix) * (size_t)E);
        if (!lx) die("out of memory");
        for (int e = 0; e < E; e++) lx[e] = (heat_ix){e, heat_of(&c->C, l, e)};
        qsort(lx, (size_t)E, sizeof *lx, heat_desc);
        for (int q = 0; q < E; q++) {
            int e = lx[q].e;
            size_t ix = (size_t)l * E + e;
            const tlm_packet_loc *loc = tlm_manifest_find(&sm, TLM_PK_EXPERT, (uint16_t)l, (uint32_t)e);
            double sz = loc ? (double)loc->total_len : 0;
            double rd = (c->pl[ix].flags & 3u) ? 0.0 : lx[q].heat * sz;   /* pinned/pruned: no streaming */
            int best = 0;
            double bc = INFINITY;
            for (int d = 0; d < ND; d++) {
                double cost;
                if (rd > 0) {   /* streamed: balance expected reads, spread co-firing experts */
                    double pen = 0;
                    for (int q2 = 0; q2 < q; q2++) {
                        int e2 = lx[q2].e;
                        if (c->pl[(size_t)l * E + e2].drive != d) continue;
                        pen += c->C.ntok ? (double)c->C.co[((size_t)l * E + e) * E + e2] / (double)c->C.ntok * sz : 0;
                    }
                    /* balance by expected read time, not bytes (research/03 rule 3) */
                    cost = (load[d] + rd + 0.5 * pen) / c->drive_gbs[d] + 1e-12 * bytes[d];
                } else {        /* pinned, pruned or never selected: balance capacity */
                    cost = bytes[d] + 1e-12 * load[d];
                }
                if (cost < bc) { bc = cost; best = d; }
            }
            c->pl[ix].drive = best;
            load[best] += rd;
            bytes[best] += sz;
        }
        free(lx);
    }
    /* replicas: warm middle, on the least loaded other drive; reads then
     * split between the two copies (shorter queue wins at run time) */
    c->n_replicated = 0;
    for (int i = n_pin; i < n_pin + n_rep && i < nh; i++) {
        int ix = hx[i].e;
        if (!(hx[i].heat > 0)) break;
        int pd = c->pl[ix].drive, best = -1;
        for (int d = 0; d < ND; d++)
            if (d != pd && (best < 0 || load[d] / c->drive_gbs[d] < load[best] / c->drive_gbs[best])) best = d;
        if (best < 0) continue;
        const tlm_packet_loc *loc = tlm_manifest_find(&sm, TLM_PK_EXPERT, (uint16_t)(ix / E), (uint32_t)(ix % E));
        double rd = hx[i].heat * (loc ? (double)loc->total_len : 0);
        /* the shorter queue wins at run time (P-31): reads split roughly in
         * proportion to the two drives' bandwidth */
        double moved = rd * c->drive_gbs[best] / (c->drive_gbs[pd] + c->drive_gbs[best]);
        load[pd] -= moved;
        load[best] += moved;
        bytes[best] += loc ? (double)loc->total_len : 0;
        c->pl[ix].replica = best;
        c->n_replicated++;
    }
    free(hx);

    /* write: global packets first, then layer-major so router-lookahead
     * prefetch of layer l+1 touches nearby offsets (research/03 rule 5) */
    char paths[TLM_MAX_DRIVES][1024];
    const char *srcp[1];
    tlm_manifest_drive_path(&sm, 0, paths[0], sizeof paths[0]);
    srcp[0] = paths[0];
    tlm_io_opts io = {TLM_IO_PREAD, 0, 2, 32};
    tlm_io *sio = tlm_io_open(srcp, 1, &io);
    if (!sio) die("%s", tlm_errmsg());
    const char *dp[TLM_MAX_DRIVES];
    for (int d = 0; d < ND; d++) dp[d] = c->drive[d];
    tlm_store_writer *w = tlm_writer_open(c->name, dp, ND);
    if (!w) die("%s", tlm_errmsg());
    tlm_packet_loc *locs = (tlm_packet_loc *)calloc(sm.n_packets, sizeof *locs);
    if (!locs) die("out of memory");
    uint32_t n = 0;
    double dense_bytes[TLM_MAX_DRIVES] = {0};
    for (int pass = 0; pass < 2 + L; pass++) {
        for (uint32_t i = 0; i < sm.n_packets; i++) {
            const tlm_packet_loc *s = &sm.packets[i];
            int want = pass == 0 ? s->type == TLM_PK_META
                     : pass == 1 ? s->type == TLM_PK_EMBED
                                 : (s->layer == pass - 2 && (s->type == TLM_PK_DENSE || s->type == TLM_PK_EXPERT));
            if (!want) continue;
            uint8_t *buf = (uint8_t *)tlm_alloc_aligned((size_t)s->total_len, TLM_ALIGN_IO);
            if (!buf) die("out of memory");
            const void *payload;
            uint64_t plen;
            if (tlm_packet_read(sio, s, buf, &payload, &plen)) die("place: %s", tlm_errmsg());
            const tlm_packet_hdr *h = (const tlm_packet_hdr *)buf;
            int drive, rep = -1;
            uint16_t flags = 0;
            float heat = 0;
            if (s->type == TLM_PK_EXPERT) {
                const place_t *p = &c->pl[(size_t)s->layer * E + s->id];
                drive = p->drive;
                rep = p->replica;
                flags = p->flags;
                heat = p->heat;
            } else {   /* resident packets: spread by bytes so load reads all drives */
                drive = 0;
                for (int d = 1; d < ND; d++) if (dense_bytes[d] < dense_bytes[drive]) drive = d;
                dense_bytes[drive] += (double)s->total_len;
            }
            uint32_t item_off[3];
            memcpy(item_off, (const uint8_t *)buf + offsetof(tlm_packet_hdr, item_off), sizeof item_off);
            if (tlm_writer_add(w, (tlm_packet_type)s->type, s->layer, s->id, h->dtype, payload, plen, item_off,
                               drive, rep, &locs[n]))
                die("place: %s", tlm_errmsg());
            locs[n].flags = flags;
            locs[n].heat = heat;
            n++;
            tlm_free_aligned(buf);
        }
    }
    tlm_io_close(sio);
    if (tlm_writer_finish(w, 1)) die("place: %s", tlm_errmsg());   /* P-05 */
    tlm_writer_free(w);

    memset(out_m, 0, sizeof *out_m);
    out_m->magic = TLM_MANIFEST_MAGIC;
    out_m->version = TLM_STORE_VERSION;
    out_m->geom = c->g;
    snprintf(out_m->name, sizeof out_m->name, "%s", c->name);
    snprintf(out_m->profile, sizeof out_m->profile, "%s", PROF_NAME[c->profile]);
    out_m->n_drives = (uint32_t)ND;
    for (int d = 0; d < ND; d++) snprintf(out_m->drives[d], sizeof out_m->drives[d], "%s", c->drive[d]);
    memcpy(out_m->source_sha256, c->src_sha, 32);
    memcpy(out_m->corpus_sha256, c->corp_sha, 32);
    out_m->n_packets = n;
    out_m->packets = locs;
    out_m->n_pruned = (uint32_t)c->n_pruned;
    tlm_manifest_free(&sm);
    c->t[8] = now_s() - t0;
}

/* ---- pass 9 ------------------------------------------------------------------------ */
static void pass_validate(cc *c) {
    double t0 = now_s();
    rtm fin;
    if (rtm_open(&fin, c->tmpman, c->pool)) die("validate: %s", tlm_errmsg());
    /* the compiled store must compute exactly what the gate measured */
    const evalres *expect = c->have_A ? &c->A : &c->R;
    evalcmp same;
    if (eval_run(fin.mdl, &c->ev, expect, NULL, &same)) die("validate: %s", tlm_errmsg());
    if (!same.identical)
        die("validate: the placed store does not reproduce the %s evaluation bit for bit (top-1 %.4f)",
            c->have_A ? "accepted" : "reference", same.top1);
    if (eval_run(fin.mdl, &c->ev, &c->R, NULL, &c->val)) die("validate: %s", tlm_errmsg());
    /* greedy free run from the first eval prompt, compared with pass 2 */
    rtm ref;
    if (rtm_open(&ref, c->refman, c->pool)) die("validate: %s", tlm_errmsg());
    int np = c->ev.start[1] - c->ev.start[0];
    if (np > 16) np = 16;
    int a[32], b[32], na = 0, nb = 0;
    tlm_kv *k1 = tlm_kv_new(fin.mdl, 64), *k2 = tlm_kv_new(ref.mdl, 64);
    tlm_sampler gs = {0.0f, 1.0f, 0.0f, 0};
    float *lg = (float *)malloc(sizeof(float) * c->g.vocab);
    if (!lg) die("out of memory");
    for (int pass = 0; pass < 2; pass++) {
        tlm_model *m = pass ? ref.mdl : fin.mdl;
        tlm_kv *kv = pass ? k2 : k1;
        int *out = pass ? b : a, *cnt = pass ? &nb : &na;
        if (tlm_model_prefill(m, kv, c->ev.tok + c->ev.start[0], np, lg)) die("validate: %s", tlm_errmsg());
        for (int i = 0; i < 32; i++) {
            int t = tlm_sample(&gs, NULL, lg, (int)c->g.vocab);
            out[(*cnt)++] = t;
            if (i + 1 < 32 && tlm_model_step(m, kv, t, lg)) die("validate: %s", tlm_errmsg());
        }
    }
    free(lg);
    c->free_run_identical = na == nb && !memcmp(a, b, sizeof(int) * (size_t)na);
    tlm_kv_free(k1);
    tlm_kv_free(k2);
    rtm_close(&ref);
    rtm_close(&fin);
    if (c->profile == PROF_EXACT && (!c->val.identical || !c->free_run_identical))
        die("validate: exact profile is not token-identical to the pass-2 reference");
    if (c->profile != PROF_EXACT && c->val.top1 < c->gate)
        die("validate: top-1 agreement %.4f below the %s gate %.2f", c->val.top1, PROF_NAME[c->profile], c->gate);
    c->t[9] = now_s() - t0;
}

/* ---- report -------------------------------------------------------------------------- */
static char *build_report(cc *c, const tlm_manifest *m) {
    const tlm_model_geom *g = &c->g;
    int E = (int)g->n_experts;
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    if (!f) die("out of memory");
    tlm_jw w;
    tlm_jw_init(&w, f);
    char hx[65];
    tlm_jw_begin_obj(&w);
    tlm_jw_str(&w, "tlmc_version", TLMC_VERSION);
    tlm_jw_str(&w, "name", c->name);
    tlm_jw_str(&w, "profile", PROF_NAME[c->profile]);
    tlm_jw_str(&w, "source", c->src);
    hex32(c->src_sha, hx);
    tlm_jw_str(&w, "source_sha256", hx);
    tlm_jw_int(&w, "source_files", c->n_src_files);
    hex32(c->corp_sha, hx);
    tlm_jw_str(&w, "corpus_sha256", hx);
    hex32(c->ref_sha, hx);
    tlm_jw_str(&w, "reference_table_sha256", hx);
    tlm_jw_key_obj(&w, "geometry");
    tlm_jw_int(&w, "layers", g->n_layers);
    tlm_jw_int(&w, "d_model", g->d_model);
    tlm_jw_int(&w, "heads", g->n_heads);
    tlm_jw_int(&w, "kv_heads", g->n_kv_heads);
    tlm_jw_int(&w, "head_dim", g->head_dim);
    tlm_jw_int(&w, "vocab", g->vocab);
    tlm_jw_int(&w, "experts", g->n_experts);
    tlm_jw_int(&w, "top_k", g->top_k);
    tlm_jw_int(&w, "d_expert", g->d_expert);
    tlm_jw_int(&w, "d_shared", g->d_shared);
    tlm_jw_str(&w, "router", g->router == 2 ? "sigmoid+bias" : g->router == 1 ? "softmax-topk" : "softmax-topk-renorm");
    tlm_jw_end_obj(&w);
    tlm_jw_key_obj(&w, "precision");
    tlm_jw_str(&w, "dense", tlm_dtype_name(c->dense_dt));
    tlm_jw_str(&w, "experts", tlm_dtype_name(c->exp_dt));
    tlm_jw_str(&w, "router", "f32");
    tlm_jw_str(&w, "norms", "f32");
    tlm_jw_int(&w, "demoted_q5_k", c->prec_applied ? c->n_demoted : 0);
    tlm_jw_int(&w, "proposed_q5_k", c->n_demoted);
    if (c->n_demoted) tlm_jw_num(&w, "plan_top1", round6(c->prec_cmp.top1));
    tlm_jw_end_obj(&w);
    tlm_jw_key_obj(&w, "calibration");
    tlm_jw_int(&w, "tokens", (int64_t)c->C.ntok);
    tlm_jw_int(&w, "sequences", c->cal.nseq);
    tlm_jw_str(&w, "source", c->cal_synthetic ? "synthetic" : c->corpus_path);
    tlm_jw_bool(&w, "reused", c->reused_cal);
    tlm_jw_bool(&w, "below_200k_warning", c->C.ntok < MIN_CALIB_WARN);
    tlm_jw_begin_arr(&w, "class_prior");
    for (int k = 0; k < NCLS; k++) tlm_jw_int(&w, NULL, (int64_t)c->C.prior[k]);
    tlm_jw_end_arr(&w);
    tlm_jw_end_obj(&w);
    tlm_jw_key_obj(&w, "prune");
    tlm_jw_str(&w, "content_default", DEPTH_NAME[c->depth]);
    tlm_jw_num(&w, "cap", round6(c->prune_cap));
    tlm_jw_num(&w, "ratio", round6(c->prune_ratio));
    tlm_jw_int(&w, "pruned", c->n_pruned);
    tlm_jw_num(&w, "gate_top1", c->profile == PROF_EXACT ? 1.0 : c->gate);
    fprintf(f, ",\"attempts\":%s", c->attempts_json ? c->attempts_json : "[]");
    tlm_jw_end_obj(&w);
    /* placement summary per drive */
    tlm_jw_key_obj(&w, "placement");
    tlm_jw_int(&w, "pinned", c->n_pinned);
    tlm_jw_int(&w, "replicated", c->n_replicated);
    tlm_jw_begin_arr(&w, "drives");
    for (uint32_t d = 0; d < m->n_drives; d++) {
        uint64_t pk = 0, by = 0, rep = 0;
        double ld = 0;
        for (uint32_t i = 0; i < m->n_packets; i++) {
            const tlm_packet_loc *l = &m->packets[i];
            int prim = l->drive == d, sec = l->replica_drive == d;
            if (!prim && !sec) continue;
            pk++;
            by += l->total_len;
            rep += sec;
            if (l->type == TLM_PK_EXPERT && !(l->flags & 3u)) {
                double share = 1.0;
                if (l->replica_drive != 0xFF) {
                    double a = c->drive_gbs[l->drive], b = c->drive_gbs[l->replica_drive];
                    share = (prim ? a : b) / (a + b);
                }
                ld += (double)l->heat * (double)l->total_len * share;
            }
        }
        tlm_jw_begin_obj(&w);
        tlm_jw_str(&w, "path", m->drives[d]);
        tlm_jw_int(&w, "packets", (int64_t)pk);
        tlm_jw_int(&w, "bytes", (int64_t)by);
        tlm_jw_int(&w, "replicas", (int64_t)rep);
        tlm_jw_num(&w, "expected_read_bytes_per_token", round6(ld));
        if (d < (uint32_t)c->n_drives && c->drive_probed[d]) tlm_jw_num(&w, "read_gbs", round6(c->drive_gbs[d]));
        else tlm_jw_null(&w, "read_gbs");
        tlm_jw_end_obj(&w);
    }
    tlm_jw_end_arr(&w);
    tlm_jw_end_obj(&w);
    tlm_jw_key_obj(&w, "eval");
    tlm_jw_int(&w, "tokens", c->val.npos);
    tlm_jw_str(&w, "source", c->ev_synthetic ? "synthetic" : (c->eval_path ? c->eval_path : "corpus holdout (20%)"));
    tlm_jw_num(&w, "top1_agreement", round6(c->val.top1));
    if (isfinite(c->val.kl)) tlm_jw_num(&w, "mean_kl", round6(c->val.kl));
    else tlm_jw_null(&w, "mean_kl");
    tlm_jw_bool(&w, "bit_identical", c->val.identical);
    tlm_jw_bool(&w, "free_run_identical", c->free_run_identical);
    tlm_jw_num(&w, "gate_top1", c->profile == PROF_EXACT ? 1.0 : c->gate);
    tlm_jw_bool(&w, "pass", 1);
    tlm_jw_end_obj(&w);
    tlm_jw_key_obj(&w, "decisions");
    tlm_jw_str(&w, "backend", "rules");
    tlm_jw_str(&w, "log", tlm_decide_log_file(c->dec));
    tlm_jw_int(&w, "asked", c->n_decisions);
    tlm_jw_int(&w, "applied", c->n_applied);
    tlm_jw_int(&w, "safe_default", c->n_defaulted);
    tlm_jw_end_obj(&w);
    tlm_jw_key_obj(&w, "timing_s");
    static const char *const tn[10] = {"", "ingest", "quantise", "calibrate", "precision", "prune",
                                       "", "", "place", "validate"};
    for (int i = 1; i < 10; i++) if (tn[i][0]) tlm_jw_num(&w, tn[i], round6(c->t[i]));
    tlm_jw_end_obj(&w);
    /* the atlas: one row per expert */
    tlm_jw_begin_arr(&w, "experts");
    for (uint32_t l = 0; l < g->n_layers; l++) {
        tlm_jw_begin_arr(&w, NULL);
        for (int e = 0; e < E; e++) {
            size_t ix = (size_t)l * E + e;
            int top;
            double h = class_entropy(&c->C, (int)l, e, &top);
            const tlm_packet_loc *loc = tlm_manifest_find(m, TLM_PK_EXPERT, (uint16_t)l, (uint32_t)e);
            tlm_jw_begin_obj(&w);
            tlm_jw_int(&w, "id", e);
            tlm_jw_int(&w, "count", (int64_t)c->C.s[ix].count);
            tlm_jw_num(&w, "heat", round6(heat_of(&c->C, (int)l, e)));
            tlm_jw_num(&w, "mean_gate", round6(gate_of(&c->C, (int)l, e)));
            tlm_jw_num(&w, "saliency", round6(saliency_of(&c->C, (int)l, e)));
            tlm_jw_int(&w, "top1", (int64_t)c->C.s[ix].top1);
            tlm_jw_str(&w, "class", h >= 0 ? CLS_NAME[top] : "unknown");
            if (h >= 0) tlm_jw_num(&w, "class_entropy", round6(h));
            tlm_jw_str(&w, "specialisation", SPEC_OPTS[c->spec[ix] >= 0 && c->spec[ix] < 4 ? c->spec[ix] : 0]);
            tlm_jw_num(&w, "p_needed", round6(c->p_needed ? c->p_needed[ix] : 1.0));
            tlm_jw_str(&w, "dtype", tlm_dtype_name(c->ex_dt[ix]));
            tlm_jw_int(&w, "drive", loc ? loc->drive : -1);
            tlm_jw_int(&w, "replica", loc && loc->replica_drive != 0xFF ? loc->replica_drive : -1);
            tlm_jw_bool(&w, "pinned", loc && (loc->flags & 1u));
            tlm_jw_bool(&w, "pruned", loc && (loc->flags & 2u));
            tlm_jw_end_obj(&w);
        }
        tlm_jw_end_arr(&w);
    }
    tlm_jw_end_arr(&w);
    tlm_jw_end_obj(&w);
    fputc('\n', f);
    if (fclose(f)) die("report: write error");
    return buf;
}

static void print_summary(const tlm_manifest *m, FILE *out);

static int cmd_compile(int argc, char **argv) {
    cc *c = (cc *)calloc(1, sizeof *c);
    if (!c) die("out of memory");
    const char *drives_arg = NULL, *prof = NULL, *rep = "auto";
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--out")) c->out = need(argc, argv, &i);
        else if (!strcmp(a, "--drives")) drives_arg = need(argc, argv, &i);
        else if (!strcmp(a, "--experts")) c->exp_name = need(argc, argv, &i);
        else if (!strcmp(a, "--dense")) c->dense_name = need(argc, argv, &i);
        else if (!strcmp(a, "--profile")) prof = need(argc, argv, &i);
        else if (!strcmp(a, "--replicas")) rep = need(argc, argv, &i);
        else if (!strcmp(a, "--corpus")) c->corpus_path = need(argc, argv, &i);
        else if (!strcmp(a, "--eval")) c->eval_path = need(argc, argv, &i);
        else if (!strcmp(a, "--etc")) c->etc = need(argc, argv, &i);
        else if (!strcmp(a, "--report")) c->print_report = 1;
        else if (!strcmp(a, "--fresh")) c->fresh = 1;
        else if (!strcmp(a, "--threads")) c->threads = (int)num_arg(need(argc, argv, &i), a, 1, 4096);
        else if (!strcmp(a, "--calib-tokens")) c->calib_tokens = num_arg(need(argc, argv, &i), a, 2, 1L << 30);
        else if (!strcmp(a, "--eval-tokens")) c->eval_tokens = num_arg(need(argc, argv, &i), a, 2, 1L << 30);
        else if (a[0] == '-') die("compile: unknown option %s", a);
        else if (!c->src) c->src = a;
        else die("compile: unexpected argument %s", a);
    }
    if (!c->src || !c->out || !drives_arg) die("usage: tlmc compile SRC --out MANIFEST --drives d1,d2,... [options]");

    /* settings: /etc/tierlm (or --etc / TIERLM_ETC), then flags */
    c->cfg = tlm_cfg_new();
    if (!c->cfg) die("out of memory");
    const char *etc = c->etc ? c->etc : getenv("TIERLM_ETC");
    if (tlm_cfg_load_etc(c->cfg, etc && *etc ? etc : TLM_ETC_DEFAULT, 0))
        die("config: %s", tlm_errmsg());
    if (!prof) prof = tlm_cfg_str(c->cfg, "compile.profile");
    if (!prof) prof = "exact";
    c->profile = -1;
    for (int i = 0; i < 3; i++) if (!strcasecmp(prof, PROF_NAME[i])) c->profile = i;
    if (c->profile < 0) die("--profile: exact, balanced or aggressive");
    c->gate = PROF_GATE[c->profile];
    if (!strcmp(rep, "auto")) c->replicas = 1;
    else if (!strcmp(rep, "none")) c->replicas = 0;
    else die("--replicas: auto or none");
    if (!c->exp_name) c->exp_name = tlm_cfg_str(c->cfg, "quant.experts");
    if (!c->exp_name) c->exp_name = "q6_k";
    if (!c->dense_name) c->dense_name = tlm_cfg_str(c->cfg, "quant.dense");
    if (!c->dense_name) c->dense_name = "q8_0";
    if (!c->threads) c->threads = tlm_cfg_is_auto(c->cfg, "cpu.threads") ? 0 : (int)tlm_cfg_int(c->cfg, "cpu.threads", 0);
    c->depth = depth_of(tlm_cfg_str(c->cfg, "content.default"));

    /* name and work dir next to the manifest */
    const char *base = strrchr(c->out, '/');
    base = base ? base + 1 : c->out;
    snprintf(c->name, sizeof c->name, "%s", base);
    char *dot = strrchr(c->name, '.');
    if (dot && dot != c->name) *dot = 0;
    for (char *p = c->name; *p; p++)
        if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9') || *p == '-' ||
              *p == '_' || *p == '.'))
            *p = '_';
    if (!c->name[0] || !strcmp(c->name, ".") || !strcmp(c->name, "..")) die("--out: cannot derive a store name");
    char outdir[1024];
    snprintf(outdir, sizeof outdir, "%s", c->out);
    char *sl = strrchr(outdir, '/');
    if (sl) *sl = 0; else snprintf(outdir, sizeof outdir, ".");
    if (tlm_mkdirs(outdir, 0755)) die("%s", tlm_errmsg());
    snprintf(c->work, sizeof c->work, "%s/%s.work", outdir, c->name);
    snprintf(c->refdir, sizeof c->refdir, "%s/ref", c->work);
    snprintf(c->candir, sizeof c->candir, "%s/cand", c->work);
    snprintf(c->refman, sizeof c->refman, "%s/ref.tlmm", c->refdir);
    snprintf(c->canman, sizeof c->canman, "%s/cand.tlmm", c->candir);
    snprintf(c->tmpman, sizeof c->tmpman, "%s/%s.tlmm", c->work, c->name);
    if (tlm_mkdirs(c->work, 0755)) die("%s", tlm_errmsg());

    /* drives: absolute paths, created now */
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s", drives_arg);
    for (char *tok = strtok(tmp, ","); tok; tok = strtok(NULL, ",")) {
        while (*tok == ' ') tok++;
        if (!*tok) continue;
        if (c->n_drives == TLM_MAX_DRIVES) die("--drives: at most %d drives", TLM_MAX_DRIVES);
        if (tlm_mkdirs(tok, 0755)) die("%s", tlm_errmsg());
        char rp[PATH_MAX];
        if (!realpath(tok, rp)) die("--drives: %s: %s", tok, strerror(errno));
        size_t rl = strlen(rp);
        if (rl >= sizeof c->drive[0]) die("--drives: path too long: %s", rp);
        memcpy(c->drive[c->n_drives++], rp, rl + 1);
    }
    if (!c->n_drives) die("--drives: no drive directories");

    tlm_decide_cfg dc;
    memset(&dc, 0, sizeof dc);
    const char *be = tlm_cfg_str(c->cfg, "decide.backend");
    dc.backend = be && !strcmp(be, "encoder") ? TLM_BE_ENCODER : be && !strcmp(be, "self") ? TLM_BE_SELF : TLM_BE_RULES;
    dc.auto_threshold = tlm_cfg_frac(c->cfg, "decide.auto_threshold", 0.90);
    dc.ask_threshold = tlm_cfg_frac(c->cfg, "decide.ask_threshold", 0.60);
    char logp[1300];
    if (tlm_cfg_origin_of(c->cfg, "decide.log", NULL) != TLM_ORIGIN_DEFAULT && tlm_cfg_str(c->cfg, "decide.log"))
        snprintf(logp, sizeof logp, "%s", tlm_cfg_str(c->cfg, "decide.log"));
    else
        snprintf(logp, sizeof logp, "%s/decisions.jsonl", c->work);
    dc.log_path = logp;
    c->dec = tlm_decide_open(&dc);
    if (!c->dec) die("decide: %s", tlm_errmsg());
    tlm_decide_add_builtin_rules(c->dec);

    printf("tlmc: compiling %s -> %s (profile %s, %d drive%s)\n", c->src, c->out, PROF_NAME[c->profile], c->n_drives,
           c->n_drives > 1 ? "s" : "");
    pass_ingest(c);                                                       /* 1 */
    size_t nx = (size_t)c->g.n_layers * c->g.n_experts;
    c->protect = (uint8_t *)calloc(nx, 1);
    c->exclude = (uint8_t *)calloc(nx, 1);
    if (!c->protect || !c->exclude) die("out of memory");
    parse_ids(c, "experts.protect", c->protect);
    parse_ids(c, "experts.exclude", c->exclude);
    pass_reference(c);                                                    /* 2 */
    printf("tlmc: pass 2 reference store %s (%s experts, %s dense)%s\n", c->refdir, tlm_dtype_name(c->exp_dt),
           tlm_dtype_name(c->dense_dt), c->reused_ref ? " [reused]" : "");
    load_tokens(c);
    c->pool = tlm_pool_new(c->threads, 0);
    if (!c->pool) die("out of memory");
    rtm ref;
    if (rtm_open(&ref, c->refman, c->pool)) die("reference store: %s", tlm_errmsg());
    pass_calibrate(c, &ref);                                              /* 3 */
    printf("tlmc: pass 3 calibrated on %llu tokens%s, reference eval %d positions\n",
           (unsigned long long)c->C.ntok, c->reused_cal ? " [reused]" : "", c->R.npos);
    int use_cand = 0;
    pass_precision(c, &use_cand);                                         /* 4 */
    if (use_cand) {
        rtm_close(&ref);
        if (rtm_open(&ref, c->canman, c->pool)) die("candidate store: %s", tlm_errmsg());
    }
    pass_prune(c, &ref);                                                  /* 5 */
    rtm_close(&ref);
    if (c->n_demoted && !c->have_A && use_cand) {   /* precision changed, nothing pruned */
        rtm cand;
        if (rtm_open(&cand, c->canman, c->pool)) die("%s", tlm_errmsg());
        evalcmp d;
        if (eval_run(cand.mdl, &c->ev, NULL, &c->A, &d)) die("%s", tlm_errmsg());
        c->have_A = 1;
        rtm_close(&cand);
    }
    tlm_manifest fm;                                                      /* 6, 7, 8 */
    pass_place(c, use_cand ? c->canman : c->refman, &fm);
    if (tlm_manifest_save(&fm, c->tmpman)) die("%s", tlm_errmsg());
    pass_validate(c);                                                     /* 9 */
    tlm_manifest_free(&fm);
    if (tlm_manifest_load(&fm, c->tmpman)) die("%s", tlm_errmsg());
    fm.report_json = build_report(c, &fm);
    if (tlm_manifest_save(&fm, c->out)) die("seal: %s", tlm_errmsg());   /* P-08 atomic */
    char rp[1300];
    snprintf(rp, sizeof rp, "%s/report.json", c->work);
    tlm_write_file_atomic(rp, fm.report_json, strlen(fm.report_json));
    unlink(c->tmpman);
    print_summary(&fm, stdout);
    if (c->print_report) fputs(fm.report_json, stdout);
    tlm_manifest_free(&fm);

    /* teardown */
    tlm_pool_free(c->pool);
    tlm_decide_close(c->dec);
    tlm_st_close(c->st);
    tlm_cfg_free(c->cfg);
    calib_free(&c->C);
    evalres_free(&c->R);
    evalres_free(&c->A);
    tokset_free(&c->cal);
    tokset_free(&c->ev);
    free(c->ex_dt);
    free(c->pruned);
    free(c->protect);
    free(c->exclude);
    free(c->spec);
    free(c->p_needed);
    free(c->pl);
    free(c->attempts_json);
    free(c);
    return 0;
}

/* ======================================================================
 * report / atlas
 * ==================================================================== */
static void fmt_bytes(uint64_t b, char *out, size_t cap) {
    if (b >= (1ull << 30)) snprintf(out, cap, "%.2f GiB", (double)b / (1 << 30));
    else if (b >= (1ull << 20)) snprintf(out, cap, "%.2f MiB", (double)b / (1 << 20));
    else snprintf(out, cap, "%.1f KiB", (double)b / 1024);
}

static void print_summary(const tlm_manifest *m, FILE *out) {
    const tlm_model_geom *g = &m->geom;
    char hx[65], b1[32];
    tlm_json *j = NULL;
    const tlm_jval *r = NULL;
    if (m->report_json) {
        char err[200];
        j = tlm_json_parse(m->report_json, strlen(m->report_json), err, sizeof err);
        r = j ? tlm_json_root(j) : NULL;
    }
    fprintf(out, "store       %s (profile %s)\n", m->name, m->profile);
    hex32(m->source_sha256, hx);
    fprintf(out, "source      %s\n            sha256 %s\n", r ? tlm_jstr(jget(r, "source"), "?") : "?", hx);
    fprintf(out, "geometry    %u layers, d_model %u, %u/%u heads x %u, %u experts top-%u (d_expert %u), shared %u, "
            "vocab %u, router %s\n", g->n_layers, g->d_model, g->n_heads, g->n_kv_heads, g->head_dim, g->n_experts,
            g->top_k, g->d_expert, g->d_shared, g->vocab,
            g->router == 2 ? "sigmoid+bias" : g->router == 1 ? "softmax top-k" : "softmax top-k renorm");
    if (r) {
        const tlm_jval *p = jget(r, "precision");
        fprintf(out, "precision   dense %s, experts %s, router/norms f32", tlm_jstr(jget(p, "dense"), "?"),
                tlm_jstr(jget(p, "experts"), "?"));
        if (tlm_jint(jget(p, "demoted_q5_k"), 0)) fprintf(out, ", %lld experts demoted to q5_k", (long long)tlm_jint(jget(p, "demoted_q5_k"), 0));
        else if (tlm_jint(jget(p, "proposed_q5_k"), 0)) fprintf(out, " (q5_k plan rolled back by the gate)");
        fputc('\n', out);
        const tlm_jval *cal = jget(r, "calibration");
        fprintf(out, "calibrate   %lld tokens in %lld sequences (%s)%s\n", (long long)tlm_jint(jget(cal, "tokens"), 0),
                (long long)tlm_jint(jget(cal, "sequences"), 0), tlm_jstr(jget(cal, "source"), "?"),
                tlm_jbool(jget(cal, "below_200k_warning"), 0) ? " - below 200k, statistics may overfit" : "");
        const tlm_jval *pr = jget(r, "prune");
        long long np = tlm_jint(jget(pr, "pruned"), 0);
        if (!strcmp(m->profile, "exact")) fprintf(out, "prune       none (exact profile)\n");
        else {
            fprintf(out, "prune       %lld of %u experts (%.0f%% of cap %.0f%%), content.default=%s\n", np,
                    g->n_layers * g->n_experts, 100 * tlm_jnum(jget(pr, "ratio"), 0), 100 * tlm_jnum(jget(pr, "cap"), 0),
                    tlm_jstr(jget(pr, "content_default"), "standard"));
            const tlm_jval *at = jget(pr, "attempts");
            for (size_t i = 0; i < tlm_jlen(at); i++) {
                const tlm_jval *a = tlm_jat(at, i);
                if (jget(a, "top1"))
                    fprintf(out, "            attempt %.0f%%: %lld experts, top-1 %.2f%%, KL %.4g -> %s\n",
                            100 * tlm_jnum(jget(a, "ratio"), 0), (long long)tlm_jint(jget(a, "pruned"), 0),
                            100 * tlm_jnum(jget(a, "top1"), 0), tlm_jnum(jget(a, "mean_kl"), 0),
                            tlm_jbool(jget(a, "pass"), 0) ? "pass" : "fail");
                else
                    fprintf(out, "            attempt %.0f%%: %s\n", 100 * tlm_jnum(jget(a, "ratio"), 0),
                            tlm_jstr(jget(a, "note"), ""));
            }
        }
    }
    fprintf(out, "placement   %u packets on %u drive%s\n", m->n_packets, m->n_drives, m->n_drives > 1 ? "s" : "");
    double tot_ld = 0, ld[TLM_MAX_DRIVES] = {0}, gb[TLM_MAX_DRIVES];
    for (uint32_t d = 0; d < m->n_drives; d++) {
        gb[d] = r ? tlm_jnum(jget(tlm_jat(jget(jget(r, "placement"), "drives"), d), "read_gbs"), 0) : 0;
        if (!(gb[d] > 0)) gb[d] = 1.0;
    }
    for (uint32_t i = 0; i < m->n_packets; i++) {   /* replicated reads split by bandwidth */
        const tlm_packet_loc *l = &m->packets[i];
        if (l->type != TLM_PK_EXPERT || (l->flags & 3u)) continue;
        double x = (double)l->heat * (double)l->total_len;
        if (l->replica_drive == 0xFF) { ld[l->drive] += x; continue; }
        double a = gb[l->drive], b = gb[l->replica_drive];
        ld[l->drive] += x * a / (a + b);
        ld[l->replica_drive] += x * b / (a + b);
    }
    for (uint32_t d = 0; d < m->n_drives; d++) tot_ld += ld[d];
    int pinned = 0, pruned = 0, repl = 0;
    for (uint32_t i = 0; i < m->n_packets; i++) {
        pinned += (m->packets[i].flags & 1u) != 0;
        pruned += (m->packets[i].flags & 2u) != 0;
        repl += m->packets[i].replica_drive != 0xFF;
    }
    for (uint32_t d = 0; d < m->n_drives; d++) {
        uint64_t by = 0;
        int pk = 0, rp = 0;
        for (uint32_t i = 0; i < m->n_packets; i++) {
            if (m->packets[i].drive == d) { pk++; by += m->packets[i].total_len; }
            if (m->packets[i].replica_drive == d) { rp++; by += m->packets[i].total_len; }
        }
        fmt_bytes(by, b1, sizeof b1);
        double gbs = r ? tlm_jnum(jget(tlm_jat(jget(jget(r, "placement"), "drives"), d), "read_gbs"), 0) : 0;
        char bw[48] = "";
        if (gbs > 0) snprintf(bw, sizeof bw, " (probed %.2f GB/s)", gbs);
        fprintf(out, "  drive %u   %s%s: %d packets + %d replicas, %s, %.0f%% of expected expert reads\n", d,
                m->drives[d], bw, pk, rp, b1, tot_ld > 0 ? 100 * ld[d] / tot_ld : 0.0);
    }
    fprintf(out, "            %d pinned (hottest), %d replicated (warm middle), %d pruned\n", pinned, repl, pruned);
    if (r) {
        const tlm_jval *ev = jget(r, "eval");
        fprintf(out, "validate    %lld eval tokens vs pass-2 reference: top-1 agreement %.2f%%, mean KL %.4g, "
                "bit-identical logits %s, greedy free run identical %s\n", (long long)tlm_jint(jget(ev, "tokens"), 0),
                100 * tlm_jnum(jget(ev, "top1_agreement"), 0), tlm_jnum(jget(ev, "mean_kl"), 0),
                tlm_jbool(jget(ev, "bit_identical"), 0) ? "yes" : "no",
                tlm_jbool(jget(ev, "free_run_identical"), 0) ? "yes" : "no");
        const tlm_jval *dc = jget(r, "decisions");
        fprintf(out, "decisions   %lld asked (%lld applied, %lld safe default), log %s\n",
                (long long)tlm_jint(jget(dc, "asked"), 0), (long long)tlm_jint(jget(dc, "applied"), 0),
                (long long)tlm_jint(jget(dc, "safe_default"), 0), tlm_jstr(jget(dc, "log"), "-"));
        const tlm_jval *t = jget(r, "timing_s");
        if (t && t->type == TLM_J_OBJ) {
            fprintf(out, "time        ");
            for (size_t i = 0; i < t->u.obj.n; i++)
                fprintf(out, "%s%s %.2fs", i ? ", " : "", t->u.obj.keys[i], tlm_jnum(t->u.obj.vals[i], 0));
            fputc('\n', out);
        }
    }
    tlm_json_free(j);
}

static int cmd_report(int argc, char **argv) {
    const char *path = NULL;
    int json = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--json")) json = 1;
        else if (argv[i][0] == '-') die("report: unknown option %s", argv[i]);
        else path = argv[i];
    }
    if (!path) die("usage: tlmc report MANIFEST [--json]");
    tlm_manifest m;
    if (tlm_manifest_load(&m, path)) die("%s", tlm_errmsg());
    if (json) fputs(m.report_json ? m.report_json : "{}\n", stdout);
    else print_summary(&m, stdout);
    tlm_manifest_free(&m);
    return 0;
}

typedef struct { int l, e; double heat, sal; const tlm_jval *v; } atlas_row;
static int g_atlas_sort;   /* 0 heat, 1 saliency, 2 id */
static int atlas_cmp(const void *a, const void *b) {
    const atlas_row *x = (const atlas_row *)a, *y = (const atlas_row *)b;
    if (g_atlas_sort == 0 && x->heat != y->heat) return x->heat > y->heat ? -1 : 1;
    if (g_atlas_sort == 1 && x->sal != y->sal) return x->sal > y->sal ? -1 : 1;
    if (x->l != y->l) return x->l - y->l;
    return x->e - y->e;
}

static int cmd_atlas(int argc, char **argv) {
    const char *path = NULL, *cls = NULL;
    long layer = -1, top = 0;
    g_atlas_sort = 2;
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--layer")) layer = num_arg(need(argc, argv, &i), a, 0, 65535);
        else if (!strcmp(a, "--top")) top = num_arg(need(argc, argv, &i), a, 1, 1L << 30);
        else if (!strcmp(a, "--class") || !strcmp(a, "--type")) cls = need(argc, argv, &i);
        else if (!strcmp(a, "--sort")) {
            const char *v = need(argc, argv, &i);
            g_atlas_sort = !strcmp(v, "heat") ? 0 : !strcmp(v, "saliency") ? 1 : !strcmp(v, "id") ? 2 : -1;
            if (g_atlas_sort < 0) die("--sort: heat, saliency or id");
        } else if (a[0] == '-') die("atlas: unknown option %s", a);
        else path = a;
    }
    if (!path) die("usage: tlmc atlas MANIFEST [--layer N] [--sort heat|saliency|id] [--class C] [--top N]");
    tlm_manifest m;
    if (tlm_manifest_load(&m, path)) die("%s", tlm_errmsg());
    if (!m.report_json) die("%s: no compiler report in this manifest", path);
    char err[200];
    tlm_json *j = tlm_json_parse(m.report_json, strlen(m.report_json), err, sizeof err);
    if (!j) die("%s: report: %s", path, err);
    const tlm_jval *ex = jget(tlm_json_root(j), "experts");
    size_t cap = 0;
    for (size_t l = 0; l < tlm_jlen(ex); l++) cap += tlm_jlen(tlm_jat(ex, l));
    atlas_row *rows = (atlas_row *)malloc(sizeof(atlas_row) * (cap ? cap : 1));
    if (!rows) die("out of memory");
    size_t n = 0;
    for (size_t l = 0; l < tlm_jlen(ex); l++) {
        if (layer >= 0 && (long)l != layer) continue;
        const tlm_jval *lv = tlm_jat(ex, l);
        for (size_t e = 0; e < tlm_jlen(lv); e++) {
            const tlm_jval *v = tlm_jat(lv, e);
            if (cls && strcmp(tlm_jstr(jget(v, "class"), ""), cls) != 0) continue;
            rows[n++] = (atlas_row){(int)l, (int)tlm_jint(jget(v, "id"), (int64_t)e), tlm_jnum(jget(v, "heat"), 0),
                                    tlm_jnum(jget(v, "saliency"), 0), v};
        }
    }
    qsort(rows, n, sizeof *rows, atlas_cmp);
    if (top > 0 && (size_t)top < n) n = (size_t)top;
    printf("expert atlas of %s (profile %s); class = most enriched input byte class (no content labels yet)\n",
           m.name, m.profile);
    printf("%-8s %7s %8s %9s %9s %6s %-7s %-11s %7s %-6s %5s %4s %s\n", "expert", "count", "heat", "gate",
           "saliency", "top1", "class", "spec", "needed", "dtype", "drive", "rep", "flags");
    for (size_t i = 0; i < n; i++) {
        const tlm_jval *v = rows[i].v;
        char id[24], rpl[24];
        snprintf(id, sizeof id, "%d:%d", rows[i].l, rows[i].e);
        long long rr = tlm_jint(jget(v, "replica"), -1);
        if (rr >= 0) snprintf(rpl, sizeof rpl, "%lld", rr); else snprintf(rpl, sizeof rpl, "-");
        printf("%-8s %7lld %8.4f %9.4f %9.4f %6lld %-7s %-11s %7.3f %-6s %5lld %4s %s%s\n", id,
               (long long)tlm_jint(jget(v, "count"), 0), rows[i].heat, tlm_jnum(jget(v, "mean_gate"), 0), rows[i].sal,
               (long long)tlm_jint(jget(v, "top1"), 0), tlm_jstr(jget(v, "class"), "?"),
               tlm_jstr(jget(v, "specialisation"), "?"), tlm_jnum(jget(v, "p_needed"), 1),
               tlm_jstr(jget(v, "dtype"), "?"), (long long)tlm_jint(jget(v, "drive"), -1), rpl,
               tlm_jbool(jget(v, "pinned"), 0) ? "pinned " : "", tlm_jbool(jget(v, "pruned"), 0) ? "pruned" : "");
    }
    if (n == 0) printf("(no experts match)\n");
    free(rows);
    tlm_json_free(j);
    tlm_manifest_free(&m);
    return 0;
}

static void usage(FILE *f) {
    fprintf(f,
            "tlmc %s - tierlm model compiler\n"
            "  tlmc synth   --out DIR [--layers N] [--d-model N] [--heads N] [--kv-heads N] [--head-dim N]\n"
            "               [--experts N] [--top-k N] [--d-expert N] [--shared N] [--vocab N] [--seed N]\n"
            "               [--router softmax|sigmoid] [--qk-norm yes|no]\n"
            "  tlmc compile SRC --out MANIFEST --drives d1,d2,... [--experts q6_k] [--dense q8_0]\n"
            "               [--profile exact|balanced|aggressive] [--replicas auto|none] [--corpus FILE]\n"
            "               [--eval FILE] [--etc DIR] [--report] [--threads N] [--calib-tokens N]\n"
            "               [--eval-tokens N] [--fresh]\n"
            "  tlmc report  MANIFEST [--json]\n"
            "  tlmc atlas   MANIFEST [--layer N] [--sort heat|saliency|id] [--class C] [--top N]\n",
            TLMC_VERSION);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);   /* progress lines interleave with errors */
    tlm_log_level(TLM_LOG_WARN);
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "help")) {
        usage(argc < 2 ? stderr : stdout);
        return argc < 2 ? 2 : 0;
    }
    if (!strcmp(argv[1], "synth")) return cmd_synth(argc, argv);
    if (!strcmp(argv[1], "compile")) return cmd_compile(argc, argv);
    if (!strcmp(argv[1], "report")) return cmd_report(argc, argv);
    if (!strcmp(argv[1], "atlas")) return cmd_atlas(argc, argv);
    usage(stderr);
    return 2;
}
