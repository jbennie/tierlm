/* tlm_model.h - config-driven MoE decoder over a compiled store.
 *
 * Patterns: P-79 pre-norm decoder skeleton, P-80 RMSNorm, P-81 RoPE (NeoX
 * half-split layout), P-61 GQA attention, P-73/P-74 routers (softmax top-k
 * with/without renorm, sigmoid + bias), P-75 SwiGLU experts + shared expert,
 * P-89 position-addressed KV, P-94 sampler, C-13 fixed combine order
 * (ascending top-k rank), P-33 router lookahead into the store.
 *
 * Numerics: activations f32; RMSNorm sum of squares in double; attention
 * scores and softmax in f32 with max subtraction; expert outputs combined in
 * ascending rank order. Results do not depend on thread count, cache size,
 * prefetch or I/O backend.
 */
#ifndef TLM_MODEL_H
#define TLM_MODEL_H

#include <stdint.h>
#include "tlm_packet.h"
#include "tlm_store.h"
#include "tlm_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tlm_model tlm_model;

typedef struct {
    int lookahead;       /* router-lookahead prefetch depth in layers (0 = off, 1 default) */
    int lookahead_k;     /* experts to hint per layer (default top_k) */
    double interest;     /* nav.interest_aggressiveness: "this question needs" (0 = legacy fixed breadth) */
} tlm_model_opts;

/* Loads every DENSE/META/EMBED packet into RAM (verified). The store serves
 * EXPERT packets. Pruned experts (manifest flag) are masked out of routing. */
tlm_model *tlm_model_open(const tlm_manifest *m, tlm_io *io, tlm_store *store,
                          tlm_pool *pool, const tlm_model_opts *o);
void       tlm_model_close(tlm_model *mdl);
const tlm_model_geom *tlm_model_geom_of(const tlm_model *mdl);

/* Apply HF YaRN (transformers' modeling_rope_utils._compute_yarn_parameters)
 * to a plain RoPE inv_freq table [head_dim/2] in place. The f32 arithmetic
 * mirrors HF's float32 path: the correction range (floor/ceil when
 * `truncate` is set) is computed in double, the ramp and the
 * extrapolate/interpolate mix in f32. `*attn_factor` receives the cos/sin
 * attention scale (1 + 0.1*ln(factor) for factor > 1, else 1.0). */
int tlm_yarn_transform(float *inv_freq, int half, uint32_t head_dim, double theta,
                       double factor, double beta_fast, double beta_slow,
                       uint32_t orig_len, int truncate, float *attn_factor);

/* KV cache for one sequence. */
typedef struct tlm_kv tlm_kv;
tlm_kv *tlm_kv_new(const tlm_model *mdl, int max_ctx);
void    tlm_kv_free(tlm_kv *kv);
void    tlm_kv_reset(tlm_kv *kv);
int     tlm_kv_len(const tlm_kv *kv);

/* Run one token at the next position; writes vocab logits when logits != NULL. */
int tlm_model_step(tlm_model *mdl, tlm_kv *kv, int token, float *logits);
/* Feed several tokens (prefill). Logits for the last token only. */
int tlm_model_prefill(tlm_model *mdl, tlm_kv *kv, const int *tokens, int n, float *logits);

/* Routing trace hook for calibration (tlmc pass 3). Called once per MoE
 * layer per token, after routing and expert compute, on the caller thread.
 * `ids` and `weights` are the k selected experts in rank order;
 * `out_norm` is ||weight_i * expert_i(x)||_2 (REAP saliency input). */
typedef void (*tlm_route_hook)(void *ctx, int layer, int pos, int token,
                               const int *ids, const float *weights,
                               const float *out_norm, int k);
void tlm_model_set_route_hook(tlm_model *mdl, tlm_route_hook fn, void *ctx);

/* Sampler (P-94): temperature 0 = greedy argmax (ties -> lowest id). */
typedef struct {
    float    temperature, top_p, min_p;
    uint64_t seed;
} tlm_sampler;
int tlm_sample(const tlm_sampler *s, uint64_t *rng_state, const float *logits, int vocab);

/* Generate up to max_new tokens greedily or sampled; calls on_token for
 * each; stops on any token in stop[] (n_stop may be 0). Returns count. */
typedef int (*tlm_token_fn)(void *ctx, int token);
int tlm_generate(tlm_model *mdl, tlm_kv *kv, const int *prompt, int n_prompt,
                 int max_new, const tlm_sampler *s, const int *stop, int n_stop,
                 tlm_token_fn on_token, void *ctx);

/* Per-step timing and counters for --stats. */
typedef struct {
    uint64_t tokens, attn_ns, moe_ns, expert_wait_ns, other_ns;
} tlm_model_stats;
void tlm_model_get_stats(const tlm_model *mdl, tlm_model_stats *out);

/* ---- additions (tlm_model.c) ------------------------------------------
 * Drive i of the manifest must be file i of `io` (open the io with
 * tlm_manifest_drive_path(m, i, ...) for i in [0, n_drives)).
 * One model runs one sequence step at a time (not reentrant). */
void tlm_model_reset_stats(tlm_model *mdl);
/* Replace the routing mask: pruned[layer * n_experts + e] != 0 removes that
 * expert from routing (softmax/top-k over the survivors). NULL restores the
 * all-present mask. Experts without a packet stay masked. Fails (mask
 * unchanged) if a layer would keep fewer than top_k experts. Used by the
 * compiler's prune gate without rewriting the manifest. */
int  tlm_model_set_expert_mask(tlm_model *mdl, const uint8_t *pruned);
/* Current mask, n_layers * n_experts bytes (1 = not routable). */
const uint8_t *tlm_model_expert_mask(const tlm_model *mdl);
int  tlm_kv_capacity(const tlm_kv *kv);

/* ---- notes (tlm_model.c) ------------------------------------------------
 * Packets: DENSE layer l holds attn_norm, ffn_norm, attn_q [H*hd, D],
 *   attn_k/attn_v [KV*hd, D], attn_o [D, H*hd] and optional q_norm/k_norm
 *   [1, hd] (per-head qk-norm, applied when present). The FFN is either a
 *   router + experts (router [E, D] and optional router_bias [1, E],
 *   shared_gate/shared_up [S, D], shared_down [D, S], shared_gate_inp [1, D]
 *   sigmoid scalar gate) or, for a dense layer (first_k_dense_replace /
 *   mlp_only_layers), a resident SwiGLU block: ffn_gate/ffn_up [d_ffn, D],
 *   ffn_down [D, d_ffn]; a dense layer carries no router and no EXPERT
 *   packets, and d_ffn is read from the ffn_gate shape. META: output_norm,
 *   lm_head [V, D]. EMBED: tok_embd [V, D]. Vectors may be any dtype
 *   (converted to f32 once); matrices any row-quantised dtype except
 *   Q8_K/FP8.
 *   EXPERT: payload = gate [de, D] | up [de, D] | down [D, de] rows at
 *   item_off[0..2], dtype from the packet header (may differ per expert).
 * tlm_model_open reads every resident packet (CRC-checked, replica fallback)
 *   and the first 4 KiB (header) of every unpruned EXPERT packet.
 * The store must offer >= top_k unpinned slots (see tlm_store.h).
 * tlm_model_step: hints for layer l+1..l+lookahead are issued only after
 *   layer l's experts are referenced (P-34); a failed expert load returns its
 *   error (TLM_EFORMAT / TLM_EIO) with every reference released.
 * tlm_generate: the prompt is prefilled, then each sampled token is passed
 *   to on_token; it stops after max_new tokens, a stop token (delivered
 *   first), on_token returning non-zero, or a full KV. The last sampled token
 *   is not fed back. Returns the count or a negative TLM_E* code.
 * tlm_sample: rng_state may be NULL for greedy; *rng_state == 0 is seeded
 *   from s->seed (splitmix64). min_p = 0 and top_p = 1 reproduce plain
 *   temperature sampling.
 */

#ifdef __cplusplus
}
#endif
#endif
