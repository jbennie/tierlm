/* tlm_decide.h - embedded Jev-style typed decision engine.
 *
 * Three primitives: CHOICE (pick one option), SCORE (ordered rubric),
 * NOUL (probability a statement is true). Answers always come back typed,
 * with a probability per option and a confidence. A confidence policy maps
 * answers to apply / review / safe-default, and deterministic hard rules
 * (applied by the caller) always have the last word.
 * Design: research/04-decision-engine.md.
 *
 * Backends: RULES (deterministic, built in), SELF (the loaded model scores
 * options; supplied as a callback), ENCODER (embedded decision model; not
 * yet ported - returns TLM_EUNSUP).
 */
#ifndef TLM_DECIDE_H
#define TLM_DECIDE_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TLM_DECIDE_MAX_OPTIONS 32

typedef enum { TLM_Q_CHOICE = 0, TLM_Q_SCORE = 1, TLM_Q_NOUL = 2 } tlm_qtype;
typedef enum { TLM_BE_RULES = 0, TLM_BE_SELF = 1, TLM_BE_ENCODER = 2 } tlm_backend;
typedef enum { TLM_POLICY_APPLY = 0, TLM_POLICY_REVIEW = 1, TLM_POLICY_DEFAULT = 2 } tlm_policy_outcome;

typedef struct {
    const char  *id;            /* "expert.domain", "expert.needed_for_usecase", ... */
    tlm_qtype    type;
    const char  *question;
    const char **options;       /* CHOICE labels / SCORE levels low->high; NULL for NOUL */
    const char **criteria;      /* optional definition per option */
    int          n_options;
    int          safe_default;  /* option index (or 1/0 for NOUL true/false) used when uncertain */
} tlm_question;

typedef struct {
    int      choice;            /* option index; NOUL: 1 if p >= 0.5 else 0 */
    float    prob[TLM_DECIDE_MAX_OPTIONS];   /* NOUL: prob[0] = P(true) */
    float    confidence;        /* 1 - normalised entropy; NOUL: |2p-1| */
    tlm_backend used;
    tlm_policy_outcome outcome;
    int      final_choice;      /* after policy (safe default applied when needed) */
    uint64_t decision_id;
} tlm_answer;

/* State is a flat list of named numeric features plus optional text. The
 * rules backend reads features; model backends read the text rendering. */
typedef struct { const char *name; double value; } tlm_feature;
typedef struct {
    const tlm_feature *features; int n_features;
    const char *text;           /* optional free text (e.g. decoded token windows) */
} tlm_state;

/* A rule maps (question id, state) to a probability distribution. Return 1
 * if the rule handled the question, 0 to pass. */
typedef int (*tlm_rule_fn)(const tlm_question *q, const tlm_state *s, float *prob, void *ctx);
/* SELF backend callback: score each option's text given the state text. */
typedef int (*tlm_self_fn)(const tlm_question *q, const tlm_state *s, float *prob, void *ctx);

typedef struct {
    tlm_backend backend;        /* preferred backend */
    double auto_threshold;      /* default 0.90 */
    double ask_threshold;       /* default 0.60 */
    int    interactive;         /* review band: 1 = return REVIEW, 0 = take safe default */
    const char *log_path;       /* JSONL audit log; NULL = no log */
    tlm_self_fn self_fn; void *self_ctx;
} tlm_decide_cfg;

typedef struct tlm_decider tlm_decider;
tlm_decider *tlm_decide_open(const tlm_decide_cfg *cfg);
void         tlm_decide_close(tlm_decider *d);
int          tlm_decide_add_rule(tlm_decider *d, const char *question_id_prefix, tlm_rule_fn fn, void *ctx);
/* Evaluate nq independent questions against one state. */
int          tlm_decide(tlm_decider *d, const tlm_state *s, const tlm_question *qs, int nq, tlm_answer *out);
/* Re-print a logged decision. */
int          tlm_decide_explain(const char *log_path, uint64_t decision_id, FILE *out);

/* Built-in rules used by tlmc (registered by tlm_decide_add_builtin_rules):
 *   expert.specialisation  SCORE  {generalist, broad, focused, narrow}
 *                          features: label_entropy (0..1)
 *   expert.needed_for_usecase NOUL features: saliency_kept (0..1), mass_excluded (0..1),
 *                          mass_deep (0..1), protected (0/1)
 *   expert.precision_tier  CHOICE {q8_0, q6_k, q5_k} features: heat, saliency, depth (0..3)
 */
void tlm_decide_add_builtin_rules(tlm_decider *d);

/* ---- additions (tlm_decide.c) ------------------------------------------
 *
 * Backends and fallback:
 *   - RULES: registered rules whose prefix is a prefix of the question id are
 *     tried longest prefix first (then registration order); the first that
 *     returns 1 answers. No rule -> uniform distribution (confidence 0), which
 *     the policy turns into the safe default.
 *   - SELF: cfg.self_fn returns 1 when it filled prob, 0 to pass, < 0 on
 *     error; pass/error/missing callback falls back to RULES (noted in log).
 *   - ENCODER: not ported; returns TLM_EUNSUP internally, logs one warning
 *     per decider and falls back to RULES (noted in the audit record).
 * Normalisation: CHOICE/SCORE probabilities are clamped to >= 0 and scaled to
 * sum 1 (all-zero or non-finite -> uniform). NOUL: prob[0] = P(true) clamped
 * to [0,1]; prob[1] is set to 1 - P(true) for convenience.
 * Confidence: CHOICE/SCORE 1 - H(p)/ln(n) (n = n_options; 1 when n == 1);
 * NOUL |2p - 1|. choice = argmax (lowest index on ties); NOUL 1 iff p >= 0.5.
 * Policy: conf >= auto -> APPLY (final = choice). ask <= conf < auto -> if a
 * SELF callback exists and SELF did not produce the answer, re-ask with SELF
 * and re-evaluate; still in band -> REVIEW when interactive, else DEFAULT.
 * conf < ask -> DEFAULT. For REVIEW and DEFAULT final_choice = safe_default
 * (the caller replaces it after a review). Thresholds <= 0 or > 1 in the cfg
 * take the defaults 0.90 / 0.60.
 * Audit log: one JSON object per answer, appended with a single write:
 *   {"decision_id":"<u64>","ts":"<UTC ISO-8601>","question_id":..,"type":..,
 *    "question":..,"options":[..],"safe_default":n,"state_hash":"<fnv1a64>",
 *    "features":{name:value,..},"text":"<first 256 bytes>","probs":[..],
 *    "confidence":c,"choice":n,"backend":"rules|self|encoder",
 *    "requested_backend":..,"note":..,"outcome":"apply|review|default",
 *    "final_choice":n,"final_label":".."}
 * log_path may name a file or a directory (then <dir>/decisions.jsonl);
 * parent directories are created. tlm_decide_open fails if it cannot open it.
 *
 * Built-in rules (exact formulas; L(x) = 1/(1+e^-x), K(t,k) = exp(-(t-k)^2/0.18)
 * normalised over the options, i.e. a Gaussian kernel with sigma 0.3 around a
 * latent position t on the option scale):
 *   expert.specialisation: t = 3 * (1 - clamp(label_entropy, 0, 1)), p_k = K(t,k)
 *       (entropy 1 -> generalist, 2/3 -> broad, 1/3 -> focused, 0 -> narrow).
 *       Passes when label_entropy is missing.
 *   expert.needed_for_usecase: protected >= 0.5 -> P = 0.9999; otherwise
 *       P(needed) = 1 - 0.99 * L((mass_excluded - 0.8)/0.02)
 *                            * L((0.05 - mass_deep)/0.01)
 *                            * L((0.10 - saliency_kept)/0.02)
 *       (missing features: saliency_kept 1, mass_excluded 0, mass_deep 0,
 *       protected 0, i.e. "needed").
 *   expert.precision_tier: options {q8_0, q6_k, q5_k}; u = (heat + saliency)/2
 *       (each clamped to [0,1], missing = 0), depth rounded to 0..3 (missing =
 *       2): t = 2 (exclude), 2 - L((u-0.6)/0.05) (shallow), 1 (standard),
 *       1 - L((u-0.7)/0.05) (deep); p_k = K(t,k).
 *   any id (prefix ""): if the state carries features "p.<option label>"
 *       (CHOICE/SCORE) or "p.true" (NOUL), they are used as the distribution.
 */
const char *tlm_backend_name(tlm_backend b);            /* "rules", "self", "encoder" */
const char *tlm_policy_name(tlm_policy_outcome o);      /* "apply", "review", "default" */
const char *tlm_qtype_name(tlm_qtype t);                /* "choice", "score", "noul" */
/* Confidence of an (already normalised) distribution, as defined above. */
float  tlm_decide_confidence(const tlm_question *q, const float *prob);
double tlm_state_feature(const tlm_state *s, const char *name, double dflt);
const char *tlm_decide_log_file(const tlm_decider *d);  /* resolved log file or NULL */
/* The built-in rules, exposed for direct tests and reuse. */
int tlm_rule_specialisation(const tlm_question *q, const tlm_state *s, float *prob, void *ctx);
int tlm_rule_needed(const tlm_question *q, const tlm_state *s, float *prob, void *ctx);
int tlm_rule_precision_tier(const tlm_question *q, const tlm_state *s, float *prob, void *ctx);
int tlm_rule_feature_probs(const tlm_question *q, const tlm_state *s, float *prob, void *ctx);

#ifdef __cplusplus
}
#endif
#endif
