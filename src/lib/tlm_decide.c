/* tlm_decide.c - typed decision engine: rules / self / encoder backends,
 * confidence policy, JSONL audit log and explain.
 * Design: research/04-decision-engine.md; behaviour documented in tlm_decide.h.
 */
#include "tlm_decide.h"
#include "tlm_json.h"
#include "tlm_platform.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    char *prefix;
    size_t plen;
    tlm_rule_fn fn;
    void *ctx;
} rule_ent;

struct tlm_decider {
    tlm_decide_cfg cfg;
    char *log_file;
    int log_fd;
    rule_ent *rules;
    int nrules, cap;
    uint64_t last_id;
    int encoder_warned;
};

const char *tlm_backend_name(tlm_backend b) {
    switch (b) {
    case TLM_BE_RULES: return "rules";
    case TLM_BE_SELF: return "self";
    case TLM_BE_ENCODER: return "encoder";
    }
    return "?";
}
const char *tlm_policy_name(tlm_policy_outcome o) {
    switch (o) {
    case TLM_POLICY_APPLY: return "apply";
    case TLM_POLICY_REVIEW: return "review";
    case TLM_POLICY_DEFAULT: return "default";
    }
    return "?";
}
const char *tlm_qtype_name(tlm_qtype t) {
    switch (t) {
    case TLM_Q_CHOICE: return "choice";
    case TLM_Q_SCORE: return "score";
    case TLM_Q_NOUL: return "noul";
    }
    return "?";
}

double tlm_state_feature(const tlm_state *s, const char *name, double dflt) {
    if (!s || !name || !s->features) return dflt;
    for (int i = s->n_features - 1; i >= 0; i--)          /* last one wins */
        if (s->features[i].name && strcmp(s->features[i].name, name) == 0) return s->features[i].value;
    return dflt;
}

static int has_feature(const tlm_state *s, const char *name) {
    return !isnan(tlm_state_feature(s, name, NAN));
}

/* ---- log path ----------------------------------------------------------- */
static char *resolve_log(const char *path) {
    if (!path || !*path) return NULL;
    size_t n = strlen(path);
    struct stat st;
    int is_dir = path[n - 1] == '/' || (stat(path, &st) == 0 && S_ISDIR(st.st_mode));
    char *out = (char *)malloc(n + 32);
    if (!out) return NULL;
    if (is_dir) snprintf(out, n + 32, "%s%sdecisions.jsonl", path, path[n - 1] == '/' ? "" : "/");
    else snprintf(out, n + 32, "%s", path);
    return out;
}

const char *tlm_decide_log_file(const tlm_decider *d) { return d ? d->log_file : NULL; }

/* ---- open / close / rules ---------------------------------------------- */
tlm_decider *tlm_decide_open(const tlm_decide_cfg *cfg) {
    tlm_decider *d = (tlm_decider *)calloc(1, sizeof *d);
    if (!d) { tlm_seterr(TLM_ENOMEM, "decide: out of memory"); return NULL; }
    if (cfg) d->cfg = *cfg;
    if (!(d->cfg.auto_threshold > 0 && d->cfg.auto_threshold <= 1)) d->cfg.auto_threshold = 0.90;
    if (!(d->cfg.ask_threshold > 0 && d->cfg.ask_threshold <= 1)) d->cfg.ask_threshold = 0.60;
    if (d->cfg.ask_threshold > d->cfg.auto_threshold) {
        tlm_seterr(TLM_EINVAL, "decide: ask_threshold %.3f exceeds auto_threshold %.3f",
                   d->cfg.ask_threshold, d->cfg.auto_threshold);
        free(d);
        return NULL;
    }
    if ((unsigned)d->cfg.backend > TLM_BE_ENCODER) {
        tlm_seterr(TLM_EINVAL, "decide: unknown backend %d", (int)d->cfg.backend);
        free(d);
        return NULL;
    }
    d->log_fd = -1;
    d->cfg.log_path = NULL;                 /* do not keep the caller's pointer */
    if (cfg && cfg->log_path && *cfg->log_path) {
        d->log_file = resolve_log(cfg->log_path);
        if (!d->log_file) { tlm_seterr(TLM_ENOMEM, "decide: out of memory"); free(d); return NULL; }
        char *slash = strrchr(d->log_file, '/');
        if (slash && slash != d->log_file) {
            *slash = 0;
            int rc = tlm_mkdirs(d->log_file, 0755);
            *slash = '/';
            if (rc != 0) { free(d->log_file); free(d); return NULL; }
        }
        d->log_fd = open(d->log_file, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (d->log_fd < 0) {
            tlm_seterr(TLM_EIO, "decide: cannot open audit log %s: %s", d->log_file, strerror(errno));
            free(d->log_file);
            free(d);
            return NULL;
        }
    }
    return d;
}

void tlm_decide_close(tlm_decider *d) {
    if (!d) return;
    if (d->log_fd >= 0) close(d->log_fd);
    for (int i = 0; i < d->nrules; i++) free(d->rules[i].prefix);
    free(d->rules);
    free(d->log_file);
    free(d);
}

int tlm_decide_add_rule(tlm_decider *d, const char *prefix, tlm_rule_fn fn, void *ctx) {
    if (!d || !fn) return tlm_seterr(TLM_EINVAL, "decide: rule needs a function");
    if (!prefix) prefix = "";
    if (d->nrules == d->cap) {
        int nc = d->cap ? d->cap * 2 : 8;
        rule_ent *r = (rule_ent *)realloc(d->rules, (size_t)nc * sizeof *r);
        if (!r) return tlm_seterr(TLM_ENOMEM, "decide: out of memory");
        d->rules = r;
        d->cap = nc;
    }
    size_t n = strlen(prefix);
    char *p = (char *)malloc(n + 1);
    if (!p) return tlm_seterr(TLM_ENOMEM, "decide: out of memory");
    memcpy(p, prefix, n + 1);
    d->rules[d->nrules++] = (rule_ent){p, n, fn, ctx};
    return 0;
}

/* ---- maths -------------------------------------------------------------- */
static double logistic(double x) { return 1.0 / (1.0 + exp(-x)); }
static double clamp01(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

static int nopts(const tlm_question *q) { return q->type == TLM_Q_NOUL ? 2 : q->n_options; }

static void normalise(const tlm_question *q, float *prob) {
    if (q->type == TLM_Q_NOUL) {
        float p = prob[0];
        if (!isfinite(p)) p = 0.5f;
        if (p < 0) p = 0;
        if (p > 1) p = 1;
        prob[0] = p;
        prob[1] = 1.0f - p;
        for (int i = 2; i < TLM_DECIDE_MAX_OPTIONS; i++) prob[i] = 0;
        return;
    }
    int n = q->n_options;
    double sum = 0;
    for (int i = 0; i < n; i++) {
        if (!isfinite(prob[i]) || prob[i] < 0) prob[i] = 0;
        sum += prob[i];
    }
    if (!(sum > 0) || !isfinite(sum)) {
        for (int i = 0; i < n; i++) prob[i] = 1.0f / (float)n;
    } else {
        for (int i = 0; i < n; i++) prob[i] = (float)(prob[i] / sum);
    }
    for (int i = n; i < TLM_DECIDE_MAX_OPTIONS; i++) prob[i] = 0;
}

float tlm_decide_confidence(const tlm_question *q, const float *prob) {
    if (!q || !prob) return 0;
    if (q->type == TLM_Q_NOUL) {
        double p = clamp01(prob[0]);
        return (float)fabs(2 * p - 1);
    }
    int n = q->n_options;
    if (n <= 1) return 1.0f;
    double h = 0;
    for (int i = 0; i < n; i++) {
        double p = prob[i];
        if (p > 0) h -= p * log(p);
    }
    double c = 1.0 - h / log((double)n);
    return (float)clamp01(c);
}

static int argmax(const tlm_question *q, const float *prob) {
    if (q->type == TLM_Q_NOUL) return prob[0] >= 0.5f ? 1 : 0;
    int best = 0;
    for (int i = 1; i < q->n_options; i++)
        if (prob[i] > prob[best]) best = i;
    return best;
}

/* Gaussian kernel (sigma 0.3) around latent position t over n levels. */
static void kernel_probs(double t, int n, float *prob) {
    double sum = 0, w[TLM_DECIDE_MAX_OPTIONS];
    for (int k = 0; k < n; k++) {
        double d = t - k;
        w[k] = exp(-(d * d) / 0.18);
        sum += w[k];
    }
    for (int k = 0; k < n; k++) prob[k] = (float)(w[k] / sum);
}

/* ---- built-in rules ----------------------------------------------------- */
int tlm_rule_specialisation(const tlm_question *q, const tlm_state *s, float *prob, void *ctx) {
    (void)ctx;
    if (q->type == TLM_Q_NOUL || q->n_options != 4 || !has_feature(s, "label_entropy")) return 0;
    double e = clamp01(tlm_state_feature(s, "label_entropy", 1.0));
    kernel_probs(3.0 * (1.0 - e), 4, prob);
    return 1;
}

int tlm_rule_needed(const tlm_question *q, const tlm_state *s, float *prob, void *ctx) {
    (void)ctx;
    if (q->type != TLM_Q_NOUL) return 0;
    if (tlm_state_feature(s, "protected", 0) >= 0.5) {
        prob[0] = 0.9999f;
        return 1;
    }
    double sal = clamp01(tlm_state_feature(s, "saliency_kept", 1.0));
    double mx = clamp01(tlm_state_feature(s, "mass_excluded", 0.0));
    double md = clamp01(tlm_state_feature(s, "mass_deep", 0.0));
    double not_needed = 0.99 * logistic((mx - 0.8) / 0.02) * logistic((0.05 - md) / 0.01) *
                        logistic((0.10 - sal) / 0.02);
    prob[0] = (float)(1.0 - not_needed);
    return 1;
}

int tlm_rule_precision_tier(const tlm_question *q, const tlm_state *s, float *prob, void *ctx) {
    (void)ctx;
    if (q->type == TLM_Q_NOUL || q->n_options != 3) return 0;
    double heat = clamp01(tlm_state_feature(s, "heat", 0.0));
    double sal = clamp01(tlm_state_feature(s, "saliency", 0.0));
    double dep = tlm_state_feature(s, "depth", 2.0);
    if (!isfinite(dep)) dep = 2.0;
    int depth = (int)lround(dep < 0 ? 0 : dep > 3 ? 3 : dep);
    double u = 0.5 * (heat + sal);
    double t;
    switch (depth) {
    case 0: t = 2.0; break;
    case 1: t = 2.0 - logistic((u - 0.6) / 0.05); break;
    case 2: t = 1.0; break;
    default: t = 1.0 - logistic((u - 0.7) / 0.05); break;
    }
    kernel_probs(t, 3, prob);
    return 1;
}

int tlm_rule_feature_probs(const tlm_question *q, const tlm_state *s, float *prob, void *ctx) {
    (void)ctx;
    char name[160];
    if (q->type == TLM_Q_NOUL) {
        if (!has_feature(s, "p.true")) return 0;
        prob[0] = (float)tlm_state_feature(s, "p.true", 0.5);
        return 1;
    }
    if (!q->options) return 0;
    int found = 0;
    for (int i = 0; i < q->n_options; i++) {
        snprintf(name, sizeof name, "p.%s", q->options[i] ? q->options[i] : "");
        double v = tlm_state_feature(s, name, NAN);
        if (!isnan(v)) found++;
        prob[i] = isnan(v) ? 0.0f : (float)v;
    }
    return found > 0;
}

void tlm_decide_add_builtin_rules(tlm_decider *d) {
    tlm_decide_add_rule(d, "expert.specialisation", tlm_rule_specialisation, NULL);
    tlm_decide_add_rule(d, "expert.needed_for_usecase", tlm_rule_needed, NULL);
    tlm_decide_add_rule(d, "expert.precision_tier", tlm_rule_precision_tier, NULL);
    tlm_decide_add_rule(d, "", tlm_rule_feature_probs, NULL);
}

/* ---- backends ----------------------------------------------------------- */
static int run_rules(tlm_decider *d, const tlm_question *q, const tlm_state *s, float *prob) {
    const char *id = q->id ? q->id : "";
    if (d->nrules == 0) return 0;
    /* matching rules ordered by longest prefix, then registration order */
    int *idx = (int *)malloc((size_t)d->nrules * sizeof *idx);
    if (!idx) return 0;
    int n = 0;
    for (int i = 0; i < d->nrules; i++)
        if (strncmp(id, d->rules[i].prefix, d->rules[i].plen) == 0) idx[n++] = i;
    for (int i = 1; i < n; i++) {                 /* insertion sort, stable */
        int v = idx[i], j = i - 1;
        while (j >= 0 && d->rules[idx[j]].plen < d->rules[v].plen) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = v;
    }
    int handled = 0;
    for (int i = 0; i < n && !handled; i++) {
        float tmp[TLM_DECIDE_MAX_OPTIONS] = {0};
        const rule_ent *r = &d->rules[idx[i]];
        if (r->fn(q, s, tmp, r->ctx) == 1) {
            memcpy(prob, tmp, sizeof tmp);
            handled = 1;
        }
    }
    free(idx);
    return handled;
}

static int run_encoder(tlm_decider *d, const tlm_question *q, const tlm_state *s, float *prob) {
    (void)d; (void)q; (void)s; (void)prob;
    return tlm_seterr(TLM_EUNSUP, "decide: encoder backend is not ported yet");
}

/* Fills prob from the requested backend with fallback; returns backend used. */
static tlm_backend run_backend(tlm_decider *d, tlm_backend want, const tlm_question *q,
                               const tlm_state *s, float *prob, char *note, size_t notecap) {
    for (int i = 0; i < TLM_DECIDE_MAX_OPTIONS; i++) prob[i] = 0;
    if (want == TLM_BE_ENCODER) {
        int rc = run_encoder(d, q, s, prob);
        if (rc == TLM_EUNSUP) {
            if (!d->encoder_warned) {
                tlm_log(TLM_LOG_WARN, "decide: encoder backend not built (%s); falling back to rules", tlm_errmsg());
                d->encoder_warned = 1;
            }
            snprintf(note, notecap, "encoder backend not built; fell back to rules");
        }
    } else if (want == TLM_BE_SELF) {
        if (d->cfg.self_fn) {
            int rc = d->cfg.self_fn(q, s, prob, d->cfg.self_ctx);
            if (rc == 1) return TLM_BE_SELF;
            snprintf(note, notecap, "self backend %s; fell back to rules", rc < 0 ? "failed" : "passed");
        } else {
            snprintf(note, notecap, "no self backend callback; fell back to rules");
        }
        for (int i = 0; i < TLM_DECIDE_MAX_OPTIONS; i++) prob[i] = 0;
    }
    if (!run_rules(d, q, s, prob)) {
        size_t l = strlen(note);
        snprintf(note + l, notecap - l, "%sno rule matched; uniform", l ? "; " : "");
        if (q->type == TLM_Q_NOUL) prob[0] = 0.5f;
        else for (int i = 0; i < q->n_options; i++) prob[i] = 1.0f;
    }
    return TLM_BE_RULES;
}

/* ---- audit log ---------------------------------------------------------- */
static uint64_t next_id(tlm_decider *d) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t us = (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
    uint64_t id = (us << 12) | ((uint64_t)getpid() & 0xFFFu);
    if (id <= d->last_id) id = d->last_id + 0x1000u;
    d->last_id = id;
    return id;
}

static void iso_now(char *out, size_t cap) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    time_t t = ts.tv_sec;
    gmtime_r(&t, &tm);
    char b[32];
    strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%S", &tm);
    snprintf(out, cap, "%s.%03ldZ", b, ts.tv_nsec / 1000000L);
}

static uint64_t fnv1a(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}

static uint64_t state_hash(const tlm_state *s) {
    uint64_t h = 0xcbf29ce484222325ull;
    if (!s) return h;
    char buf[64];
    for (int i = 0; i < s->n_features; i++) {
        const char *nm = s->features[i].name ? s->features[i].name : "";
        h = fnv1a(h, nm, strlen(nm) + 1);
        int n = snprintf(buf, sizeof buf, "%.17g", s->features[i].value);
        h = fnv1a(h, buf, (size_t)n + 1);
    }
    if (s->text) h = fnv1a(h, s->text, strlen(s->text));
    return h;
}

static const char *opt_label(const tlm_question *q, int i) {
    if (q->type == TLM_Q_NOUL) return i ? "true" : "false";
    if (q->options && i >= 0 && i < q->n_options && q->options[i]) return q->options[i];
    return "?";
}

static int write_audit(tlm_decider *d, const tlm_question *q, const tlm_state *s, const tlm_answer *a,
                       tlm_backend requested, const char *note) {
    if (d->log_fd < 0) return 0;
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    if (!f) return tlm_seterr(TLM_ENOMEM, "decide: out of memory");
    tlm_jw w;
    tlm_jw_init(&w, f);
    char tmp[64];
    tlm_jw_begin_obj(&w);
    snprintf(tmp, sizeof tmp, "%" PRIu64, a->decision_id);
    tlm_jw_str(&w, "decision_id", tmp);
    iso_now(tmp, sizeof tmp);
    tlm_jw_str(&w, "ts", tmp);
    tlm_jw_str(&w, "question_id", q->id ? q->id : "");
    tlm_jw_str(&w, "type", tlm_qtype_name(q->type));
    if (q->question) tlm_jw_str(&w, "question", q->question);
    if (q->type != TLM_Q_NOUL) {
        tlm_jw_begin_arr(&w, "options");
        for (int i = 0; i < q->n_options; i++) tlm_jw_str(&w, NULL, opt_label(q, i));
        tlm_jw_end_arr(&w);
        if (q->criteria) {
            tlm_jw_begin_arr(&w, "criteria");
            for (int i = 0; i < q->n_options; i++) tlm_jw_str(&w, NULL, q->criteria[i] ? q->criteria[i] : "");
            tlm_jw_end_arr(&w);
        }
    }
    tlm_jw_int(&w, "safe_default", q->safe_default);
    snprintf(tmp, sizeof tmp, "%016" PRIx64, state_hash(s));
    tlm_jw_str(&w, "state_hash", tmp);
    tlm_jw_key_obj(&w, "features");
    if (s)
        for (int i = 0; i < s->n_features; i++)
            tlm_jw_num(&w, s->features[i].name ? s->features[i].name : "", s->features[i].value);
    tlm_jw_end_obj(&w);
    if (s && s->text) {
        size_t tl = strlen(s->text);
        if (tl > 256) {
            tl = 256;
            while (tl > 0 && ((unsigned char)s->text[tl] & 0xC0) == 0x80) tl--;   /* UTF-8 boundary */
        }
        char *t = (char *)malloc(tl + 1);
        if (t) {
            memcpy(t, s->text, tl);
            t[tl] = 0;
            tlm_jw_str(&w, "text", t);
            free(t);
        }
    }
    tlm_jw_begin_arr(&w, "probs");
    for (int i = 0; i < nopts(q); i++) tlm_jw_num(&w, NULL, (double)a->prob[i]);
    tlm_jw_end_arr(&w);
    tlm_jw_num(&w, "confidence", (double)a->confidence);
    tlm_jw_int(&w, "choice", a->choice);
    tlm_jw_str(&w, "choice_label", opt_label(q, a->choice));
    tlm_jw_str(&w, "backend", tlm_backend_name(a->used));
    tlm_jw_str(&w, "requested_backend", tlm_backend_name(requested));
    if (note && *note) tlm_jw_str(&w, "note", note);
    tlm_jw_num(&w, "auto_threshold", d->cfg.auto_threshold);
    tlm_jw_num(&w, "ask_threshold", d->cfg.ask_threshold);
    tlm_jw_str(&w, "outcome", tlm_policy_name(a->outcome));
    tlm_jw_int(&w, "final_choice", a->final_choice);
    tlm_jw_str(&w, "final_label", opt_label(q, a->final_choice));
    tlm_jw_end_obj(&w);
    fputc('\n', f);
    fclose(f);
    int rc = tlm_write_full(d->log_fd, buf, len);
    free(buf);
    if (rc != 0) return tlm_seterr(TLM_EIO, "decide: audit log write to %s failed", d->log_file);
    return 0;
}

/* ---- decide ------------------------------------------------------------- */
static int check_question(const tlm_question *q) {
    if (!q->id || !*q->id) return tlm_seterr(TLM_EINVAL, "decide: question without id");
    if ((unsigned)q->type > TLM_Q_NOUL) return tlm_seterr(TLM_EINVAL, "decide: %s: bad type", q->id);
    if (q->type == TLM_Q_NOUL) {
        if (q->safe_default != 0 && q->safe_default != 1)
            return tlm_seterr(TLM_EINVAL, "decide: %s: NOUL safe_default must be 0 or 1", q->id);
        return 0;
    }
    if (q->n_options < 1 || q->n_options > TLM_DECIDE_MAX_OPTIONS)
        return tlm_seterr(TLM_EINVAL, "decide: %s: n_options %d outside 1..%d", q->id, q->n_options,
                          TLM_DECIDE_MAX_OPTIONS);
    if (q->safe_default < 0 || q->safe_default >= q->n_options)
        return tlm_seterr(TLM_EINVAL, "decide: %s: safe_default %d outside 0..%d", q->id, q->safe_default,
                          q->n_options - 1);
    return 0;
}

static void evaluate(const tlm_question *q, tlm_answer *a) {
    normalise(q, a->prob);
    a->confidence = tlm_decide_confidence(q, a->prob);
    a->choice = argmax(q, a->prob);
}

int tlm_decide(tlm_decider *d, const tlm_state *s, const tlm_question *qs, int nq, tlm_answer *out) {
    if (!d || (!qs && nq > 0) || (!out && nq > 0) || nq < 0) return tlm_seterr(TLM_EINVAL, "decide: bad arguments");
    for (int i = 0; i < nq; i++) {
        int rc = check_question(&qs[i]);
        if (rc) return rc;
    }
    int first_err = 0;
    for (int i = 0; i < nq; i++) {
        const tlm_question *q = &qs[i];
        tlm_answer *a = &out[i];
        memset(a, 0, sizeof *a);
        char note[256] = "";
        tlm_backend want = d->cfg.backend;
        a->used = run_backend(d, want, q, s, a->prob, note, sizeof note);
        evaluate(q, a);
        double conf = a->confidence;
        if (conf < d->cfg.auto_threshold && conf >= d->cfg.ask_threshold && a->used != TLM_BE_SELF &&
            d->cfg.self_fn) {
            float p2[TLM_DECIDE_MAX_OPTIONS] = {0};
            int rc = d->cfg.self_fn(q, s, p2, d->cfg.self_ctx);
            size_t l = strlen(note);
            if (rc == 1) {
                memcpy(a->prob, p2, sizeof p2);
                a->used = TLM_BE_SELF;
                evaluate(q, a);
                conf = a->confidence;
                snprintf(note + l, sizeof note - l, "%sre-asked with self backend", l ? "; " : "");
            } else {
                snprintf(note + l, sizeof note - l, "%sself re-ask unavailable", l ? "; " : "");
            }
        }
        if (conf >= d->cfg.auto_threshold) a->outcome = TLM_POLICY_APPLY;
        else if (conf >= d->cfg.ask_threshold) a->outcome = d->cfg.interactive ? TLM_POLICY_REVIEW : TLM_POLICY_DEFAULT;
        else a->outcome = TLM_POLICY_DEFAULT;
        a->final_choice = a->outcome == TLM_POLICY_APPLY ? a->choice : q->safe_default;
        a->decision_id = next_id(d);
        int rc = write_audit(d, q, s, a, want, note);
        if (rc && !first_err) first_err = rc;
    }
    return first_err;
}

/* ---- explain ------------------------------------------------------------ */
static void print_bar(FILE *out, double p) {
    int n = (int)lround(clamp01(p) * 20);
    for (int i = 0; i < 20; i++) fputc(i < n ? '#' : '.', out);
}

static void explain_record(const tlm_jval *r, FILE *out) {
    const char *type = tlm_jstr(tlm_jobj_get(r, "type"), "?");
    int is_noul = strcmp(type, "noul") == 0;
    fprintf(out, "decision %s  %s\n", tlm_jstr(tlm_jobj_get(r, "decision_id"), "?"),
            tlm_jstr(tlm_jobj_get(r, "ts"), ""));
    fprintf(out, "  question   %s (%s)\n", tlm_jstr(tlm_jobj_get(r, "question_id"), "?"), type);
    const char *qt = tlm_jstr(tlm_jobj_get(r, "question"), NULL);
    if (qt) fprintf(out, "             %s\n", qt);
    const tlm_jval *feat = tlm_jobj_get(r, "features");
    fprintf(out, "  state      hash %s", tlm_jstr(tlm_jobj_get(r, "state_hash"), "?"));
    if (feat && feat->type == TLM_J_OBJ && feat->u.obj.n) {
        fprintf(out, "; features:");
        for (size_t i = 0; i < feat->u.obj.n; i++)
            fprintf(out, " %s=%g", feat->u.obj.keys[i], tlm_jnum(feat->u.obj.vals[i], NAN));
    }
    fputc('\n', out);
    const char *text = tlm_jstr(tlm_jobj_get(r, "text"), NULL);
    if (text) fprintf(out, "  text       %.120s%s\n", text, strlen(text) > 120 ? "..." : "");
    fprintf(out, "  backend    %s (requested %s)\n", tlm_jstr(tlm_jobj_get(r, "backend"), "?"),
            tlm_jstr(tlm_jobj_get(r, "requested_backend"), "?"));
    const char *note = tlm_jstr(tlm_jobj_get(r, "note"), NULL);
    if (note) fprintf(out, "  note       %s\n", note);
    const tlm_jval *probs = tlm_jobj_get(r, "probs");
    const tlm_jval *opts = tlm_jobj_get(r, "options");
    int64_t choice = tlm_jint(tlm_jobj_get(r, "choice"), -1);
    if (is_noul) {
        double p = tlm_jnum(tlm_jat(probs, 0), NAN);
        fprintf(out, "  answer     P(true) = %.4f  ", p);
        print_bar(out, p);
        fputc('\n', out);
    } else {
        fprintf(out, "  answer     ");
        for (size_t i = 0; i < tlm_jlen(probs); i++) {
            double p = tlm_jnum(tlm_jat(probs, i), NAN);
            if (i) fprintf(out, "             ");
            fprintf(out, "%c %-18s %.4f  ", (int64_t)i == choice ? '*' : ' ', tlm_jstr(tlm_jat(opts, i), "?"), p);
            print_bar(out, p);
            fputc('\n', out);
        }
    }
    fprintf(out, "  confidence %.4f (auto >= %.2f, ask >= %.2f)\n", tlm_jnum(tlm_jobj_get(r, "confidence"), NAN),
            tlm_jnum(tlm_jobj_get(r, "auto_threshold"), NAN), tlm_jnum(tlm_jobj_get(r, "ask_threshold"), NAN));
    fprintf(out, "  policy     %s -> final: %s (safe default %" PRId64 ")\n",
            tlm_jstr(tlm_jobj_get(r, "outcome"), "?"), tlm_jstr(tlm_jobj_get(r, "final_label"), "?"),
            tlm_jint(tlm_jobj_get(r, "safe_default"), -1));
}

int tlm_decide_explain(const char *log_path, uint64_t decision_id, FILE *out) {
    if (!log_path || !out) return tlm_seterr(TLM_EINVAL, "explain: no log path");
    char *file = resolve_log(log_path);
    if (!file) return tlm_seterr(TLM_ENOMEM, "explain: out of memory");
    FILE *f = fopen(file, "r");
    if (!f) {
        int rc = tlm_seterr(TLM_ENOTFOUND, "explain: cannot open %s: %s", file, strerror(errno));
        free(file);
        return rc;
    }
    char want[32];
    snprintf(want, sizeof want, "%" PRIu64, decision_id);
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    int found = 0;
    while (!found && (n = getline(&line, &cap, f)) > 0) {
        if (!strstr(line, want)) continue;
        char err[256];
        tlm_json *j = tlm_json_parse(line, (size_t)n, err, sizeof err);
        if (!j) continue;
        const tlm_jval *r = tlm_json_root(j);
        const tlm_jval *idv = tlm_jobj_get(r, "decision_id");
        int match = 0;
        if (idv && idv->type == TLM_J_STR) match = strcmp(idv->u.str.s, want) == 0;
        else if (idv && idv->type == TLM_J_NUM && idv->u.num.is_int) match = (uint64_t)idv->u.num.i == decision_id;
        if (match) {
            explain_record(r, out);
            found = 1;
        }
        tlm_json_free(j);
    }
    free(line);
    fclose(f);
    int rc = found ? 0 : tlm_seterr(TLM_ENOTFOUND, "explain: decision %s not found in %s", want, file);
    free(file);
    return rc;
}
