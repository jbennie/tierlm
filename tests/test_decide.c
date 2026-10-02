/* test_decide.c - decision engine: primitives, confidence maths, policy
 * bands, safe default, backend fallback, rule ordering, audit log and
 * explain round trip, built-in rules. */
#include "tlm_decide.h"
#include "tlm_json.h"
#include "tlm_platform.h"
#include "tlm_test.h"

#include <ftw.h>
#include <inttypes.h>
#include <stdint.h>
#include <unistd.h>

static char g_tmp[512];

static int rm_cb(const char *p, const struct stat *st, int flag, struct FTW *f) {
    (void)st; (void)flag; (void)f;
    return remove(p);
}

/* A rule that returns the distribution stored in ctx. */
typedef struct { float p[8]; int calls; } fixed_ctx;
static int fixed_rule(const tlm_question *q, const tlm_state *s, float *prob, void *ctx) {
    (void)q; (void)s;
    fixed_ctx *f = (fixed_ctx *)ctx;
    f->calls++;
    memcpy(prob, f->p, sizeof f->p);
    return 1;
}
static int pass_rule(const tlm_question *q, const tlm_state *s, float *prob, void *ctx) {
    (void)q; (void)s; (void)prob;
    (*(int *)ctx)++;
    return 0;
}

static const char *SPEC[] = {"generalist", "broad", "focused", "narrow"};
static const char *TIER[] = {"q8_0", "q6_k", "q5_k"};
static const char *ABC[] = {"a", "b", "c"};

static tlm_question noul(const char *id, int safe) {
    tlm_question q = {id, TLM_Q_NOUL, "statement", NULL, NULL, 0, safe};
    return q;
}

static tlm_decider *open_rules(int interactive) {
    tlm_decide_cfg cfg = {TLM_BE_RULES, 0.90, 0.60, interactive, NULL, NULL, NULL};
    return tlm_decide_open(&cfg);
}

static void test_confidence_math(void) {
    tlm_question q3 = {"x", TLM_Q_CHOICE, NULL, ABC, NULL, 3, 0};
    float uni[3] = {1.0f / 3, 1.0f / 3, 1.0f / 3};
    float one[3] = {0, 1, 0};
    float mix[3] = {0.7f, 0.2f, 0.1f};
    CHECK_NEAR(tlm_decide_confidence(&q3, uni), 0.0, 1e-6);
    CHECK_NEAR(tlm_decide_confidence(&q3, one), 1.0, 1e-6);
    double h = -(0.7 * log(0.7) + 0.2 * log(0.2) + 0.1 * log(0.1));
    CHECK_NEAR(tlm_decide_confidence(&q3, mix), 1.0 - h / log(3.0), 1e-6);
    tlm_question q1 = {"x", TLM_Q_CHOICE, NULL, ABC, NULL, 1, 0};
    float single[1] = {1};
    CHECK_NEAR(tlm_decide_confidence(&q1, single), 1.0, 0);
    tlm_question n = noul("x", 1);
    float p[2] = {0.9f, 0.1f};
    CHECK_NEAR(tlm_decide_confidence(&n, p), 0.8, 1e-6);
    p[0] = 0.5f;
    CHECK_NEAR(tlm_decide_confidence(&n, p), 0.0, 1e-6);
    p[0] = 0.0f;
    CHECK_NEAR(tlm_decide_confidence(&n, p), 1.0, 1e-6);
}

static void test_primitives_and_normalisation(void) {
    tlm_decider *d = open_rules(0);
    CHECK(d != NULL);
    fixed_ctx fc = {{2, 6, 2}, 0};                 /* unnormalised: 0.2 0.6 0.2 */
    tlm_decide_add_rule(d, "choice.", fixed_rule, &fc);
    tlm_question qs[3] = {
        {"choice.one", TLM_Q_CHOICE, "pick", ABC, NULL, 3, 0},
        {"choice.score", TLM_Q_SCORE, "rate", ABC, NULL, 3, 2},
        noul("noul.unhandled", 1),
    };
    tlm_answer a[3];
    CHECK_EQ_INT(tlm_decide(d, NULL, qs, 3, a), 0);
    CHECK_EQ_INT(fc.calls, 2);
    CHECK_EQ_INT(a[0].choice, 1);
    CHECK_NEAR(a[0].prob[0] + a[0].prob[1] + a[0].prob[2], 1.0, 1e-6);
    CHECK_NEAR(a[0].prob[1], 0.6, 1e-6);
    CHECK_EQ_INT(a[0].used, TLM_BE_RULES);
    CHECK_EQ_INT(a[1].choice, 1);
    CHECK_EQ_INT(a[1].outcome, TLM_POLICY_DEFAULT);          /* conf ~0.13 */
    CHECK_EQ_INT(a[1].final_choice, 2);                      /* safe default */
    /* no rule: uniform, confidence 0, safe default */
    CHECK_NEAR(a[2].prob[0], 0.5, 1e-6);
    CHECK_NEAR(a[2].confidence, 0, 1e-6);
    CHECK_EQ_INT(a[2].outcome, TLM_POLICY_DEFAULT);
    CHECK_EQ_INT(a[2].final_choice, 1);
    CHECK(a[0].decision_id != a[1].decision_id && a[1].decision_id < a[2].decision_id);
    /* negative / NaN probabilities are clamped; all-zero becomes uniform */
    fixed_ctx bad = {{-1, NAN, 0}, 0};
    tlm_decide_add_rule(d, "choice.bad", fixed_rule, &bad);
    tlm_question qb = {"choice.bad", TLM_Q_CHOICE, NULL, ABC, NULL, 3, 0};
    CHECK_EQ_INT(tlm_decide(d, NULL, &qb, 1, a), 0);
    CHECK_NEAR(a[0].prob[0], 1.0 / 3, 1e-6);
    CHECK_NEAR(a[0].confidence, 0, 1e-6);
    /* invalid questions are rejected before anything is decided */
    tlm_question inv[2] = {{"ok", TLM_Q_CHOICE, NULL, ABC, NULL, 3, 0}, {"bad", TLM_Q_CHOICE, NULL, ABC, NULL, 3, 3}};
    CHECK_EQ_INT(tlm_decide(d, NULL, inv, 2, a), TLM_EINVAL);
    tlm_question inv2 = {"bad", TLM_Q_SCORE, NULL, NULL, NULL, 0, 0};
    CHECK_EQ_INT(tlm_decide(d, NULL, &inv2, 1, a), TLM_EINVAL);
    tlm_question inv3 = noul("bad", 2);
    CHECK_EQ_INT(tlm_decide(d, NULL, &inv3, 1, a), TLM_EINVAL);
    tlm_question inv4 = {NULL, TLM_Q_NOUL, NULL, NULL, NULL, 0, 0};
    CHECK_EQ_INT(tlm_decide(d, NULL, &inv4, 1, a), TLM_EINVAL);
    tlm_decide_close(d);
}

static void test_policy_bands(void) {
    for (int interactive = 0; interactive <= 1; interactive++) {
        tlm_decider *d = open_rules(interactive);
        fixed_ctx fc = {{0}, 0};
        tlm_decide_add_rule(d, "p", fixed_rule, &fc);
        tlm_question q = noul("p.x", 1);
        tlm_answer a;
        fc.p[0] = 0.02f;                                     /* conf 0.96 -> apply false */
        tlm_decide(d, NULL, &q, 1, &a);
        CHECK_EQ_INT(a.outcome, TLM_POLICY_APPLY);
        CHECK_EQ_INT(a.choice, 0);
        CHECK_EQ_INT(a.final_choice, 0);
        fc.p[0] = 0.96f;                                     /* conf 0.92 -> apply true */
        tlm_decide(d, NULL, &q, 1, &a);
        CHECK_EQ_INT(a.outcome, TLM_POLICY_APPLY);
        CHECK_EQ_INT(a.final_choice, 1);
        fc.p[0] = 0.15f;                                     /* conf 0.70 -> band */
        tlm_decide(d, NULL, &q, 1, &a);
        CHECK_EQ_INT(a.outcome, interactive ? TLM_POLICY_REVIEW : TLM_POLICY_DEFAULT);
        CHECK_EQ_INT(a.choice, 0);
        CHECK_EQ_INT(a.final_choice, 1);                     /* safe default until reviewed */
        fc.p[0] = 0.40f;                                     /* conf 0.20 -> default */
        tlm_decide(d, NULL, &q, 1, &a);
        CHECK_EQ_INT(a.outcome, TLM_POLICY_DEFAULT);
        CHECK_EQ_INT(a.final_choice, 1);
        tlm_decide_close(d);
    }
    /* thresholds from cfg; defaults for 0; ask > auto is refused */
    tlm_decide_cfg cfg = {TLM_BE_RULES, 0.99, 0.10, 0, NULL, NULL, NULL};
    tlm_decider *d = tlm_decide_open(&cfg);
    fixed_ctx fc = {{0.97f}, 0};
    tlm_decide_add_rule(d, "", fixed_rule, &fc);
    tlm_question q = noul("t", 0);
    tlm_answer a;
    tlm_decide(d, NULL, &q, 1, &a);
    CHECK_EQ_INT(a.outcome, TLM_POLICY_DEFAULT);             /* 0.94 < 0.99, not interactive */
    tlm_decide_close(d);
    tlm_decide_cfg bad = {TLM_BE_RULES, 0.5, 0.7, 0, NULL, NULL, NULL};
    CHECK(tlm_decide_open(&bad) == NULL);
    d = tlm_decide_open(NULL);
    CHECK(d != NULL);
    tlm_decide_close(d);
}

static int self_conf(const tlm_question *q, const tlm_state *s, float *prob, void *ctx) {
    (void)q; (void)s;
    int *calls = (int *)ctx;
    (*calls)++;
    prob[0] = 0.99f;
    return 1;
}
static int self_pass(const tlm_question *q, const tlm_state *s, float *prob, void *ctx) {
    (void)q; (void)s; (void)prob;
    (*(int *)ctx)++;
    return 0;
}

static void test_backends(void) {
    int calls = 0;
    tlm_answer a;
    tlm_question q = noul("b.x", 1);
    fixed_ctx fc = {{0.15f}, 0};                             /* band answer from rules */

    /* SELF preferred */
    tlm_decide_cfg cfg = {TLM_BE_SELF, 0.9, 0.6, 0, NULL, self_conf, &calls};
    tlm_decider *d = tlm_decide_open(&cfg);
    tlm_decide_add_rule(d, "", fixed_rule, &fc);
    tlm_decide(d, NULL, &q, 1, &a);
    CHECK_EQ_INT(a.used, TLM_BE_SELF);
    CHECK_EQ_INT(calls, 1);
    CHECK_EQ_INT(fc.calls, 0);
    CHECK_EQ_INT(a.outcome, TLM_POLICY_APPLY);
    tlm_decide_close(d);

    /* RULES preferred, band -> re-ask with SELF */
    calls = 0;
    cfg.backend = TLM_BE_RULES;
    d = tlm_decide_open(&cfg);
    tlm_decide_add_rule(d, "", fixed_rule, &fc);
    tlm_decide(d, NULL, &q, 1, &a);
    CHECK_EQ_INT(fc.calls, 1);
    CHECK_EQ_INT(calls, 1);
    CHECK_EQ_INT(a.used, TLM_BE_SELF);
    CHECK_EQ_INT(a.outcome, TLM_POLICY_APPLY);
    CHECK_EQ_INT(a.final_choice, 1);
    tlm_decide_close(d);

    /* SELF passes -> falls back to rules */
    calls = 0;
    fc.calls = 0;
    tlm_decide_cfg cfg2 = {TLM_BE_SELF, 0.9, 0.6, 0, NULL, self_pass, &calls};
    d = tlm_decide_open(&cfg2);
    tlm_decide_add_rule(d, "", fixed_rule, &fc);
    tlm_decide(d, NULL, &q, 1, &a);
    CHECK_EQ_INT(a.used, TLM_BE_RULES);
    CHECK_EQ_INT(fc.calls, 1);
    tlm_decide_close(d);

    /* ENCODER is not ported: falls back to rules */
    fc.p[0] = 0.01f;
    tlm_decide_cfg cfg3 = {TLM_BE_ENCODER, 0.9, 0.6, 0, NULL, NULL, NULL};
    d = tlm_decide_open(&cfg3);
    tlm_decide_add_rule(d, "", fixed_rule, &fc);
    tlm_log_level(TLM_LOG_ERROR);                            /* silence the one-time warning */
    CHECK_EQ_INT(tlm_decide(d, NULL, &q, 1, &a), 0);
    tlm_log_level(TLM_LOG_WARN);
    CHECK_EQ_INT(a.used, TLM_BE_RULES);
    CHECK_EQ_INT(a.outcome, TLM_POLICY_APPLY);
    tlm_decide_close(d);
}

static void test_rule_ordering(void) {
    tlm_decider *d = open_rules(0);
    fixed_ctx general = {{1, 0, 0}, 0}, specific = {{0, 0, 1}, 0};
    int passes = 0;
    tlm_decide_add_rule(d, "expert.", fixed_rule, &general);
    tlm_decide_add_rule(d, "expert.domain", fixed_rule, &specific);
    tlm_decide_add_rule(d, "expert.domain.x", pass_rule, &passes);
    tlm_question q = {"expert.domain.x", TLM_Q_CHOICE, NULL, ABC, NULL, 3, 0};
    tlm_answer a;
    tlm_decide(d, NULL, &q, 1, &a);
    CHECK_EQ_INT(passes, 1);                                  /* longest tried first, passed */
    CHECK_EQ_INT(specific.calls, 1);
    CHECK_EQ_INT(general.calls, 0);
    CHECK_EQ_INT(a.choice, 2);
    tlm_question q2 = {"expert.other", TLM_Q_CHOICE, NULL, ABC, NULL, 3, 0};
    tlm_decide(d, NULL, &q2, 1, &a);
    CHECK_EQ_INT(general.calls, 1);
    CHECK_EQ_INT(a.choice, 0);
    tlm_question q3 = {"prompt.depth", TLM_Q_CHOICE, NULL, ABC, NULL, 3, 1};
    tlm_decide(d, NULL, &q3, 1, &a);
    CHECK_EQ_INT(a.outcome, TLM_POLICY_DEFAULT);
    CHECK_EQ_INT(a.final_choice, 1);
    tlm_decide_close(d);
}

static double needed_p(double sal, double mx, double md, double prot) {
    tlm_feature f[4] = {{"saliency_kept", sal}, {"mass_excluded", mx}, {"mass_deep", md}, {"protected", prot}};
    tlm_state s = {f, 4, NULL};
    tlm_question q = noul("expert.needed_for_usecase", 1);
    float p[TLM_DECIDE_MAX_OPTIONS] = {0};
    CHECK_EQ_INT(tlm_rule_needed(&q, &s, p, NULL), 1);
    return p[0];
}

static void test_builtin_rules(void) {
    tlm_decider *d = open_rules(0);
    tlm_decide_add_builtin_rules(d);
    tlm_answer a;

    /* specialisation from label entropy */
    tlm_question qs = {"expert.specialisation", TLM_Q_SCORE, NULL, SPEC, NULL, 4, 0};
    const double ent[4] = {1.0, 2.0 / 3, 1.0 / 3, 0.0};
    for (int lvl = 0; lvl < 4; lvl++) {
        tlm_feature f = {"label_entropy", ent[lvl]};
        tlm_state s = {&f, 1, NULL};
        tlm_decide(d, &s, &qs, 1, &a);
        CHECK_EQ_INT(a.choice, lvl);
        CHECK(a.confidence >= 0.9f);
        CHECK_EQ_INT(a.outcome, TLM_POLICY_APPLY);
    }
    {
        tlm_feature f = {"label_entropy", 0.5};                   /* between broad and focused */
        tlm_state s = {&f, 1, NULL};
        tlm_decide(d, &s, &qs, 1, &a);
        CHECK(a.confidence < 0.6f);
        CHECK_EQ_INT(a.outcome, TLM_POLICY_DEFAULT);
        CHECK_EQ_INT(a.final_choice, 0);                          /* safe: generalist */
        double prev = -1;                                         /* expected level falls with entropy */
        for (int i = 0; i <= 20; i++) {
            tlm_feature g = {"label_entropy", 1.0 - i / 20.0};
            tlm_state t = {&g, 1, NULL};
            float p[TLM_DECIDE_MAX_OPTIONS] = {0};
            tlm_rule_specialisation(&qs, &t, p, NULL);
            double e = p[1] + 2 * p[2] + 3 * p[3];
            CHECK(e >= prev - 1e-9);
            prev = e;
        }
        tlm_state none = {NULL, 0, NULL};
        float p[TLM_DECIDE_MAX_OPTIONS] = {0};
        CHECK_EQ_INT(tlm_rule_specialisation(&qs, &none, p, NULL), 0);
    }

    /* needed for use case */
    CHECK(needed_p(0.0, 1.0, 0.0, 1.0) > 0.999);                  /* protected */
    CHECK(needed_p(0.6, 0.95, 0.0, 0.0) > 0.999);                 /* high saliency on kept types */
    CHECK(needed_p(0.02, 0.5, 0.0, 0.0) > 0.99);                  /* not mostly excluded */
    CHECK(needed_p(0.02, 0.95, 0.2, 0.0) > 0.99);                 /* deep mass present */
    double lo = needed_p(0.02, 0.9, 0.0, 0.0);
    CHECK(lo < 0.05);
    CHECK(needed_p(0.0, 0.98, 0.0, 0.0) < 0.03);
    {
        double prev = 2;
        for (int i = 0; i <= 50; i++) {                           /* smooth, monotone in mass_excluded */
            double p = needed_p(0.02, i / 50.0, 0.0, 0.0);
            CHECK(p <= prev + 1e-9);
            prev = p;
        }
        double prev2 = -1;
        for (int i = 0; i <= 50; i++) {                           /* monotone in saliency */
            double p = needed_p(i / 100.0, 0.9, 0.0, 0.0);
            CHECK(p >= prev2 - 1e-9);
            prev2 = p;
        }
    }
    {
        tlm_feature f[3] = {{"saliency_kept", 0.02}, {"mass_excluded", 0.9}, {"mass_deep", 0.0}};
        tlm_state s = {f, 3, NULL};
        tlm_question q = noul("expert.needed_for_usecase", 1);
        tlm_decide(d, &s, &q, 1, &a);
        CHECK_EQ_INT(a.choice, 0);
        CHECK(a.confidence >= 0.9f);
        CHECK_EQ_INT(a.outcome, TLM_POLICY_APPLY);
        CHECK_EQ_INT(a.final_choice, 0);                          /* prune candidate */
        tlm_feature g[3] = {{"saliency_kept", 0.08}, {"mass_excluded", 0.82}, {"mass_deep", 0.0}};
        tlm_state t = {g, 3, NULL};
        tlm_decide(d, &t, &q, 1, &a);                             /* borderline -> keep */
        CHECK(a.outcome != TLM_POLICY_APPLY || a.final_choice == 1);
        CHECK_EQ_INT(a.final_choice, 1);
    }

    /* precision tier */
    struct { double heat, sal, depth; int want; } pt[] = {
        {0.95, 0.9, 3, 0},   /* deep + hot -> q8_0 */
        {0.2, 0.1, 3, 1},    /* deep + cold -> q6_k */
        {0.99, 0.99, 2, 1},  /* standard -> q6_k */
        {0.0, 0.0, 2, 1},
        {0.1, 0.1, 1, 2},    /* shallow, unimportant -> q5_k */
        {0.9, 0.9, 1, 1},    /* shallow but important -> q6_k */
        {0.99, 0.99, 0, 2},  /* excluded -> lowest */
    };
    tlm_question qp = {"expert.precision_tier", TLM_Q_CHOICE, NULL, TIER, NULL, 3, 0};
    for (size_t i = 0; i < sizeof pt / sizeof pt[0]; i++) {
        tlm_feature f[3] = {{"heat", pt[i].heat}, {"saliency", pt[i].sal}, {"depth", pt[i].depth}};
        tlm_state s = {f, 3, NULL};
        tlm_decide(d, &s, &qp, 1, &a);
        if (a.choice != pt[i].want) fprintf(stderr, "    precision case %zu: got %d\n", i, a.choice);
        CHECK_EQ_INT(a.choice, pt[i].want);
        CHECK(a.confidence >= 0.9f);
    }

    /* generic feature distribution */
    const char *dom[] = {"code.python", "sql", "generalist"};
    tlm_question qd = {"expert.domain", TLM_Q_CHOICE, NULL, dom, NULL, 3, 2};
    tlm_feature f[2] = {{"p.code.python", 0.97}, {"p.sql", 0.03}};
    tlm_state s = {f, 2, NULL};
    tlm_decide(d, &s, &qd, 1, &a);
    CHECK_EQ_INT(a.choice, 0);
    CHECK_NEAR(a.prob[0], 0.97, 1e-6);
    CHECK_NEAR(a.confidence, 1.0 - (-(0.97 * log(0.97) + 0.03 * log(0.03)) / log(3.0)), 1e-5);
    CHECK_EQ_INT(a.outcome, TLM_POLICY_DEFAULT);             /* 0.877 < 0.90, unattended */
    CHECK_EQ_INT(a.final_choice, 2);
    f[0].value = 0.99;
    f[1].value = 0.01;
    tlm_decide(d, &s, &qd, 1, &a);
    CHECK(a.confidence > 0.94f);
    CHECK_EQ_INT(a.outcome, TLM_POLICY_APPLY);
    CHECK_EQ_INT(a.final_choice, 0);
    tlm_feature pt2 = {"p.true", 0.3};
    tlm_state s2 = {&pt2, 1, NULL};
    tlm_question qe = noul("eval.equivalent", 0);
    tlm_decide(d, &s2, &qe, 1, &a);
    CHECK_NEAR(a.prob[0], 0.3, 1e-6);
    CHECK_NEAR(a.prob[1], 0.7, 1e-6);
    CHECK_EQ_INT(a.outcome, TLM_POLICY_DEFAULT);
    tlm_decide_close(d);
}

static void test_audit_and_explain(void) {
    char dir[600];
    snprintf(dir, sizeof dir, "%s/decisions", g_tmp);       /* created on open */
    tlm_decide_cfg cfg = {TLM_BE_ENCODER, 0.9, 0.6, 0, NULL, NULL, NULL};
    char dirslash[620];
    snprintf(dirslash, sizeof dirslash, "%s/", dir);
    cfg.log_path = dirslash;
    tlm_log_level(TLM_LOG_ERROR);
    tlm_decider *d = tlm_decide_open(&cfg);
    CHECK(d != NULL);
    if (!d) return;
    tlm_decide_add_builtin_rules(d);
    char logfile[700];
    snprintf(logfile, sizeof logfile, "%s/decisions.jsonl", dir);
    CHECK(strcmp(tlm_decide_log_file(d), logfile) == 0);
    tlm_feature f[4] = {{"saliency_kept", 0.02}, {"mass_excluded", 0.9}, {"mass_deep", 0.0}, {"label_entropy", 0.1}};
    tlm_state s = {f, 4, "layer 17 expert 44: def foo(x):\n    return \"bar\""};
    const char *crit[] = {"mixed", "few", "one area", "one narrow area"};
    tlm_question qs[2] = {
        noul("expert.needed_for_usecase", 1),
        {"expert.specialisation", TLM_Q_SCORE, "How specialised is this expert?", SPEC, crit, 4, 0},
    };
    qs[0].question = "This expert is needed for the selected content types";
    tlm_answer a[2];
    CHECK_EQ_INT(tlm_decide(d, &s, qs, 2, a), 0);
    tlm_decide_close(d);
    tlm_log_level(TLM_LOG_WARN);

    /* every line is a JSON object with the documented fields */
    FILE *fp = fopen(logfile, "r");
    CHECK(fp != NULL);
    if (!fp) return;
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    int lines = 0;
    while ((n = getline(&line, &cap, fp)) > 0) {
        char err[256];
        tlm_json *j = tlm_json_parse(line, (size_t)n, err, sizeof err);
        if (!j) fprintf(stderr, "    %s\n", err);
        CHECK(j != NULL);
        if (!j) continue;
        const tlm_jval *r = tlm_json_root(j);
        char want[32];
        snprintf(want, sizeof want, "%" PRIu64, a[lines].decision_id);
        CHECK(strcmp(tlm_jstr(tlm_jobj_get(r, "decision_id"), ""), want) == 0);
        CHECK(strcmp(tlm_jstr(tlm_jobj_get(r, "question_id"), ""), qs[lines].id) == 0);
        CHECK(strlen(tlm_jstr(tlm_jobj_get(r, "ts"), "")) == 24);
        CHECK(strcmp(tlm_jstr(tlm_jobj_get(r, "backend"), ""), "rules") == 0);
        CHECK(strcmp(tlm_jstr(tlm_jobj_get(r, "requested_backend"), ""), "encoder") == 0);
        CHECK(strstr(tlm_jstr(tlm_jobj_get(r, "note"), ""), "encoder") != NULL);
        CHECK_NEAR(tlm_jnum(tlm_jobj_get(tlm_jobj_get(r, "features"), "mass_excluded"), 0), 0.9, 1e-12);
        CHECK(strcmp(tlm_jstr(tlm_jobj_get(r, "outcome"), ""), tlm_policy_name(a[lines].outcome)) == 0);
        CHECK_EQ_INT(tlm_jint(tlm_jobj_get(r, "final_choice"), -9), a[lines].final_choice);
        CHECK(strlen(tlm_jstr(tlm_jobj_get(r, "state_hash"), "")) == 16);
        CHECK(strstr(tlm_jstr(tlm_jobj_get(r, "text"), ""), "def foo") != NULL);
        const tlm_jval *probs = tlm_jobj_get(r, "probs");
        CHECK_EQ_INT(tlm_jlen(probs), lines == 0 ? 2 : 4);
        CHECK_NEAR(tlm_jnum(tlm_jat(probs, 0), -1), a[lines].prob[0], 1e-6);
        tlm_json_free(j);
        lines++;
    }
    free(line);
    fclose(fp);
    CHECK_EQ_INT(lines, 2);

    /* explain finds each decision and prints the policy outcome */
    for (int i = 0; i < 2; i++) {
        char *buf = NULL;
        size_t len = 0;
        FILE *out = open_memstream(&buf, &len);
        CHECK_EQ_INT(tlm_decide_explain(dir, a[i].decision_id, out), 0);
        fclose(out);
        char want[64];
        snprintf(want, sizeof want, "decision %" PRIu64, a[i].decision_id);
        CHECK(strstr(buf, want) != NULL);
        CHECK(strstr(buf, qs[i].id) != NULL);
        CHECK(strstr(buf, "policy") != NULL);
        CHECK(strstr(buf, tlm_policy_name(a[i].outcome)) != NULL);
        if (i == 1) CHECK(strstr(buf, "* narrow") != NULL);
        else CHECK(strstr(buf, "P(true)") != NULL);
        free(buf);
    }
    FILE *devnull = fopen("/dev/null", "w");
    CHECK_EQ_INT(tlm_decide_explain(logfile, 12345, devnull), TLM_ENOTFOUND);
    CHECK_EQ_INT(tlm_decide_explain("/nonexistent/tlm/log.jsonl", 1, devnull), TLM_ENOTFOUND);
    fclose(devnull);

    /* unwritable log location fails at open */
    tlm_decide_cfg bad = {TLM_BE_RULES, 0.9, 0.6, 0, "/proc/tlm-no/decisions.jsonl", NULL, NULL};
    CHECK(tlm_decide_open(&bad) == NULL);
}

int main(void) {
    printf("test_decide\n");
    const char *td = getenv("TMPDIR");
    snprintf(g_tmp, sizeof g_tmp, "%s/tlm-test-decide-XXXXXX", td && *td ? td : "/tmp");
    if (!mkdtemp(g_tmp)) { perror("mkdtemp"); return 2; }
    RUN(test_confidence_math);
    RUN(test_primitives_and_normalisation);
    RUN(test_policy_bands);
    RUN(test_backends);
    RUN(test_rule_ordering);
    RUN(test_builtin_rules);
    RUN(test_audit_and_explain);
    nftw(g_tmp, rm_cb, 16, FTW_DEPTH | FTW_PHYS);
    TEST_MAIN_END();
}
