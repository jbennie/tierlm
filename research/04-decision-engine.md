# Decision engine and content selection

Oct 2, 2026 · @Jay

## At a glance

Embed a Jev-style typed decision engine, `tlm-decide`, inside `tlmc` and `tierlm-setup`. Every judgment call in building and pruning becomes a typed question with a fixed answer set and a calibrated probability, instead of a hard-coded heuristic or a free-text guess.

- **Three primitives, as in Jev:**
  - **Choice** picks one option from a set.
  - **Score** places the input on an ordered rubric.
  - **Noul** gives the probability that a statement is true. Answers always come back in the declared type, with a probability for every option.
- **Backends, all local and offline:**
  - An embedded encoder decision model (Laya / OpenJevX class, \~0.4B parameters, Apache-2.0) ported to our C kernels, for high-volume decisions.
  - "Self" scoring with the model being compiled, for the few decisions where its own knowledge matters most.
  - A deterministic rule layer that always has the last word on hard limits.
- **What it decides:**
  - What content each expert specialises in, which feeds an expert atlas.
  - Which experts the selected use case needs.
  - Precision tier per expert.
  - How the calibration prompts break down by content type.
  - Whether two eval answers are equivalent when no exact checker exists.
- **Policy:**
  - Confident answers are applied automatically.
  - Middling ones are shown to you in the setup tool or compile run.
  - Uncertain ones fall back to the safe default: keep the expert, higher precision.
  - Every decision is logged with its inputs, probabilities and outcome, and `tlmc explain` replays any of them.
- **Setup gets an "Experts and content" section:** set each content type (code by language, maths, law, medicine, languages, tool calling, …) to `exclude`, `shallow`, `standard` or `deep`. Pin or ban individual experts by id. See the estimated size and speed effect before anything is written.

## Which backend to embed

Jev itself is a hosted API with no released weights, so it cannot be embedded; the open Laya family and its OpenJevX derivative implement the same idea and can be ported to C.

| Option | What it is | Size and speed | Fit for tlmc |
| --- | --- | --- | --- |
| [Jev](https://www.cometapi.com/models/typesafe-ai/jev) (TypeSafe AI) | Closed, hosted System One model; Choice, Score, Noul; 64k context | Vendor-reported 70–500 ms per request; weights and size not published | Not embeddable; would send model internals to a cloud API. Use only its request shape. |
| [Laya](https://www.orcarouter.ai/blog/laya-decision-model-explained) (ConvAI) | Open-weight, non-autoregressive decision model on ModernBERT-large; English, multilingual and typed-decisions checkpoints | 421M parameters (multilingual 322M), 512–1,024 token context, Apache-2.0 | Embed. Its authors call it a base to specialise rather than a zero-shot engine, so we fine-tune it on our own decision traces. |
| [OpenJevX](https://huggingface.co/muthuishere/openjevx) | Laya derivative with a dynamic typed-decision head, Jev-compatible request shape | 0.4B parameters; 0.774 accuracy on the typed-decisions benchmark; \~23 ms per 5-question case on an RTX 4090 | Embed as the starting checkpoint for the encoder backend. |
| Self (the model being compiled) | Score each allowed answer's token probability with the big model, no generation (colibri's Brio mode does this) | No extra weights; 1–5 s per decision on CPU for short states | Use for low-volume, knowledge-heavy calls: naming what an expert specialises in. |
| Rules | Deterministic checks in C | Microseconds | Always applied last: RAM fit, explicit include/exclude, safety rules. |

**Default wiring:**

- The encoder backend handles volume: labelling \~250k calibration prompts by content type, and first-pass expert labels.
- The self backend re-checks expert labels whose encoder confidence is below 0.9.
- Rules gate everything.

The encoder runs on the CPU in Q8\_0 (\~420 MB). Its 512–1,024 token context is enough for the short "state" summaries the compiler builds.

## How a decision is made

&#91;embedded content: decision flow · two backends, one policy, hard rules last\]

Measured numbers bypass the engine and go straight to the rules; the engine only supplies the judgments that cannot be computed, and the eval gate can still undo them.

## C API and data shapes

One header, `decide.h`, shared by `tlmc` and `tierlm-setup`; the request mirrors Jev's state-plus-questions shape so decision sets can be tested against any Jev-compatible server too.

```c
typedef enum { TLM_Q_CHOICE, TLM_Q_SCORE, TLM_Q_NOUL } tlm_qtype;
typedef enum { TLM_BE_ENCODER, TLM_BE_SELF, TLM_BE_RULES_ONLY } tlm_backend;

typedef struct {
    const char  *id;            /* "expert.domain", "prune.needed", ... */
    tlm_qtype    type;
    const char  *question;      /* instruction text */
    const char **options;       /* CHOICE: labels; SCORE: rubric levels low->high */
    const char **criteria;      /* optional one-line definition per option */
    int          n_options;     /* NOUL: 0 */
} tlm_question;

typedef struct {
    int    choice;              /* index into options, -1 for NOUL */
    float  prob[32];            /* per option; NOUL: prob[0] = P(true) */
    float  confidence;          /* 1 - normalised entropy; NOUL: |2p-1| */
    tlm_backend used;
    uint64_t    decision_id;    /* key into the audit log */
} tlm_answer;

tlm_decider *tlm_decide_open(const tlm_decide_cfg *cfg);   /* loads encoder, policy */
int  tlm_decide(tlm_decider *, const char *state_json,
                const tlm_question *qs, int nq, tlm_answer *out);
int  tlm_decide_explain(tlm_decider *, uint64_t decision_id, FILE *out);
void tlm_decide_close(tlm_decider *);
```

**Rules for callers:**

- Questions in one call are independent, as in Jev. Chained decisions are separate calls joined by compiler code.
- Option names carry their definition in `criteria`, not just a label. Typed decision models have been shown to follow the option name rather than the definition bound to it ([arXiv 2609.26758](https://arxiv.org/pdf/2609.26758)), so option names are written to mean what they say; avoid bare yes/no labels.
- Every answer is appended to `/var/lib/tierlm/decisions/<model>.jsonl` with `decision_id`, the question, a hash of the state, probabilities, backend, policy outcome and timestamp.

## Decisions the compiler delegates

Numbers that can be measured exactly (heat, REAP saliency, bytes, bandwidth) stay as numbers; the engine is only asked what cannot be computed, and its answers become features alongside those numbers.

| Pass | Question id | Type | State given to the engine | Options | Used for |
| --- | --- | --- | --- | --- | --- |
| 3 Calibrate | `prompt.content_type` | Choice (multi-question) | One calibration prompt, truncated to 1k tokens | The content taxonomy (see setup section) | Weighting calibration by your use case; coverage report |
| 3 Calibrate | `prompt.depth` | Score | Same prompt | surface, working, expert, research | How deep each content type is used |
| 3 Calibrate | `expert.domain` | Choice | Expert summary: top 20 activating token windows from calibration, decoded; layer; heat; co-firing experts' labels | Taxonomy + `generalist` | Expert atlas, exclusion, protection |
| 3 Calibrate | `expert.specialisation` | Score | Same summary + label distribution entropy | generalist, broad, focused, narrow | Only `focused`/`narrow` experts can be pruned by content exclusion |
| 4 Precision | `expert.precision_tier` | Choice | Heat, saliency, domain, domain depth setting, size | q8\_0, q6\_k, q5\_k | Mixed precision in `balanced`/`aggressive` |
| 5 Prune | `expert.needed_for_usecase` | Noul | Expert label, specialisation, saliency on kept domains, your content settings | statement: "This expert is needed for the selected content types" | Prune candidate list, ranked with REAP saliency |
| 9 Validate | `eval.equivalent` | Noul | Reference answer and compiled-model answer on a held-out prompt | statement: "The second answer is equivalent in substance to the first" | Task-score gate when no exact checker exists |
| setup | `setup.recommend.<key>` | Choice | Probe results + current value + allowed values | Allowed values of that key | Recommendation shown next to deterministic rules' value |

Ranking for pruning combines both kinds of evidence: candidate score = (1 − normalised REAP saliency on kept domains) × P(not needed). Only experts above the profile's cut-off are tried, and the eval gate decides whether the prune stands.

## Confidence policy and hard rules

The engine advises; deterministic rules and your explicit settings decide. A probabilistic answer never overrides a fact the compiler can check exactly.

| Confidence | Action | Example |
| --- | --- | --- |
| ≥ `decide.auto_threshold` (default 0.90) | Apply automatically, log it | Expert 17:178 labelled `code.python`, 0.94 |
| Between `decide.ask_threshold` (0.60) and auto | Re-ask with the self backend; if still in the band, queue for review (interactive) or take the safe default (unattended) | `prune.needed` P = 0.30 for a law expert when you set law to `shallow` |
| < `decide.ask_threshold` | Safe default: keep the expert, keep the higher precision, label `unknown` | Mixed-domain expert with flat probabilities |

**Hard rules, applied after every decision and never overridden:**

- Your explicit `experts.protect` and `experts.exclude` ids always win.
- Router, norms, attention, shared experts, MTP head, embeddings and lm\_head are never pruned or dropped below Q8\_0.
- At least `prune.min_keep_per_layer` of experts per layer survive (default 0.5), and never fewer than top-k × 4.
- An expert labelled `generalist` or `broad` is never removed by a content exclusion, only by REAP saliency under the eval gate.
- In the `exact` profile the engine runs for labelling and the atlas only; it changes nothing.
- The eval gate is final: a decision set that fails it is rolled back regardless of confidence.

**Calibration of the engine itself:** before its answers are trusted on a new model family, `tlmc decide-calibrate` checks 200 expert labels with the self backend and reports agreement. If agreement is below 85%, the auto threshold is raised to 0.95 for that model.

## Setup: experts and content

`tierlm-setup` gains an **Experts and content** section. You set how deeply each content type must be supported, or exclude it. The compiler turns that into calibration weighting, pinning, precision, replicas and, outside the `exact` profile, pruning.

**Depth levels**

| Level | Calibration weight | Pruning of its specialist experts | Precision | Pinning and replicas |
| --- | --- | --- | --- | --- |
| `exclude` | 0 (still evaluated, for the report) | Allowed: candidates if specialisation is `focused`/`narrow` | Lowest allowed by profile | Never pinned |
| `shallow` | 0.25× | Allowed for `narrow` only | Q5\_K allowed | Low priority |
| `standard` (default) | 1× | REAP and gate only | Q6\_K | By heat |
| `deep` | 2× | Never, even if cold | Q6\_K, or Q8\_0 for the hottest | Pin priority and replicas first |

**Taxonomy** (two levels; you can set a parent or any child, and children inherit). The list is a file, `/etc/tierlm/content-taxonomy.conf`, so you can add types.

- `code`: `python`, `c_cpp`, `rust`, `go`, `js_ts`, `java`, `sql`, `shell`, `other`
- `math`, `science`, `engineering`
- `law`, `medicine`, `finance`
- `lang`: `en`, `zh`, `ja`, `ko`, `de`, `fr`, `es`, `other`
- `creative`, `chat`
- `tools`: `json`, `function_calling`, `agentic_traces`
- `reasoning` (long chain-of-thought)

**Keys written to `/etc/tierlm/tierlm.conf`** (or a per-use-case file under `conf.d`):

```conf
# -- Experts and content --------------------------------------------
content.default            = standard
content.code               = deep
content.code.python        = deep
content.code.sql           = deep
content.tools              = deep
content.math               = standard
content.lang.en            = deep
content.lang.zh            = shallow
content.lang.ja            = exclude
content.law                = exclude
content.medicine           = exclude
content.creative           = shallow

experts.protect            = 12:178, 17:44        # layer:expert, never pruned or demoted
experts.exclude            =                      # explicit removals, still eval-gated
prune.max_ratio            = 0.25                 # cap regardless of exclusions
prune.min_keep_per_layer   = 0.50

decide.backend             = encoder              # encoder | self | rules
decide.encoder_model       = /usr/share/tierlm/decide/openjevx-q8_0.tlmd
decide.auto_threshold      = 0.90
decide.ask_threshold       = 0.60
decide.log                 = /var/lib/tierlm/decisions
```

**Interactive flow**

1. The section opens with a content breakdown of your calibration corpus. The encoder labels each prompt, for example 58% code (python 31%, sql 12%), 14% tools, 11% chat. Recommended depths are pre-filled from it: frequent types `deep`, absent types `shallow`.
2. You change any row. If a compiled atlas exists for the model, each row also shows its experts: specialists, generalists, GB on disk and the estimated effect of excluding it. Example: "law: 412 narrow experts, \~5.6 GB, −3% store size, no change to decode speed".
3. A second screen lists individual experts for the types you touched, sorted by heat, with their label and confidence. Mark any as protect or exclude.
4. The summary shows the projected compile result before writing: store size, RAM-resident share, expected hit rate, and which profile the choices require. Exclusions need `balanced` or `aggressive`; in `exact` they only change calibration weighting and pinning.

```sh
sudo tierlm-setup --section content                 # just this section
sudo tierlm-setup --section content --model ~/tlm/dsv4f-coding.tlmm   # with the atlas
tlmc atlas  ~/tlm/dsv4f-coding.tlmm --type law      # inspect experts for one type
```

## How an exclusion becomes pruning

Excluding a content type never deletes experts directly. It marks the specialists for that type as candidates, and the same REAP-plus-gate loop as any other prune decides what actually goes.

This caution has a reason. Experts are rarely clean single-topic units. In colibri's GLM-5.2 atlas, one Python-region expert measured 20.2% Python, 14.6% JSON, 14.2% conversation and 13.3% SQL. Pruning by topic alone would cut capability you kept.

1. **Candidate:** an expert qualifies only if all of these hold:
   - `expert.specialisation` is `focused` or `narrow`.
   - Its label mass in excluded types is ≥ 0.8.
   - Its label mass in `deep` types is < 0.05.
2. **Rank:** candidates are sorted by (1 − REAP saliency measured on the kept-type calibration set) × P(not needed for the selected content types).
3. **Cap:** remove from the top of the list until `prune.max_ratio` or the per-layer floor is reached.
4. **Renormalise:** router logits are renormalised over the surviving experts, the same way REAP does.
5. **Gate:** the held-out eval runs on kept types only. Pass → keep. Fail → put back the last 5 points of the ratio and retry.
6. **Report:** `tlmc report` lists, per content type, the experts removed, GB saved, and the measured change on kept types. It also runs the excluded types' eval and shows how much they degraded, so you know what you gave up. That figure is reported, not gated.
7. **Reversible:** unpruned packets stay on the drives when space allows, so `tlmc compile --reuse` with new content settings only re-runs passes 5, 8 and 9.

Two safeguards against overfitting:

- The calibration corpus must contain at least 2% of every type you mark `standard` or `deep`. The setup tool warns when a type you care about is thin in your logs.
- REAP's own results support the approach at scale. Its authors report near-lossless code generation and tool calling with half the experts of Qwen3-Coder-480B and Kimi-K2 removed, but those were measured on the retained domains, the same thing our gate checks.

## Changes to the milestone plan

The decision engine adds about four weeks. Most of it is porting the ModernBERT encoder and building the labelled traces needed to fine-tune it.

| Milestone | Added | Extra time |
| --- | --- | --- |
| M1 Foundations | `decide.h` API, audit log, rules backend; Jev-compatible JSON request format for tests | +0.5 week |
| M3 Fast CPU kernels | ModernBERT-large encoder forward pass in C (alternating local/global attention, GeGLU) on the existing Q8\_0 kernels; OpenJevX checkpoint converted to `.tlmd`; parity test against its reference outputs | +1.5 weeks |
| M4 Expert streaming | Self backend (option scoring with the loaded model); calibration labelling; expert atlas; `tlmc atlas` | +1 week |
| M6 Ship v1 | Prune ranking with decisions; setup **Experts and content** section with atlas preview; `tlmc explain`; fine-tune recipe for the encoder on our own decision logs (offline, Python allowed) | +1 week |

Gate added: on 500 hand-checked experts from DeepSeek V4 Flash, the label for every decision that was applied automatically agrees with the self backend ≥ 90% of the time.

## Sources

- [CometAPI: Jev model overview (TypeSafe AI)](https://www.cometapi.com/models/typesafe-ai/jev)
- [OpenJevX on Hugging Face](https://huggingface.co/muthuishere/openjevx)
- [Laya explained (ConvAI Innovations)](https://www.orcarouter.ai/blog/laya-decision-model-explained); [Laya model family](https://www.llmreference.com/model-family/laya)
- [Type-Safe Is Not Error-Free (arXiv 2609.26758)](https://arxiv.org/pdf/2609.26758)
- [REAP (arXiv 2510.13999)](https://arxiv.org/abs/2510.13999v1)
- [JustVugg/colibri](https://github.com/JustVugg/colibri) (expert atlas, Brio option-scoring mode)
