# Stage navigation: maps, waymarkers, instinct, and the decision weight

A generation is a journey. This note fixes the vocabulary for how the runtime
navigates it, what decides the local route, and when analysis can be skipped
entirely. It is the in-box companion to `cluster.md` (which assumes this
navigation exists and adds a remote drafter on top of it).

## 1. Navigation primitives

| Primitive   | What it is                                        | tierlm                                                                          | Timescale          |
|-------------|---------------------------------------------------|---------------------------------------------------------------------------------|--------------------|
| **Map**     | where everything is; the global layout            | expert placement, pins, replicas, atlas; heat-balanced                          | static, build-time |
| **Waymarker**| the per-position probability distribution over expert choices | router output `M->rp` (full softmax/sigmoid, every layer, every token) | dynamic, per token |
| **Transport**| how bytes move                                    | I/O backend (pread/io_uring), slot cache, prefetch queue                        | per read           |
| **Paths**   | the route actually taken                          | realized top-k per layer, per token                                             | the journey itself |

Each is optimized to the nature of its scale: the map minimizes expected I/O
over the whole workload (placement by heat); the waymarker decides the local
breadth (how much of the distribution to act on); transport matches the stage's
I/O profile; the path is what is realized. The runtime today optimizes the map
(build-time) and uses the waymarker only as a fixed top-k projection. This note
adds the missing layer: a **decision weight** that reads the waymarker and the
heat and chooses the breadth.

## 2. Decision vocabulary

`tlm_decide`'s confidence policy already *is* a use/avoid/adapt/explore
selector for coarse decisions:

- **APPLY** (high confidence) = well-trodden map → follow it, act decisively.
- **REVIEW** (mid) = locality of specific interest → surface it, explore options.
- **DEFAULT** (low) = avoid → safe default, don't go there.

**Do not route per-token decisions through `tlm_decide`.** Typed answers, a
confidence policy, and a JSONL audit write per decision, millions of times per
generation, is a category error and a throughput disaster. The split:

- **Coarse — decision engine, once per journey (logged, confidence-gated):**
  pick the *strategy* for this trip. "Well-trodden domain → follow the map
  (narrow, high-confidence) or locality of interest → explore (wider sampling,
  speculation on)?" A CHOICE with a safe default = current behavior.
- **Fine — inline, deterministic, per token:** *execute* the strategy. The
  decision weight below. No logging, no policy, just arithmetic.

## 3. Stages

- **Prefill** — the well-trodden map to a known destination (the prompt is
  given). Today this is literally "the decode step repeated" (P-92), with the
  same one-layer lookahead as decode. There is no prefill-specific transport
  yet; P-36 (coverage-gated whole-next-layer) is consensus-adopted but not
  built.
- **Decode (bulk)** — the journey along the map. Lookahead (P-33) keeps one
  layer ahead.
- **Last mile** — the locality of specific interest, where you *explore*. This
  is M3 speculation, and the useful refinement from this framing is that
  exploration should be **stage-gated** (enabled near the end, or when the
  coarse decision judges the query to be a locality of interest), not uniform
  always-on.

## 4. Instinct

**Instinct is when you skip detailed analysis because the answer is almost
always this way.** It is a first-class concept, not a threshold. Two sources:

- **The instinct vector** — a per-layer vector of *dynamically learned*
  routing-shape statistics. Its core component is the EMA of the top-1 share of
  routed mass (how peaked this layer's waymarker tends to be), learned at
  runtime as tokens flow (`moe()` observes the real route each layer).
- **The graph** — layers are nodes; the lookahead link (l → l+1) is the edge.
  The per-node instinct vector plus the per-node given heat *is* the dynamic
  instinct vector graph. Cross-layer recall (how often l's prediction covered
  l+1's real top-k) is the edge weight; it is telemetry for now, action later.

Instinct does not replace the waymarker; it decides *how much* of the
waymarker to act on. A peaked waymarker + hot layer = the answer is obvious =
jump (narrow). A flat waymarker + cold layer = deliberate (widen).

## 5. The decision weight (three terms)

The local navigation decision — how much breadth to prefetch — is a single
weight **W** in [0,1] that is the product of exactly three terms:

| Term                    | Symbol | Source                          | Role        | Direction        |
|-------------------------|--------|---------------------------------|-------------|------------------|
| **Interest aggressiveness** | `α`  | policy, `nav.interest_aggressiveness` | the driver | high → widen |
| **Instinct vector**     | `I`    | dynamic, learned per layer       | dampener    | high (peaked) → narrow |
| **Model heat**          | `H`    | given, manifest `loc.heat` per layer (the hormone) | dampener | high (hot) → narrow |

In human terms the weight asks three things at once:

- **"This question needs"** (`α`) — how much exploration *this* journey demands. It
  is a **per-journey** quantity: the coarse decision (or a heuristic on the
  prompt) sets it. A config default stands in until the coarse decision is
  built; the code keeps it a per-model field so a per-journey setter is a
  small step away.
- **"My experience tells me"** (`I`) — what I have learned doing this at
  runtime: the instinct vector.
- **"My training tells me"** (`H`) — what I was calibrated on: the given heat
  prior.

Breadth is how much to *explore* right now, given what this question needs,
tempered by what experience and training already say is obvious.

```
W = α · (1 − I) · (1 − H)        ∈ [0, α] ⊆ [0, 1]
k_eff = top_k + round(W · (lookahead_k − top_k))
```

- `α` is the only user-settable term — it is a policy choice (how boldly to
  chase the interest). `I` is measured (learned) and `H` is given (calibrated);
  neither is a choice, so neither is a knob. Three terms, one weight, no
  per-term fudge factors.
- `α = 0` → `W = 0` → always narrow to top_k: this is the **off** state and the
  legacy default, so the feature is strictly opt-in.
- `α > 0` with a peaked, hot waymarker (`I, H → 1`) → `W → 0` → instinct
  (jump). With a flat, cold waymarker (`I, H → 0`) → `W → α` → deliberate
  (widen toward `lookahead_k`).
- **Heat as hormone:** `H` *lowers the bar* for instinct. A well-trodden (hot)
  region is trusted more — the same way a hormone shifts priority. The given
  heat is the compiler's calibration selection frequency, already sealed in the
  manifest (`tlm_packet_loc.heat`, the same value the store reads for pins).

Prefetch breadth never changes output (P-34), so W is a pure speed/efficiency
dial: it trades speculative reads (wasted when the waymarker is peaked) against
miss coverage (needed when it is flat).

## 6. Gaps and A/B plan

1. **Runtime negative waymarker** — a probability floor on speculative loads
   (don't reserve a slot or issue a read for `p[e]` below a threshold). Prior
   art: P-34 (speculation never hurts demand) and P-36 (coverage gating). A/B on
   doris: a few floor values on the miss-bound run; measure wasted-read bytes and
   tok/s.
2. **Stage navigator** — a thin mode selector (prefill / decode / last-mile)
   that picks transport + waymarker strategy per stage; the decision engine
   makes the coarse choice per journey. The decision weight above is its first
   in-box instance.
3. **Stage-gated last-mile exploration** — M3 speculation enabled only near the
   end or when the coarse decision flags a locality of interest.

Queued for doris (M2): the `--lookahead-k` breadth sweep (8/12/16/24) on the
1G-miss run, now that the knob is exposed; then the decision-weight A/B
(α = 0 vs 0.5 vs 1.0, with heat from the real manifest) to measure whether the
three-term weight beats fixed breadth on a real store.

## 7. Implemented here

- `nav.interest_aggressiveness` config key + `--interest` CLI flag.
- Per-layer **instinct vector** (dynamic `share_ema`, learned in `moe()`) and
  per-layer **model heat** (given, from the manifest) in `tlm_model`.
- The three-term decision weight in `tlm_model` lookahead: `α = 0` is the
  legacy fixed-breadth path (default, bit-identical to before); `α > 0` engages
  the weight. Invariance is preserved by construction (breadth never changes
  output); `test_model` runs the invariance suite with `α > 0`.
