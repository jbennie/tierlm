# tierlm cluster design: prediction slaves, expert hosts and other arrangements

Oct 2, 2026 · status: proposal for review · inputs: fragments/distributed.md (F-dist-01..20),
fragments/CONSENSUS.md, research/01-04, web sources in section 5

## At a glance

- **The requested arrangement works on the 10GbE you have today and needs no RDMA.** A
  prediction slave only sends advice (expert ids, draft tokens) and the master never waits for
  it, so the link's round trip (~25-60 µs) is tiny next to an NVMe expert read (~1.3-2.5 ms).
  Bandwidth is not an issue either: well under 1 MB per token in either direction.
- **It is lossless by construction.** The master always computes its own routing and verifies
  every draft. Predictions only reorder and start NVMe reads earlier; drafts only change how
  many tokens one verify pass commits. A dead, slow or wrong slave changes speed, never output.
- **Its gain is capped by the I/O stall that survives the in-box router lookahead**, and the
  reference projects measured several predictors that improved recall but lost end to end
  (P-35, P-38, P-39) and an external drafter that made a streamed MoE 4.5x slower (P-88). So
  the first step is a single-box measurement with a perfect-oracle replay that bounds what any
  prediction slave can buy, before any network code.
- **If the slave has a lot of RAM, hosting experts beats predicting them.** For DeepSeek V4
  Flash the master already caches ~65% of experts; a slave with ~64 GB free holds the rest, so
  the pair runs the model fully from RAM. That costs 43 round trips per token (est. 3.5-4.5 ms
  on 10GbE TCP, about 1 ms or less on 25/100GbE RoCE) and removes most NVMe misses. It also makes
  speculation cheaper, because verifying extra tokens then reads RAM, not disk.
- **The onboard NICs cannot do RDMA.** The board's 10GbE is an Intel X710-AT2 (Intel's RDMA
  driver covers only E810 and X722). Red Hat deprecated Soft-RoCE in RHEL 8 with removal planned
  for RHEL 9, and the RHEL 9 RDMA guide documents only Soft-iWARP, as a Technology Preview (check
  `modinfo rdma_rxe siw` on 10.2). The USB4 ports (ASMedia ASM4242) carry TCP through
  thunderbolt-net, not verbs. Real RDMA means an add-in ConnectX card per box.
  Plan TCP first and keep verbs as a second backend behind the same interface (ds4's pattern,
  F-dist-08).
- **Ranking for a first experiment:** (1) single-box measurement plus the transport skeleton;
  (2) expert host if a slave with at least 64 GB is available, otherwise the prediction slave;
  (3) layer pipeline only as a fallback on high-latency links; tensor parallel only after RoCE
  NICs; never remote RAM as a cache tier on this master (four local NVMe out-read 10GbE about
  20 to 40 times).

## 0. Ground truth this design uses

### 0.1 Links available

| Link | Hardware on/for this box | Small-message RTT | Payload bandwidth | RDMA? | Source / status |
|---|---|---|---|---|---|
| 10GbE TCP | onboard Intel X710-AT2, 2 ports | ~27-57 µs without busy polling; ~13-16 µs with it | ~1.17 GB/s (9.41 Gb/s measured on a 10GbE direct link) | no (X710 has no RDMA engine) | netdev 2.1 busy-poll paper: 17.5k-77k single-flow TCP_RR transactions/s on ixgbe/mlx4 10G NICs, i.e. 13-57 µs RTT; 9.41 Gb/s in the llama.cpp RPC benchmark repo |
| 10GbE soft RDMA | same NIC + Soft-iWARP (`siw`, RHEL TP) or a self-built `rdma_rxe` | ~20-70 µs | ~0.5-0.6 GB/s | verbs API, CPU transport | Level1Techs USB4 cluster notes: Soft-RoCE send 25-67 µs, write 17-71 µs, 4.3 Gb/s. Use it to develop the verbs path, not for speed |
| 25GbE RoCEv2 | add-in ConnectX-4 Lx / ConnectX-5 + DAC | ~3-6 µs | ~2.9 GB/s | yes | RAMCloud verbs table (RoCE RTT 4.5-9.6 µs on 2010-era cards); ConnectX-5 `ibv_rc_pingpong` ~4.6-4.9 µs |
| 100GbE RoCEv2 | add-in ConnectX-5/6 + DAC | ~3-5 µs | ~11-12 GB/s | yes | same; eRPC reports 2.3 µs median RPC latency on lossy Ethernet |
| USB4 40G | onboard ASMedia ASM4242, 2 Type-C | est. 50-100 µs (ICMP 35-500 µs measured on AMD USB4 links) | 2.5-4.4 GB/s (20-35 Gb/s measured on AMD hosts) | no hardware RDMA on Linux | Level1Techs notes; a CachyOS user reports ASM4242 Linux↔macOS link cycling on kernel 7.2 and points to an upstream fix (commit 1881f2efbf7f), Aug 2026: test the cable link before relying on it |

Apple's RDMA over Thunderbolt 5 (macOS 26.2) shows what a low-latency link buys: under 50 µs
vs ~300 µs for TCP in Jeff Geerling's tests, 4 Mac Studios at 31.9 tok/s on Qwen3 235B with
exo/RDMA vs 15.2 tok/s with llama.cpp over TCP (20.4 tok/s on one machine). Linux has no
equivalent over USB4 today.

Measure, do not trust these: every slave link gets a `PROBE` at connect (64 B and 8/64 KiB
ping-pongs, p50/p99, plus a 256 MiB stream) and roles are assigned from the measured numbers
(pulsar's rule, F-dist-14: labels lie).

### 0.2 Model parameters used in the budgets

Running example: DeepSeek V4 Flash, the recommended flagship for this box (research/02).
Shape from the ds4 model table: 43 layers, d_model D = 4096, vocabulary 129,280. Assumed for
sizing: k = 6 routed experts per token per layer, one expert ≈ 16 MB at stored precision
(colibri quotes 18.8 MB for V4.1 Flash), KV ≈ 48 KiB per token (MLA-style latent, research/02),
master token time ≈ 100 ms (8-12 tok/s estimate in research/02, ~65% of experts cached).

Derived (all estimates, to be replaced by Phase 0 measurements):

| Quantity | Value |
|---|---|
| per-layer time on the master, t_L = T_tok / L | ~2.3 ms |
| expert bytes routed per token, L × k × S_e | ~4.1 GB |
| expert bytes missed per token at 65% hits | ~1.45 GB |
| one expert read, Gen4 drive (~6.5 GB/s) / Gen5 (~12 GB/s) | ~2.5 ms / ~1.3 ms |
| NVMe budget per token, 4 drives × 100 ms | ~2.4 GB (Gen4) to ~4.5 GB (Gen5) |
| extra NVMe bytes per additional verified draft token (union growth, ~40% overlap with the previous token, 35% miss) | ~0.7-0.9 GB |
| one bf16 hidden vector | 8 KiB |
| one f32 hidden vector | 16 KiB |

Two consequences drive the rest of this document:

1. **Network latency is small next to disk latency.** One 10GbE TCP round trip (~40 µs) is
   1.6% of one Gen4 expert read. Anything that turns a disk read into a network message wins,
   and anything that only needs to arrive "a layer early" does not care about RDMA.
2. **Speculation is expensive on a streamed MoE.** Each extra verified token can cost
   ~0.7-0.9 GB of reads against a ~1 GB per-token headroom. Drafts help only when they are
   accepted at a high rate or their experts are already resident (P-88, research/02's union
   guard).

## 1. The requested arrangement: prediction on slaves, primary on master

### 1.1 Principles

1. **The master is complete.** It holds the whole compiled model on its own NVMe and can run
   alone at any moment (M = 0 slaves is a supported configuration). Slaves add speed only.
2. **Decision path only.** What crosses the wire is tokens, expert ids, confidences, small
   vectors (a hidden state, KV rows), never weights.
3. **Lossless.** The master recomputes everything that affects output: routing, logits,
   sampling, acceptance.
4. **Never wait.** Advice that arrives late is dropped. The master has no code path that
   blocks on a prediction slave.
5. **Logical clocks.** Every message names the token position, layer and epoch it is about.
   Staleness is decided by comparing these to where the master is, not by synchronised wall
   clocks.
6. **Same safety rules as in-box prefetch** (P-34): remote advice enters the same bounded,
   lowest-priority prefetch queue as the in-box router lookahead, can never evict a pinned,
   in-use or demand-loading expert, and is dropped first when demand reads are queued.

### 1.2 Roles

| Role | Runs | Receives | Sends | Needs on the slave |
|---|---|---|---|---|
| **M** master | target model at compiled precision, sampler, verifier, expert cache, NVMe reads | advice | committed tokens, optional anchors and KV rows | the box described in research/02 |
| **P** predictor (routing oracle) | a *shadow* of the target (same model, low-bit, own RAM) or a *trunk-only* predictor | COMMIT, optional ACT_PUSH and KV_PAGE | ROUTE_PRED, optional DRAFT_RESP | shadow: ~90-100 GB for V4 Flash at Q2 (est.); trunk-only: dense weights + routers, a few GB |
| **D** drafter | MTP head, a small draft LM, n-gram/corpus drafter, or the P shadow's own head | DRAFT_REQ (+ last hidden for MTP) | DRAFT_RESP | small |
| **H** heat predictor | decayed expert-heat model across sessions, cache-plan suggestions | ROUTE_TRACE | HEAT_HINT | tiny |
| **E** expert host (section 2.3) | a subset of experts in RAM, computes them on request | EXPERT_REQ | EXPERT_RESP | RAM for its shard |

One slave can take several roles (P + D + H is natural: the shadow is also a drafter, and heat
is a by-product of its routing). Roles are assigned by the master from CAPS and probe results
and re-evaluated between requests, never mid-request.

### 1.3 What each prediction slave runs

**P, shadow variant (recommended P).** tlmc compiles a second, low-bit profile of the same
checkpoint (Q2-class experts, Q4 dense, same tokenizer and routing code). The slave runs it as
an ordinary decoder, one step behind or ahead of the master:

- On every `COMMIT(pos, token)` it runs the shadow forward for that exact input token and
  streams `ROUTE_PRED` for each layer as soon as that layer's router has run (chunks of
  2-4 layers). Because the shadow is low-bit and resident, it runs ~3x faster than the master
  (est. ~30 ms/token vs ~100 ms), so after the first couple of layers its predictions arrive
  ahead of the master.
- Before the commit arrives it has already run its *own draft* of that token (its greedy
  pick from the previous step). When the master's committed token equals that draft, the
  predictions for the whole next forward are already on the master, a full token early.
- Optional **KV mirroring**: the master streams its exact KV rows (`KV_PAGE`, ~48 KiB per token)
  so the shadow's attention reads exact keys and values; only the shadow's weights are
  approximate. A/B it: it costs the master one memcpy-sized send per token.
- Optional **anchoring**: the master sends its exact hidden state at chosen layers
  (`ACT_PUSH`, 8 KiB, e.g. every 4th layer); the shadow restarts from that state and predicts
  the next few layers of the current token with drift reset. This is the user's "router
  lookahead running layer l+1..l+n ahead", done with exact state instead of the in-box
  one-layer heuristic.
- It also emits `DRAFT_RESP` from its own head (it is the same model at low bits, so acceptance
  should be high) together with `union_bytes_est`: how many non-resident expert bytes the master
  would read if it verified each prefix of the draft (the shadow knows its routing; the master's
  residency map is mirrored through heat/cache summaries). This lets the master choose draft
  length by expected time, which is exactly what failed in P-88.

Prior art for the shadow idea: MoE-SpeQ uses an INT4 copy of the MoE to predict the target's
experts (90.9% hard + soft matches on Qwen-MoE); OD-MoE runs a quantized "shadow" model in
parallel on another node with periodic token/KV alignment (recall 99.94% FP16, 97.34% INT8,
95.67% NF4, over 1 Gb/s Ethernet). Neither is our model, hardware or bit width: Q2 recall on
V4 Flash must be measured (Phase 0).

**P, trunk-only variant (for slaves too small for a shadow).** The slave holds only the dense
trunk: attention, shared experts, norms, routers (a few GB for V4 Flash at Q4, est.). Given an
anchor `h_l` and mirrored KV it runs attention + shared expert for layers l+1..l+n, skips routed
experts, and applies each router. This generalises colibri's in-box "two-step" (shared expert
then next router, +2.3-3% recall on top of 71.6%). Its accuracy beyond one layer is unknown and
may decay fast; it is an experiment, gated in Phase 0 by offline simulation on the master.

**D drafter.** Useful only if it beats the master's own MTP head, which is cheap (one extra
layer of ~43). A remote MTP head saves the master ~2% of a token and adds a round trip plus an
8 KiB hidden per round, so it is not worth a slave on its own. A remote drafter pays when it is
*bigger* (a 7-14B dense draft LM, the P shadow) or *wider* (SpecExec-style trees), and only
under the union-cost negotiation above.

**H heat predictor.** Consumes the master's actual routing (`ROUTE_TRACE`, ~0.5 KB/token),
keeps decayed per-expert heat by session and content type, and every few hundred tokens
suggests pin/unpin sets (`HEAT_HINT`). Low value at first; it mostly moves the existing
heat/LFU logic (P-07 family) off the master's hot path and lets it span sessions and machines.

### 1.4 Messages and bytes per token

V4 Flash sizing, 40-byte header (section 4.2) included.

| Message | Direction | Size | When | Per token (typical) | On master's critical path? |
|---|---|---|---|---|---|
| COMMIT | M→P,D | 68 B for one token (+4 B per extra committed token) | after each verify | 68-84 B | no (fire and forget) |
| ROUTE_PRED | P→M | 48 B + 18 B per layer (k = 6: 6 × u16 id + 6 × u8 confidence) | streamed in 2-4 layer chunks | ~0.9 KB (one forward) to ~4 KB (with draft tokens) | no |
| ACT_PUSH | M→P | 8,240 B (bf16 D = 4096) | every s-th layer (s = 4 default) | ~90 KB (s = 4) to ~350 KB (s = 1) | no (non-blocking, droppable) |
| KV_PAGE (mirror) | M→P | 60 B + 43 × ~1,152 B | once per committed token | ~48 KiB | no |
| DRAFT_REQ | M→D | 48 B token-only; 8,240 B with bf16 last hidden (MTP) | once per verify round, or never (proactive mode) | 48 B-8.2 KB | no |
| DRAFT_RESP | D,P→M | 48 B + 6 B per draft token (token + fp16 q) + 4 B per union-cost entry | per round | ~90 B (chain of 4), ~0.6 KB (64-node tree) | no: used only if it arrives before the verify batch is built |
| ROUTE_TRACE | M→H | 48 B + 12 B per layer | per token, batched | ~0.56 KB | no |
| HEAT_HINT | H→M | 48 B + 4 B per entry | every ~256 tokens | ~0 | no |
| HEARTBEAT | both | 64 B | 10 Hz | ~0 | no |
| CANCEL | M→P,D | 64 B | on rewind/new request | ~0 | no |

Total master→slave traffic with KV mirroring and anchors every 4th layer: ~140 KB/token, i.e.
1.4 MB/s at 10 tok/s or 4.2 MB/s at 30 tok/s, about 0.1-0.4% of one 10GbE port. Slave→master is
a few KB/token.

### 1.5 Timing

One decode step, master at ~10 tok/s (t_L ≈ 2.3 ms), shadow at ~3x the master's speed, 10GbE TCP
(one-way ~20-30 µs). Time runs downwards.

```
 MASTER compute                MASTER io (4 NVMe)       link (10GbE)        SLAVE P (shadow + KV mirror)
 ──────────────                ──────────────────       ────────────        ────────────────────────────
 verify step t-1 done:
 commit x_t (sampled)
  ├─ COMMIT(pos=t, x_t, ep) ───────────────────────────────────────────▶   x_t == my draft? (yes ~70-85%)
  ├─ KV_PAGE(pos=t-1 rows) ────────────────────────────────────────────▶   (yes) predictions for forward t
  │                                                                           were sent during step t-1 ─┐
  │                             ◀──────────── ROUTE_PRED(pos=t, L0..L42, src=shadow-draft) ────────────┘
  │                             (already queued at low priority; in-box
  │                              lookahead still runs for every layer)
 L0  attention, route (exact)   reads for L0 misses (demand)                 (no) run shadow fwd(x_t):
 L0  experts                    reads for L2..L5 (predicted, low prio)        L0..L2 routers done at ~2 ms
  ├─ ACT_PUSH(pos=t, L=0, h0) ─────────────────────────────────────────▶   anchor: restart from h0
 L1  ...                                                                     ◀── ROUTE_PRED(pos=t, L1..L4)
 L2  (predicted experts already resident: no stall)                          continue shadow to L42, then
 ...                                                                          draft d1..dg from shadow head,
 L42 experts                                                                  run shadow fwd(d1) → predict
 head, sample x_{t+1}                                                         forward t+1 (sent early)
 verify batch built from:   ◀─────────── DRAFT_RESP(pos=t+1, d1..dg, q, union_bytes per prefix) ─────
   own MTP drafts or slave drafts, whichever is better by expected time;
   anything that arrived late is discarded, nothing is awaited
  ├─ COMMIT(pos=t+1, accepted span, ep) ───────────────────────────────▶   (next step)
```

Lead time for a predicted expert at layer l of the forward the shadow is computing *now*:

```
lead(l) = l × t_L(master) − ( one_way + (l + 1) × t_L(shadow) )
        ≈ l × 2.3 ms − (0.03 ms + (l + 1) × 0.7 ms)
l = 1: 0.87 ms   (less than one Gen4 read: the in-box lookahead covers layer 1)
l = 2: 2.5 ms    (about one Gen4 read)
l ≥ 3: ≥ 4 ms    (comfortable)
```

When the shadow's draft for x_t was right (the common case if acceptance is high), lead is
about one whole token (~100 ms) for every layer. The network contributes 0.03 ms of that;
RDMA would change nothing here.

### 1.6 How the master validates (lossless)

- **Routing:** at every layer the master runs its own router on its own exact state and uses
  that result. A prediction only decides which reads start early and in what order. A wrong
  prediction costs a wasted read and possibly an eviction (bounded by P-34 rules), never a
  different expert in the computation.
- **Drafts:** verified by the normal acceptance rule (P-86): greedy exact match, or exact
  sampling with a point-mass drafter (accept with p_target(d), otherwise resample with d
  excluded), or the standard min(1, p/q) rule when the drafter supplies q. Draft and verify
  use the same kernels (P-87). Output is token-identical to running with no slave, for the same
  seed.
- **Tests (gate for every phase):**
  - teacher-forced token signatures identical with no slave, with a healthy slave, with a slave
    killed or SIGSTOPped mid-generation (ds4's QA pattern, F-dist-09), and with a fuzzing slave
    that returns random but well-formed ids and drafts;
  - prefix hash on every COMMIT (F-dist-11) so a slave on the wrong timeline is detected and
    resynced, not trusted.

### 1.7 When a slave is slow or absent

| Situation | Master behaviour |
|---|---|
| ROUTE_PRED arrives for (pos, layer) the master has already entered or passed | drop on arrival, count as `late` |
| ROUTE_PRED arrives with lead < one expected read time | still queued (it may help partially), lowest priority |
| DRAFT_RESP not present when the verify batch is assembled | use the master's own MTP draft (or none); nothing waits. Optional `draft_wait_us`, default 0 |
| epoch in a message ≠ current epoch (rewind, rejected drafts, new request) | drop |
| send buffer to a slave above a threshold | drop new ACT_PUSH/KV_PAGE (advisory), keep COMMIT |
| slave lag > 2 tokens (from heartbeats) | stop ACT_PUSH and KV_PAGE to it; keep COMMIT so it can catch up |
| 3 missed heartbeats (300 ms) or socket error | mark absent; stop all sends; prefetch runs on in-box lookahead only |
| slave reconnects | HELLO/CAPS again; master sends COMMIT with prefix hash; slave rebuilds its shadow KV by replaying the transcript (shadow prefill) or receives bulk KV_PAGEs on the bulk channel; advice resumes when its lag is ≤ 1 token |
| slave sends malformed frames | ERROR, close, mark absent (never parse past a failed check; F-dist-02) |

The compute thread never touches a socket. A network thread per slave (pinned to one core, out
of the 24-thread compute pool) moves messages through single-producer/single-consumer rings; the
compute thread only enqueues and polls the advice queue between layers.

### 1.8 Link budget

Per-hop numbers from 0.1; costs per token for V4 Flash (L = 43). "Critical path" means time the
master spends waiting on the network per token.

| Arrangement | Network events per token on the critical path | 10GbE TCP | 10GbE soft RDMA | 25GbE RoCE | 100GbE RoCE | USB4 TCP |
|---|---|---|---|---|---|---|
| **Prediction slave (this section)** | none | 0 | 0 | 0 | 0 | 0 |
| ...its lead-time loss (one-way + serialisation of a 1 KB message) | per prediction | ~0.03 ms | ~0.03 ms | ~0.003 ms | ~0.002 ms | ~0.05 ms |
| ...its bandwidth use (~140 KB/token) at 30 tok/s | | 0.4% | 0.8% | 0.15% | 0.04% | 0.1-0.2% |
| Remote drafter, if the master chose to wait for it | 1 round trip per verify round | 0.03-0.06 ms | 0.02-0.07 ms | ~0.005 ms | ~0.004 ms | 0.05-0.1 ms |
| Expert host, 1 slave, exact mode (2.3) | 43 round trips + ~2.3 MB | 3.5-4.5 ms | 6-8 ms | ~1 ms | ~0.4 ms | 2.5-5 ms |
| Tensor-parallel pair (2.2), 1-2 exchanges/layer of 16 KiB f32 | 43-86 exchanges | 1.5-3.9 ms | 3-6 ms | 0.3-0.8 ms | 0.2-0.4 ms | 1.5-5 ms |
| Two-stage pipeline (2.1) | 2 one-way hops of 8-16 KiB | ~0.06 ms | ~0.1 ms | ~0.01 ms | ~0.01 ms | ~0.1 ms |
| Remote RAM as cache tier (2.7), all 1.45 GB of misses remote | bulk | ~1.2 s | ~2.5 s | ~0.5 s | ~0.13 s | 0.33-0.58 s |

Compare with the token itself: ~100 ms at 10 tok/s, ~40 ms at 25 tok/s. The prediction slave
is the only arrangement whose cost is zero on every link; the expert host is affordable on
10GbE (~4% of a 100 ms token) and becomes cheap on RoCE; the remote RAM tier is hopeless next to
four local NVMe drives (~24-45 GB/s).

### 1.9 What the prior art says will go wrong, and the counter-measures

| Risk | Evidence | Counter-measure in this design |
|---|---|---|
| Better recall, worse speed: extra reads evict experts that were about to be used | P-35: slipstream's coupling predictor +14-20 pp recall in replay, -8% end to end; naive coupled prefetch -5 to -15% | advice uses the in-box speculative queue and its no-evict rules; per-token speculative byte budget (NVMe budget − demand bytes); gate on end-to-end tok/s, never on recall |
| Hint storms starve demand reads | P-34: apus's hints starved demand fills 5:1; pulsar queued ~177 GiB of speculative reads in prefill | bounded queue, demand first, drop on backlog, prediction disabled during prefill unless coverage-gated (P-36) |
| Predictors that look plausible but are inaccurate | P-38 (previous-token reuse 25.6% hit, -18%), P-39 (MLP 31%, early routing 53% and -38%) | Phase 0 measures each predictor offline on the master's own traces before any slave is built |
| External drafts make a streamed MoE slower | P-88: flashmoe's external 35B draft at 33% acceptance was 4.5x slower | union-cost estimate per draft prefix; master picks draft length by expected time; MTP remains the default drafter |
| A slave drifts onto another timeline | ds4 built prefix hashes for exactly this (F-dist-11) | prefix hash on COMMIT, epochs on every message, resync by transcript replay |
| Network thread steals compute | — | one pinned core out of 24; decode is RAM-bandwidth bound, so the loss should be small (measure) |

## 2. Catalogue of arrangements

Notation: M = master (this box), S = slave, E = expert, h = hidden vector (D values), L = layers.
Fit ratings are for this hardware: 24-core/128 GB master with four NVMe, 10GbE now, RoCE later.

### 2.1 Pipeline parallel (layer split)

```
 tokens ─▶ [ M: layers 0..20 ] ──h (8-16 KiB)──▶ [ S: layers 21..42 + head ] ──logits/token──▶ M
```

- **Crosses the wire:** one hidden vector per token per stage boundary (8 KiB bf16 or 16 KiB
  f32), plus the sampled token back. Prefill: chunks of hidden rows.
- **Wins:** capacity. Each box keeps its layers resident, so a model larger than one box's RAM
  stops streaming from NVMe. Few hops per token, so it tolerates slow links (prima.cpp runs it
  over home Wi-Fi). Long prefills overlap across stages (F-dist-13).
- **Loses:** one decode stream gets no parallelism: each box idles while the other computes
  (ds4: "not a guaranteed decode speedup"). Measured elsewhere: llama.cpp RPC over a 10GbE direct
  link took a 72B model from 11.1 to 5.9 tok/s when split across two machines; four Macs running
  llama.cpp over Thunderbolt TCP were slower than one (15.2 vs 20.4 tok/s).
- **Fit:** good fallback for USB4 or other high-latency links when the model does not fit; worse
  than the expert host on a decent link, which gives the same capacity plus overlap. Prefer it
  for models where per-layer round trips are too costly (very deep models, slow links).
- **Prior art:** ds4 pipeline mode (F-dist-10, -11, -13, -18); prima.cpp (piped-ring, Halda
  scheduler); Petals (fault-tolerant chains of servers, client-side replay); PipeInfer and
  FlowSpec (speculation to fill pipeline bubbles).

### 2.2 Tensor parallel

```
            ┌── M: attention part A, experts 0..127 ──┐
 token ─▶   │          exchange f32 partials           │  ×1-2 per layer, lockstep
            └── S: attention part B, experts 128..255 ┘
```

- **Crosses the wire:** a D-wide f32 partial sum per exchange point, 1-2 per layer, both
  directions; optionally half the logits per token (~253 KiB for a 129k vocabulary, F-dist-16).
- **Wins:** decode on two equal boxes, because each reads half the weights per token. ds4: two
  Sparks over RoCE averaged 21.9 tok/s on V4.1 Flash Q2; exo with RDMA over TB5 reached 1.56x on
  4 Macs (Qwen3 235B).
- **Loses:** needs near-identical boxes (every layer waits for the slower side, F-dist-14);
  lockstep means a stall on either side stalls the token, and there is no partial fallback; 2L
  latency-bound exchanges per token, so TCP links eat the gain at higher token rates.
- **Fit:** poor on 10GbE TCP (est. 1.7-3.9 ms/token of exchanges, plus lockstep fragility);
  reasonable on 25/100GbE RoCE with a twin box. Not a first experiment.
- **Prior art:** ds4 network TP (F-dist-05..08, -17); exo (MLX distributed, RDMA over TB5);
  distributed-llama (power-of-two nodes, Q80 activation sync); kt-kernel NUMA split inside one
  box (F-dist-15).

### 2.3 Expert parallel: experts sharded across nodes' RAM ("expert host")

```
 M: attention, routers, shared expert, KV, sampling, full model on NVMe (fallback)
   per MoE layer:  route ─▶ split routed experts by owner
                    ├─ local experts (RAM cache / NVMe) ───────────────┐
                    └─ EXPERT_REQ{x once, (e, rows)} ──▶ S: compute   │──▶ combine in canonical order
                       ◀── EXPERT_RESP{f32 y per expert} ──────────────┘
```

- **Crosses the wire:** per MoE layer, the input rows once (in the exact operand format the
  local kernel consumes, e.g. Q8 blocks, ~4.3 KiB for D = 4096) plus the expert list; back, one
  f32 output row per (expert, row) (16 KiB). V4 Flash decode with 3 of 6 experts remote: ~53 KB
  per layer, ~2.3 MB per token.
- **Exactness:** the master sums expert outputs in a canonical order (by routing rank or expert
  id) whichever node computed them, so the result does not depend on placement or on which
  slave is alive; that is why outputs return per expert in f32 and not as a per-node partial sum
  (a per-node partial is half the bytes but ties bitwise output to the placement). Requires the
  same build and numerics profile on both sides (HELLO's `numerics_hash`) and colibri's parity
  gate (F-dist-04).
- **Deadline and fallback (hedged):** the master issues remote requests first, computes its local
  experts, then waits for remote results only up to `p99 × 2` of that slave's measured service
  time; past it, it computes the missing experts itself (RAM if cached, else NVMe). Output is
  unchanged; only speed suffers. A slave that misses deadlines repeatedly loses its shard.
- **Wins:** whenever the pair's RAM holds what the master alone would stream. V4 Flash
  (~150-165 GB) fits master + one 64 GB-free slave entirely, so per-token NVMe misses (~1.45 GB,
  the dominant cost) mostly disappear for ~3.5-4.5 ms of network on 10GbE. Both boxes compute
  experts at the same time (unlike the pipeline). And it makes speculation cheap: verifying more
  draft tokens then reads RAM on two boxes, not NVMe.
- **Loses:** for models that already fit the master's RAM (network cost with nothing saved); on
  links with RTT ≫ 100 µs (43 round trips per token); when routing is so skewed that one owner
  is always busy (use heat-aware placement, below).
- **Placement:** tlmc computes owner(layer, expert) from heat and co-activation traces (hot
  experts on the master; co-activated experts on the same slave to cut per-layer fan-out, as in
  Speculative MoE's expert pre-grouping); colibri's `(expert + layer) mod N` is the naive
  baseline. Optionally replicate the hottest remote experts on two slaves and hedge (2.10).
- **Fit:** the best raw speed-up available on 10GbE if any slave has ~64 GB free; the per-layer
  round trip is the same shape as research/02's per-layer GPU KV offload (20-40 µs per layer).
- **Prior art:** colibri cluster mode (F-dist-03: one batch-union request per layer per worker,
  persistent TCP; visits workers sequentially and fail-stops, both fixed here); ds4's expert-owning
  TP pair; MegaScale-Infer (attention/FFN disaggregation, M2N, ping-pong micro-batches);
  Speculative MoE (token/expert pre-scheduling, 89% route speculation accuracy).

### 2.4 Disaggregated prefill / decode

```
 long prompt ─▶ S (prefill node: compute-heavy, batch) ──KV_PAGE stream──▶ M (decode node)
```

- **Crosses the wire:** the KV cache of the prompt (~48 KiB per token for V4 Flash: a 32k prompt
  is ~1.5 GB, ~1.3 s on 10GbE, ~0.4 s on USB4, ~0.13 s on 100GbE), streamed per layer while
  prefill proceeds.
- **Wins:** when prefill is much faster on the other node (a GPU box) or when several users
  share the master and a long prompt would freeze everyone's decode (DistServe, Splitwise,
  Mooncake). The transfer is small next to CPU prefill time for long prompts.
- **Loses:** a single user on two equal CPU boxes: prefill is no faster on the slave, and the
  master idles while waiting.
- **Fit:** low now (single user, CPU slaves); medium if a GPU slave or multi-user serving
  appears. The research/02 GPU KV design already gives the master local attention offload.
- **Prior art:** DistServe, Splitwise, Mooncake (KV-centric, RDMA transfer engine), LMCache,
  NIXL; ds4's topology-neutral snapshots (F-dist-18).

### 2.5 KV-cache tier node

```
 M: hot KV pages (RAM / GPU) ◀── page in by position ── S: cold KV pages (RAM), prefix cache shared across sessions
```

- **Crosses the wire:** KV pages (64-token pages, ~3 MB each for V4 Flash) on demand or
  prefetched by position; or, for attention-near-KV, the query and per-head partial softmax
  statistics (lossless merge with log-sum-exp).
- **Wins:** very long contexts and many saved sessions (agentic use: re-open a 100k-token
  session without re-prefill); a shared prefix cache across sessions and boxes.
- **Loses:** attention-near-KV for MLA: the absorbed query is large (DeepSeek V3-class: 128
  heads × 576 values, ~147 KB per layer at bf16), so 43 per-layer round trips of that size cost
  ~5 ms/token on 10GbE; page-in of cold pages stalls decode unless prefetched.
- **Fit:** medium later; the master's disk KV store (research/02) covers most of it locally.
- **Prior art:** LMCache, Mooncake store, ds4 disk KV/snapshots.

### 2.6 Remote RAM as a cache tier between NVMe and local RAM

```
 M miss ─▶ local RAM cache? ─no─▶ remote RAM (RDMA READ of the expert's bytes) ─no─▶ local NVMe
```

- **Crosses the wire:** whole experts (~16 MB each).
- **Wins:** on boxes with one slow SSD and a fast link (classic far memory: Infiniswap,
  Fastswap, AIFM), or with ≥ 2 × 100GbE when the local drives are saturated, as an extra parallel
  source rather than a replacement.
- **Loses:** here. Four local NVMe give ~24-45 GB/s; 10GbE gives 1.17 GB/s (+3-5% as an extra
  source), 100GbE ~11.5 GB/s. One 16 MB expert takes ~14 ms over 10GbE vs ~1.3-2.5 ms from one
  local drive.
- **Fit:** reject for this master. If a slave has the RAM, run the experts there (2.3: move
  compute to the data, 53 KB per layer instead of 16 MB per expert).
- **Prior art:** memory-disaggregation survey (Infiniswap, Fastswap, AIFM, LegoOS).

### 2.7 Cooperative distributed expert cache (heat-sharded across cluster RAM)

```
            global directory: (layer, expert) ─▶ {owner node, replicas, heat}
 M evicts expert e ─▶ "forward" e to the peer with spare RAM (N-chance style) ─▶ later misses on e go to that peer as EXPERT_REQ
```

- **Crosses the wire:** directory updates (tiny), occasional expert transfers when ownership
  moves (16 MB each, background, bulk channel), and EXPERT_REQ/RESP traffic as in 2.3.
- **Wins:** a dynamic version of 2.3 for several slaves and shifting workloads: the cluster's
  aggregate RAM behaves like one cache with heat-based placement; an expert evicted by the
  master is not lost but moves to a peer (Dahlin's cooperative caching, adapted).
- **Loses:** complexity; ownership churn costs bulk bandwidth; static tlmc placement may be
  just as good for a single user.
- **Fit:** phase 5 (multi-slave). Start with static placement (2.3), add evict-to-peer only if
  traces show the hot set drifting.
- **Prior art:** cooperative caching (Dahlin et al., OSDI 1994); SlimCaching and OD-MoE for edge
  MoE (expert placement and on-demand loading across nodes).

### 2.8 Speculative draft farm

```
 M ── COMMIT ─▶ S1 (shadow), S2 (small LM), S3 (n-gram/corpus), ...
 M ◀─ DRAFT_RESP{tokens, q, union_bytes per prefix} ── each
 M: merge into one token tree or pick the proposal with the best (expected accepted tokens / expected verify time)
```

- **Crosses the wire:** tokens out, token trees back (bytes to a few KB per round).
- **Wins:** when the target is offloaded and verify of many tokens costs about the same as one
  (SpecExec: 10-20 accepted tokens per step, 4-6 tok/s for 70B offloaded on consumer GPUs); with
  proactive drafting during verification and network time (SpecEdge, SwiftSpec, DSI, PipeInfer);
  with diverse drafters whose strengths differ by content (code vs prose).
- **Loses:** on a streamed MoE each verified token adds its own experts (~0.7-0.9 GB per extra
  token here); flashmoe measured 4.5x slower with an external drafter at 33% acceptance (P-88).
- **Fit:** low until the expert host makes verify cheap; then medium. Always gate by measured
  acceptance and union cost.
- **Prior art:** DSI (speculation parallelism; proved faster than both speculative and
  non-speculative decoding for any drafter, given enough parallel instances), SpecExec,
  PipeInfer, FlowSpec, SpecEdge, SwiftSpec, Mirror-SD, Lookahead decoding, SP-MoE (drafting to
  prefetch the verify union), MoE-SpeQ.

### 2.9 Hedged / racing reads

```
 M needs e ─▶ ask owner S1 ─(no reply by p95)─▶ also ask replica S2 / start local NVMe read ─▶ take first, CANCEL the other
```

- **Crosses the wire:** a duplicate EXPERT_REQ (KBs) and a CANCEL.
- **Wins:** tail latency: a slave that hiccups (GC, page cache, another tenant) no longer stalls
  a layer. Dean and Barroso: deferring the second request to the 95th-percentile latency caps
  the extra load near 5%; tied requests cut Google's 99.9th percentile by 40%.
- **Loses:** if hedges fire too early they double load on links and drives exactly when the
  system is busy.
- **Fit:** built into 2.3 from the start, with local compute as the always-available second
  source; replica hedging only once there are two slaves.
- **Prior art:** "The Tail at Scale" (CACM 2013).

### 2.10 Lookahead routing oracle node

This is role P of section 1 on a dedicated box. The design choice is which predictor it runs:
the shadow (best expected accuracy, needs ~100 GB for V4 Flash, does not scale to GLM-5.2 at
744B), the trunk-only predictor (small, unknown accuracy beyond one layer) or a hybrid (trunk
plus a low-bit copy of only the hottest experts). Fit: depends on Phase 0's oracle replay.
Prior art: Pre-gated MoE, MoE-Infinity, AdapMoE (in-box predictors); MoE-SpeQ, OD-MoE (shadow
models); colibri's in-box lookahead (P-33).

### 2.11 Decision-engine and compiler offload (tlmc on slaves)

```
 tlmc (M) ─ JOB{calibration shard, eval shard, label batch, cache-simulation} ─▶ S1..Sn ─ JOB_RESULT ─▶ tlmc
```

- **Crosses the wire:** job descriptions and results: routing traces (~0.5 KB/token), labels,
  eval scores, simulated hit rates. Model bytes only if a slave lacks the model.
- **Wins:** always, in wall-clock compile time: profiling routing on ~250k calibration prompts,
  held-out quality gates, decision-engine encoder labelling (0.4B encoder, research/04) and
  cache/placement simulations are all embarrassingly parallel and latency-insensitive.
- **Loses:** nothing at inference time; slaves need the model (or the encoder) locally.
- **Fit:** excellent, and the safest first consumer of the transport (exercises framing,
  identity, failure handling with no effect on inference).
- **Prior art:** standard batch farming; colibri's analysis tools and slipstream's trace replay
  show which jobs exist.

### 2.12 Other arrangements worth knowing

| Arrangement | One line | Fit here |
|---|---|---|
| Data-parallel replicas + request routing | each box runs its own copy; route whole requests (slipstream's scored router, F-dist-20) | multi-user only |
| Attention/FFN disaggregation with micro-batch ping-pong | MegaScale-Infer: attention nodes and expert nodes trade micro-batches so both stay busy; needs ≥ 3-4 micro-batches | multi-user only |
| Early-exit hints to drafters (novel here) | the master pushes its hidden state at layer L−k (`ACT_PUSH`); the slave applies its own copy of the final norm + LM head to it (a logit lens, costing the master nothing) and pre-drafts a branch for each top-κ candidate, so a draft for the true next token is ready when the master samples it (after Mirror-SD, which uses an early exit inside the target) | phase 3 experiment |
| Union-aware draft negotiation (novel here) | drafters return expected expert bytes per draft prefix; the master picks draft length by expected time | phase 3, with any drafter |
| Two-level sampling for a vocabulary-split head (novel here) | each rank sends (max, log-sum-exp, local top-k) instead of half the logits; pick the half by its mass, sample inside it: exact, a few hundred bytes instead of ~253 KiB | only with TP |
| Evict-to-peer | the master forwards evicted experts to a slave with spare RAM instead of dropping them | with 2.7 |
| Speculative pipeline fill | run drafted tokens through idle pipeline stages (PipeInfer, FlowSpec) | only with 2.1; union cost applies |

## 3. Recommendation and phased plan

### 3.1 Ranking for a first experiment

| Rank | Experiment | Needs | Expected effect (estimate) | Risk to output | Effort | Link sensitivity |
|---|---|---|---|---|---|---|
| 1 | **Phase 0 measurement + transport skeleton** (oracle replay; tlmc job offload as first network user) | master only; any second box for the transport | none directly; decides everything below | none | low | none |
| 2a | **Expert host** (2.3) on one slave | a slave with ≥ 64 GB free RAM | V4 Flash fully RAM-resident across two boxes; plausibly 1.3-2x over streaming | none if the parity gate passes; deadline fallback | medium (colibri-sized core, plus fallback and parity) | high (43 RTT/token) but fine on 10GbE up to ~20-25 tok/s |
| 2b | **Prediction slave** (section 1), shadow or trunk variant | shadow: ~100 GB slave; trunk: a few GB | bounded by Phase 0's oracle replay; plausibly 1.1-1.4x if exposed stalls are large | none (advisory) | medium-high (shadow profile in tlmc, sync, prefetch integration) | none |
| 3 | Pipeline split (2.1) | slave RAM | capacity on bad links; no decode overlap | none | medium | low |
| 4 | Draft farm / remote drafter (2.8) | slave compute | low while experts stream; better after 2a | none (verified) | medium | low |
| 5 | Tensor-parallel pair (2.2) | twin box + RoCE NICs | 1.5-1.8x decode on twins (est.) | none if parity passes; lockstep fragility | high | very high |
| 6 | Prefill/decode split, KV tier (2.4, 2.5) | GPU slave or multi-user | long prompts / many users | none | medium | bandwidth |
| 7 | Remote RAM tier (2.6) | 100GbE+ | ≤ +5% on 10GbE | none | medium | extreme |

Choose between 2a and 2b with Phase 0's numbers:

```
if slave_free_ram ≥ (model_bytes − master_cache_budget) and exposed I/O stall ≥ 25% of token time:
    start with the expert host (2a)            # removes the misses instead of hiding them
elif oracle_replay_speedup ≥ 1.15:
    start with the prediction slave (2b)       # hides the misses that remain
else:
    neither pays on this model: use slaves for tlmc jobs; revisit with a bigger model
```

For the user's stated goal (prediction on slaves) the plan below builds 2b in full, but places
2a beside it because the same transport, identity, deadline and parity machinery serves both.

### 3.2 Phases and gates

**Phase 0: single-box truth (no network code).**
- Instrument per-token time: compute, exposed I/O stall per layer, prefetch lead time per read,
  in-box lookahead recall at 1..4 layers ahead (P-33 measures 71.6% at one layer on colibri's
  models; flashmoe measured ~0 cross-layer correlation on another, C-04).
- **Perfect-oracle replay:** record routing for a fixed prompt set, then replay with prefetch
  driven by the true future routing at lead times of 1, 2, 4 layers and 1 token. This is the
  upper bound for any prediction slave.
- **Simulate the slave on the master:** run the Q2 shadow and the trunk-only predictor offline
  on the same traces; record recall versus lead time; run the replay again with their
  predictions (with realistic delay added).
- Measure verify union growth per draft token and MTP acceptance.
- **Gate:** continue to 2b only if oracle replay ≥ 1.15x tok/s and a simulated predictor gets
  ≥ 60% of the oracle's gain. Continue to 2a only if exposed stall ≥ 25% of token time.

**Phase 1: transport and protocol skeleton (TCP).**
- `tln` library: header, HELLO/CAPS/PROBE/HEARTBEAT/CANCEL/ERROR, three channels, network
  thread, SPSC rings, identity checks (F-dist-01), validated decoding (F-dist-02), bounded waits
  (F-dist-09), prefix hashes (F-dist-11).
- First user: tlmc JOB offload (calibration traces, cache simulations).
- **Gate:** loopback and real 10GbE p50/p99 RTT reported for 64 B / 8 KiB / 64 KiB; a fuzzer
  over every decoder runs clean under ASan/UBSan; kill and SIGSTOP of a peer mid-job never hangs
  the master; byte-exact wire transcripts frozen as golden files.

**Phase 2b: prediction slave MVP.**
- COMMIT + ROUTE_PRED from a shadow running on committed tokens only (no anchors, no KV mirror),
  feeding the existing speculative prefetch queue with its safety rules.
- **Gate:** token-identical output in all fault scenarios of 1.6; tok/s ≥ +10% over the in-box
  lookahead alone on the Phase 0 prompt set; demand-read latency p99 not worse; report recall,
  late fraction and wasted bytes. Fails → keep the code, disable by default.

**Phase 2a: expert host MVP** (in parallel if a big-RAM slave exists).
- EXPERT_REQ/RESP with rows sent once, per-expert f32 outputs, canonical combine, deadline +
  local fallback, static tlmc placement.
- **Gate:** colibri-style parity (identical teacher-forced signatures, every weight format);
  ≥ 1.3x tok/s on a model larger than the master's RAM; worst-case token time with the slave
  SIGSTOPped ≤ single-box time + deadline.

**Phase 3: refinements, each gated at ≥ +5% or removed.**
- P: KV mirroring, anchoring with ACT_PUSH, own drafts + union-cost negotiation, early-exit
  pre-drafting, heat hints.
- E: hedging with replicas, heat-aware re-placement, batch-union across sessions.

**Phase 4: RDMA backend (when ConnectX cards are installed).**
- Verbs backend behind the same transport interface: RC queue pairs per channel, inline sends
  for small advice, pre-posted receive rings with credits, RDMA WRITE with immediate for bulk
  (KV pages, expert results), dlopen'd verbs, full-size warm-up, TCP fallback (F-dist-08).
- **Gate:** per-hop RTT ≤ 10 µs; expert host network time ≤ 1 ms/token; then evaluate TP.
  Prediction roles stay on TCP unless measurements show otherwise.

**Phase 5: more slaves.** Multi-slave placement, cooperative cache, draft farm, P/D split, KV
tier; role assignment from probes and CAPS, re-evaluated between requests.

## 4. Protocol sketch

Prefix `tln_` (tierlm net). C11, little-endian on the wire (all supported hosts are), fixed-size
records with explicit lengths, no pointers, no implicit padding.

### 4.1 Rules

- Identity before work: HELLO is exchanged both ways and checked field by field before any other
  message is accepted (F-dist-01).
- Every length is bounded by the negotiated maximum and validated before allocation; every
  count, layer, position and expert id is range-checked; unknown flags are errors (F-dist-02).
- Logical clocks (`pos`, `layer`, `epoch`) on every message; wall-clock deadlines exist only
  inside one process.
- Advisory messages carry `TLN_F_ADVISORY`: either side may drop them at any time.
- Three channels per peer so a bulk transfer can never delay a critical message (TCP head-of-line
  blocking): CRIT (EXPERT_*, DRAFT_*, COMMIT, control), ADVISORY (ROUTE_PRED, ACT_PUSH,
  HEAT_HINT, ROUTE_TRACE), BULK (KV_PAGE, JOB payloads, snapshots).

### 4.2 Header

```c
#include <stdint.h>

#define TLN_MAGIC        0x314E4C54u   /* bytes 'T','L','N','1' on the wire */
#define TLN_VER_MAJOR    1u
#define TLN_VER_MINOR    0u
#define TLN_HARD_MAX     (64u << 20)   /* absolute payload cap; HELLO negotiates lower */
#define TLN_LAYER_NONE   0xFFFFu

enum tln_type {
    /* link scope */
    TLN_HELLO       = 1,  TLN_CAPS        = 2,  TLN_HEARTBEAT = 3,
    TLN_CANCEL      = 4,  TLN_ERROR       = 5,  TLN_BYE       = 6,  TLN_PROBE = 7,
    /* decode advice and drafting */
    TLN_COMMIT      = 16, TLN_DRAFT_REQ   = 17, TLN_DRAFT_RESP = 18,
    TLN_ROUTE_PRED  = 19, TLN_ROUTE_TRACE = 20, TLN_HEAT_HINT  = 21,
    TLN_ACT_PUSH    = 22,
    /* delegated compute and state */
    TLN_EXPERT_REQ  = 32, TLN_EXPERT_RESP = 33,
    TLN_KV_PAGE     = 48,
    /* batch work (tlmc) */
    TLN_JOB         = 64, TLN_JOB_RESULT  = 65,
};

enum tln_flag {
    TLN_F_RESPONSE = 1u << 0,   /* `ref` names the request being answered */
    TLN_F_ADVISORY = 1u << 1,   /* may be dropped by either side */
    TLN_F_OPTIONAL = 1u << 2,   /* a receiver that does not know `type` ignores it */
    TLN_F_MORE     = 1u << 3,   /* fragment; more of the same logical message follows */
    TLN_F_CRC      = 1u << 4,   /* payload ends with crc32c (bulk channel) */
};

typedef struct tln_hdr {        /* 40 bytes */
    uint32_t magic;
    uint8_t  ver_major;
    uint8_t  ver_minor;
    uint8_t  type;              /* enum tln_type */
    uint8_t  flags;             /* enum tln_flag */
    uint32_t len;               /* payload bytes following the header */
    uint32_t session;           /* 0 = link scope */
    uint64_t seq;               /* per (peer, channel, direction), starts at 1 */
    uint64_t ref;               /* seq answered or cancelled; 0 if none */
    uint32_t pos;               /* token position the message is about */
    uint16_t layer;             /* layer it is about, TLN_LAYER_NONE if whole token */
    uint16_t epoch;             /* bumped on rewind, rejected drafts, resync, new request */
} tln_hdr;
_Static_assert(sizeof(tln_hdr) == 40, "tln_hdr is a wire format");
```

### 4.3 Payload records

All records start at offset 40, are followed by the variable parts named in their comments,
and may grow at the end in later minor versions (a receiver reads `min(len, known size)` and
skips the rest). The header and every record below compile without padding under
`gcc -std=c11 -Wpadded` (header 40 B, HELLO 104 B, CAPS 96 B, COMMIT 24 B, the rest 8-24 B).

```c
enum tln_role { TLN_R_PREDICT = 1u<<0, TLN_R_DRAFT = 1u<<1, TLN_R_HEAT = 1u<<2,
                TLN_R_EXPERT  = 1u<<3, TLN_R_KV    = 1u<<4, TLN_R_JOB  = 1u<<5 };

typedef struct {                 /* HELLO */
    uint16_t proto_min, proto_max;   /* (major << 8) | minor */
    uint16_t is_master;              /* exactly one side sets 1 */
    uint16_t name_len;               /* node name bytes follow the record */
    uint8_t  model_hash[32];         /* tlmc manifest hash of the target model */
    uint8_t  numerics_hash[16];      /* kernel family + reduction order; must match for TLN_R_EXPERT */
    uint32_t n_layer, d_model, n_vocab, n_expert, top_k, kv_row_bytes;
    uint32_t max_payload;            /* largest payload this side accepts */
    uint32_t ctx_max;
    uint64_t roles_offered;          /* enum tln_role bits */
    uint64_t nonce;                  /* optional pre-shared-key challenge */
} tln_hello;

typedef struct {                 /* CAPS (may be re-sent when capacity changes) */
    uint64_t roles_active;
    uint8_t  shadow_hash[32];        /* tlmc hash of the shadow profile; zero if none */
    uint32_t max_inflight;           /* concurrent CRIT requests accepted */
    uint32_t shadow_us_per_layer;    /* measured at startup */
    uint32_t expert_us_per_row;      /* measured at startup */
    uint32_t n_ranges;               /* tln_expert_range[n_ranges] follow */
    uint64_t ram_free;
    uint8_t  psk_mac[32];            /* HMAC over the peer's nonce, if a key is configured */
} tln_caps;
typedef struct { uint16_t layer_first, layer_count, expert_first, expert_count; } tln_expert_range;

typedef struct {                 /* COMMIT: tokens now part of the master's history */
    uint32_t pos0, n;                /* int32 tokens[n] follow */
    uint64_t prefix_hash;            /* hash of tokens[0..pos0) */
    uint64_t result_hash;            /* hash of tokens[0..pos0+n) */
} tln_commit;

typedef struct {                 /* DRAFT_REQ (optional when drafters run proactively) */
    uint16_t max_len;                /* chain length or tree node budget */
    uint8_t  shape;                  /* 0 chain, 1 tree */
    uint8_t  hidden_dtype;           /* 0 none (token-only drafter), 1 bf16, 2 f32 (MTP) */
    uint32_t wait_hint_us;           /* how long the master may keep the slot open; usually 0 */
} tln_draft_req;                     /* hidden[d_model] follows if hidden_dtype != 0 */

typedef struct {                 /* DRAFT_RESP */
    uint16_t n;                      /* tokens (chain) or nodes (tree) */
    uint8_t  source;                 /* 0 shadow, 1 MTP, 2 n-gram/corpus, 3 small LM */
    uint8_t  has_q;                  /* fp16 drafter probability per token follows */
    uint32_t union_n;                /* entries in union_bytes[] (one per prefix length) */
} tln_draft_resp;    /* int32 tok[n]; int16 parent[n] if tree; uint16 q[n] if has_q; uint32 union_bytes[union_n] */

typedef struct {                 /* ROUTE_PRED: predicted experts for hdr.pos + tok_offset */
    uint16_t layer_first, n_layers;
    uint8_t  k;                      /* ids per layer */
    uint8_t  source;                 /* 0 shadow on committed token, 1 shadow on own draft, 2 anchored, 3 trunk-only, 4 heat */
    uint8_t  tok_offset;             /* 0 = token at hdr.pos; 1.. = draft tokens after it */
    uint8_t  reserved;
} tln_route_pred;                    /* per layer: uint16 expert[k]; uint8 conf[k] (0..255) */

typedef struct {                 /* ROUTE_TRACE: what the master actually routed */
    uint16_t layer_first, n_layers;
    uint8_t  k, reserved[3];
} tln_route_trace;                   /* per layer: uint16 expert[k] */

typedef struct {                 /* HEAT_HINT */
    uint32_t n;                      /* entries follow: uint16 layer, uint16 expert, int8 action (+pin/-unpin), uint8 strength */
    uint32_t valid_tokens;           /* advice horizon */
} tln_heat_hint;

typedef struct {                 /* ACT_PUSH: exact hidden state for anchoring */
    uint8_t  point;                  /* 0 post-attention residual, 1 post-layer residual */
    uint8_t  dtype;                  /* 1 bf16, 2 f32 */
    uint16_t reserved;
    uint32_t d_model;                /* values follow */
} tln_act_push;

typedef struct {                 /* EXPERT_REQ (CRIT channel) */
    uint16_t n_rows, n_items;
    uint8_t  act_codec;              /* exact operand format of the local kernel (e.g. Q8 blocks) */
    uint8_t  out_codec;              /* f32 in exact mode */
    uint16_t reserved;
    uint32_t deadline_us;            /* master will fall back after this; slave may skip late work */
} tln_expert_req;    /* rows[n_rows] in act_codec; then per item: uint16 expert, uint16 n, uint16 row_idx[n] */

typedef struct {                 /* EXPERT_RESP */
    uint16_t n_items, status;        /* status 0 ok; nonzero = partial, see per-item status */
    uint32_t reserved;
} tln_expert_resp;   /* per item: uint16 expert, uint16 n, uint16 status, uint16 pad, then n rows in out_codec */

typedef struct {                 /* KV_PAGE (BULK channel, or ADVISORY for mirroring) */
    uint16_t layer_first, n_layers;
    uint32_t pos_first, n_pos;
    uint32_t row_bytes;
    uint8_t  codec;                  /* as stored by the master: f16, fp8 + scale, ... */
    uint8_t  reserved[3];
} tln_kv_page;                       /* rows follow; crc32c trailer when TLN_F_CRC */

typedef struct {                 /* HEARTBEAT */
    uint64_t mono_ns;                /* sender's monotonic clock, for RTT only */
    uint32_t last_pos_done;          /* slave: last committed pos fully processed */
    uint32_t lag_tokens;
    uint16_t load_permille, queue_depth;
    uint32_t flags;
} tln_heartbeat;

typedef struct {                 /* CANCEL */
    uint64_t ref_first, ref_last;    /* seq range; 0,0 = everything */
    uint16_t below_epoch;            /* also drop all work tagged with an older epoch */
    uint16_t reason;
    uint32_t reserved;
} tln_cancel;

typedef struct {                 /* PROBE: link measurement at connect and on demand */
    uint32_t round, n_rounds;
    uint32_t echo_bytes;             /* responder pads its reply to this size */
    uint32_t reserved;
} tln_probe;
```

### 4.4 Transport abstraction

```c
#include <sys/uio.h>

typedef enum { TLN_CH_CRIT = 0, TLN_CH_ADVISORY = 1, TLN_CH_BULK = 2, TLN_N_CH = 3 } tln_channel;

typedef struct tln_link   tln_link;      /* one peer, all channels */
typedef struct tln_msg {                 /* received message, payload in a pooled buffer */
    tln_hdr     hdr;
    const void *payload;
    tln_channel ch;
    void      (*release)(struct tln_msg *);
} tln_msg;
typedef struct { void *base; size_t len; uint64_t handle; } tln_region;   /* TCP: plain memory; verbs: MR */
typedef struct { uint64_t addr; uint32_t key; uint32_t len; } tln_remote_region;

enum { TLN_SEND_NOWAIT = 1u << 0 };      /* return -EAGAIN instead of blocking (all advisory sends) */

typedef struct tln_transport_ops {
    const char *name;                                            /* "tcp", "verbs" */
    int  (*open)(tln_link *l, const char *addr, int port, int listen);
    int  (*send)(tln_link *l, tln_channel ch, const tln_hdr *h,
                 const struct iovec *iov, int iovcnt, unsigned flags);
    int  (*poll)(tln_link *l, tln_msg **out, int max, int64_t timeout_ns);   /* progress + completions */
    int  (*reg)(tln_link *l, void *base, size_t len, tln_region *r);
    int  (*put)(tln_link *l, const tln_region *src, size_t off, size_t len,  /* one-sided bulk write; */
                const tln_remote_region *dst, size_t dst_off, uint32_t imm); /* TCP emulates it with a framed copy */
    void (*close)(tln_link *l);
} tln_transport_ops;
```

The application never sees sockets or queue pairs. One network thread per link calls `poll`,
dispatches into per-role SPSC rings, and performs sends queued by the compute thread.

**TCP backend (phase 1).** Three sockets per peer, one per channel. `TCP_NODELAY` everywhere;
`SO_BUSY_POLL` (or a spinning `recv` with `MSG_DONTWAIT`) on CRIT during decode; large
`SO_SNDBUF`/`SO_RCVBUF` and `MSG_ZEROCOPY` (or io_uring send-zc) on BULK; symmetric exchanges
interleave send and receive in one loop (F-dist-07); advisory sends use `MSG_DONTWAIT` and drop
on `EAGAIN`. Bind to the direct-link interfaces (two 10GbE ports allow two slaves without a
switch; for one slave, the second port can carry BULK). io_uring zero-copy receive (Linux 6.15+)
is a later optimisation for BULK only.

**Verbs backend (phase 4).** One RC queue pair per channel. Small advice with inline SEND;
receive rings of ≥ 64 entries per channel with credit-based flow control inside the backend's
own framing (invisible to the application header); EXPERT_RESP and KV_PAGE via RDMA WRITE with
immediate into registered slabs (completion = arrival); libibverbs loaded with `dlopen`, device
chosen by the IP on the direct link, a full-size warm-up exchange, at most three queue-pair
recreations, then TCP (F-dist-08). Soft-iWARP or a self-built Soft-RoCE on the onboard X710 is
for developing this backend before the cards arrive; it is not expected to beat tuned TCP.

### 4.5 Versioning and compatibility

- `ver_major` must match exactly; a mismatch is refused in HELLO with an ERROR naming both
  versions ("update all peers together", ds4's rule).
- `ver_minor`: both sides run at `min(minor)`. New message types in a minor version are sent
  with `TLN_F_OPTIONAL` until both sides advertise them. Records only grow at the end.
- Model compatibility: `model_hash` must match for every role that touches the target model;
  the shadow's own manifest records the `model_hash` it was derived from, so a stale shadow is
  refused. `numerics_hash` must match for TLN_R_EXPERT (bitwise-equal results are part of the
  contract).
- Golden byte-exact transcripts for every message type and version are part of the test suite
  (colibri froze its gateway codec this way); decoders are fuzzed because every input is
  untrusted.

### 4.6 Security

Like ds4 and colibri, this assumes a trusted, directly cabled network: no encryption. Minimum
hygiene still applies: listen only on the direct-link interfaces; an optional pre-shared key
answered with an HMAC in CAPS to stop accidental cross-talk between clusters (not a defence
against a hostile LAN); never accept file paths, commands or code; hard caps on every size and
rate. A public or shared network would need slipstream's model (signed identities, sealed
payloads, replay cache, admission limits) and is out of scope.

## 5. Sources

Reference code read for this design (read only; summarised in fragments/distributed.md):
- ds4 (MIT): ds4_tp.c/.h, ds4_distributed.c/.h, docs/DISTRIBUTED.md, docs/DGX_SPARK.md,
  docs/SPECULATIVE_DECODING.md, QA_BEFORE_RELEASES.md, run-nvidia-tp-server.sh
- colibri (Apache-2.0): c/colibri.c (cluster mode), c/cluster.py, c/tests/test_cluster_*.{py,c},
  README.md, CHANGELOG.md
- pulsar (MIT): crates/kernels/src/lib.rs, crates/engine/src/lib.rs,
  docs/prefill-chunk-pipeline-design.md
- slipstream (MIT): crates/p2p-router, crates/p2p-net, docs/plans/2026-08-02-public-community-mesh-design.md
- ktransformers kt-kernel (Apache-2.0): operators/moe-tp.hpp
- apus, flash-moe, monolite: searched; no cross-machine inference code
- tierlm: fragments/CONSENSUS.md (P-33..39, P-84..88), research/01-04

Hardware and links:
- ASUS Pro WS WRX90E-SAGE SE specifications: https://www.asus.com/us/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/techspec/
- ASUS product page (Intel X710-AT2): https://www.asus.com/no/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/
- Intel Linux RDMA driver for the E810 and X722: https://www.intel.com/content/www/us/en/download/19632/29875/linux-rdma-driver-for-the-e810-and-x722-intel-ethernet-controllers.html
- RHEL 8.8 release notes, deprecated kernel functionality (Soft-RoCE): https://docs.redhat.com/ja/documentation/red_hat_enterprise_linux/8/html/8.8_release_notes/deprecated-functionality-kernel
- RHEL 9, Configuring InfiniBand and RDMA networks (Soft-iWARP Technology Preview): https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html-single/configuring_infiniband_and_rdma_networks
- Level1Techs, Ryzen AI Halo USB4 clustering with RDMA, testing notes (Jul-Aug 2026): https://forum.level1techs.com/t/ryzen-ai-halo-usb4-clustering-with-rdma-testing-notes/253117
- CachyOS forum, ASM4242 Thunderbolt networking on Linux 7.2 (Aug 2026): https://discuss.cachyos.org/t/asm4242-thunderbolt-networking-unusable-on-linux-cachyos-7-2-0-1/34826
- Linux kernel, USB4 and Thunderbolt (networking over the cable): https://docs.kernel.org/admin-guide/thunderbolt.html
- Gigazine on Jeff Geerling's 4x Mac Studio RDMA-over-Thunderbolt tests (Dec 2025): https://gigazine.net/gsc_news/en/20251222-mac-rdma-over-thunderbolt-1-5tb
- Busy polling next generation (netdev 2.1): https://www.netdevconf.org/2.1/papers/BusyPollingNextGen.pdf
- RAMCloud, InfiniBand verbs performance: https://ramcloud.atlassian.net/wiki/spaces/RAM/pages/6848596/Infiniband+Verbs+Performance
- NVIDIA developer forum, RoCEv2 performance test (ConnectX-5): https://forums.developer.nvidia.com/t/performance-test-with-rocev2/241928
- Kalia et al., Datacenter RPCs can be General and Fast (eRPC, NSDI 2019): https://www.usenix.org/conference/nsdi19/presentation/kalia
- Soft-RoCE overview (OFA 2017): https://www.openfabrics.org/images/eventpresos/2017presentations/205_SoftRoCE_LLiss.pdf
- NVIDIA RDMA Aware Networks Programming User Manual: https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17/key+concepts
- rdma-core (libibverbs, rxe, siw): https://github.com/linux-rdma/rdma-core
- perftest (ib_send_lat, ib_write_bw): https://github.com/linux-rdma/perftest
- io_uring zero-copy receive: https://docs.kernel.org/6.15/networking/iou-zcrx.html

Distributed inference systems:
- llama.cpp RPC: https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc
- llama.cpp distributed benchmarks over 10GbE (M2 Ultra + DGX Spark): https://github.com/kjaiswal/llama-cpp-distributed-benchmarks
- exo: https://github.com/exo-explore/exo
- distributed-llama: https://github.com/b4rtaz/distributed-llama
- prima.cpp (ICLR 2026): https://arxiv.org/abs/2504.08791
- Petals: https://arxiv.org/abs/2209.01188 and https://arxiv.org/abs/2312.08361
- MegaScale-Infer: https://arxiv.org/abs/2504.02263
- DistServe: https://arxiv.org/abs/2401.09670
- Splitwise: https://arxiv.org/abs/2311.18677
- Mooncake: https://arxiv.org/abs/2407.00079
- LMCache: https://arxiv.org/abs/2510.09665
- vLLM NIXL connector: https://docs.vllm.ai/en/v0.25.0/features/nixl_connector_usage/

Speculative decoding across devices:
- DSI, Distributed Speculative Inference (ICLR 2025): https://arxiv.org/abs/2405.14105
- PipeInfer: https://arxiv.org/abs/2407.11798
- FlowSpec: https://arxiv.org/abs/2507.02620
- SpecExec: https://arxiv.org/abs/2406.02532
- Lookahead decoding: https://arxiv.org/abs/2402.02057
- SpecEdge: https://arxiv.org/abs/2505.17052
- SLED: https://arxiv.org/abs/2506.09397
- SwiftSpec: https://arxiv.org/abs/2506.11309
- Mirror Speculative Decoding: https://arxiv.org/abs/2510.13161

MoE expert prediction, placement and caching:
- SP-MoE: https://arxiv.org/abs/2510.10302
- MoE-SpeQ: https://arxiv.org/abs/2511.14102
- OD-MoE: https://arxiv.org/abs/2512.03927
- Speculative MoE: https://arxiv.org/abs/2503.04398
- Pre-gated MoE (ISCA 2024): https://arxiv.org/abs/2308.12066
- MoE-Infinity: https://arxiv.org/abs/2401.14361
- AdapMoE: https://arxiv.org/abs/2408.10284
- SlimCaching: https://arxiv.org/abs/2507.06567

Far memory, cooperative caching, tail latency:
- Memory disaggregation: advances and open challenges: https://arxiv.org/abs/2305.03943
- Infiniswap (NSDI 2017): https://rtcl.eecs.umich.edu/rtclweb/assets/publications/2017/infiniswap-nsdi17.pdf
- Dahlin et al., Cooperative Caching (OSDI 1994): https://www.usenix.org/legacy/publications/library/proceedings/osdi/dahlin.html
- The Tail at Scale, summary: https://blog.acolyer.org/2015/01/15/the-tail-at-scale/
