# Many slow hosts, one large reasoning cluster

Status: design, not built. Numbers marked *est.* are engineering estimates, not measurements.

## The question

A single box of CPU and RAM is slow. Links between boxes are slow too, with high
latency. If we accept that latency, can a large pool of slow hosts give more total
throughput? And can the pool grow over VPNs into a very large reasoning cluster?

## Short answer

Yes for **throughput**, no for **single-stream speed**.

- One token of one conversation must pass through every layer in order. Each cross-host hop
  adds a round trip, so splitting one model across hosts on a WAN makes each stream slower,
  never faster. Example: 8 pipeline stages over a VPN at 30 ms per hop gives at least 240 ms
  per token, so ≤ 4 tok/s for that stream, however fast the hosts are.
- Total work done per second, across many independent sequences, does grow with the pool. The
  trick is to cut the work at boundaries where very little has to cross the network, and to keep
  many sequences in flight so latency is hidden rather than paid.
- tierlm changes what a "slow host" can do. With NVMe expert streaming, one cheap box with
  64 GB RAM and a fast SSD can hold a whole ~280B MoE and run it slowly (≈1–3 tok/s *est.*).
  So the pool does not have to split the model at all. Each host (or each LAN island) runs the
  **whole** model, and the WAN only carries work units: prompts, branches, scores and
  final answers.

Reasoning quality can be bought with breadth: many samples, many branches, many agents.
That is the workload where a big pool of slow hosts wins.

## Three ways to use a big pool

```
A. Replicas (data parallel)        B. Pipeline over WAN           C. Islands
   one full model per host            layers split over hosts        LAN groups run A/B/EP inside,
                                                                     WAN carries only work units
 master/scheduler                  client -> h1 -> h2 -> ... -> hN   +---------island 1---------+
   |  prompts / branches  ^          (activations every token,       | h  h  h  (10GbE / RDMA)   |
   v                      |           every hop)                     +---------------------------+
 [h1] [h2] [h3] ... [hN]  results                                          ^  work units  v
                                                                     +---------island 2---------+
```

| | A. Replicas | B. Pipeline over WAN | C. Islands |
| --- | --- | --- | --- |
| What crosses the WAN | Prompts, branch prefixes, results (KB per task) | One hidden vector per token per hop (8–14 KB) | Work units only |
| Single-stream speed | Same as one host | Worse than one host: ≥ hops × RTT per token | Same as one island |
| Total throughput | ≈ Σ host rates (linear) | Grows with sequences in flight, capped by the slowest stage | ≈ Σ island rates |
| Host must hold | Whole model (fine with tierlm: RAM + NVMe) | Only its layers | Whole model per island |
| Survives a host leaving | Re-queue its tasks | Pipeline stalls until a replacement takes the layers | Re-queue |
| Best for | Breadth: sampling, voting, search, agent swarms, batch evaluation | Hosts too small to hold the model even with NVMe | Mixed fleets: some LANs, some WAN |

Recommendation: **A for WAN, C when some hosts share a LAN, B only as a last resort.** Petals,
the best-known over-the-internet system, uses B. It reports ~0.8–2.3 steps/s single stream for
70B–176B models, rising from 70 to 254 tokens/s with batching
([Petals](https://arxiv.org/abs/2312.08361)). Prime Intellect found synchronous pipeline
schedules a strong baseline for inference over public networks, and that asynchronous
micro-batching does not raise throughput in memory-bound decoding
([Prime Intellect](https://www.primeintellect.ai/blog/inference)). Both confirm that B buys
throughput, not speed.

## Why batching on each host matters more than more hosts

On a MoE, the expensive part of a token is reading its routed experts from RAM or NVMe. If one
host decodes B sequences together, it reads the **union** of their experts once. That costs much
less than B separate reads (the batch-union pattern P-32/P-36 in `fragments/CONSENSUS.md`).
So each host should run several branches of the same problem at once, not one.

*est.* for DeepSeek V4 Flash on a 64 GB host with a Gen4 NVMe:

| Branches decoded together on one host | Expert bytes per step vs one branch | Aggregate tok/s (*est.*) |
| --- | --- | --- |
| 1 | 1.0× | 1–3 |
| 4 | ≈2–3× | 3–7 |
| 8 | ≈3–5× | 5–10 |

A pool of 20 such hosts ≈ 100–200 tok/s of a 284B model in aggregate (*est.*), with each
individual stream still at host speed.

## Turning throughput into better reasoning

The master becomes a **scheduler and judge**, not the main compute.

1. **Parallel sampling with a judge.** N hosts each produce candidate answers for the same
   prompt. The master scores them with a verifier: tests for code, an exact checker for maths,
   or the decision engine's Score/Noul primitives (`docs/design` decision engine) for open
   questions. It returns the best, or a majority vote.
2. **Branch-and-prune search (tree of thought / MCTS).** The master holds the tree. Hosts expand
   nodes: a node is "prefix + continue for ≤ T tokens". The master scores children and sends the
   promising ones out again. Only token prefixes and scores cross the WAN.
3. **Map-reduce reasoning.** Split a big task (a codebase, a document set) into chunks, reason
   over each chunk on a different host, then reduce on the master or a strong island.
4. **Agent swarms.** Long-running agents, one or more per host, each with its own KV cache,
   coordinated through a shared task board on the master.
5. **Speculative breadth for the master's own stream.** The design in `cluster.md` §1, done
   over a LAN. Over a WAN only coarse work survives the latency, such as "draft the next
   paragraph", not "draft the next 4 tokens".

## The work unit

Everything that crosses the WAN is a small, idempotent, self-describing task:

```
TASK  { task_id, model_manifest_sha256, prompt_or_prefix_tokens (or a content hash
        the host already has), sampling {temp, top_p, seed}, max_tokens, stop,
        deadline_ms, return {tokens | text | logprobs | score} }
RESULT{ task_id, host_id, tokens, finish_reason, timings, expert_hit_rate,
        signature }
```

- **Same model everywhere.** A host accepts a task only if it holds the exact compiled manifest
  (same SHA-256), so results are comparable and replayable.
- **Prefixes by hash.** A long shared prompt is sent once per host and then referenced by hash.
  Hosts keep it in their own prefix/KV cache. Shipping KV over a WAN loses: an 8k-token MLA prefix
  is ~400 MB, while re-prefilling it locally costs seconds.
- **Seeds make work checkable.** With greedy or seeded sampling, any result can be re-run on
  another host and compared token for token. This defends against faulty or dishonest hosts.

## Scheduling over slow, unreliable links

- **Pull, not push.** Hosts ask for work when they have capacity (work stealing), so a slow
  host simply takes less.
- **Deadlines and hedging.** If a task is not done by its deadline, re-issue it to another host
  and take whichever answer arrives first. The cost is wasted compute, never a wrong answer.
- **Batch per host.** The scheduler hands each host several related branches at once, so it gets
  the batch-union saving above.
- **Locality.** Tasks that share a prefix go to hosts that already hold it, much like
  cache-aware routing in a CDN.
- **Back-pressure.** The master's judge (verifier, decision engine) is the bottleneck at large N.
  Batch its scoring too, and cap tasks in flight.

## Over VPNs

- **Transport:** WireGuard (or Tailscale/Headscale) mesh, with the work protocol over TCP or QUIC
  inside it. Latency of 20–100 ms is fine for work units and fatal for per-token or per-layer
  traffic.
- **Trust:** a host sees prompts in plain text. Treat every host as able to read the work it
  receives. Keep private work on owned hosts, and mark tasks `private` so the scheduler only
  sends them to trusted islands.
- **Integrity:** sign results, spot-check by deterministic replay (e.g. 2% of tasks re-run
  elsewhere), and drop hosts that disagree.
- **Heterogeneity:** hosts advertise capabilities (RAM, measured tok/s for the manifest, NVMe
  speed, island id). The scheduler weights them by measured throughput, not by labels.
- **Churn:** nothing is lost when a host disappears, because tasks are idempotent and re-queued.

## What will not work, and why

- **Expert-parallel or tensor-parallel across a WAN.** These need a round trip per layer
  (43–80 per token), so 30 ms × 60 layers ≈ 1.8 s per token. Keep them inside a LAN island.
- **Remote RAM over WAN as an expert cache.** A WAN is 100× slower than a local NVMe.
- **Expecting one long chain of thought to get faster.** Breadth parallelises; depth does not.
  Methods that convert breadth into depth exist (parallel thinking then merging, speculative
  paragraphs), but expect modest gains.

## How this fits the build plan

| Phase | Adds | Gate |
| --- | --- | --- |
| S0 | Local multi-branch decode on one host (B branches, batch-union expert reads) | Aggregate tok/s at B=8 ≥ 3× B=1 on DeepSeek V4 Flash (*target*) |
| S1 | Work-unit protocol (`TASK`/`RESULT`) on the `cluster.md` transport, scheduler with pull + deadlines + hedging, on a LAN | 4 hosts give ≥ 3.5× one host's aggregate |
| S2 | Judge on the master: verifier hooks + decision-engine scoring; best-of-N and majority vote | Quality gain on a held-out reasoning set at fixed wall time vs one host |
| S3 | WireGuard mesh, capability adverts, signed results, replay spot-checks, private/public task classes | Throughput scales with hosts at 50 ms RTT; no wrong result accepted in fault-injection tests |
| S4 | Tree search and map-reduce drivers | Measured win on a long-horizon task suite |

## Sources

- [Petals: Distributed Inference and Fine-tuning of LLMs Over The Internet (arXiv 2312.08361)](https://arxiv.org/abs/2312.08361)
- [Prime Intellect: decentralized inference](https://www.primeintellect.ai/blog/inference)
- [ParaThinker: native parallel thinking (arXiv 2509.04475)](https://arxiv.org/pdf/2509.04475)
- [Learning to reason across parallel samples (arXiv 2506.09014)](https://arxiv.org/pdf/2506.09014)
- `docs/design/cluster.md` (LAN arrangements, transport and message set) and `fragments/distributed.md`
