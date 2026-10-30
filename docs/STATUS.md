# Status

Last updated 2026-10-30. This page is the hand-over point: read it before picking up work.

## What exists and is tested

`make test` (unit tests, then `tests/setup.sh`, then `tests/e2e.sh`) and `make asan` both pass from a
clean tree. The code is about 16k lines of C in `src/`, plus 6k lines of tests.

| Area | State | Tested by |
| --- | --- | --- |
| Platform: errors, logging, aligned/huge-page memory, O_DIRECT twin fds, full-read loops, atomic writes, CPU features, L3 topology, cgroup-aware RAM, CRC32C | done | all tests use it; CRC check value `e3069283` |
| Config: `key = value` files, `include`, conf.d, `TIERLM_<KEY>` env, origin-ranked precedence, typed schema with ranges, `content.*` / `hardware.*` prefixes | done | `test_cfg`, `tests/setup.sh` |
| `tierlm-setup`: probe (CPU, RAM, drives, DRAM bandwidth), recommendations with reasons, interactive / `--auto` / `--answers`, Experts-and-content section, `--apply-system` snippets, `--show` | done | `tests/setup.sh` (10 groups) |
| Quant formats Q8_0, Q4_K, Q5_K, Q6_K, Q8_K activations, MXFP4, FP8 E4M3 (128×128 scales), F16/BF16; scalar, AVX2 and AVX-512 VNNI kernels | done; integer paths are bit-identical across tiers | `test_quant` (8040 single-row and 7074 matvec comparisons, 0 mismatches) |
| Packet format: 64-byte header, 4 KiB alignment, CRC32C per packet, manifest with SHA-256 over the packet table, replicas, pin and prune flags | done | `test_packet` |
| Async I/O: persistent pread pool per drive, optional io_uring, O_DIRECT switch, replica reads (shorter queue wins, fall back on error) | done | `test_io` |
| Expert store: fixed slot arena, reserve → loading → ready → released, global LFU with decay, pins, warm sidecar, prefetch that never displaces demand | done | `test_store`, the e2e tiny-vs-large cache check |
| Worker pool: chunk claiming, deterministic split, L3-grouped pinning | done | `test_pool` |
| safetensors reader (single and sharded) and writer (F32/F16/BF16/F8_E4M3 raw tensors); `tlm_st_read_fp8_block` decodes E4M3 weights with their `<name>_scale_inv` 128×128 F32 block-scale grid | done | `test_st` (writer round-trips, block-scale decode incl. partial blocks; refused: missing/non-F32/misshaped scales, 1-D FP8, E4M3 NaN codes) |
| MoE decoder: RMSNorm, GQA with qk-norm, RoPE (NeoX + YaRN), softmax and sigmoid+bias routers, SwiGLU experts + shared expert, dense-FFN layers (`first_k_dense_replace`/`mlp_only_layers`, resident SwiGLU, no router), fixed combine order, KV cache, sampler, router-lookahead prefetch, expert mask | done | `test_model` (invariance across threads, cache sizes, kernels; dense prefix vs naive reference) |
| BPE tokenizer: Qwen3 byte-level BPE (`tlm_bpe` encode/decode), NFC + Unicode tables, Qwen3 chat template, `--tokenizer` run wiring | done | `test_bpe` (round-trip + HF parity on a mixed corpus) |
| Decision engine: Choice / Score / Noul questions, rules backend, SELF callback, confidence policy, JSONL audit log | done (rules) | `test_decide` |
| `tlmc`: synth (incl. `--fp8` block-scale and `--dense-layers` checkpoints), compile passes 1–9 (exact / balanced / aggressive) over BF16/F16/F32 and FP8 E4M3 + 128×128 block-scale checkpoints, resume, REAP pruning with an eval gate, Q5_K demotion, heat-balanced placement with pins and replicas, report, atlas | done for Qwen3-MoE-like checkpoints (mixed dense + routed) | `tests/e2e.sh` (incl. the `synth --fp8` compile + run step; dense-prefix synth/compile/run) |
| `tierlm`: run, bench, config show/check, missing-drive fallback to replicas | done | `tests/e2e.sh` |

Kernel throughput measured in the build container (single thread, AVX-512, -O2) was 20.4 GB/s for
Q6_K, 17.2 GB/s for Q8_0 and 15.0 GB/s for Q4_K, counted as weight bytes per second. These are
**not** WRX90 numbers. Nothing has been measured on the target machine yet.

## Known gaps

The core real-model path (M1) is done. The gaps below are the remaining edges: other model
families (MLA, native MXFP4), calibration on real tokens, and the not-yet-built features (MTP,
GPU KV, batching, cluster).

1. **Calibration corpus still uses byte-level tokens.** The *runtime* tokenizer is now real Qwen3
   BPE with the chat template (P-127, `test_bpe` parity gates pass), so `tierlm run --prompt`
   tokenises like HF. But `tlmc compile --corpus` still maps bytes to tokens
   (`tokset_from_bytes`, `byte % vocab`), so routing-heat and expert-saliency calibration
   statistics are not yet based on real tokens.
2. **Narrow ingest.** `tlmc compile` takes BF16/F16/F32 safetensors, and FP8 E4M3 safetensors with
   f32 128×128 block scales, for Qwen3-MoE / Qwen2-MoE style decoders: no attention biases. RoPE
   is plain NeoX or YaRN (`rope_scaling` type `yarn` is ingested and applied; other scaling types
   are refused). Dense-FFN prefix layers (`first_k_dense_replace`, `mlp_only_layers`) are
   supported: those layers become resident SwiGLU DENSE packets (no router, no experts), mixed
   with routed MoE layers. It refuses the following:
   - FP8 checkpoints with single-byte UE8M0 block scales (DeepSeek-V4 class)
   - MLA attention (DeepSeek V3/V4), GGUF input, and native MXFP4 experts
3. **No MTP speculation.** `spec.mode = mtp` is accepted by the config schema, but the runtime warns
   and decodes without it. The packet type (`TLM_PKT_MTP`) and `geom.mtp_layers` exist.
4. **No GPU KV.** `kv.device = hip:N` is accepted, and the runtime warns that the KV stays in RAM.
   The KV cache is f32, sized for one sequence.
5. **Single sequence.** There is no batching, no server and no prompt or prefix cache. Prefill runs
   token by token (`tlm_model_prefill` loops over `step`).
6. **Decision engine: rules only.** The ENCODER backend (an embedded Laya / OpenJevX-class model)
   is a stub that falls back to rules and records the fallback in the audit log.
7. **Cluster and swarm are designs only** (`docs/design/cluster.md`, `docs/design/swarm.md`).
   There is no `tlm_net` yet.

### Small API issues found by the builders

- `tlm_store_is_ready()` does not fill `ref->payload`. Callers must use `tlm_store_wait()` to get
  the pointer. Either document this in the header or add `tlm_store_poll()`.
- SHA-256 is `static` in `tlm_packet.c`, and `tlmc.c` carries its own copy. Export
  `tlm_sha256_*` from the library.
- `tlm_question` options and criteria should be `const char *const *`.
- The packed `tlm_packet_hdr` triggers `-Waddress-of-packed-member` when `item_off` is passed by
  pointer. Copy it out instead, or unpack the struct in memory.

### Compiler limitations (by design for this slice)

- Calibration below 200k tokens only produces a warning; it is not refused. The synthetic default
  is 2048 tokens.
- The eval gate checks top-1 agreement. Mean KL is reported but not gated.
- Passes 6–7 use the row-major layout with one packet per expert. There are no bundles and no MTP
  packets yet.
- The precision plan only demotes (Q6_K → Q5_K). It never promotes.
- Resume reuses passes 2–3 only.
- Resident (dense) packets have no replicas, so the drive that holds them cannot be missing.
- Content classes in the atlas come from input byte classes, because the encoder backend that
  would label content is not ported. Content settings act through `content.default` only.
- Model open reads one 4 KiB header per unpruned expert, because dtype and offsets are not in the
  manifest. Adding them to the manifest would remove these reads.
- Not built: `tlmc replan`, `tierlm serve`, chat mode.

## Milestones

The order follows `research/01-review-and-plan.md` as amended by `research/03-two-stage-design.md`.
Each milestone ends on a gate.

**M1 — complete (2026-10-30):** BPE tokenizer + chat template (P-127, `test_bpe` parity gates pass), YaRN RoPE,
dense-prefix layers (`first_k_dense_replace`/`mlp_only_layers`) and FP8 E4M3 block-scale ingest
are integrated. The teacher-forced oracle (P-121) now runs a real model: Qwen3-30B-A3B compiled
with `tlmc` to `~/tierlm-m1/q30b.tlmm` (48 layers, 128 experts top-8, Q8_0 experts + dense),
and the strict gate compares tierlm's per-position argmax against a full BF16 HF forward on this
box (`tools/oracle_ref.py --dtype bf16`, ~90 s; the old "GPU box betty" note is obsolete).

**Oracle-gate result (152-token prompt, BF16 reference):** 142/152 positions match. The 10
mismatches are all exact or BF16-precision near-ties in the reference (margin 0.0 or 0.125, the
smallest resolvable BF16 logit gap) — quantization noise, not a bug. The initial pos0 (1-token
context) mismatch was root-caused to the Q6_K expert precision: SiLU multiplies a large gate
(~47) by a small up (~0), amplifying the 1.76 % Q6_K weight error to a 12.5 % MoE-delta error,
which the next attention layer magnifies into a routing inversion. Recompiling the experts at
Q8_0 (the highest supported expert tier; Q8_K/FP8 are rejected for experts) closed pos0 and two
more positions, leaving only tie noise.

All five M1 deliverables are delivered and verified: `make test` (unit + e2e, including the BPE
parity, FP8, dense-prefix, YaRN and oracle steps) and `make asan` pass from a clean tree, and the
strict oracle gate (per-position argmax vs a full BF16 HF forward) holds at 142/152 with only
tie-noise misses.

| # | Milestone | Main work | Gate |
| --- | --- | --- | --- |
| **M1** ✅ | Real model bring-up | BPE tokenizer + chat template (P-127); FP8 block-scale ingest; dense-prefix layers; YaRN RoPE; teacher-forced oracle against HF transformers (P-121) on Qwen3-30B-A3B | per-position argmax matches the reference; tokenizer parity on a mixed corpus |
| M2 | Measure on the WRX90 | `make PROFILE=wrx90`; `tierlm-setup --drive` on all four NVMe; `fio` baseline; `tierlm bench` on Qwen3-235B-A22B at Q6_K with cache sweeps; io_uring vs pread and O_DIRECT A/B (C-02, C-03) | ≥ 85% of `fio` bandwidth on misses; decode tok/s recorded with hit rate |
| M3 | MTP speculation | MTP head packets (Q8_0, P-84); chain drafting with adaptive depth 0–4; batched verify; pause below 70% acceptance (P-85); expert-budget-aware draft length | output identical to non-speculative greedy; speed-up measured |
| M4 | GPU KV on the R9700s | optional `dlopen` ROCm backend: Q/K/V shipped per layer, paged KV in ≤ 10% VRAM, CPU fallback | bit-for-bit vs the CPU KV at f32; context length vs `kv.vram_frac` |
| M5 | Flagship family | MLA + compressed KV, native MXFP4 experts, DSA indexer (DeepSeek V4 Flash class) | teacher-forced match; ≥ 1 tok/s warm decode |
| M6 | Cluster S0–S1 | `tlm_net` framing (P-128, P-129, P-135, P-137); prediction slaves over 10GbE; replica swarm work units | master output unchanged with or without slaves (P-131) |

## Decisions taken without asking

- **Licence:** Apache-2.0 (patent grant, compatible with the Apache/MIT reference projects).
  Change `LICENSE` and `NOTICE` if you prefer MIT.
- **Default compile profile:** `exact` (no pruning). `balanced` and `aggressive` are opt-in and
  gated.
- **Default I/O backend:** the pread pool. Change it to io_uring only after M2 measures a gain.
