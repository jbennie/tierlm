# Status

Last updated 2026-10-02. This page is the hand-over point: read it before picking up work.

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
| safetensors reader (single and sharded) and writer | done | `test_st` |
| MoE decoder: RMSNorm, GQA with qk-norm, RoPE (NeoX), softmax and sigmoid+bias routers, SwiGLU experts + shared expert, fixed combine order, KV cache, sampler, router-lookahead prefetch, expert mask | done | `test_model` (invariance across threads, cache sizes, kernels) |
| Decision engine: Choice / Score / Noul questions, rules backend, SELF callback, confidence policy, JSONL audit log | done (rules) | `test_decide` |
| `tlmc`: synth, compile passes 1–9 (exact / balanced / aggressive), resume, REAP pruning with an eval gate, Q5_K demotion, heat-balanced placement with pins and replicas, report, atlas | done for Qwen3-MoE-like checkpoints | `tests/e2e.sh` |
| `tierlm`: run, bench, config show/check, missing-drive fallback to replicas | done | `tests/e2e.sh` |

Kernel throughput measured in the build container (single thread, AVX-512, -O2) was 20.4 GB/s for
Q6_K, 17.2 GB/s for Q8_0 and 15.0 GB/s for Q4_K, counted as weight bytes per second. These are
**not** WRX90 numbers. Nothing has been measured on the target machine yet.

## Known gaps

These are the gaps that stop tierlm from running a real model today.

1. **No real tokenizer.** Prompts are byte-level (`token = byte mod vocab`), which only means
   something for synthetic models. Calibration corpora go through the same byte mapping. You can
   use `tierlm run --tokens 1,2,3` with ids from an external tokenizer, but `tlmc --corpus` needs
   the real tokenizer before its calibration statistics mean anything. This needs BPE plus the
   chat template, with the parity gates of P-127.
2. **Narrow ingest.** `tlmc compile` takes BF16/F16/F32 safetensors for Qwen3-MoE / Qwen2-MoE style
   decoders: every layer MoE, no attention biases, plain RoPE (`rope_scaling` is ignored with a
   warning). It refuses the following:
   - FP8 checkpoints (block-scale ingest not wired)
   - dense-FFN prefix layers (`first_k_dense_replace`, `mlp_only_layers`)
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

## Next milestones

The order follows `research/01-review-and-plan.md` as amended by `research/03-two-stage-design.md`.
Each milestone ends on a gate.

| # | Milestone | Main work | Gate |
| --- | --- | --- | --- |
| M1 | Real model bring-up | BPE tokenizer + chat template (P-127); FP8 block-scale ingest; dense-prefix layers; YaRN RoPE; teacher-forced oracle against HF transformers (P-121) on Qwen3-30B-A3B | per-position argmax matches the reference; tokenizer parity on a mixed corpus |
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
