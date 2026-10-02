# CPU/RAM/NVMe MoE Inference Engines — Top 5 Review and C Implementation Plan

Oct 2, 2026 · @Jay

## Executive summary

The excitement is real and it is about one trick: big Mixture-of-Experts (MoE) models only touch a few percent of their weights per token, so the routed experts can live on NVMe and be paged into RAM on demand. Between March and July 2026 at least eight engines appeared that do this; five matter.

**Top 5, ranked for our purpose (a CPU-first engine in C):**

1. **colibri** (JustVugg) — pure C, zero deps, CPU-first, VRAM/RAM/NVMe treated as one hierarchy, nine model families. The closest thing to what we want to build. \~38.7k stars, Apache-2.0.
2. **DwarfStar / ds4** (antirez) — C, deliberately narrow, GPU-first (Metal/CUDA/ROCm) with SSD streaming as the larger-than-RAM mode. Best engineering discipline and integration testing. \~22.8k stars, MIT.
3. **KTransformers** (kvcache-ai / Tsinghua) — the mature RAM tier: experts on CPU with AMX/AVX-512/AVX2 kernels, attention on GPU. No NVMe streaming, but the best CPU expert kernels. \~18.6k stars, Apache-2.0.
4. **flash-moe** (danveloper) — the project that started the X wave: Qwen3.5-397B on a 48 GB MacBook at \~4.4 tok/s. Mac-only, single model, now abandoned, but its 90-experiment log is the best record of what does and does not work.
5. **Pulsar** (giannisanni) — Rust + CUDA, but the cleanest storage path of the group: io\_uring + O\_DIRECT at queue depth 32, a four-tier cache and touch-count admission. Small project, big ideas.

**Recommendation:** build a single-file-per-model C11 engine in the colibri/ds4 mould, CPU-only first, with an explicit expert pager (bounded RAM cache, io\_uring/O\_DIRECT reads, one-layer-ahead prefetch) and a scalar CPU oracle that every optimisation must match. Target one model first (Qwen3.6-35B-A3B to bring up, then DeepSeek V4 Flash or GLM-5.3-Flash). Realistic outcome on a 64–128 GB desktop with a Gen4/Gen5 NVMe: 1–3 tok/s on 300B-class models, 10+ tok/s on 35B-class models. The plan below is six milestones, about 18 weeks for one engineer or 10–12 weeks for two.

Hardware-specific defaults for the Threadripper PRO 9965WX workstation: Your WRX90 build

## Why this works, and the number that limits it

Decode speed on these engines is set by how many expert bytes miss the RAM cache per token, divided by disk bandwidth. Total parameter count barely matters: MONOLITE notes that going from gpt-oss-20B to 120B grows the model 6× but per-token expert traffic only about 1.5×.

```latex
\text{bytes/token} = L_{moe} \times k \times S_{expert} \times (1 - h)
\qquad
\text{tok/s}_{max} \approx \frac{BW_{ssd}}{\text{bytes/token}}
```

where L = MoE layers, k = experts routed per token, S = bytes per expert at the stored precision, h = cache hit rate. Dense parts (attention, shared experts, router, embeddings) must stay resident; they cost RAM, not disk.

| Engine, model, machine | Expert traffic per token | Reported decode |
| --- | --- | --- |
| flash-moe, Qwen3.5-397B 4-bit, M3 Max 48 GB | 60 × 4 × 6.75 MB, \~71% page-cache hits | 4.4 tok/s |
| colibri, GLM-5.2 744B int4, 128 GB CPU-only desktop | \~12.7 GB routed per token before cache | \~1.8 tok/s warm |
| apus, GLM-5.3-Flash FP8, M1 Pro 32 GB, 2 GiB cache | \~7.9 GiB per token cold | \~0.085 tok/s |
| monolite, gpt-oss-120B, 30 GB RAM + 2×12 GB GPUs | small; pager hides nearly all disk latency | 9.74 tok/s (llama.cpp mmap: 5.18) |

Three consequences shape the design: RAM for the expert cache is the biggest lever, disk bandwidth is the second, and anything that silently changes the model (fewer experts per token, harsher quantisation) is a quality trade, not an optimisation.

## The top 5 projects

### 1. colibri — pure C, CPU-first, one memory hierarchy

[JustVugg/colibri](https://github.com/JustVugg/colibri) · C (one `.c` per model family over shared headers) · Apache-2.0 · \~38.7k stars, 2,800+ commits, v1.12.

Runs 744B to 2.8T-parameter MoEs (GLM-5.2/5.3, Kimi K3, DeepSeek V4/V4.1 Flash, Qwen3.6/3.8, Inkling, OLMoE) with no GPU required. Dense weights stay resident in RAM at int4; routed experts (each expert's gate/up/down stored adjacent, one `pread`) stream from NVMe through a per-layer LRU, a learned pinned hot-set built from usage history (`.coli_usage`), and an optional VRAM tier. A router-lookahead thread prefetches the next layer's experts; the project measures routing as about 71.6% predictable one layer ahead. Batched positions read each unique expert once (batch-union). Optional CUDA, Metal and Vulkan backends.

Pros:

- Architecture is exactly our target: pure C, no BLAS, no Python at runtime, CPU path is first-class.
- Strong semantics rule: placement may change speed, never precision or routing.
- Practical I/O work: optional `O_DIRECT`, async I/O pool overlapping reads with compute, weighted multi-SSD mirroring (+37.5% decode measured with two drives).
- Compressed MLA KV state (576 floats/token vs 32,768) persisted to disk, so conversations reopen with zero re-prefill.
- Token-exact validation against a `transformers` oracle in CI.

Cons:

- Breadth is now its weakness: nine engines, many env-var knobs, and the README admits mechanisms often land in one engine and not its siblings.
- Python launcher and gateway around the C engine; the one-time converter is Python.
- Speed claims are honest but modest on CPU-only (\~1.8 tok/s warm on GLM-5.2 with 128 GB); MTP speculation measured a 32% loss at \~85% expert hit rate.

### 2. DwarfStar (ds4) — narrow, disciplined, GPU-first

[antirez/ds4](https://github.com/antirez/ds4) · C with Metal/CUDA/ROCm backends · MIT · \~22.8k stars, \~800 commits.

A self-contained engine for a short list of models (DeepSeek V4/V4.1 Flash and PRO, GLM 5.2/5.3/5.3-Flash, Qwen3.8 Flash Next). Not a general GGUF runner: it uses GGUFs the project produces. Resident inference is the default; `--ssd-streaming` keeps a bounded routed-expert cache and reads misses from the GGUF, with expert preloading, a per-model hotlist and optional full resident prefix layers. Measured on a 128 GB M5 Max: GLM 5.3 Flash Q4\_K (178 GiB) at \~12–15 tok/s generation with streaming; DeepSeek V4 Flash Q2 resident at \~26 tok/s.

Pros:

- Best example of "one engine, few models, tested in integration": model loading, prompt rendering, tool calls, KV state, server and a native coding agent are built and tested together.
- Disk KV cache for sessions (`~/.ds4/kvcache`) and prompt-prefix reuse, which matters a lot when prefill is slow.
- Clear documentation of what the cache budget does and does not cover.
- Keeps GGUF quant layouts and CPU dot logic from llama.cpp under MIT, so we can study the same tables.

Cons:

- CPU is not a first-class backend; streaming targets Metal first, then CUDA/ROCm.
- Model support is deliberately opportunistic; models can be dropped.
- Beta quality and fast-moving; heavy AI-assisted development by the author's own disclosure.

### 3. KTransformers — the mature CPU expert kernels

[kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) · Python + C++ kernels (kt-kernel), integrated into SGLang · Apache-2.0 · \~18.6k stars, v0.7 (Aug 2026).

CPU/GPU hybrid: attention, KV and dense layers on the GPU, routed experts in system RAM computed on the CPU. Its value is the kernels: Intel AMX INT4/INT8 with tiling-aware repacked weight layouts, AVX-512 (now including AMD Zen 4/5 without AMX in v0.7) and AVX2-only paths (since v0.5.3). Day-0 support for most open MoEs (DeepSeek V3/V4, Kimi K2.x, GLM-5/5.2, Qwen3-MoE, MiniMax).

Pros:

- Fastest CPU-side expert compute of the group; prefill up to \~20× over baselines on AMX hardware in the SGLang integration.
- Kernel selection by work size (AMX for prefill batches, vector kernels for decode) is a pattern we should copy.
- NUMA-aware placement for multi-socket servers.

Cons:

- No NVMe tier: the whole expert set must fit in RAM (500 GB+ DDR5 for DeepSeek-V3-class models).
- Python/C++/SGLang stack; nowhere near "compile like llama.cpp".
- Best numbers need Sapphire Rapids-class Xeons.

### 4. flash-moe — the proof of concept that started the wave

[danveloper/flash-moe](https://github.com/danveloper/flash-moe) · C / Objective-C / Metal (\~7,000-line `infer.m`) · no licence file · \~4k stars, no commits since March 2026.

Runs Qwen3.5-397B-A17B (209 GB at 4-bit) on a 48 GB M3 Max at 4.36 tok/s, keeping \~5.5 GB resident. Only the active experts are loaded via parallel `pread()` on GCD dispatch groups; the OS page cache provides \~71% hits. Inspired by Apple's "LLM in a Flash" paper.

Pros:

- Simplest possible design that works: dense part via mmap, experts via `pread`, trust the OS cache.
- Publishes negative results: F\_RDADVISE prefetch net 0%, temporal expert prediction −18%, expert file clustering 0%, MTP speculation break-even.

Cons:

- Hard-coded to one model and Apple Silicon.
- Cuts active experts from 10 to 4 per token to fit the SSD budget, a real quality loss that critics called misleading when stacked on 2-bit/4-bit quantisation.
- No licence, abandoned.

### 5. Pulsar — the best storage path

[giannisanni/pulsar](https://github.com/giannisanni/pulsar) · Rust + CUDA (successor to the C engine NeutronStar) · MIT · small (11 stars) but well documented.

Runs Hy3 295B at 7.2 tok/s and GLM-5.2 743B at 2.0 tok/s on two 16 GB consumer GPUs, 30 GB RAM and one Gen5 NVMe, versus 0.64–0.70 and 0.40 for the ds4-lineage C engine on the same box. Each expert slab resolves through four tiers: resident on spare GPUs, a VRAM hot pool with touch-count admission, a host LFU cache persisted to a `.warm` sidecar, then io\_uring + O\_DIRECT at queue depth 32 with each completion uploaded while other reads are in flight. A background thread prefetches the next layer by running its router on the current layer's input.

Pros:

- The I/O design to copy: io\_uring, O\_DIRECT, deep queue, completion-driven pipelining.
- Touch-count admission keeps a stable hot set instead of thrashing.
- Warm-start sidecar: the second run bulk-loads the hot set in \~3 s.
- Measures hardware (per-card bandwidth) instead of trusting labels.
- Kernels receive explicit per-slot pointers and never consult global cache state, which keeps paging and compute decoupled.

Cons:

- Not C, Linux + NVIDIA only; the CPU is only a staging area.
- Very young; correctness is certified against ds4 rather than an independent reference.
- No DSA indexer yet, so GLM context is limited to 2048.

## Side by side

colibri is the only one of the five that is both C and CPU-first with an NVMe tier; the others each excel at one layer of the stack.

| Project | Language | Licence | Stars | NVMe tier | CPU compute | GPU | Models | Best reported decode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [colibri](https://github.com/JustVugg/colibri) | C | Apache-2.0 | \~38.7k | Yes, LRU + learned pins + lookahead | First-class | Optional CUDA/Metal/Vulkan | 9 families | \~1.8 tok/s GLM-5.2 744B, 128 GB CPU-only |
| [DwarfStar](https://github.com/antirez/ds4) | C | MIT | \~22.8k | Yes, bounded cache + hotlist | Secondary | Metal/CUDA/ROCm primary | \~6 hand-picked | \~12–15 tok/s GLM 5.3 Flash streamed, M5 Max 128 GB |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | Python + C++ | Apache-2.0 | \~18.6k | No (experts in RAM) | Best (AMX/AVX-512/AVX2) | Required for attention | Most open MoEs | Prefill up to \~20× baselines on AMX |
| [flash-moe](https://github.com/danveloper/flash-moe) | C/Obj-C/Metal | None | \~4k | Yes, pread + OS page cache | No | Metal only | 1 | 4.36 tok/s Qwen3.5-397B, M3 Max 48 GB |
| [Pulsar](https://github.com/giannisanni/pulsar) | Rust + CUDA | MIT | 11 | Yes, io\_uring + O\_DIRECT, 4 tiers | Staging only | CUDA required | 2 | 7.2 tok/s Hy3 295B, 2×16 GB GPUs |

Worth reading but not in the top 5:

- [MONOLITE](https://github.com/Pioshin/monolite) — C + CUDA, gpt-oss only. Explicit NVMe/RAM/VRAM pager, custom `.mnl`/`.experts` format with 4K-aligned reads, and a scalar CPU oracle that caught silent CUDA races and a RoPE/YaRN bug. Beats llama.cpp mmap by 88% when the model does not fit.
- [apus-glm5.3-flash](https://github.com/bricesommers/apus-glm5.3-flash) — C11, NEON/AVX2/scalar, macOS/Linux/Windows, libc + pthreads only. Bitwise-exact against the HF reference on the full 306 GiB model. Source-available non-commercial licence, so read for ideas only.
- llama.cpp itself — `--override-tensor` plus mmap is the baseline every project above measures against.

## Wider landscape

Since the first pass, the idea has moved into mainstream stacks: SGLang shipped an SSD Expert Pack path, several llama.cpp forks stream experts, and a 2026 research line shows speculative decoding on MoE needs expert-aware verification to pay off. None changes the top 5, but three ideas are now in the plan: direct I/O into the final buffer, Q5\_K/Q6\_K-friendly expert layout, and union-aware speculation.

| Project | Type | What it adds | Notable number |
| --- | --- | --- | --- |
| [SGLang SSD Expert Pack](https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/) | Python/CUDA serving stack, Aug 2026 | Offline repack so each (layer, expert) is one aligned block; O\_DIRECT straight into pinned buffers, removing the page-cache copy; byte-budgeted LFU/LRU VRAM cache; fails closed if alignment is wrong | DeepSeek V4 Flash decode \~1.9 tok/s on 32 GB RAM + RTX 5090, 6.6–6.9× an Ollama baseline |
| [Slipstream](https://github.com/Schero94/slipstream) | llama.cpp fork (MIT), macOS app, engine builds on Linux | Sidecar expert file, per-layer CLOCK-LRU-K cache, RAM admission gate, experts executed straight from cache slots (+13–24%), MTP on Qwen and DFlash on Laguna; only Q4\_K/Q5\_K/Q6\_K experts | Qwen3.6-35B-A3B Q4: 5.5 → 19 tok/s as cache grows 2 → 14 GiB |
| [llama.cpp discussion #27149](https://github.com/ggml-org/llama.cpp/discussions/27149) and the [moe-stream fork](https://dev.to/anyesh/running-a-26b-moe-on-an-8-gb-jetson-by-streaming-experts-from-ssd) | Upstream proposal and a graph-level patch | Expert streaming hooked into llama.cpp's graph so it works for every GGUF MoE; output bit-identical to the normal path | 26B MoE on an 8 GB Jetson |
| [gmlx](https://pypi.org/project/gmlx/) | Apple Silicon GGUF runner | `--stream-experts`, popularity-managed GPU arena, MTP verify path tuned for long context | MTP lift 1.4–1.8× held from 17k to 110k tokens |
| [AcceptMoE](https://arxiv.org/abs/2608.02989), MoE-Spec, EVICT | Research, 2026 | Verification cost on MoE = union of experts over draft positions; shrink that union with commitment-weighted, cache-aware expert sets | 2.06× over standard speculation under offloading, 73.6–77.1% less expert traffic, −0.27 points accuracy |
| [DFlash](https://arxiv.org/html/2602.06036v2) | Block-diffusion drafter | Drafts a whole block in one pass; beats EAGLE-3 on GPUs | \~4.9× over autoregressive, 2.4× over EAGLE-3 at equal budget |

KTransformers' 2026 releases (v0.5.3 in April, v0.7 in August) also widened its CPU kernels to AVX-512 without AMX (AMD Zen 4/5) and AVX2-only CPUs, which makes it the natural benchmark baseline for the RAM-resident case on AMD hardware.

## Lessons to carry into our design

Every successful project made the same three choices: specialise per model, page experts explicitly rather than trusting mmap, and gate every optimisation on a slow exact reference.

**Copy:**

1. **Narrow by model, shared by mechanism.** One `.c` per model family; tokenizer, file reader, expert store and kernels in shared headers (colibri, ds4). A fix in a shared header reaches every model.
2. **Explicit expert pager, not mmap.** monolite +88% and Pulsar 10× over generic paths came from owning the cache. Keep dense weights mmap'd or loaded; page experts yourself.
3. **Store each expert contiguously, 4 KiB-aligned.** gate, up and down adjacent so one read fetches an expert (colibri, monolite).
4. **io\_uring + O\_DIRECT, deep queue.** Queue depth \~32, overlap reads with compute (Pulsar). Make O\_DIRECT a measured option: colibri saw +34% on some drives and nothing on QLC/DRAM-less ones.
5. **One-layer-ahead prefetch from the router.** Run the next layer's router on the current hidden state; \~70% predictable (colibri, Pulsar, apus).
6. **Admission control and warm starts.** Touch-count admission or LFU beats plain LRU; persist the hot set to a sidecar so run two starts warm (Pulsar, colibri `.coli_usage`).
7. **Batch-union in prefill.** Read each unique expert once per chunk.
8. **Disk KV/prefix cache.** Prefill on streamed models is slow, so never redo it (ds4, colibri).
9. **Scalar CPU oracle that never leaves the tree.** Token-exact against HF `transformers` (monolite, apus).
10. **`doctor` / `plan` commands** that measure RAM, disk bandwidth and SIMD, and size the cache before running (monolite, colibri).

**Avoid:**

- Reducing top-k to hit a speed number (flash-moe 10→4).
- Speculative decoding by default: at partial residency it often loses (colibri −32% at 85% hits; DeepSeek drafts cost more than they saved). Ship it off, measure, enable per workload.
- Speculative OS hints: F\_RDADVISE and temporal prediction were net zero or negative in flash-moe.
- Breadth before depth: colibri's nine engines drift apart.

## Our engine: goals and non-goals

Working name **tierlm**: a C11 engine that runs MoE models larger than RAM on a CPU-only machine, builds with `make` and a C compiler, and never changes model semantics to go faster.

**Goals**

- Build: `make` with gcc or clang, C11 + pthreads + libc only; optional `liburing` on Linux (falls back to a pread thread pool). No BLAS, no Python at runtime, no C++.
- Platforms: Linux x86-64 (AVX2, AVX-512 VNNI optional), macOS arm64 (NEON), Windows x86-64 via MinGW in a later milestone.
- Memory: run any supported model with dense weights + KV + a minimum expert cache in RAM; the rest on NVMe.
- Exactness: greedy output token-identical to the scalar oracle at every cache size, thread count and prefetch setting.
- Interfaces: CLI (`run`, `chat`, `doctor`, `plan`, `bench`) and an OpenAI-compatible `/v1/chat/completions` server written in C.

**Non-goals for v1**

- GPU backends (leave a backend interface; add CUDA/Metal later).
- Multi-user batching and concurrent requests: one request, one KV cache.
- Training, LoRA, vision encoders, dense (non-MoE) models beyond what bring-up needs.
- Reading arbitrary GGUFs: we convert into our own container.

**Reference hardware**

| Tier | CPU | RAM | Disk |
| --- | --- | --- | --- |
| Minimum | 8 cores, AVX2 | 32 GB | PCIe Gen4 NVMe, ≥ 5 GB/s |
| Target | 16 cores, AVX-512 | 64–128 GB DDR5 | Gen4/Gen5 NVMe, 7–12 GB/s |
| Mac | M-series Pro/Max | 48–128 GB | internal SSD |

**Model sequence** (an assumption to confirm, see comment)

1. OLMoE-1B-7B — smallest standard MoE, for bring-up and the oracle.
2. Qwen3-30B-A3B — standard GQA MoE (128 experts, top-8); fits in RAM, so it is the benchmark against llama.cpp.
3. DeepSeek V4 Flash (284B total / 13B active) — the flagship streamed target; needs MLA and the DSA indexer.

**Performance targets**

- Qwen3-30B-A3B fully resident, 4-bit: ≥ 80% of llama.cpp decode on the same machine.
- DeepSeek V4 Flash on the Target tier with 64 GB: ≥ 1 tok/s warm decode, ≥ 70% expert-cache hit rate after warm-up.
- Disk path: ≥ 85% of `fio` sequential-read bandwidth on 4–8 MB expert reads.

## Architecture specification

The engine is a set of single-purpose C headers plus one `.c` file per model family; the expert store is the only component that touches the disk during decode.

**Source tree**

```
tierlm/
  Makefile                 make, make AVX512=1, make test, make oracle
  src/
    tlm.h                  public API: tlm_load, tlm_prefill, tlm_decode, tlm_free
    platform.h             compat: aligned alloc, O_DIRECT/F_NOCACHE, threads, timers
    tlmf.h                 container reader (header, tensor index, mmap of dense part)
    tensor.h               tensor view: dtype, shape, strides, data ptr
    quant.h                block formats + dequant (Q8_0, Q4_K-style, FP8-E4M3, MXFP4)
    kern_scalar.h          reference kernels (the oracle path, never removed)
    kern_avx2.h            AVX2/FMA matvec, matmul, quant dots
    kern_avx512.h          AVX-512 / VNNI variants
    kern_neon.h            NEON / dotprod variants
    threadpool.h           fixed worker pool, barrier, parallel-for over rows
    expert_store.h         slab cache, admission, eviction, pinning, stats
    io.h                   io_uring backend + pread-pool fallback
    prefetch.h             router-lookahead prefetcher thread
    kv.h                   KV cache (GQA and MLA-compressed), disk snapshot
    tok.h                  BPE tokenizer (tiktoken-style ranks) + chat templates
    sample.h               greedy, temperature, top-p, min-p
    json.h, http.h         minimal JSON + HTTP/1.1 for the server
  models/
    olmoe.c  qwen3moe.c  dsv4flash.c   one forward graph each
  tools/
    convert.py             safetensors -> .tlmf (offline only)
    oracle.py              HF transformers reference dumps
  tests/
```

&#91;embedded content: per-layer decode path · compute, store, I/O\]

Three threads cooperate per layer: compute asks the store for slots, the prefetcher hints the next layer's experts, and the I/O thread is the only reader of the disk.

**Module contracts**

| Module | Owns | Must never |
| --- | --- | --- |
| `models/*.c` | Layer order, attention variant, router math, which tensors are experts | Do I/O or know whether an expert is cached |
| `expert_store.h` | Which experts are in RAM, slot allocation, eviction, pins, hit statistics | Change bytes; dequantise; reorder routing |
| `io.h` | Submitting and completing aligned reads into slot buffers | Block the compute thread except in `es_wait` |
| `prefetch.h` | Guessing next-layer experts and issuing non-blocking requests | Evict a pinned slot or one in use this layer |
| `kern_*.h` | Arithmetic on given pointers | Allocate, or read global state |
| `kv.h` | KV layout, append, truncate, save/restore to disk | Depend on the model file being open |

**Per-token decode path** (one MoE layer)

1. Attention block on resident weights; append to KV.
2. Router: `logits = W_r · h`, top-k with the model's exact scoring (softmax or sigmoid, bias, group limits).
3. `es_acquire(layer, ids[k])` returns a ready slot or a pending ticket per expert; misses go to `io.h` immediately.
4. Compute shared expert and every already-resident routed expert while misses are in flight.
5. `es_wait` on remaining tickets, compute them, combine with router weights in the model's fixed order.
6. `es_release` the slots; the prefetcher has already run layer l+1's router on `h` and queued its likely misses.

The weighted sum in step 5 always runs in ascending top-k order regardless of which expert finished first, so arrival order can never change the floating-point result.

## File format: `.tlmf`

One file (or a directory of shards with the same layout) split into a small header, a dense region that is mmap'd, and an expert region where every routed expert is one contiguous, 4 KiB-aligned record read with a single I/O.

This single-file container is superseded by the packet store produced by the model compiler: Two-stage design

**Layout**

```
[0]            tlmf_header      (4096 B, padded)
[4096]         metadata JSON    (config, tokenizer, chat template, quant recipe)
[align 4096]   tensor index     (tlmf_tensor[n_tensors])
[align 2 MiB]  DENSE REGION     embeddings, norms, attention, router, shared experts, lm_head
[align 4096]   expert table     (tlmf_expert[n_moe_layers * n_experts])
[align 4096]   EXPERT REGION    layer-major, expert-minor; each record = gate|up|down
```

**Header and index structs** (little-endian, fixed width)

```c
typedef struct {
    char     magic[8];          /* "TLMF\0\0\0\1" */
    uint32_t version;           /* 1 */
    uint32_t arch;              /* TLM_ARCH_OLMOE, _QWEN3MOE, _DSV4FLASH */
    uint64_t meta_off, meta_len;
    uint64_t tindex_off;  uint32_t n_tensors;
    uint64_t dense_off,  dense_len;
    uint64_t etable_off; uint32_t n_moe_layers, n_experts, top_k;
    uint64_t expert_off, expert_len;
    uint32_t align;             /* 4096 */
    uint8_t  sha256_dense[32];
    uint8_t  reserved[4096 - 128];
} tlmf_header;

typedef struct {
    char     name[96];          /* "blk.12.attn_q.weight" */
    uint32_t dtype;             /* TLM_F32, _BF16, _Q8_0, _Q4K, _FP8E4M3_B128, _MXFP4 */
    uint32_t ndim;  uint64_t shape[4];
    uint64_t offset, nbytes;    /* within dense region */
} tlmf_tensor;

typedef struct {
    uint64_t offset;            /* absolute, multiple of 4096 */
    uint32_t nbytes;            /* padded to 4096 */
    uint32_t off_up, off_down;  /* gate starts at 0 */
    uint32_t dtype;
    uint32_t crc32c;            /* verified on first load, optional thereafter */
} tlmf_expert;
```

**Rules**

- Experts keep the checkpoint's native precision when it is already low (FP8 blocks, MXFP4); the converter repacks, it does not requantise them unless a recipe asks to.
- Within a record, each matrix is stored in the kernel's preferred interleaved block order, so no repacking happens at load or decode time.
- The converter streams shard by shard and never needs the full source on disk (copy colibri and apus).
- A `.tlmf.warm` sidecar stores per-expert hit counts and the last pinned set; it is advisory and safe to delete.
- Optional mirror files for multi-SSD reads must be byte-identical; checked by size and header hash at open.

## Expert store, cache and I/O

The expert store is a fixed pool of equal-size, page-aligned RAM slots allocated once at start; experts move between disk and slots, never through malloc, and a slot's contents are only ever an exact copy of the file record.

**API**

```c
typedef struct es es_t;
typedef struct { int slot; int ticket; } es_ref;   /* ticket < 0 => ready */

es_t  *es_open(const tlmf *f, size_t budget_bytes, const es_opts *o);
es_ref es_acquire(es_t *, int layer, int expert);  /* never blocks */
const void *es_wait(es_t *, es_ref);               /* blocks until bytes ready */
void   es_release(es_t *, es_ref);
void   es_prefetch(es_t *, int layer, const int *ids, int n); /* hint, may drop */
void   es_pin(es_t *, int layer, int expert);       /* from .warm or policy */
void   es_stats(const es_t *, es_stats_t *out);     /* hits, misses, bytes, stall us */
void   es_save_warm(const es_t *, const char *path);
```

**Slot states:** `FREE → LOADING → READY(refcnt) → EVICTABLE`. A slot with refcnt > 0 or marked for the current layer is never evicted.

**Policy (v1)**

- Budget: `--cache auto` = RAM total − dense − KV(ctx) − scratch − OS reserve (default 4 GB, or 10%). `plan` prints the result before loading.
- Eviction: segmented LRU per layer group (probation + protected), protected segment entered after a second hit. Equivalent in spirit to Pulsar's touch-count admission and cheap to implement.
- Pins: on start, load the top N experts from `.tlmf.warm` up to `--pin-frac` of budget (default 30%). Pinned slots are excluded from eviction.
- Prefetch never evicts protected or pinned slots and is dropped when the in-flight queue exceeds half its depth.
- Prefill: compute the union of experts across the chunk (batch-union), sort by file offset, read each once.

**I/O backend**

- Linux: `io_uring` with registered buffers (the slot pool) and registered file, queue depth 32 (tunable 8–128), `O_DIRECT` default on, `--no-direct` falls back to buffered reads with `posix_fadvise(DONTNEED)` after use.
- Fallback (macOS, Windows, no liburing): a pool of 8–16 I/O threads doing `pread` into slots; on macOS `F_NOCACHE` on the fd.
- One expert = one read. Records larger than 8 MiB are split into 2–4 MiB reads submitted together.
- Multi-SSD: weighted round-robin across mirrors by bandwidth measured at open; a mirror read error falls back to the primary.
- Completions are reaped by a dedicated I/O thread that flips slot state and signals a per-slot futex/condvar.

**Prefetcher**

A dedicated thread receives the post-attention hidden state of layer l, applies layer l+1's router (norm + `W_r` matvec, \~0.1% of layer FLOPs), takes top-k and calls `es_prefetch`. It must use the same kernel as the real router so its guess is the true routing whenever the hidden state matches. Hit rate of the prefetch is reported separately.

**Metrics surfaced by `--stats`:** hit rate per layer, MB read per token, disk GB/s achieved, compute-thread stall µs per token, prefetch accuracy, pinned fraction.

## Compute kernels, quantisation and threading

Decode on CPU is memory-bandwidth bound, so kernels are int8-activation dot products over block-quantised weights, chosen at runtime by CPU feature detection, with one scalar reference for each.

**Quantisation formats (v1)**

| Format | Use | Block | Notes |
| --- | --- | --- | --- |
| BF16 / F32 | Norms, router, small tensors | — | Router stays at checkpoint precision to keep routing exact |
| Q8\_0 | Attention, shared experts, embeddings | 32 values + f16 scale | Same layout as GGML so results can be cross-checked |
| Q4\_K-style | Routed experts for models shipped in BF16 | 256 super-block, 6-bit sub-scales | Converter option, with quality gate |
| FP8-E4M3, 128×128 blocks | Experts for FP8 checkpoints | per-block F32 scale | Dequant to BF16/F32 in-register, no requantisation |
| MXFP4 | Experts for MXFP4/fp4 checkpoints (DeepSeek V4) | 32 values + E8M0 scale | Native, no conversion |

Activations are quantised per row to Q8 (32-value blocks) once per matmul and reused by all experts that read the same input.

**Kernels**

- `matvec_q(W, x_q8, y, rows)` for decode; `matmul_q(W, X_q8, Y, rows, n_tok)` for prefill with a 4×N register tile.
- Variants: scalar (oracle), AVX2+FMA, AVX-512BW/VNNI (`vpdpbusd`), NEON with `sdot`. Dispatch by function pointer table filled at start from `cpuid` / `getauxval`.
- Expert FFN fused: `gate` and `up` matvecs share one pass over `x_q8`, SiLU/SwiGLU applied in registers, result re-quantised for `down`.
- Attention: RoPE (with YaRN where the model uses it), GQA with Q8 or F16 KV; MLA with the absorbed-weights trick and compressed latent KV for DeepSeek; DSA top-k indexer for long context in milestone 6.

**Threading**

- One fixed pool of `n_cores` compute workers (physical cores, not SMT siblings by default), one I/O reaper thread, one prefetcher thread.
- Parallelism is over output rows within a matmul; for routed experts, the k experts of a layer are split across workers when k ≥ workers/2, otherwise each expert's rows are split.
- Each worker reduces its own rows; no cross-thread float reductions, so results are independent of thread count.
- NUMA (later): interleave dense weights across nodes; pin expert slots to the node of the worker that computes them.

## Correctness, testing and benchmarking

No optimisation merges unless the fast path reproduces the scalar oracle's tokens, and the oracle itself matches Hugging Face `transformers` token for token on the bring-up models.

**Test gates** (each a `make` target, all run in CI on Linux x86-64 and macOS arm64)

| Target | Checks | Pass rule |
| --- | --- | --- |
| `test-tok` | Tokenizer + chat template vs `tokenizer.json` and Jinja-rendered goldens | Byte-identical on 100k probe strings |
| `test-quant` | Each dequant + dot kernel vs scalar, random blocks | Max abs error 0 for int paths; ≤ 1 ulp after F32 accumulate |
| `test-layer` | One layer forward vs `oracle.py` dumps | Cosine ≥ 0.9999, max rel error ≤ 1e-3 |
| `test-e2e` | Teacher-forced 64 positions vs HF | Top-1 match on all positions except documented near-ties (< 0.05 logit) |
| `test-store` | Expert store under random acquire/release, tiny budget, fault injection | Bytes in slot == file record (crc32c), no deadlock in 10^6 ops |
| `test-invariance` | Same prompt with cache 1%, 10%, 100%; threads 1/4/16; prefetch on/off; direct on/off | Bitwise-identical greedy tokens |
| `test-server` | OpenAI API scripted fixtures, streaming SSE | Exact JSON shapes |

Also run every C test three times: once normal, once with `-fsanitize=address,undefined` and once under ThreadSanitizer for `expert_store.h` and `io.h`.

**Benchmark protocol** (`tierlm bench`, results committed as JSON)

Record hardware, commit, model file hash, command, prompt, cache state (cold after `echo 3 > drop_caches` / warm after one pass), prefill tok/s, decode tok/s, TTFT, hit rate, GB read per token, disk GB/s and stall µs per token. Always compare against `llama.cpp` on the same box and the same quant. One variable changes per run.

## Milestone plan

Six milestones, about 18 weeks for one experienced C engineer (or \~10–12 weeks for two working on kernels and the store in parallel); each ends on a gate that must pass before the next starts.

&#91;embedded content: roadmap · 6 milestones, a gate each\]

M4 is the heart of the project: everything after it assumes streaming is exact and near disk speed.

1. **M1 Foundations (weeks 1–3)**
   - Makefile with `make`, `make AVX512=1`, `make test`, `make asan`; CI on Linux and macOS.
   - `platform.h`, `threadpool.h`, `tensor.h`, `tlmf.h` reader, `tools/convert.py` writer (streaming, shard by shard).
   - `tok.h` BPE + chat template rendering; `kern_scalar.h` for every op.
   - Exit: `test-tok` byte-exact; OLMoE converted and re-read with matching tensor hashes.
2. **M2 Oracle bring-up (weeks 4–5)**
   - `models/olmoe.c` on the scalar path; `tools/oracle.py` dumps per-layer activations and logits.
   - CLI `run` and `chat` with greedy and temperature sampling.
   - Exit: `test-layer` and `test-e2e` pass against HF.
3. **M3 Fast CPU kernels (weeks 6–8)**
   - Q8\_0, Q4\_K-style, FP8 and MXFP4 dot kernels in AVX2, AVX-512/VNNI and NEON with runtime dispatch; fused SwiGLU expert FFN; prefill matmul tiling.
   - `models/qwen3moe.c`, fully resident.
   - Exit: `test-quant` green; decode ≥ 80% of llama.cpp on the same quant and machine.
4. **M4 Expert streaming (weeks 9–12)**
   - `expert_store.h`, `io.h` (io\_uring + pread fallback), `prefetch.h`, `.tlmf.warm` sidecar, batch-union prefill, `--stats`.
   - `doctor` (SIMD, RAM, sustained disk read) and `plan` (cache sizing).
   - Exit: `test-store` and `test-invariance` bitwise green; Qwen3-30B at a 10% cache runs at ≥ 85% of `fio` bandwidth on misses.
5. **M5 Flagship model (weeks 13–16)**
   - `models/dsv4flash.c`: MLA with compressed KV, native MXFP4 experts, FP8 dense, DSA top-k indexer.
   - Exit: teacher-forced match vs HF on a short prompt; ≥ 1 tok/s warm decode with 64 GB RAM on a Gen4 NVMe.
6. **M6 Ship v1 (weeks 17–18)**
   - C HTTP server with `/v1/chat/completions` and SSE; disk KV and prompt-prefix cache; Windows (MinGW) build with the pread backend.
   - Exit: `test-server` green; v1.0 tagged with benchmark JSON for three machines.

After v1: multi-SSD mirroring, NUMA placement, a CUDA or Metal backend for attention and hot experts, and MTP speculation behind a measured flag.

## Risks and open questions

The largest risk is that the target machine's RAM, not the disk, sets the speed: below a \~60–70% hit rate, decode collapses toward one disk round-trip per expert.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Too little RAM for a useful cache on the flagship model | < 0.3 tok/s, unusable for interactive work | `plan` refuses or warns; pick a smaller flagship (Qwen3.8 Flash Next class) for 32 GB machines |
| Consumer SSD sustained-read throttling (QLC, DRAM-less, thermal) | Bandwidth halves after minutes | `doctor` runs a 60 s sustained read test; document recommended drives |
| Architecture churn: new attention variants (MLA, DSA, KDA, gated delta) every few months | Flagship goes stale | Keep model code in one file per family; budget a model bring-up as \~3–4 weeks |
| Licence contamination | Can't ship | Only MIT/Apache sources may be adapted (llama.cpp/GGML quant tables, ds4, colibri, Pulsar kernels with attribution); apus (PolyForm NC) and flash-moe (no licence) are read-only references |
| Prefetch thread competes with compute for memory bandwidth | Slower than no prefetch on some hosts | Off-switch, measured A/B in `bench` |
| io\_uring unavailable (old kernels, containers, Windows) | Lower I/O efficiency | pread pool fallback is a first-class, tested path |

**Open questions to settle in milestone 1**

- Do we accept a Python converter, or must conversion also be C? (Plan assumes Python offline, C at runtime.)
- Is Windows needed for v1 or can it wait for milestone 6?
- Is a GPU backend expected within the first release, or strictly later?

## Sources

- [JustVugg/colibri](https://github.com/JustVugg/colibri) — README, techniques, benchmarks, model roster
- [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4) and its [SSD streaming guide](https://github.com/antirez/ds4/blob/main/docs/SSD_STREAMING.md)
- [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers); [Phoronix on KTransformers 0.7](https://phoronix.com/news/KTransformers-0.7); [LMSYS: KTransformers in SGLang](https://www.lmsys.org/blog/2025-10-22-KTransformers)
- [danveloper/flash-moe](https://github.com/danveloper/flash-moe); [InsiderLLM: Your SSD is the new GPU](https://buttondown.com/insiderllm/archive/your-ssd-is-the-new-gpu/)
- [giannisanni/pulsar](https://github.com/giannisanni/pulsar)
- [Pioshin/monolite](https://github.com/Pioshin/monolite)
- [bricesommers/apus-glm5.3-flash](https://github.com/bricesommers/apus-glm5.3-flash)
- [GitHub topic: ssd-streaming](https://github.com/topics/ssd-streaming)

Added in the wider pass:

- [LMSYS: SGLang SSD Expert Pack](https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/)
- [Schero94/slipstream](https://github.com/Schero94/slipstream)
- [llama.cpp discussion #27149: expert-aware SSD streaming](https://github.com/ggml-org/llama.cpp/discussions/27149); [moe-stream on Jetson](https://dev.to/anyesh/running-a-26b-moe-on-an-8-gb-jetson-by-streaming-experts-from-ssd)
- [gmlx on PyPI](https://pypi.org/project/gmlx/)
- [AcceptMoE (arXiv 2608.02989)](https://arxiv.org/abs/2608.02989); [DFlash (arXiv 2602.06036)](https://arxiv.org/html/2602.06036v2)
- [Phoronix: KTransformers 0.5.3 AVX2](https://www.phoronix.com/news/KTransformers-0.5.3)
