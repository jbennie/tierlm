# From fragments to libtlm

How the 172 fragments and 145 scored patterns in `fragments/` were reduced to one common
library (`libtlm`) plus three glue binaries. Each module lists the patterns it implements
(`P-xx` ids from `fragments/CONSENSUS.md`) and the verdict that justified it.

Rule used: build `adopt` patterns by default; build `adopt-tested` patterns behind a config
switch with a benchmark gate; leave `experimental` patterns as hooks only; never build
`reject` patterns. Where projects contradict each other, the contradiction id (`C-xx`) is noted
and the switch defaults to the safer side.

## Module map

| Module | Purpose | Patterns (verdict) | Defaults chosen |
| --- | --- | --- | --- |
| `tlm_platform` | Aligned allocation, monotonic time, full-read I/O helpers, O_DIRECT open, CPU feature detection, RAM snapshot (MemAvailable + cgroup) | P-30 full-read discipline (adopt), P-25 cache-bypass switch (adopt), P-55 CPU dispatch (adopt), P-107 memory snapshot (adopt) | O_DIRECT off unless `io.direct = yes` (C-02: drive-dependent) |
| `tlm_cfg` | `key = value` config with includes, precedence (defaults < files < env < flags), typed getters with ranges | planner/config patterns P-116 static memory plan (adopt) | Unknown keys warn |
| `tlm_json` | Strict, arena-backed JSON reader for safetensors headers, manifests and the server | file-reader patterns (adopt) | Depth and size limits |
| `tlm_st` | safetensors reader: header validation, tensor index, range reads | P-03 validated container (adopt), file readers (adopt, format-defined) | Every offset bounds-checked |
| `tlm_pool` | Fixed worker pool, parallel-for over row ranges, no cross-thread float reductions | P-23 persistent workers (adopt), P-43 fixed reduction order (adopt) | Threads = physical cores |
| `tlm_quant` | Block formats and dot products: Q8_0, Q4_K, Q5_K, Q6_K, Q8_K activations, MXFP4, FP8 E4M3; scalar reference + AVX2 + AVX-512 with runtime dispatch | P-40 GGUF layouts (adopt), P-42 quantise activations once (adopt), P-43 bit-exact SIMD (adopt), P-44 fold offsets via group sums (adopt), P-45 sign trick (adopt), P-47 FP8 (adopt), P-48 MXFP4 (adopt), P-50 keep packed (adopt) | Integer paths must equal scalar bit for bit; tests enforce it |
| `tlm_packet` | Packet header, manifest, CRC32C, writer and reader | P-01 one expert = one record (adopt), P-03 validated container (adopt), P-04 resumable conversion (adopt), P-05 verify bytes (adopt), P-08 tmp + rename (adopt) | 4 KiB alignment, CRC per packet |
| `tlm_io` | Read backend: persistent pread worker pool; io_uring when built with liburing | P-23 pread pool (adopt), P-24 io_uring (adopt-tested), P-29 batch-union (adopt), P-31 multi-drive replicas (adopt-tested) | pread pool default; io_uring opt-in until measured (C-03) |
| `tlm_store` | Expert slot cache: fixed aligned slots, reserve/publish/release lifecycle, global frequency-aware eviction with decay, pins, warm sidecar, prefetch that never hurts demand | P-10 bounded cache (adopt), P-12 slot lifecycle (adopt), P-15 frequency eviction (adopt), P-18 global cache (adopt), P-19 warm pins (adopt), P-34 speculation never hurts demand (adopt), P-33 router lookahead (adopt) | Global cache (C-07); packed bytes in slots (C-08) |
| `tlm_model` | Ops (RMSNorm, RoPE, GQA attention, router variants, SwiGLU expert FFN, combine in fixed order), KV cache, sampler, a config-driven MoE decoder | P-61 GQA (adopt), P-73 sigmoid router (adopt), P-75 SwiGLU + shared expert (adopt), P-79 decoder skeleton (adopt), P-80 RMSNorm (adopt), P-81 RoPE (adopt), P-89 KV (adopt), P-94 sampler (adopt) | f32 accumulation; experts summed in ascending top-k order (C-13) |
| `tlm_decide` | Typed decision API (Choice, Score, Noul), rules backend, confidence policy, audit log; encoder backend is a stub until ported | decision-engine design (`research/04-decision-engine.md`) | Rules only; safe default = keep |
| `tlm_net` (not started) | Cluster message header and framing (TCP first) | P-128 handshake, P-129 typed frames, P-131 parity gate, P-135 bounded waits, P-137 prefix hash, P-144 registry (adopt); P-136/P-138..P-140 (adopt-tested) | Planned; design in `docs/design/cluster.md` and `docs/design/swarm.md` |

## Glue binaries

| Binary | Role | Uses |
| --- | --- | --- |
| `tierlm-setup` | Probe the machine, recommend settings, write `/etc/tierlm/tierlm.conf` and `hardware.conf`; Experts-and-content section | platform, cfg, decide |
| `tlmc` | Model compiler: ingest safetensors → reference quantise → calibrate (routing heat, REAP saliency) → precision plan → optional prune (decision engine + gate) → segment into packets → place across drives → validate and seal manifest | st, json, quant, packet, model, decide, pool |
| `tierlm` | Runtime: load manifest, map dense packets, pin hot experts, stream the rest, generate; `config show/check`; `bench` | cfg, packet, io, store, model, pool |

## Not built (rejected by evidence)

P-09 expert file compression/re-layout, P-14 reserved HOT sub-tier, P-27 advisory read-ahead
hints, P-28 mmap for cold experts, P-35 coupling tables, P-38 temporal prediction, P-39 learned
routing predictors, P-53 2-bit production experts, P-88 external draft model on a streamed MoE.

## Provenance

Every module header names the pattern ids it implements. `fragments/` records where each
pattern came from (`Origin:` lines). No code was copied from any reference project; the library
was written from the pseudocode fragments and public format specifications (GGUF block layouts,
safetensors, OCP MX).
