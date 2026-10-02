# Architecture

This page shows how the code fits together. The pattern-by-pattern justification of each module is
in `DISTILLED.md`. The full rationale is in `research/03-two-stage-design.md`.

## Two stages, one library

```
             offline, slow, once per model                         online, every run
 ┌────────────────────────────────────────────────┐   ┌──────────────────────────────────────┐
 │ tlmc compile                                   │   │ tierlm run / bench                   │
 │  1 ingest (config.json + safetensors, SHA-256) │   │  open manifest, check SHA-256        │
 │  2 reference quantise  → scratch store         │   │  load DENSE/EMBED/META → RAM         │
 │  3 calibrate (runs the runtime with a hook)    │   │  warm: pin compiler's hot set        │
 │  4 precision plan  ┐                           │   │  per token, per layer:               │
 │  5 prune (REAP)    ├─ tlm_decide + eval gate   │──▶│    router → top-k ids                │
 │  6/7 pre-build + segment into packets          │   │    store_acquire (hit or NVMe read)  │
 │  8 place across drives (heat, pins, replicas)  │   │    prefetch layer l+1 (lookahead)    │
 │  9 validate through the runtime, seal manifest │   │    SwiGLU experts, fixed-order sum   │
 └────────────────────────────────────────────────┘   └──────────────────────────────────────┘
        writes: <name>.tlmm (manifest)                    reads: manifest + packet files
                <drive_i>/<name>.tlmp (packets)
                <name>.work/ (scratch, decisions.jsonl)
```

`tierlm-setup` sits beside both stages. It writes `/etc/tierlm/{tierlm,hardware}.conf`, which both
binaries read through `tlm_cfg`.

## Library layering

Lower layers never include higher ones.

```
 binaries     tierlm-setup          tlmc                         tierlm
                  │                   │                            │
 engine           │        tlm_decide │  tlm_model ────────────────┤
                  │            │      │   │   │                    │
 storage          │            │      │ tlm_store ── tlm_io        │
                  │            │      │   │                        │
 formats      tlm_cfg      tlm_json ─ tlm_st   tlm_packet ── tlm_quant (+ kern_avx2, kern_avx512)
                  │            │        │          │             │
 base         ────┴──────── tlm_platform ─────────┴───── tlm_pool
```

| Module | One-line role | Key invariant |
| --- | --- | --- |
| `tlm_platform` | errors, logging, memory, files, CPU and topology, CRC32C | `tlm_pread_full` never returns a short read silently |
| `tlm_cfg` + `tlm_cfg_schema` | layered `key = value` config with a typed schema | later layer wins; every value remembers its origin |
| `tlm_json` | strict JSON reader/writer | depth and size limits; no recursion blow-ups |
| `tlm_st` | safetensors reader/writer | every offset bounds-checked before any read |
| `tlm_quant` + kernels | block formats, quantise, dot products, matvec | AVX2/AVX-512 integer results equal scalar bit for bit |
| `tlm_packet` | 64-byte header, packet writer/reader, manifest | publish only after CRC; manifest saved by tmp + rename |
| `tlm_io` | per-drive pread pool or io_uring, replica choice | every accepted request gets exactly one callback |
| `tlm_store` | expert slot cache with LFU + decay, pins, prefetch | a slot in use or loading is never evicted; prefetch never blocks demand |
| `tlm_pool` | persistent workers, deterministic parallel-for | the chunk split depends only on (n, grain, size) |
| `tlm_model` | MoE decoder, KV cache, sampler | logits do not depend on threads, cache size, prefetch, I/O or kernel tier |
| `tlm_decide` | typed questions, rules backend, audit log | low confidence → the safe default (keep the expert) |

## The determinism contract

This is the main correctness property, and most tests exist to protect it:

> For a given manifest and prompt, the output tokens and logits are bit-identical regardless of
> expert cache size (down to `top_k` slots), prefetch on or off, thread count, I/O backend,
> O_DIRECT, which replica served a read, and the kernel tier (scalar, AVX2 or AVX-512).

These choices make that possible:

- Integer dot products, with activation offsets folded through group sums (P-44).
- Each worker owns whole output rows, so there are no cross-thread float sums (P-43).
- `-ffp-contract=off`, and no `-ffast-math`.
- RMSNorm sums in double.
- Experts are combined in ascending top-k rank order (C-13).
- Prefill is the decode step repeated.
- The cache stores packed bytes exactly as they sit on disk (P-50).

`tlmc compile --profile exact` checks this end to end in pass 9, against the pass-2 reference.

## Packet store on disk

- **Header:** each packet starts with a 64-byte header containing:
  - magic `TLMP` and the packet type
  - layer, id, dtype and layout
  - payload length, CRC32C and up to 3 item offsets
- **Alignment:** packets are 4 KiB aligned so O_DIRECT works.
- **Types:**
  - DENSE (1)
  - EXPERT (2)
  - BUNDLE (3)
  - MTP (4)
  - EMBED (5)
  - TABLE (6)
  - META (7)
  - TOKEN (8)
- **Manifest (`.tlmm`):** holds the following:
  - model geometry
  - one location per packet: drive, offset, length and flags (bit 0 pinned, bit 1 pruned)
  - an optional replica location
  - per-expert heat
  - the sealed compile report (JSON)
  - a SHA-256 over the packet table
- **Placement:**
  - packets are laid out layer-major on each drive
  - experts are spread so that the expected read bytes per token are balanced across drives
  - the warm middle is replicated on the least-loaded other drive, and the hottest experts are
    pinned in RAM

## Where the decision engine is used

`tlmc` asks three typed questions: `expert.specialisation` (Score), `expert.precision_tier`
(Choice) and `expert.needed_for_usecase` (Noul). The rules backend answers them from calibration
features, and the confidence policy then does one of the following:

- **0.90 or more:** apply the answer.
- **0.60 to 0.90:** review. The engine re-asks with SELF when that backend is available.
  Otherwise it takes the safe default (non-interactive) or returns REVIEW (interactive).
- **Below 0.60:** take the safe default.

Hard rules sit outside the model and cannot be overridden: `experts.protect`, deep content,
`prune.max_ratio`, `prune.min_keep_per_layer`, and never fewer than 4 × top_k experts per layer.

Every answer is appended to `decisions.jsonl` together with its features, so a later compile, or a
person, can audit it. See `research/04-decision-engine.md`.
