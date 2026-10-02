# Two-stage design: model compiler and runtime

Oct 2, 2026 · @Jay

## At a glance

Split tierlm into a slow, offline **model compiler** (`tlmc`) that does every expensive decision once, and a fast **runtime** (`tierlm`) that only loads and executes what the compiler laid out. The runtime gets simpler and faster; the compiler is where the intelligence lives.

- **Stage 1, `tlmc compile`:** any supported checkpoint (safetensors BF16/FP8/MXFP4, or GGUF) + a hardware profile + a use-case profile → a packet store spread across the four dedicated drives, plus a manifest. It profiles routing on your own prompts, chooses precision per tensor group, optionally prunes experts under a quality gate, pre-builds kernel-native layouts, cuts the model into aligned packets and places them for parallel reads.
- **Stage 2, `tierlm run`:** opens the manifest, maps dense packets into RAM, pins the hot experts the compiler chose, and streams the rest with no layout work, no repacking and no guessing at start-up.
- **Default profile is exact:** quantisation only, no pruning, output identical to the reference quantised model. Pruning and heat-based mixed precision are opt-in and must pass a held-out quality gate before the compiler will write them.
- **Capacity:** the four 256 GB drives give \~860 GiB (\~920 GB) of usable packet space after filesystem overhead and a 10% free-space reserve. That holds DeepSeek V4 Flash with every expert stored twice, or GLM-5.2 at Q5\_K once, or a small library of two or three mid-size models.
- **The same compiled store is portable:** if the hardware changes, `tlmc replan` re-derives placement and pinning from the stored profiles in minutes, without re-quantising.

## How the two stages fit

&#91;embedded content: two-stage flow · 9 compiler passes, 4 drives, runtime\]

Everything above the drives runs once and can take hours; everything below runs on every launch and must take seconds.

## Compiler inputs

The compiler takes four inputs; the last three are small JSON files, so the same model can be compiled differently for different machines or jobs.

| Input | Contents | Produced by |
| --- | --- | --- |
| Source model | HF safetensors (BF16, FP8 blocks, MXFP4) or GGUF, config, tokenizer, chat template | Downloaded shard by shard; never needs the whole source on disk at once |
| Hardware profile | RAM total and usable, measured DRAM GB/s, CCD layout, SIMD level, drive list with measured sustained GB/s and capacity, GPU KV budget | `tierlm-setup --probe → /etc/tierlm/hardware.conf` |
| Use-case profile | Calibration corpus (your real prompts: e.g. coding-agent transcripts, chat), held-out eval set, typical context length, sampling mode, which tools/templates are used | You, with `tlmc corpus` helpers to sample from logs |
| Quality budget | Profile (`exact`, `balanced`, `aggressive`), max allowed drops: KL divergence, top-1 agreement, task score on the held-out set | Defaults per profile; overridable |

Rule: the calibration and evaluation sets never overlap, and the compiler refuses a use-case profile with fewer than \~200k calibration tokens, because routing statistics from tiny samples overfit (colibri saw exactly this with learned pins).

## Compiler passes

Nine passes run in order; each writes its result into a work directory so an interrupted compile resumes at the last finished pass.

1. **Ingest and verify.** Read config and tokenizer, hash every source shard, select the architecture adapter (`models/*.c` exposes a tensor map: which tensors are dense, routed experts, shared experts, router, MTP, embeddings, n-gram tables). Emits `ingest.json`.
2. **Reference quantise.** Convert per the precision plan's starting point (Q6\_K experts, Q8\_0 dense, F32 router, native FP4 kept) into a scratch store. This is the "exact" model every later pass is measured against. Emits the scratch packets and `ref.sha256`.
3. **Calibrate.** Run the runtime itself over the calibration corpus with full tracing. Record per (layer, expert): selection count, mean gate value, mean output norm (REAP saliency), co-activation counts within a layer, next-layer predictability of the router, activation statistics for imatrix, MTP acceptance by draft position. Emits `routing.bin`, `imatrix.dat`, `mtp_stats.json`.
4. **Precision plan.** Final per-tensor formats. In `exact`, unchanged from pass 2. In `balanced`, imatrix-guided Q5\_K for the coldest experts so that the model fits a target size. Never touches router, norms or the MTP head. Emits `precision.json`.
5. **Prune (optional).** REAP-style: rank experts per layer by router-weighted activation norm on the calibration set, drop the lowest up to the profile's ratio, renormalise router weights over the survivors. Re-run the held-out eval; if any gate fails, back off the ratio by 5 points and retry. Emits `prune.json` and an eval report.
6. **Pre-build.** Everything the runtime would otherwise compute at load: kernel-native interleaved blocks for AVX-512 tiles; gate and up fused into one interleaved matrix; MLA absorbed projections pre-multiplied; RoPE/YaRN tables; tokenizer merge tables in a flat array; chat template compiled to a token program. Emits final packet payloads.
7. **Segment.** Cut payloads into packets (types in the next section). Small experts that the calibration shows firing together are bundled so one read fetches them; large experts stay one packet each.
8. **Place.** Decide RAM-pinned set, drive for each packet, and replicas (rules in the placement section). Write packets with O\_DIRECT, 4 KiB aligned, then read back and verify every CRC.
9. **Validate and seal.** Run the held-out set through the compiled store via the real runtime and compare to pass 2's reference: token agreement, KL, task score. Write the manifest with all stats and the eval report, then mark the store read-only.

Time for DeepSeek V4 Flash on this box is dominated by passes 3 and 9 (running the model over \~200k calibration and \~50k eval tokens): expect several hours, once.

Passes 3, 4, 5 and 9 delegate their judgment calls to the embedded decision engine: Decision engine

## Packet format

Every byte the runtime reads is inside a packet: a 64-byte header followed by a payload, padded to 4 KiB, addressable by drive, offset and length from the manifest alone.

| Packet type | Holds | Where it ends up at run time |
| --- | --- | --- |
| `DENSE` | Attention, norms, router, shared experts, dense FFN for one layer | RAM (hugepages), loaded once |
| `EXPERT` | One routed expert: fused gate+up, then down, kernel-native layout | RAM slot cache or streamed |
| `BUNDLE` | 2–8 small experts of one layer that co-fire often | Read together in one I/O, split into slots |
| `MTP` | The MTP/NextN layer, Q8\_0 | RAM, loaded once |
| `EMBED` | Embedding rows in pages of 4,096 tokens | RAM, or paged for very large vocabularies |
| `TABLE` | Large lookup tables (n-gram/engram memories) as row-addressable pages | Read by row on demand |
| `META` | Tokenizer arrays, chat-template program, RoPE tables | RAM |

```c
typedef struct {              /* 64 bytes, little-endian */
    uint32_t magic;           /* 'TLMP' */
    uint16_t type;            /* DENSE, EXPERT, BUNDLE, MTP, EMBED, TABLE, META */
    uint16_t version;
    uint16_t layer;           /* 0xFFFF if not per-layer */
    uint16_t n_items;         /* experts in a BUNDLE, else 1 */
    uint32_t first_id;        /* expert id, or first row */
    uint32_t dtype;           /* Q6_K, Q5_K, Q8_0, MXFP4, F16, ... */
    uint32_t layout;          /* kernel layout id, e.g. AVX512_16ROW_INTERLEAVED */
    uint64_t payload_len;     /* bytes before padding */
    uint32_t crc32c;          /* of the payload */
    uint32_t item_off[3];     /* sub-offsets: up/down, or bundle members */
    uint8_t  reserved[16];   /* pads the header to 64 bytes */
} tlm_packet_hdr;
```

**Manifest (`model.tlmm`, on the OS drive):** model config; precision, prune and pre-build decisions; a packet table `(type, layer, id) → {drive, offset, len, replica_drive, replica_offset}`; the pinned set; initial `.warm` heat from calibration; hardware and use-case profile hashes; eval report; and a SHA-256 over the whole packet table. The runtime trusts nothing that is not in the manifest.

## Placement on four 256 GB drives

With the drives dedicated, the compiler can spend spare capacity on replicas of busy experts, which turns idle space into read bandwidth.

**Usable space:** 4 × 256 GB ≈ 4 × 238 GiB after formatting; keeping 10% free for SSD write and garbage-collection headroom leaves \~214 GiB per drive, \~860 GiB (\~920 GB) in total. Space left below is out of that \~920 GB.

| Model as compiled | Store size | Copies of routed experts | Space left |
| --- | --- | --- | --- |
| DeepSeek V4 Flash, native FP4 experts | \~150 GB | 2 (every expert on two drives) | \~620 GB |
| Qwen3.8-Flash-Next, Q5\_K | \~88 GB + n-gram tables | 1 (fully RAM-resident at run time) | room for others |
| GLM-5.3-Flash, Q6\_K | \~265 GB | 1 + replicas of the hottest 40% | \~560 GB |
| GLM-5.2, Q5\_K | \~510 GB | 1 + replicas of the hottest 15% | \~340 GB |
| GLM-5.2, Q5\_K, REAP 40% pruned (balanced) | \~310 GB | 1 + replicas of the hottest 60% | \~435 GB |

**Rules the placer follows**

1. **RAM first.** The pinned set is the highest-heat experts that fit the hardware profile's cache budget × `pin_frac`. Pinned experts still get one disk copy but no replica.
2. **Spread each layer.** Within a layer, experts with high co-activation go to different drives (graph colouring on the co-activation matrix), so a token's top-k reads hit all four queues at once.
3. **Balance by expected load, not by bytes.** Each drive's expected read rate = Σ heat × miss probability × size; greedy bin-packing keeps the four within 5% of each other, weighted by each drive's measured bandwidth.
4. **Replicate the warm middle.** Experts too cold to pin but hot enough to miss often get a second copy on the least-loaded other drive; the runtime reads from whichever queue is shorter.
5. **Order for prefetch.** Within a drive, packets are written layer-major, so the router-lookahead prefetch of layer l+1 touches nearby LBAs.
6. **Bundle only when it pays.** Experts under \~1 MiB that co-fire above a threshold are packed into `BUNDLE`s; above that size, separate packets read in parallel are faster.
7. **Libraries are allowed.** Several compiled models can share the drives; each has its own packet files per drive and its own manifest.

## Pruning and quality gates

Pruning is the one pass that changes the model, so it is opt-in, use-case specific and must pass a held-out gate. REAP is the method to use: it scores experts by router gate value times output norm, keeps the router's independent control over survivors, and its authors report near-lossless code generation and tool calling on Qwen3-Coder-480B and Kimi-K2 with 50% of experts removed ([REAP](https://arxiv.org/abs/2510.13999v1)). Merging experts is worse for generation, so `tlmc` does not offer it.

| Profile | What may change | Prune ratio | Gate on the held-out set |
| --- | --- | --- | --- |
| `exact` (default) | Nothing beyond the precision plan | 0 | Token-identical to the reference quantised model |
| `balanced` | Coldest experts drop to Q5\_K; REAP pruning | Up to 25% | Top-1 agreement ≥ 97%, mean KL ≤ 0.02, task score within 1 point |
| `aggressive` | As balanced | Up to 50% | Top-1 agreement ≥ 93%, mean KL ≤ 0.05, task score within 2 points |

- Pruning ratios are per layer; early and late layers usually tolerate less, so the compiler searches per layer under the global ratio.
- A pruned store records the pruned expert ids; the runtime renormalises router logits over survivors exactly as the compiler did during evaluation.
- Because pruning is fitted to your corpus, the manifest carries the corpus hash, and `tierlm` prints a warning when it is used for a different workload label.
- The original, unpruned packets can stay on the drives side by side if space allows, so switching back is a manifest change.

## Stage 2: what the runtime loads and checks

The runtime does no layout, quantisation or placement work; it validates the manifest against the machine, loads what the compiler said to load, and starts serving.

1. **Open and verify the manifest:** SHA-256 over the packet table, engine version, layout ids it has kernels for. Unknown layout → refuse with the exact `tlmc` command to rebuild.
2. **Check the machine against the hardware profile:** RAM, drives present and mounted, SIMD level. Small differences (less RAM, a slower drive) → recompute the pinned set and cache size at start-up from the stored heat. Missing drive → refuse, or run from replicas if every packet on that drive has one.
3. **Load resident packets:** `DENSE`, `MTP`, `META`, `EMBED` into a hugepage arena with O\_DIRECT reads across all four drives in parallel (\~10–20 GB in a few seconds), then `mlock`.
4. **Warm the cache:** read the pinned `EXPERT`/`BUNDLE` packets in drive-balanced order in the background; serving starts immediately and early misses fall through to normal streaming.
5. **Serve:** the expert store, io\_uring rings (one per drive), prefetcher, GPU KV backend and MTP described in the other tabs; the manifest's replica table lets the I/O layer pick the shorter queue per read.
6. **Learn, but don't rewrite:** the runtime updates `.warm` heat as it runs; `tlmc replan` can later fold that heat back into pinning and replicas. The packets themselves are never modified by the runtime.

Start-up target on this box: first token within \~10 s of launch for DeepSeek V4 Flash with a cold page cache.

## Command line

```sh
# once per machine (see the setup tool section below)
sudo tierlm-setup

# once per use case
tlmc corpus --from ~/agent-logs --calib 250k --eval 50k --out ~/tlm/coding.usecase.json

# stage 1: compile (resumable; --profile exact|balanced|aggressive)
tlmc compile deepseek-ai/DeepSeek-V4-Flash \
     --hw /etc/tierlm/hardware.conf --usecase ~/tlm/coding.usecase.json \
     --profile exact --experts native --dense q8_0 --mtp q8_0 \
     --drives /srv/tlm/nvme0,/srv/tlm/nvme1,/srv/tlm/nvme2,/srv/tlm/nvme3 \
     --replicas auto --out ~/tlm/dsv4f-coding.tlmm

tlmc report  ~/tlm/dsv4f-coding.tlmm      # sizes, placement per drive, eval results
tlmc replan  ~/tlm/dsv4f-coding.tlmm --hw ~/tlm/hw-new.json   # new machine, no re-quantise
tlmc list    /srv/tlm                     # compiled library on the drives

# stage 2: run
tierlm run   ~/tlm/dsv4f-coding.tlmm --chat
tierlm serve ~/tlm/dsv4f-coding.tlmm --port 8000
```

Both binaries come from the same `make`; `tlmc` is C as well, with an optional Python helper only for downloading from Hugging Face.

## Setup tool: `tierlm-setup` and `/etc/tierlm`

`tierlm-setup` probes the machine, recommends a value for every engine preference with the reason, lets you accept or change each one, and writes plain `key = value` files under `/etc/tierlm/`. The runtime and the compiler read the same files, so preferences are set once per machine, not per command.

**Files**

| Path | Contents | Written by | Mode |
| --- | --- | --- | --- |
| `/etc/tierlm/tierlm.conf` | Engine preferences (threads, cache, I/O, KV device, speculation, quantisation defaults, paths) | `tierlm-setup` | 0644 root:root |
| `/etc/tierlm/hardware.conf` | Measured facts: RAM, DRAM GB/s, CCDs, SIMD, per-drive GB/s and capacity, GPUs and VRAM; doubles as the compiler's hardware profile | `tierlm-setup --probe` | 0644 root:root |
| `/etc/tierlm/conf.d/*.conf` | Drop-in overrides, read in lexical order (e.g. `50-local.conf`) | You or config management | 0644 |
| `/etc/tierlm/system/` | Suggested `sysctl.d`, `limits.d` and `tuned` snippets; only installed with `--apply-system` | `tierlm-setup` | 0644 |
| `/etc/tierlm/tierlm.conf.bak.<timestamp>` | Previous version, kept on every write | `tierlm-setup` | 0644 |

**Precedence** (later wins): built-in defaults → `tierlm.conf` → `conf.d/*.conf` → `TIERLM_<KEY>` environment variables (dots become underscores) → command-line flags. `tierlm config show --origin` prints every effective value and which file or flag set it.

**Format**

```conf
# /etc/tierlm/tierlm.conf -- generated by tierlm-setup 1.0 on 2026-10-02
# one key per line; '#' starts a comment; later lines override earlier ones

cpu.threads            = 24          # 1 per physical core (probe: 24 cores, 4 CCDs)
cpu.thread_groups      = ccd
cpu.pin                = yes

memory.ram_reserve     = 12G
memory.expert_cache    = auto        # resolves to ~100G on this machine
memory.pin_frac        = 0.30
memory.hugepages       = thp

io.backend             = io_uring    # probe: kernel.io_uring_disabled = 0
io.direct              = yes
io.queue_depth         = 32
io.drives              = /srv/tlm/nvme0, /srv/tlm/nvme1, /srv/tlm/nvme2, /srv/tlm/nvme3
io.prefetch            = router

kv.device              = hip:0, hip:1
kv.vram_frac           = 0.05        # max 0.10
kv.dtype               = f16
kv.page_tokens         = 64

spec.mode              = mtp
spec.shape             = chain
spec.depth             = auto
spec.union_guard       = yes
spec.restrict_experts  = no          # lossy, opt-in

quant.experts          = q6_k
quant.dense            = q8_0
quant.mtp_head         = q8_0
compile.profile        = exact

paths.models           = /srv/tlm/models   # manifests
paths.kv_store         = /var/lib/tierlm/kv
paths.log              = /var/log/tierlm
server.listen          = 127.0.0.1:8000

include /etc/tierlm/conf.d/*.conf
```

Parser rules (one small C file, `cfg.h`, shared by both binaries):

- `key = value`; keys are lowercase dotted names; whitespace around `=` and at line ends is trimmed; `#` to end of line is a comment unless inside double quotes.
- Value types: integers with optional `K`/`M`/`G`/`T` suffix (powers of 1024); fractions; booleans `yes/no/on/off/true/false`; comma-separated lists; `auto`; strings, quoted only when they contain `#` or leading spaces.
- `include <glob>` pulls in files in lexical order; includes may not nest more than 4 deep.
- Unknown keys warn and are ignored; `--strict` turns warnings into errors. Out-of-range values are errors, reported with file, line and the allowed range.

**How the tool works**

1. **Probe** (read-only, no root needed): CPU topology and SIMD from `cpuid` and sysfs; DRAM bandwidth with a 10 s multi-threaded read test; each drive's sustained read for 60 s with O\_DIRECT; io\_uring availability; ROCm devices, VRAM and free VRAM via the HIP library if present. Writes `hardware.conf`.
2. **Recommend:** derives each preference from the probe using the rules in the other tabs (threads per CCD, cache size from RAM, KV budget from VRAM, quant defaults from RAM vs model sizes) and shows a table of key, recommended value, current value and the reason in one line.
3. **Select:** interactive prompts for each group (Enter accepts the recommendation), or `--auto` to accept all, or `--answers file.conf` for unattended installs.
4. **Validate and write:** builds the new file, runs the same parser and range checks the runtime uses, shows a diff, then writes atomically (temp file + `rename`) and keeps a timestamped backup. Writing to `/etc` needs root; everything before it does not.
5. **System suggestions:** prints, but does not apply, what the machine still needs (io\_uring sysctl, memlock limit, tuned profile, NVMe power state). `--apply-system` installs them into `/etc/sysctl.d/90-tierlm.conf` and `/etc/security/limits.d/90-tierlm.conf` after a confirmation.
6. **Re-run any time:** `tierlm-setup --probe` after a hardware change refreshes `hardware.conf` and flags preferences that no longer fit (for example a cache larger than the new RAM).

```sh
sudo tierlm-setup                  # probe, recommend, prompt, write /etc/tierlm/tierlm.conf
sudo tierlm-setup --auto           # accept all recommendations
tierlm-setup --probe --dry-run     # show recommendations, write nothing
tierlm config check                # validate all files, exit non-zero on error
tierlm config show --origin        # effective values and where each came from
sudo tierlm-setup --apply-system   # also install sysctl and limits snippets
```

RHEL notes: files under `/etc/tierlm` get the default `etc_t` SELinux label; the drive mount points and `/var/lib/tierlm` need `restorecon` after creation if a systemd service runs tierlm under a confined domain. The package ships `tierlm.conf` as `%config(noreplace)` so upgrades never overwrite your choices.

Plan impact: `cfg.h` and the probe land in M1 (+0.5 week); the recommendation rules grow with each milestone as features arrive.

The setup tool also has an **Experts and content** section for choosing which content types to keep, at what depth, and which to exclude: Decision engine

## Changes to the milestone plan

The split mostly moves work rather than adding it: conversion, placement and warm-start logic leave the runtime and become compiler passes. Net addition is about four weeks, mainly calibration tracing, pruning and the placer.

| Milestone | Change | Extra time |
| --- | --- | --- |
| M1 Foundations | Packet header and manifest replace `.tlmf`; `tlmc` passes 1, 2, 6, 7 (ingest, reference quantise, pre-build, segment) in C; Python only for downloads | +1 week |
| M2 Oracle bring-up | Runtime loads packets via the manifest from day one; oracle compares against pass 2's reference store | 0 |
| M4 Expert streaming | Pass 3 calibration tracing; pass 8 placer (co-activation spreading, load balancing, replicas); runtime replica-aware reads; `tlmc replan` | +2 weeks |
| M5 Flagship | Compile DeepSeek V4 Flash with the coding use case; pass 9 validation gate in CI | 0 |
| M6 Ship v1 | Pass 4 Q5\_K heat plan and pass 5 REAP pruning behind `balanced`/`aggressive`; `tlmc report` | +1 week |

This also settles the earlier open question on the converter: it becomes C (`tlmc`), so the whole toolchain builds with `make`.

## Sources

- [REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression (arXiv 2510.13999)](https://arxiv.org/abs/2510.13999v1); [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)
- [LMSYS: SGLang SSD Expert Pack](https://lmsys.org/blog/2026-08-29-sglang-ssd-expert-pack/) (offline expert-major repack, aligned O\_DIRECT reads)
- [JustVugg/colibri](https://github.com/JustVugg/colibri) (expert atlas, learned pins and their overfitting, REAP-pruned DeepSeek V4 Flash 150B, multi-SSD mirrors)
- [Schero94/slipstream](https://github.com/Schero94/slipstream) (sidecar expert file, resumable converter with CRC journal)
- [giannisanni/pulsar](https://github.com/giannisanni/pulsar) (warm-start sidecar from a popularity census)
