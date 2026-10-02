# tierlm on the WRX90 workstation

Oct 2, 2026 · @Jay

## At a glance

On this box the engine is RAM-bandwidth bound first and disk bound second: plan around \~200–250 GB/s of real DRAM bandwidth (not the 409.6 GB/s on the spec sheet), \~110 GB of RAM for weights and expert cache, and four NVMe drives read in parallel.

- **Build:** `make PROFILE=wrx90`, then sudo tierlm-setup to write /etc/tierlm/tierlm.conf — GCC 14, `-march=znver5`, AVX-512 VNNI/BF16 kernels, io\_uring + O\_DIRECT, 24 compute threads grouped by CCD, optional ROCm KV backend loaded at runtime with `dlopen` so the core stays plain C.
- **Weights stay in RAM/NVMe; only the KV cache goes to the GPUs.** The CPU computes Q/K/V, ships a few KB per layer to the R9700 that owns that layer's KV pages, and gets the attention output back. 5% of each card (1.6 GB) is the default, 10% (3.2 GB) the cap.
- **Speculation:** the model's own MTP head at Q8\_0, drafting a chain (not a tree) of adaptive depth 0–4, with draft length cut whenever the verify step would pull too many new experts from disk. Exact: output is token-identical to non-speculative decoding.
- **Quantisation:** Q6\_K routed experts by default (Q5\_K when a model only fits that way), Q8\_0 for attention, shared experts and the MTP head; models whose experts ship natively in FP4 (DeepSeek V4 Flash, Kimi K3) keep FP4, because Q6 would only add bytes.
- **Best flagship for this box:** DeepSeek V4 Flash (dense part resident, \~65% of experts cached, rest streamed), estimated 8–12 tok/s before MTP. Qwen3.8-Flash-Next at Q5\_K fits fully in RAM for a faster daily driver.
- **Two things to fix on RHEL first:** io\_uring is disabled by default in RHEL 10 (tech preview), and ROCm 7.2's support matrix lists RHEL 10.1, not 10.2, for the R9700.

## Hardware facts and what they mean

The 9965WX's four CCDs, not its eight memory channels, set the bandwidth ceiling, so thread placement matters more than on a high-core-count part.

| Component | Fact | Consequence for tierlm |
| --- | --- | --- |
| [Threadripper PRO 9965WX](https://www.techspot.com/review/3020-amd-ryzen-threadripper-9980x-9970x/) | 24 Zen 5 cores in 4 CCDs × 6; 128 MB L3 (32 MB per CCD); full-width AVX-512 | 24 compute threads, 1 per core, in 4 groups of 6; AVX-512 VNNI/BF16 kernels are the default path |
| Memory: 8-channel DDR5-6400 ECC RDIMM, 128 GB | 409.6 GB/s theoretical; on Threadripper, real bandwidth follows CCD count, and 7000-series 8-channel boxes measured \~245 GB/s ([Level1Techs](https://forum.level1techs.com/t/when-is-8-memory-channels-not-8-memory-channels/239685)) | Plan for \~200–250 GB/s; all 4 CCDs must read at once to reach it; `doctor` measures it at install |
| [ASUS Pro WS WRX90E-SAGE SE](https://www.asus.com/us/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/techspec/) | 6 × PCIe 5.0 x16 + 1 × x16 (x8 mode); 4 onboard Gen5 x4 M.2; bifurcation per ASUS table | The riser card needs its slot set to x4x4x4x4; \~4 × 12–14 GB/s if the drives are Gen5, \~4 × 7 GB/s if Gen4 |
| 4 NVMe on the riser | Independent devices, separate from the OS drive | One io\_uring ring and one expert shard per drive; no RAID layer needed |
| 2 × [Radeon AI PRO R9700](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) | 32 GB each, gfx1201 (RDNA 4), supported by ROCm | 1.6–3.2 GB per card for paged KV; compute shared with your other models, so latency jitter is possible |
| RHEL 10.2 | io\_uring is a tech preview and off by default ([RHEL-65347](https://issues.redhat.com/browse/RHEL-65347)); ROCm 7.2 lists RHEL 10.1 | Set `kernel.io_uring_disabled`; check the ROCm matrix or run the ROCm part from the 10.1 packages |

Assumption: the 128 GB is 8 × 16 GB, one DIMM per channel, running at 6400. If it is 4 × 32 GB, only four channels are populated and bandwidth halves; `doctor` will flag it.

## Which models fit, and how fast

With KV on the GPUs, about 110 GB of the 128 GB is usable for weights and expert cache (OS and other processes \~8 GB, scratch \~4 GB, headroom \~6 GB). Models under that run fully resident; larger ones stream their experts from the four drives.

| Model | Total / active | Expert format | Size | Placement | Estimated decode, no MTP |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B | 35B / 3B | Q6\_K | \~29 GB | Fully resident | 30–50 tok/s |
| Qwen3.8-Flash-Next | 125B / 6B (+ n-gram table on disk) | Q5\_K (Q6\_K is \~103 GB, too tight) | \~88 GB | Fully resident | 15–25 tok/s |
| DeepSeek V4 Flash | 284B / 13B | Native FP4 experts, Q8\_0 dense | \~150–165 GB | Dense resident, \~65% of experts cached | 8–12 tok/s |
| GLM-5.3-Flash | 321B | Q6\_K from the FP8 checkpoint | \~265 GB | \~40% cached | 3–6 tok/s |
| GLM-5.2 | 744B / 40B | Q5\_K | \~510 GB | \~20% cached | 1–2.5 tok/s |

These are engineering estimates, not measurements: RAM time = active bytes ÷ \~130 GB/s effective, plus disk time = missed expert bytes ÷ \~24 GB/s (Gen4) to \~45 GB/s (Gen5) across four drives, with partial overlap. Per-token expert traffic comes from the projects' own figures (colibri for DeepSeek V4 and GLM-5.2, apus for GLM-5.3-Flash). Replace each row with `tierlm bench` results in M4/M5.

DeepSeek V4 Flash is the recommended flagship here: its experts are already 4-bit by training, its per-token expert traffic is low, and the dense part fits easily, so it is the only 280B+ model that should stay interactive on this box.

## Default build profile: `make PROFILE=wrx90`

One profile file sets compiler flags, SIMD path, I/O backend and the optional GPU KV backend; the core binary still needs only libc, pthreads and liburing.

```make
# profiles/wrx90.mk  -- Threadripper PRO 9965WX, RHEL 10.2, 2 x R9700
CC        ?= gcc                      # GCC 14 (RHEL 10 default) or clang >= 19
ARCHFLAGS  = -march=znver5 -mtune=znver5 -mprefer-vector-width=512
OPTFLAGS   = -O3 -fno-math-errno -fno-trapping-math -fno-semantic-interposition
# no -ffast-math: reassociation breaks the bitwise invariance tests
CFLAGS    += -std=c11 -pthread $(ARCHFLAGS) $(OPTFLAGS) -flto=auto
CPPFLAGS  += -DTLM_SIMD=TLM_AVX512_VNNI_BF16 \
             -DTLM_IO=TLM_IO_URING -DTLM_IO_DIRECT=1 \
             -DTLM_HUGEPAGES=TLM_THP_MADVISE \
             -DTLM_NUMA=0                  # BIOS NPS1: one node
LDLIBS    += -luring
# GPU KV backend: tiny HIP library, built separately, loaded with dlopen()
HIP        = 1
HIPCC     ?= /opt/rocm/bin/hipcc
HIPFLAGS   = -O3 --offload-arch=gfx1201 -fPIC -shared
KVLIB      = libtlm_kv_hip.so          # absent at runtime => KV falls back to RAM
```

- AVX-512 is the primary path; the dispatcher still carries AVX2 and scalar kernels so one binary runs anywhere, and `make PROFILE=generic` drops `-march=znver5`.
- `-flto` and `-fno-semantic-interposition` help the hot matvec loops inline across headers.
- The HIP part is one `.hip` file exposing a C ABI (`kv_open`, `kv_append`, `kv_attend`, `kv_close`); the engine `dlopen`s it, so a machine without ROCm builds and runs unchanged.

## Runtime defaults

`sudo tierlm-setup` proposes these after probing the machine and writes them to `/etc/tierlm/tierlm.conf` in plain `key = value` form; any key can be overridden in `/etc/tierlm/conf.d/`, by a `TIERLM_` environment variable or on the command line. The full file format and the setup tool are in the Two-stage design tab.

```conf
# /etc/tierlm/tierlm.conf -- recommended for the WRX90 / 9965WX workstation
cpu.threads            = 24          # one per physical core; SMT siblings left for I/O and prefetch
cpu.thread_groups      = ccd         # 4 groups x 6 cores; each routed expert runs inside one CCD
cpu.pin                = yes

memory.ram_reserve     = 12G         # OS, other processes, scratch
memory.expert_cache    = auto        # ~95-105G on this box after dense weights
memory.pin_frac        = 0.30        # share of cache for the hot set
memory.hugepages       = thp         # madvise + MAP_POPULATE + mlock on the slot pool

io.backend             = io_uring
io.direct              = yes
io.queue_depth         = 32          # per drive
io.drives              = /srv/tlm/nvme0, /srv/tlm/nvme1, /srv/tlm/nvme2, /srv/tlm/nvme3
io.prefetch            = router      # run layer l+1's router on layer l's output

kv.device              = hip:0, hip:1   # layers split half per card; "cpu" to disable
kv.vram_frac           = 0.05        # per card; hard cap 0.10
kv.dtype               = f16         # f8_e4m3 doubles the context that fits
kv.page_tokens         = 64

spec.mode              = mtp
spec.shape             = chain
spec.depth             = auto        # 0..4, chosen per step by measured utility
spec.union_guard       = yes         # trim the draft when it would pull too many experts from disk
spec.restrict_experts  = no          # lossy AcceptMoE-style mode, opt-in only

quant.experts          = q6_k        # q5_k when q6_k does not fit; native fp4 kept as-is
quant.dense            = q8_0
quant.mtp_head         = q8_0
```

## KV cache in 5–10% of the R9700s

Keep the weights on the CPU side and move only attention's state and arithmetic to the GPUs: the CPU sends each layer's new q, k, v (a few KB), the card that owns that layer's KV pages appends and attends, and returns one hidden-size vector.

&#91;embedded content: per-layer attention offload · CPU and R9700\]

Shipping the KV cache back and forth would cost more than it saves; shipping activations costs a few KB per layer, so the GPU's bandwidth does the part of decode that grows with context.

**Budget and capacity** (per token, F16 KV; FP8 halves it)

| Model attention | KV per token | Fits in 3.2 GB (5% of both cards) | Fits in 6.4 GB (10% of both cards) |
| --- | --- | --- | --- |
| GQA, 48 layers × 4 KV heads × 128 (Qwen3-30B-A3B class) | 96 KiB | \~33k tokens | \~66k tokens |
| MLA latent, 576 values × 43 layers (DeepSeek-style) | \~48 KiB | \~65k tokens | \~129k tokens |
| Hybrid linear attention (Qwen3.6/3.8 class) | KV only on the few full-attention layers; recurrent state is small | most contexts | most contexts |

**Spec**

- `libtlm_kv_hip.so` allocates one fixed arena per card at start (`vram_frac × 32 GB`) and never grows; if the allocation fails, tierlm logs it and keeps KV in RAM.
- Layers split across the two cards: layers 0..L/2−1 on `hip:0`, the rest on `hip:1`. No peer-to-peer traffic.
- Pages of 64 tokens, block table per sequence (paged attention), so prefix reuse and the on-disk KV snapshot work page by page.
- Per layer: one async H2D of q,k,v from a pinned host buffer, one kernel (append + attention, GQA or MLA-absorbed variant), one D2H of o. Kernels launched through a pre-built HIP graph per layer to keep launch overhead to a few microseconds.
- Expected overhead: \~20–40 µs per layer round trip, about 1–2 ms per token at 48 layers, under 5% at 20 tok/s. Prefill sends whole chunks, so the overhead amortises.
- Shared GPUs: HIP stream at high priority; your other models keep 90–95% of VRAM. Expect some latency jitter when they are busy; `--stats` reports per-layer GPU wait.
- Save/restore: `/save` copies pages to the disk KV store so sessions reopen without re-prefill.

## MTP speculation: the variant we ship

The best variant for a CPU-bound MoE with streamed experts is the model's own MTP head, drafting a short chain whose length adapts every step and is capped by how many new experts the verify pass would touch.

**Why not the GPU-favoured options**

- On GPUs, DFlash and EAGLE-3 beat native MTP: DFlash reports \~4.9× average speedup over autoregressive and 2.4× over EAGLE-3 at equal draft budget ([DFlash](https://arxiv.org/html/2602.06036v2)). Both need a separate draft checkpoint (DFlash 3–6 GB), which does not fit a 5–10% VRAM budget and would run slowly on the CPU.
- On MoE, verification cost is the union of experts touched by all draft positions. Bigger EAGLE-3 trees made GLM-4.7-Flash slower than smaller ones ([benchmark](https://medium.com/@ineshreddy249/bigger-draft-trees-hurt-on-moe-benchmarking-eagle3-vs-mtp-on-glm-4-7-flash-74591c39a873)). With experts offloaded, standard speculative decoding was slower than plain decoding on 7 of 8 Qwen pairs and spent 65–78% of wall time waiting for expert weights ([AcceptMoE](https://arxiv.org/abs/2608.02989)). colibri measured MTP at −32% around an 85% expert hit rate.

**What we ship**

1. **Drafter:** the checkpoint's native MTP/NextN layer, stored at Q8\_0. colibri found int4 MTP heads collapse acceptance to 0–4%, so the Q5/Q6 preference never applies to the head.
2. **Shape:** a single chain, no tree. Depth chosen per step from 0 to 4.
3. **Depth controller:** keeps running estimates of acceptance per position, draft time and verify time, and picks the depth that maximises expected tokens per second; depth 0 (speculation off) whenever the best expected gain is under 5%.
4. **Union guard (lossless):** after drafting, run each draft position's router (cheap, already needed for prefetch), count routed experts not in RAM, and cut the chain at the first position where missing expert bytes exceed a budget (default: what the drives can deliver in 30% of one decode step). Every emitted token is still the target model's own, so output is unchanged.
5. **Verification:** one batched forward over the accepted chain. Dense weights are read once for all positions, which is why verifying 3–4 tokens on a memory-bound CPU costs much less than 3–4 forwards. Greedy: exact match. Sampling: standard speculative rejection sampling, distribution-exact.
6. **Prefetch hand-off:** the draft's routed experts are queued to the I/O threads before verification starts, so misses overlap the dense part of the verify pass.
7. **Opt-in lossy mode:** `restrict_experts = on` applies AcceptMoE-style verifier expert sets (commitment-weighted, self-sizing, residency-aware). AcceptMoE reports 2.06× under offloading with 73.6–77.1% less expert traffic, at a mean accuracy cost of 0.27 points; it changes the model's distribution, so it stays off by default.

Expected effect on this box: +20–60% decode on resident models (Qwen3.6/3.8), +10–35% on DeepSeek V4 Flash depending on hit rate, and automatic switch-off when it does not pay.

## Quantisation recipe for a Q5/Q6 preference

Q6\_K is the default for routed experts converted from BF16 or FP8 checkpoints; everything that steers the model (router, attention, MTP head) stays at Q8\_0 or higher, and experts that were trained in FP4 are never re-quantised upward.

| Tensor group | Format | Bits per weight | Reason |
| --- | --- | --- | --- |
| Routed experts, BF16/FP8 source | Q6\_K (default) | 6.56 | Near-lossless; the bulk of bytes, so it sets RAM fit and disk traffic |
| Routed experts, when Q6\_K does not fit | Q5\_K + imatrix | 5.5 | 16% fewer bytes than Q6\_K; imatrix matters more at 5 bits |
| Routed experts, native FP4/MXFP4 (DeepSeek V4, Kimi K3) | Keep native | \~4.25 | Trained at 4 bits; Q6 would add \~50% bytes for no quality |
| Attention, shared experts, dense FFN, embeddings, lm\_head | Q8\_0 | 8.5 | Read every token; small share of bytes; errors here compound |
| MTP / NextN head | Q8\_0 | 8.5 | Low-bit heads collapse draft acceptance |
| Router, norms, biases | F32 / BF16 | 16–32 | Routing must match the reference exactly |
| KV cache (on GPU) | F16, or FP8 E4M3 | 16 / 8 | FP8 doubles context in the same VRAM |

Kernel work this adds to M3: Q5\_K and Q6\_K dot products against Q8\_K activations with `vpdpbusd` (AVX-512 VNNI), with the 5th/6th bit planes unpacked in registers. The converter stores each expert's blocks pre-interleaved for 16-row tiles so the inner loop does aligned 64-byte loads with no repacking at run time.

## The four NVMe drives on the riser

Treat the four drives as four independent devices and let the engine stripe at expert granularity; no md RAID is needed, and each drive gets its own queue.

- **Riser slot:** set the slot to x4x4x4x4 bifurcation in BIOS (ASUS publishes the per-slot table). Put the riser in a slot not shared with either GPU so expert reads and GPU traffic do not contend.
- **Filesystem:** one XFS per drive, mounted `noatime`, holding one shard file each (`model.tlmx.0` … `.3`). The dense part and metadata live on the OS drive and are loaded once into RAM.
- **Placement:** the converter assigns each expert record whole to drive `(layer × n_experts + expert) mod 4`, so the k experts of a layer spread across all four drives and are read in parallel. Records stay 4 KiB aligned for O\_DIRECT.
- **I/O:** one io\_uring ring per drive, registered buffers (the slot pool) and registered files, queue depth 32 per drive; completions reaped by one I/O thread on an SMT sibling.
- **Hot copy (optional):** if the drives have spare space, `tierlm mirror` copies the top 20% of experts by `.warm` heat to a second drive so reads can go to whichever queue is shorter.
- **Checks in `doctor`:** sustained 60 s read per drive (catches thermal throttling on Gen5 drives without heatsinks), and a 4-drive aggregate. Target ≥ 85% of the `fio` aggregate on 4–16 MB reads.

## RHEL 10.2, BIOS and kernel checklist

Most of these are one-time settings; `tierlm doctor` checks each and prints the fix.

**BIOS**

- [ ] Memory at DDR5-6400, one DIMM per channel, all 8 channels populated
- [ ] NPS1 (one NUMA node); L3-as-NUMA off
- [ ] Riser slot bifurcation x4x4x4x4; Above 4G decoding and Resizable BAR on (for the R9700s)
- [ ] Power/determinism set to performance; PCIe ASPM off for the riser slot

**RHEL 10.2**

- [ ] Enable io\_uring (tech preview, off by default): `sysctl -w kernel.io_uring_disabled=0`, or `=1` plus `kernel.io_uring_group=<tierlm group>`; persist in `/etc/sysctl.d/90-tierlm.conf`
- [ ] `dnf install liburing-devel` (or vendor liburing in-tree if not shipped)
- [ ] `tuned-adm profile throughput-performance`
- [ ] THP set to `madvise`; raise `memlock` limit for the tierlm user so the slot pool can be `mlock`ed
- [ ] NVMe I/O scheduler `none` (default) and `nvme_core.default_ps_max_latency_us=0` to keep the drives out of deep power states
- [ ] ROCm: install the 7.2.x packages for RHEL 10.1 if 10.2 is not yet listed, or run only `libtlm_kv_hip.so` against a ROCm container; KV falls back to RAM if HIP fails to load
- [ ] Cap tierlm's GPU use so it cannot starve your other models: fixed arena allocated at start from `vram_frac`, no growth afterwards
- [ ] Run `tierlm doctor` and keep its JSON with every benchmark

## Changes to the milestone plan

This box adds about five weeks, mostly the GPU KV backend and MTP, which the generic plan had deferred past v1.

| Milestone | Added for this box | Extra time |
| --- | --- | --- |
| M1 Foundations | `profiles/wrx90.mk`; `doctor` measures DRAM bandwidth, per-drive sustained reads, CCD layout, io\_uring availability | +0.5 week |
| M3 Fast CPU kernels | Q5\_K/Q6\_K × Q8\_K AVX-512 VNNI kernels; CCD-grouped thread pool | +1 week |
| M4 Expert streaming | 4-drive expert-granular striping, per-drive rings, optional hot mirror | +0.5 week |
| M5 Flagship (DeepSeek V4 Flash) | `libtlm_kv_hip.so`: paged KV, attention kernels for GQA and MLA on gfx1201, layer split across two cards, CPU fallback | +2 weeks |
| M6 Ship v1 | Native MTP chain with depth controller and union guard; lossy expert-restriction mode behind a flag | +1.5 weeks |

Gates added: KV-on-GPU logits within 1e-3 of KV-in-RAM and greedy tokens identical on the test suite (GPU attention sums in a different order, so bitwise equality holds only within one device); MTP on vs off token-identical; DeepSeek V4 Flash ≥ 8 tok/s warm without MTP on this box.

## Sources

- [TechSpot: Threadripper 9000 review (CCD counts)](https://www.techspot.com/review/3020-amd-ryzen-threadripper-9980x-9970x/)
- [Level1Techs: When is 8 memory channels not 8 memory channels?](https://forum.level1techs.com/t/when-is-8-memory-channels-not-8-memory-channels/239685)
- [ASUS Pro WS WRX90E-SAGE SE specifications](https://www.asus.com/us/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/techspec/)
- [ROCm system requirements (R9700, gfx1201, RHEL versions)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- [AMD R9700 Linux drivers (RHEL 10.1, ROCm 7.2)](https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-ai-pro/radeon-ai-pro-r9000-series/amd-radeon-ai-pro-r9700.html)
- [RHEL-65347: io\_uring tech preview, disabled by default in RHEL 10](https://issues.redhat.com/browse/RHEL-65347)
- [AcceptMoE (arXiv 2608.02989)](https://arxiv.org/abs/2608.02989)
- [DFlash (arXiv 2602.06036)](https://arxiv.org/html/2602.06036v2)
- [Bigger draft trees hurt on MoE: EAGLE-3 vs MTP on GLM-4.7-Flash](https://medium.com/@ineshreddy249/bigger-draft-trees-hurt-on-moe-benchmarking-eagle3-vs-mtp-on-glm-4-7-flash-74591c39a873)
- [vLLM: speculative decoding on AMD GPUs](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)
- [JustVugg/colibri](https://github.com/JustVugg/colibri) (MTP int8 finding, expert traffic figures); [apus-glm5.3-flash](https://github.com/bricesommers/apus-glm5.3-flash) (GLM-5.3-Flash traffic)
