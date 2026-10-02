# Fragments: flash-moe (SSD-streamed 397B MoE on a 48 GB Apple laptop)

All pseudocode below is written fresh after reading the cited files; names are ours.
Numbers are the project's own measurements on its own hardware (M3 Max, 48 GB unified
memory, ~17.5 GB/s internal NVMe, Qwen3.5-397B-A17B, 60 layers x 512 experts).
The project has no licence file, so nothing here is copied: no code, comments or
project-specific identifiers. Where the paper and the logs disagree on a number,
both are given.

Model shape used throughout: hidden 4096, 60 layers (45 gated-delta linear-attention,
15 full-attention at every 4th layer), 512 routed experts + 1 shared expert per layer,
expert FFN 4096 -> 1024 -> 4096 with SwiGLU, affine 4-bit weights with group size 64
and bf16 scale/bias per group.

---

## F-storage-01: Offline index of where each expert slice lives in the source shards
Origin: [ds4] [pulsar] [ktransformers] [flashmoe]

Licence-of-origin: Apache-2.0 (ktransformers), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: flash-moe (no licence) expert_index.json, repack_experts.py -- read only, not copied
- Topic: storage
- Purpose: let the repacker locate one expert's slice of each stacked tensor with plain arithmetic instead of parsing every shard again.
- Inputs / outputs: input: safetensors shards where each layer's experts are stored as stacked tensors [n_experts, rows, cols]; output: JSON map layer -> component -> {shard file, absolute byte offset of expert 0, per-expert stride, per-expert size, total size, shape}.
- Pseudocode:
      index = {model_dir, per_layer: {}}
      for each shard:
          header, data_start = read safetensors header
          for each tensor name matching "<layer>.<mlp experts>.<proj>.<weight|scales|biases>":
              n_exp = shape[0]
              per_exp = tensor_bytes / n_exp
              index.per_layer[layer][proj.part] = {
                  file, abs_offset = data_start + tensor_begin,
                  stride = per_exp, size = per_exp, total = tensor_bytes, shape }
      # offset of expert e for that component:
      src_off(layer, comp, e) = index[layer][comp].abs_offset + e * index[layer][comp].stride
- Notes: nine components per expert (gate/up/down x weight/scales/biases). Stride equals size because the source stacks experts densely. The index is generated once and is tied to a specific local snapshot path. Components of one expert are scattered across (possibly different) shards, which is why repacking is needed: reading an expert straight from safetensors would be nine scattered reads.
- Variants seen elsewhere: other engines keep safetensors and coalesce only when gate/up/down happen to be adjacent; this project always repacks.

## F-storage-02: One flat file per layer, fixed-size expert records, verify-as-you-go
Origin: [colibri] [monolite] [slipstream] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (slipstream), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: flash-moe (no licence) repack_experts.py, metal_infer/repack_experts_2bit.py, paper/flash_moe.tex -- read only, not copied
- Topic: storage
- Purpose: make "load expert e of layer l" exactly one positioned read with no seek table.
- Inputs / outputs: input: the index from F-storage-01; output: 60 files, each n_experts * RECORD bytes, plus a small layout JSON (record size, component offsets/sizes/dtypes/shapes).
- Pseudocode:
      COMPONENTS = ordered list [(gate.w, U32,[1024,512]), (gate.s, BF16,[1024,64]), (gate.b, ...),
                                 (up.w ...), (up.s ...), (up.b ...),
                                 (down.w, U32,[4096,128]), (down.s, BF16,[4096,16]), (down.b ...)]
      assign each component a running offset; RECORD = sum of sizes      # 7,077,888 B at 4-bit
      check every index entry's per-expert size == expected size, else abort
      check free disk >= layers * n_experts * RECORD, else abort with a hint
      for layer in selected_layers:
          out = create file; preallocate n_experts * RECORD
          plan = [(src_fd, src_off(layer,c,e), e*RECORD + c.offset, c.size) for e, c]
          sort plan by (src_fd, src_off)          # sequential reads over the source
          for each step: data = positional_read(src); assert full; positional_write(out, data)
          spot-verify experts {0, 1, mid, last}: every component byte-equal to source; abort on mismatch
      reader: offset(e) = e * RECORD   (single file descriptor per layer)
- Notes: 3.63 GB per layer at 4-bit, 209 GB total; 3,932,160 B per expert (120 GB total) at 2-bit. No header, no padding, no alignment beyond what the fixed sizes give (all component sizes are multiples of 128 KiB). Supports layer subsets so the repack can be done in chunks when disk is short. Layer-major order means one fd per layer.
- Variants seen elsewhere: monolite/slipstream add a header, 4-16 KiB alignment and per-record CRC; flash-moe relies on fixed constants compiled into the engine.

## F-format-01: Non-expert weights as one mmap'd blob with a JSON manifest
Origin: [ds4] [monolite] [slipstream] [ktransformers] [flashmoe]

Licence-of-origin: Apache-2.0 (monolite, ktransformers), MIT (ds4, slipstream), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/extract_weights.py, metal_infer/infer.m -- read only, not copied
- Topic: format
- Purpose: keep everything that is touched every token (embeddings, attention, norms, router, shared expert, LM head; ~5.5 GB) resident and addressable by both CPU and GPU.
- Inputs / outputs: input: safetensors shards minus expert tensors; output: one binary file (tensors packed, 64-byte aligned, native dtype U32/BF16/F32) and a manifest name -> {offset, size, shape, dtype}.
- Pseudocode:
      writer: for tensor in non_expert_tensors (sorted):
                  pos = round_up(pos, 64); write bytes at pos; manifest[name] = {pos, size, shape, dtype}
      loader: base = mmap(file, read-only)
              build open-addressed hash table (power-of-two size > 4x tensor count) keyed by name hash
              wrap the whole mapping as ONE zero-copy GPU buffer
              per layer, resolve pointers once into a struct of raw pointers (q/k/v/o, norms, router, shared)
              GPU kernels address a tensor as (whole-blob buffer, byte offset = ptr - base)
- Notes: mmap is the right choice here because the blob fits in RAM and stays resident; the same project found mmap catastrophic for the cold expert files (F-io-03). Resolving pointers once per layer avoids per-token string lookups.
- Variants seen elsewhere: GGUF-style single file with tensor directory; here weights and experts are deliberately separate files.

## F-io-01: Parallel positional reads of the K routed experts
Origin: [colibri] [ds4] [monolite] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/infer.m, metal_infer/main.m, metal_infer/results.tsv, paper/flash_moe.tex -- read only, not copied
- Topic: io
- Purpose: fetch K expert records for a layer as fast as the SSD allows, straight into GPU-visible buffers.
- Inputs / outputs: layer fd, K expert ids, K destination buffers of RECORD bytes; output: per-slot valid flag.
- Pseudocode:
      # variant A: fire-and-wait with a dispatch group (used on the default path)
      start(fd, ids, dst):
          for k in 0..K-1: group_async(high_priority_queue, { res[k] = pread(fd, dst[k], RECORD, ids[k]*RECORD) })
      ... do work that does not need expert bytes (copy shared-expert inputs into GPU buffers) ...
      wait(): group_wait(forever); valid[k] = (res[k] == RECORD)

      # variant B: persistent pool of 4 workers (no create/join per layer)
      pool: mutex, cv_work, cv_done, generation counter, task array
      dispatch(tasks): lock; publish tasks; generation++; broadcast cv_work
                       wait on cv_done until all 4 workers reported; unlock
      worker(i): loop { wait until generation changes; for t in tasks[i::4]: t.res = pread(...);
                        lock; done++; if done == 4 signal cv_done }
      on short read: mark slot invalid, warn, give it zero weight in the combine
- Notes: one thread per expert (K=4 -> 4 threads). Parallel 4-way pread measured ~9x faster than sequential (NVMe queue depth). Persistent pool vs per-call dispatch_apply: 3.81 vs 3.97 ms/layer, 4.28 vs 4.09 tok/s (kept). An async variant that starts the reads right after routing only overlaps ~0.1 ms of CPU work, because nothing else useful can run before the expert compute (routing is the last step before I/O). Per-layer I/O budget at 4-bit: 4 x 6.75 MB, ~2.4 ms mixed hit/miss (1.0 ms all-warm, 5.8 ms all-cold); 2-bit: ~1.4-1.5 ms.
- Variants seen elsewhere: io_uring batch submission on Linux; here plain pread on threads was best (dispatch_io -70%, aio about -7% / equal).

## F-io-02: 2 MB-aligned destination buffers wrapped as zero-copy GPU buffers
Origin: [ds4] [apus] [flashmoe]

Licence-of-origin: MIT (ds4), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/infer.m, docs/io-and-gpu-exploration.md, paper/flash_moe.tex -- read only, not copied
- Topic: io
- Purpose: let the kernel copy/DMA page-cache data into the read buffer in large bursts and let the GPU read the same memory with no copy.
- Inputs / outputs: K slots (and a second set B for prediction experiments), each RECORD bytes rounded up to the page size.
- Pseudocode:
      for k in 0..MAX_K-1:
          raw = aligned_alloc(alignment = 2 MiB, size = round_up(RECORD, page))
          slot[k] = gpu_buffer_wrapping(raw, shared storage, no copy, no free-on-release)
      kernels bind slot[k] with byte offsets of each component inside the record
- Notes: isolated test on page-cache-resident data: 234 us / 16.8 GB/s with 2 MB alignment vs 836 us / 4.7 GB/s with 16 KB alignment (3.6x). End-to-end only ~5% because most time is real SSD latency. Free to do. 2 MB superpages themselves are not available on that OS (fixed 16 KB pages).
- Variants seen elsewhere: O_DIRECT bounce buffers aligned to 4 KiB on Linux.

## F-cache-01: "Trust the OS" -- no application expert cache at all
Origin: [flashmoe]

Licence-of-origin: no-licence (flashmoe)
- Source: flash-moe (no licence) CLAUDE.md, docs/io-and-gpu-exploration.md, paper/flash_moe.tex, results.tsv -- read only, not copied
- Topic: cache
- Purpose: decide where hot experts live; the answer measured best was "only in the kernel page cache".
- Inputs / outputs: none at runtime -- the design is the absence of a cache.
- Pseudocode:
      open each layer file normally (no cache-bypass flag, no advisory hints)
      every token, every layer: pread the K records into fixed scratch slots
      engine resident set stays fixed (~6 GB): weights blob + scratch + KV + linear-attn state
      leave the remaining ~35-42 GB to the OS page cache, which keeps the hot experts
- Notes: page cache gives ~71% expert hit rate at 4-bit (35 GB cache vs 209 GB experts). Measured history of the alternatives: GPU-buffer LRU of 500 entries (3.5 GB) 3.14 tok/s vs 2.86 with no cache (small win early on); 1000 entries 2.24; 2500 entries (9.8 GB) 2.24; 3000 entries (21 GB) 1.99; malloc zero-copy cache 2581 entries (18 GB, 52% hit) 2.10; 500 entries (3.5 GB, 37% hit) 2.77. Final comparison at 2-bit: GPU LRU + F_NOCACHE 4.11 tok/s, GPU LRU alone 4.80, no cache 5.74 (+38%). Root cause: GPU-visible shared buffers are wired, shrink the page cache, and pushed the memory compressor to 60k-130k decompressions/s (~1-2 GB/s of bandwidth) -- near zero once removed. The OS policy (recency+frequency) beat strict LRU, and a page-cache hit has no lookup cost. A later log entry shows an indexed 2500-entry GPU cache beating 1500 entries (4.35 vs 3.98 tok/s; 3500+ regressed) but still below the no-cache 5.74. Tiered I/O (cache-bypass fd for first-ever reads, normal fd for repeats, tracked with a per-layer bitset) gave marginal tok/s and identical memory pressure; dropped. Routing statistics behind this: 43-57% of a layer's 512 experts used within 30 tokens; per-layer LRU hit 49/59/68/71% at 16/32/64/128 entries; consecutive-token overlap 8-34%; cross-layer correlation ~0.
- Variants seen elsewhere: engines on Linux/CUDA keep an explicit pinned RAM/VRAM expert cache; this result is specific to unified memory where app caches compete with the page cache for the same DRAM.

## F-io-03: Read-path and hint experiments (all neutral or harmful except plain pread)
Origin: [colibri] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), no-licence (flashmoe)
- Source: flash-moe (no licence) docs/io-and-gpu-exploration.md, docs/plan-io-experiments.md, docs/optimization-experiments-q4.md, results.tsv -- read only, not copied
- Topic: io
- Purpose: record why the read path is the simplest one.
- Inputs / outputs: n/a (experiment summary as pseudocode of the decision).
- Pseudocode:
      choose read primitive:
          mmap + memcpy      -> 0.56 tok/s (5x slower): one 16 KB fault per page, ~240 faults per expert
          dispatch_io channel-> -70%: buffer object management overhead
          POSIX aio + suspend-> about -7% / no better than threaded pread
          threaded pread     -> chosen
      hints:
          F_NOCACHE          -> +3% at 2-bit only while an app cache existed; -1% vs trust-OS (5.68 vs 5.74)
          F_RDAHEAD on/off   -> 0%
          F_RDADVISE now     -> -8% (double-issued reads, NVMe contention); later also logged "no improvement"
          F_RDADVISE ahead   -> -4% (65-80% of advised experts not used next token)
          MADV_RANDOM        -> harmful (splits each read into ~5.7 x 512 KB ops)
          MADV_SEQUENTIAL    -> 0%
          MADV_WILLNEED      -> helps first touch only
      => no hints
- Notes: kernel tracing showed each ~3.9 MB pread became ~5.7 device ops, mostly 512 KB, ~46 us kernel overhead per pread (~11 ms/token over 240 reads); not fixable from user space. Idea left open: check residency first and memcpy from a mapping on hit, pread on miss. I/O efficiency at 2-bit: 4 x 3.93 MB per layer in 1.37 ms = 11.5 GB/s effective vs 17.5 GB/s peak.
- Variants seen elsewhere: Linux engines often gain from O_DIRECT + io_uring; on this OS the defaults were best.

## F-planner-01: Per-layer three-submission pipeline with deferred expert work
Origin: [flashmoe]

Licence-of-origin: no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/infer.m, CLAUDE.md, docs/optimization-experiments-q4.md, paper/flash_moe.tex -- read only, not copied
- Topic: planner
- Purpose: minimise CPU-GPU sync points per layer and let expert GPU work run while the CPU starts the next layer.
- Inputs / outputs: hidden [4096] in/out; per-layer state (KV or delta-net); K; layer fd.
- Pseudocode:
      layer(l):
          if previous layer's expert submission already produced the normalised input on GPU:
              submit S1(l) immediately            # GPU queue orders S3(l-1) before S1(l)
          else: wait S3(l-1); combine on CPU; RMSNorm on CPU; upload; submit S1(l)
          S1: attention input projections (3 matvecs full-attn / 4 linear-attn) [+ whole linear-attn step on GPU]
          wait S1                                  # implicitly also waits for S3(l-1)
          CPU: remaining attention math (full-attn softmax for short ctx, etc.)
          S2: output proj + residual add + post-attn RMSNorm + router matvec (512)
              + shared-expert gate/up + shared-expert scalar gate      (8-12 dispatches, one submit)
          wait S2
          CPU: softmax(512) -> top-K -> renormalise
          I/O: parallel pread K records (F-io-01)
          S3: per expert {gate,up} then {SwiGLU, down}; shared SwiGLU + down;
              fused combine + residual + shared sigmoid gate (F-ffn-01);
              RMSNorm with layer l+1's input-norm weight -> next layer's input buffer
          commit S3, do NOT wait; remember it as pending
      after last layer: wait pending; read hidden back; final norm + LM head
- Notes: measured 4-bit per-layer budget (4.28 ms, 3.90 tok/s before FMA kernel): S1 wait 1.22 ms (28%, includes the deferred S3), S2 wait 0.55 ms (13%), expert I/O 2.41 ms (56%), CPU 0.10 ms (2%), routing 0.003 ms. 2-bit trust-OS (2.9 ms/layer): I/O 1.37 (47%), S1 wait 0.87 (30%), S2 wait 0.45 (15%), CPU attention 0.15 (5%), other 0.06. Moving the combine onto the GPU removed a 0.83 ms CPU wait but the time reappeared in the S1 wait (0.49 -> 1.78 ms) -- kept for cleanliness. Measured negatives: removing the S2 -> CPU -> S3 bounce (routing on GPU feeding experts) 4.07 vs 3.70 ms/layer; GPU softmax+top-K 3.96 vs 3.84 ms/layer; fused GPU final-norm + LM head 4.01 vs 3.93 ms/layer; spin-polling instead of blocking wait -23% (CPU heat steals GPU budget); completion handler neutral; unretained command buffers and encoder count neutral. Key hardware finding: on unified memory SSD DMA and GPU matvec share the memory controller, so overlapping I/O with GPU work slows the GPU disproportionately; the serial GPU -> SSD -> GPU order is treated as optimal.
- Variants seen elsewhere: discrete-GPU engines overlap PCIe copies with compute via streams; that overlap did not pay here.

## F-kernel-01: Tiled 4-bit dequant matvec (one SIMD group per output row)
Origin: [flashmoe]

Licence-of-origin: no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/shaders.metal, paper/flash_moe.tex -- read only, not copied
- Topic: kernel
- Purpose: y = W x for affine-quantised W (8 nibbles per u32, group 64, bf16 scale/bias), batch 1.
- Inputs / outputs: W [out, in/8] u32; scale, bias [out, in/64] bf16; x [in] f32 (in <= 4096); y [out] f32.
- Pseudocode:
      threadgroup = 256 threads = 8 SIMD groups of 32; grid = ceil(out/8) threadgroups
      shared xs[4096]
      all 256 threads cooperatively copy x into xs; barrier          # BEFORE any early exit
      row = tg * 8 + simd_group; if row >= out: return
      acc = 0
      for word in lane, lane+32, lane+64, ... < in/8:                 # adjacent lanes read adjacent words
          g = word / 8                                                 # 8 words per 64-group
          s = bf16_to_f32(scale[row,g]); b = bf16_to_f32(bias[row,g])
          w = W[row, word]; base = word * 8
          for j in 0..7: acc += (float((w >> 4j) & 15) * s + b) * xs[base + j]
      total = simd_sum(acc); lane 0 writes y[row]
- Notes: gate/up [1024 x 4096] -> 128 threadgroups; down [4096 x 1024] -> 512. The barrier-before-bounds-check ordering is a correctness bug they hit: an early return left xs partially filled. Profile of the 2-bit variant: math 64%, bf16->f32 conversion 25%, bit extraction 19%; L1 read hit 93%; ~418 GB/s LLC bandwidth; per-matvec GPU time 37 us but wall 1.58 ms -> GPU utilisation 2.4% (data delivery bound). Storing scale/bias as f32 was suggested (would remove the conversion cost) but not done. A batched variant (K experts in one dispatch via per-expert offsets) exists. bf16->f32 is a 16-bit left shift of the bits.
- Variants seen elsewhere: llama.cpp Metal uses multiple rows per SIMD group and block-interleaved quant formats.

## F-kernel-02: FMA rearrangement of the dequant inner loop
Origin: [flashmoe]

Licence-of-origin: no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/shaders.metal, docs/optimization-experiments-q4.md, CLAUDE.md, results.tsv -- read only, not copied
- Topic: kernel
- Purpose: fold dequantise + multiply into one fused multiply-add per weight.
- Inputs / outputs: as F-kernel-01.
- Pseudocode:
      # algebra: (q*s + b) * x  ==  q * (s*x) + (b*x)
      for word ...:
          for j in 0..7:
              sx[j] = s * xs[base+j]; bx[j] = b * xs[base+j]      # per-input precompute
          for j in 0..7:
              acc += fma(float(nibble_j(w)), sx[j], bx[j])
- Notes: per weight drops from convert+mul+add+mul+add to convert+fma+add. Measured: S1 wait -5.4%, S2 wait -10.7%, GPU compute -5 to -12%; tok/s 3.90 -> 4.36 at 4-bit (README calls it +12%; the Q4 doc's heading says +2.6%, the numbers it quotes are +12%). Not applied to the 2-bit kernel in the reviewed code. Numerically slightly different rounding from the naive form; outputs reported unchanged.
- Variants seen elsewhere: some engines precompute sum(x) per group and apply bias once per group: acc += s*dot(q,x) + b*sum(x) -- fewer multiplies than per-element bx; worth testing.

## F-kernel-03: Kernel ideas that lost
Origin: [flashmoe]

Licence-of-origin: no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/shaders.metal, docs/optimization-experiments-q4.md, docs/io-and-gpu-exploration.md, docs/plan-io-experiments.md, results.tsv -- read only, not copied
- Topic: kernel
- Purpose: record measured dead ends for quantised matvec on an Apple GPU.
- Inputs / outputs: n/a.
- Pseudocode:
      LUT per group: lut[v] = v*s + b for v in 0..15; acc += lut[nibble] * x       -> -2% (dynamic register indexing serialises)
      128-bit vector weight loads (4 words per load)                               -> -3% (register pressure)
      explicit bit-field-extract intrinsic                                         -> 0% (compiler already emits it)
      fused gate+up+SwiGLU in one kernel reading x once                            -> exists, not used on the hot path
      whole expert (gate,up,SwiGLU,down) in ONE encoder to keep it in one cluster L2 -> -2% (less cross-expert parallelism)
      copy expert to GPU-private memory (hardware lossless compression) then matvec -> isolated -13.5% matvec time,
                                                                                      but 4 x 7 MB copy per layer -> -20% overall
- Notes: the default two-encoders-per-expert layout (gate+up, then SwiGLU+down) lets the scheduler interleave experts across GPU clusters, which beat L2 locality. Lesson recorded: compute is ~41% of the 4-bit layer time and bandwidth bound; kernel wins are capped.
- Variants seen elsewhere: on NVIDIA, LUT dequant in shared memory is often a win; not on this GPU.

## F-quant-01: Second-stage requantisation of experts from 4-bit to 2-bit
Origin: [colibri] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/repack_experts_2bit.py, metal_infer/shaders.metal, paper/flash_moe.tex, CLAUDE.md, results.tsv -- read only, not copied
- Topic: quant
- Purpose: halve the bytes read per expert (I/O is the bottleneck).
- Inputs / outputs: one 4-bit expert record -> one 2-bit record (same group size 64, same scale/bias shapes, weights 16 per u32).
- Pseudocode:
      for proj in [gate(1024x4096), up(1024x4096), down(4096x1024)]:
          q4 = unpack 8 nibbles per word (LSB first) -> [out, in]
          f  = q4 * s4[group] + b4[group]                    # f32
          per group of 64: lo = min(f), hi = max(f)
              s2 = (hi - lo) / 3; b2 = lo; if s2 == 0: use 1 for the divide
              q2 = clamp(round((f - b2) / s2), 0, 3)
          rmse = sqrt(mean((q2*s2 + b2 - f)^2))              # logged per proj
          pack 16 x 2-bit per word (LSB first); store s2, b2 as bf16 by truncation
      new record = 3 x (1 MiB weights + 128 KiB scales + 128 KiB biases) = 3,932,160 B (-44.5%)
      2-bit matvec kernel: same tiling as F-kernel-01, 16 values per word, 4 words per 64-group
- Notes: reported RMSE vs the 4-bit dequant: 0.0012-0.0028 (down proj worst, later layers worse). Speed: 5.55 sustained -> 5.68 with cache bypass -> 5.74 with trust-OS, 7.05 peak warm; per-layer I/O 2.6 -> 1.5 ms. Quality: fine on math/code/explanations, BUT it broke JSON / tool calling (emits backslash-escaped names instead of quoted strings), so 4-bit (4.36 tok/s) is the production setting. Lesson: per-weight RMSE and short eval prompts did not catch a structured-output regression; any low-bit expert format needs a JSON/tool-call test. min/max-only range fit (no clipping search) and bf16 truncation (no rounding) are simplifications. 1-bit/ternary listed as future work, untested.
- Variants seen elsewhere: GPTQ/AWQ-style error-compensating 2-bit, or mixed precision (keep hot experts at 4-bit).

## F-ffn-01: Expert FFN, shared expert, and fused combine + residual + sigmoid gate
Origin: [colibri] [ds4] [pulsar] [monolite] [ktransformers] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite, ktransformers), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/shaders.metal, metal_infer/infer.m -- read only, not copied
- Topic: ffn
- Purpose: compute the MoE block output and the next layer's normalised input inside the deferred GPU submission.
- Inputs / outputs: h_post [4096] (post-attn normed), h_mid [4096] (residual stream after attention), K expert records, routing weights w[K], shared gate/up [1024] (from S2), shared scalar gate logit; outputs: new hidden [4096] and normed input for layer l+1.
- Pseudocode:
      for k in valid experts:
          encoder A: g_k = Wgate_k h_post ; u_k = Wup_k h_post
          encoder B: a_k = silu(g_k) * u_k (4-wide vectorised) ; e_k = Wdown_k a_k
      shared: a_s = silu(shared_g) * shared_u ; e_s = Wdown_shared a_s
      combine kernel, one thread per hidden element i:
          moe = sum_{k < K} w[k] * e_k[i]        # fixed max of 8 output slots always bound; unused weight = 0
          hidden[i] = h_mid[i] + moe + sigmoid(shared_logit) * e_s[i]
      RMSNorm in two dispatches:
          pass 1 (1 threadgroup): ss = sum(hidden^2) via simd_sum + shared partials
          pass 2: next_in[i] = hidden[i] * rsqrt(ss/4096 + eps) * w_norm_next[i] (bf16 weight read from the blob)
- Notes: an expert whose read failed is given weight 0 rather than aborting. Routing weights are renormalised top-K softmax probabilities (F-routing-01); the shared expert has its own learned scalar sigmoid gate. Fusing the combine lets the next layer submit immediately; on the last layer the combine falls back to CPU. Encoder count per layer: 2 per expert + 2 shared + 3 combine/norm.
- Variants seen elsewhere: a single grouped-GEMM over all K experts (batched variant exists in the shader file but the per-expert encoders were kept).

## F-routing-01: Softmax top-K routing and cutting K from 10 to 4
Origin: [colibri] [ds4] [pulsar] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/infer.m, paper/flash_moe.tex, CLAUDE.md -- read only, not copied
- Topic: routing
- Purpose: pick experts per token per layer; choose the smallest K that keeps quality.
- Inputs / outputs: router logits [512] (GPU matvec in S2); output: K ids + K weights summing to 1.
- Pseudocode:
      p = softmax(logits)                          # full 512-way, max-subtracted
      ids, vals = top_k(p, K)                      # simple replace-the-current-min scan; K tiny
      w = vals / sum(vals)
      K = runtime option, default 4 (model default 10); hard ceiling 8 slots in the GPU combine
- Notes: measured K sweep (5 prompts, 200 tokens): K=10 1.20 tok/s, 2,359 MB/token; K=6 1.91, 1,415 MB, same quality; K=4 3.15, 943 MB (2-bit), "no observable degradation"; K=3 and K=2 collapse (model emits end-of-sequence after 1-2 words). Variable K across depth: K=0 for the 40 middle layers gave 10 tok/s but blank output; K=2-3 in middle layers also collapsed -> every layer needs K>=4. Quality assessment was qualitative (no perplexity/benchmark numbers), so "no degradation at K=4" should be re-measured before relying on it. Running routing on the GPU (softmax + top-K) was slower (F-planner-01).
- Variants seen elsewhere: dynamic K by cumulative probability threshold; untested here.

## F-prefetch-01: Temporal expert prediction and advisory prefetch (both rejected)
Origin: [colibri] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/infer.m, docs/plan-async-pread-pipeline.md, docs/optimization-experiments-q4.md, results.tsv -- read only, not copied
- Topic: prefetch
- Purpose: start reading layer l's experts before routing finishes, using the previous token's choice at the same layer.
- Inputs / outputs: remembered ids per layer from token t-1; a second set of K scratch slots.
- Pseudocode:
      after S1 commit (CPU otherwise idle while GPU runs):
          if generating (not prefill) and predictions exist:
              async_pread(previous_ids[l]) into slot set B
      after routing:
          wait predicted reads
          for k: if ids[k] in previous_ids[l] and that read succeeded: use slot B[match]  (hit)
                 else: queue for sync read into slot A[k]                                 (miss)
          sync parallel pread the misses
          previous_ids[l] = ids                    # only after the hit check
      advisory variant: instead of reading, issue a kernel read-ahead advisory for predicted ranges
- Notes: hit rate 25.6% (about 1 of 4); a layer only gets faster if all K hit (0.25^4 ~ 0.4%) because wall time is the max of parallel reads; wasted reads cost bandwidth -> 3.18 tok/s vs 3.90 (-18%). Advisory prefetch during GPU compute: expert I/O -31% but S2 wait +73% from DMA contending with GPU memory traffic -> net 0% over 5 prompts (3.91 tok/s). Advisory with one-token lead time: -4%; immediate: -8%.
- Variants seen elsewhere: on discrete GPUs with separate memory, speculative prefetch can pay off; here unified memory kills it.

## F-prefetch-02: Learned / early-routing predictors and cross-layer tricks (rejected)
Origin: [flashmoe]

Licence-of-origin: no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/train_predictor.py, metal_infer/infer.m, paper/flash_moe.tex, docs/plan-async-pread-pipeline.md, results.tsv -- read only, not copied
- Topic: prefetch
- Purpose: predict routing earlier than the real router.
- Inputs / outputs: logged samples (layer, K, pre-attention hidden [4096], true ids[K]).
- Pseudocode:
      logging: per layer write (layer, K, layer-input hidden, chosen ids) to a binary file
      MLP predictor per layer: hidden -> 256 -> 512 logits, multi-label target, top-4 accuracy
      early routing ("logit lens"): apply this layer's router matrix to the PRE-attention hidden,
          top-K, start loading those into an expert cache
      two-pass: route all 60 layers first, then do all I/O + compute
- Notes: MLP predictor 31% accuracy (worse than temporal). Early routing 53% accurate but 1.70 tok/s, 38% slower (cache pollution + extra matvec). Two-pass reached ~6 tok/s (MLX era) but output became repetitive -- routing depends on hidden states updated by the earlier layers, so it is simply wrong. Cross-layer expert correlation is ~0, so no layer-ahead prediction. Pipelining expert reads one layer ahead is impossible without prediction, and prediction is too weak at K=4.
- Variants seen elsewhere: some systems report 80-90% next-layer prediction on other MoE families; not reproduced for this model.

## F-attention-01: Gated delta-net linear attention step (CPU BLAS and GPU)
Origin: [colibri] [ds4] [pulsar] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/infer.m, metal_infer/shaders.metal, CLAUDE.md, results.tsv -- read only, not copied
- Topic: attention
- Purpose: one-token recurrent update for the 45 linear-attention layers.
- Inputs / outputs: projections qkv [12288] = q 2048 (16 heads x128) | k 2048 | v 8192 (64 heads x128), z [8192], alpha [64], beta [64]; persistent per-layer state S [64][128][128] f32 (4 MiB) and conv history [3][12288].
- Pseudocode:
      conv: for c in 12288: o = sum_{t<3} hist[t][c]*wc[c][t] + in[c]*wc[c][3]; out[c] = silu(o); shift hist, append in
      split q,k,v; per k-head RMS-normalise q and k (no weight); q *= 1/128 ; k *= 1/sqrt(128)
      per v-head h: decay[h] = exp(-exp(A_log[h]) * softplus(alpha[h] + dt_bias[h])); gate[h] = sigmoid(beta[h])
      for h in 64 v-heads (k-head = h / 4):
          S_h *= decay[h]                          # BLAS scale over 128x128
          m = S_h @ k                              # BLAS gemv
          d = (v_h - m) * gate[h]
          S_h += d outer k                         # BLAS rank-1 update
          o_h = S_h @ q                            # BLAS gemv
      out_h = rms_norm(o_h) * silu(z_h) * w_gnorm (weight shared by heads)
      GPU version: 64 threadgroups x 128 threads, thread = one state row: decay row while dotting with k,
                   delta, rank-1 add, dot with q (state stays in a GPU buffer per layer)
- Notes: BLAS version cut CPU attention 0.78 -> 0.28 ms/layer (+64% on attention). Later the whole chain (conv, q/k norm, decay/beta, step, gated norm) was encoded into the projection submission on GPU: 3.90 vs 4.15 ms/layer, identical output; kept as default with a CPU fallback flag. Earlier attempt at GPU state hit memory pressure (paper says 195 MB per-layer buffers). Linear-layer index = layer - (layer+1)/4. Prefill is token-by-token through the recurrence.
- Variants seen elsewhere: chunked parallel scan for prefill (not implemented here).

## F-attention-02: Full-attention layers (gated output, partial RoPE, GQA)
Origin: [colibri] [ds4] [pulsar] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/infer.m, metal_infer/shaders.metal, results.tsv -- read only, not copied
- Topic: attention
- Purpose: the 15 softmax-attention layers.
- Inputs / outputs: q_proj [32 x 512] = per head (query 256 | gate 256); k, v [2 x 256]; KV cache f32 [len][512] per layer (CPU copy + GPU mirror up to 8192 positions).
- Pseudocode:
      split per head into query and gate halves
      per-head RMSNorm of q and k with learned weights
      RoPE on first 64 of 256 dims (rotary fraction 0.25), theta 1e7, pairs (i, i+32) "half-split" style
      append k, v to cache
      if len >= 32 and len < GPU cap:      # GPU in the S2 submission
          scores[h,p] = q_h . k_{p, h/16} / 16 ; softmax per head (3-pass max/sum/scale) ; out = scores @ V
      else: same on CPU
      out *= sigmoid(gate)                  # elementwise output gate
- Notes: GPU attention for short contexts was slower and reverted (hence the >=32 threshold). GPU RoPE fused with q de-interleave and k norm plus fused projection+attention submission: 5.53 -> 4.76 ms/layer for those layers (+2% overall). 16 query heads share each KV head.
- Variants seen elsewhere: flash-attention-style single kernel; fp16 KV.

## F-planner-02: Prefill that skips expert results for non-final tokens; session reuse
Origin: [colibri] [ds4] [pulsar] [monolite] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/infer.m, paper/flash_moe.tex, results.tsv -- read only, not copied
- Topic: planner
- Purpose: cut time-to-first-token.
- Inputs / outputs: prompt ids; persistent KV caches and delta-net states.
- Pseudocode:
      embed all prompt tokens up front
      for t in prompt[0 .. n-2]:
          hidden = embed[t]
          for l in layers: layer(l)            # updates KV / recurrent state
          wait for pending expert submission but do NOT read back or combine on CPU
      last prompt token: normal path, read back, LM head
      chat/server: keep KV + recurrent state between turns; prefill only the new user tokens
- Notes: reported TTFT 5,587 -> 2,615 ms (-53%) for a short prompt; ~57 ms saved per intermediate token. Caveat: the paper describes this as skipping expert I/O for N-1 tokens, but in the reviewed loop each intermediate token still runs every layer including expert reads (only the final readback is discarded) -- the intermediate layers' expert outputs feed later layers, so they cannot be skipped without changing results. Treat the 53% as measured, the mechanism as "less sync, not less I/O".
- Variants seen elsewhere: true batched prefill (many tokens per expert read) which amortises I/O; listed as future work here.

## F-mtp-01: Speculative decoding for an SSD-streamed MoE (break-even or worse)
Origin: [flashmoe]

Licence-of-origin: no-licence (flashmoe)
- Source: flash-moe (no licence) docs/optimization-experiments-q4.md, paper/flash_moe.tex, CLAUDE.md -- read only, not copied
- Topic: mtp
- Purpose: decide whether multi-token prediction / draft models help.
- Inputs / outputs: n/a (analysis).
- Pseudocode:
      cost(verify n tokens) on dense model ~ cost(1 token)       # same weights
      cost(verify n tokens) on streamed MoE ~ union of experts over n tokens  (~ n x I/O at K=4, low overlap)
      MTP head (1 extra MoE layer): verify 2 tokens ~ 1.75x expert I/O for ~1.7 accepted -> break-even
      external draft (35B-A3B, fits RAM, 37 tok/s): 33% acceptance at 4 draft tokens -> 4.5x slower
- Notes: the MTP weights were absent from the quantised checkpoint, so this was analysed, not run. Self-speculation with K=1 listed as an idea, untested. General lesson: speculation pays when verification reuses weights; with per-token expert I/O it does not.
- Variants seen elsewhere: in-RAM MoE engines do gain from MTP.

## F-storage-03: Compression and layout experiments on expert files (all rejected)
Origin: [flashmoe]

Licence-of-origin: no-licence (flashmoe)
- Source: flash-moe (no licence) docs/optimization-experiments-q4.md, docs/plan-io-experiments.md, metal_infer/repack_experts_lz4.c, metal_infer/test_apfs_compress.sh, results.tsv -- read only, not copied
- Topic: storage
- Purpose: reduce or reorder bytes on disk.
- Inputs / outputs: per-layer expert files.
- Pseudocode:
      LZ4 per expert: file = [n_experts x {offset, comp_size, raw_size}] + blobs; worker preads comp, decompresses
      co-occurrence clustering: profile routing, permute experts so frequent co-picks are adjacent, map ids via table
      per-expert files instead of per-layer
      filesystem transparent compression; slower codec (LZFSE)
      LZ4 second-level RAM cache of compressed experts
- Notes: LZ4 209 -> 175 GB; decompress 0.17 ms/expert at ~41 GB/s; cold reads 15-24% faster in isolation but pipeline 3.55 vs 4.36 tok/s (-13%) because most reads are warm page-cache hits and 0.68 ms/layer decompress dominates. LZFSE 2.6 GB/s too slow. FS compression ~2x slower (read+decompress serialised). Per-expert files -15% (metadata). Clustering 0%: four 7 MB reads spread over 21 MB or 2.9 GB cost the same. LZ4 RAM cache: analysed as ~3% more experts reachable (steals page cache) -> skipped.
- Variants seen elsewhere: entropy-coded weights pay off when decode is on GPU or reads are mostly cold.

## F-config-01: Memory budget and fixed resident set
Origin: [colibri] [monolite] [slipstream] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (slipstream), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: flash-moe (no licence) CLAUDE.md, paper/flash_moe.tex, metal_infer/infer.m -- read only, not copied
- Topic: config
- Purpose: size everything so the engine never competes with the page cache.
- Inputs / outputs: n/a.
- Pseudocode:
      resident = weights blob 5.5 GB + GPU scratch ~0.2-0.3 GB + expert slots K x RECORD (~0.03-0.06 GB)
                 + KV (15 layers, f32) + linear state (45 x 4 MiB + conv)       ~ 6-6.5 GB
      all scratch allocated once at start; no allocation during decode
      page cache gets the rest (~35-42 GB of 48)
      model-to-RAM ratio: 209/48 = 4.4x (4-bit), 120/48 = 2.5x (2-bit)
- Notes: static allocation is what makes "trust the OS" safe. Per-token data: 4 x 60 x 6.75 MB = 1.62 GB (4-bit), 943 MB (2-bit). Paper's I/O-bound ceiling at 2-bit: 943 MB / 17.5 GB/s = 54 ms -> 18.6 tok/s theoretical; with ~71% page-cache hits the docs estimate ~9 tok/s. 4-bit practical ceiling stated as ~4.4 tok/s. Startup: native BPE tokenizer cut 3,500 -> 180 ms.
- Variants seen elsewhere: engines that budget an explicit expert cache fraction of RAM.

## F-sampling-01: Greedy decode with a thinking-token budget
Origin: [ds4] [flashmoe]

Licence-of-origin: MIT (ds4), no-licence (flashmoe)
- Source: flash-moe (no licence) metal_infer/infer.m -- read only, not copied
- Topic: sampling
- Purpose: bound the length of the model's reasoning section.
- Inputs / outputs: logits [248,320]; think-start/end token ids; budget (default 2048).
- Pseudocode:
      next = argmax(logits)
      if next == think_end: in_think = false
      if in_think: think_count++; if budget > 0 and think_count >= budget: next = think_end
      stop on either of two end-of-sequence ids
- Notes: no temperature/top-p in the reviewed engine. Forcing the end-of-think token is a cheap way to cap latency at ~5 tok/s.
- Variants seen elsewhere: soft budget by biasing the end-think logit.

## F-testing-01: Experiment discipline and quality gates
Origin: [colibri] [ds4] [monolite] [slipstream] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, slipstream), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: flash-moe (no licence) paper/flash_moe.tex, results.tsv, metal_infer/results.tsv, progress.py, metal_infer/infer.m -- read only, not copied
- Topic: testing
- Purpose: keep speed work honest.
- Inputs / outputs: TSV log rows (commit, model, params, active params, tok/s, TTFT, resident GB, keep/discard, note).
- Pseudocode:
      per experiment (<= 5 min): commit; run fixed prompt; record tok/s excluding first 5 tokens; record TTFT and RAM
          compare output text with baseline (same greedy continuation) and check coherence
          keep if faster and output unchanged/coherent, else revert; log either way
      instrumentation: per-phase timers summed over layers per token
          (deferred wait, S1 encode/wait, CPU attention, S2 encode/wait, routing, expert I/O, S3 encode)
      monitor OS memory stats (compressor decompressions/s) and kernel I/O trace during runs
      mark results measured with a since-deleted cache as non-reproducible
- Notes: 90+ experiments, ~42% discarded; the commonest failure was silently wrong output (cache eviction races, routing-buffer mapping bugs, wrong attention). Several early "best" numbers (5.29, 4.20, 4.80 tok/s) were later invalidated because they depended on a warm app-level cache that was removed. Gaps: quality checks were a handful of prompts, which missed the 2-bit JSON regression; no perplexity harness.
- Variants seen elsewhere: logit-diff tests against a reference implementation per layer.

---

## Experiment ledger

| Experiment | Change | Measured effect | Verdict |
|---|---|---|---|
| MLX baseline | Python/MLX, mmap experts | 1.03 tok/s (397B) | superseded |
| MLX packed + K=4 | per-layer packed files, K=4 | 3.14 tok/s | superseded by native engine |
| Expert pruning K=10 -> 6 -> 4 | fewer routed experts | 1.20 -> 1.91 -> 3.15 tok/s; quality unchanged (qualitative) | keep K=4 |
| K=3 / K=2 | fewer experts | output collapses to end-of-sequence | reject |
| Variable K (0 in middle 40 layers) | skip middle-layer experts | 10 tok/s, blank output; K=2-3 middle also collapses | reject |
| Two-pass routing | route all layers, then I/O+compute | ~6 tok/s, repetitive output (wrong) | reject |
| C pread extension + MLX | native reads inside Python | 18.8 GB/s alone, 2.6 GB/s with interpreter lock | reject |
| Native C/Metal engine | full rewrite | first coherent 0.28 tok/s -> 5.29 (with warm app cache, later invalidated) | keep (engine) |
| Parallel pread, 4 threads | vs sequential | ~9x faster reads | keep |
| Persistent I/O pool | vs per-call parallel-for | 3.81 vs 3.97 ms/layer; 4.28 vs 4.09 tok/s | keep |
| mmap expert files | page faults instead of pread | 0.56 tok/s (5x slower) | reject |
| dispatch_io | OS async I/O channel | -70% | reject |
| POSIX aio | kernel async reads | ~-7% / parity | reject |
| 2 MB-aligned read buffers | vs 16 KB | 3.6x faster warm memcpy; ~5% end-to-end | keep |
| GPU LRU expert cache 500 (3.5 GB) | app cache | 3.14 vs 2.86 tok/s early; later hurt vs no cache | reject (final) |
| GPU LRU 1000 / 2500 / 3000 | bigger app cache | 2.24 / 2.24 / 1.99 tok/s | reject |
| Indexed GPU cache 2500 vs 1500 | faster lookup | 4.35 vs 3.98 tok/s; 3500+ regressed | superseded by no cache |
| Malloc zero-copy cache 2581 (18 GB) | non-GPU cache | 52% hit, 2.10 tok/s | reject |
| Malloc zero-copy cache 500 (3.5 GB) | smaller | 37% hit, 2.77 tok/s | reject |
| Trust the OS | delete app cache + bypass flag | 4.11 -> 5.74 tok/s (+38%) at 2-bit; compressor 60-130k/s -> ~0 | keep (foundational) |
| F_NOCACHE (with app cache, 2-bit) | bypass page cache | 5.68 vs 5.55 (+3%) | superseded; -1% vs trust-OS |
| Tiered fds (bypass first read only) | seen-bitset | marginal, same memory pressure | reject |
| F_RDAHEAD toggle | read-ahead | 0% | reject |
| F_RDADVISE immediate | advisory | -8% / no gain | reject |
| F_RDADVISE from previous token | advisory with lead | -4% (65-80% wrong) | reject |
| F_RDADVISE during GPU compute (4-bit) | prefetch predicted | I/O -31%, S2 wait +73%, net 0% (3.91 tok/s) | reject |
| MADV_RANDOM / SEQUENTIAL / WILLNEED | mapping hints | worse / 0% / first touch only | reject |
| Temporal expert prediction | reuse t-1 ids, preload slot set B | 25.6% hit; 3.18 vs 3.90 tok/s (-18%) | reject |
| MLP routing predictor | 4096->256->512 | 31% accuracy | reject |
| Speculative early routing | router on pre-attn hidden | 53% accurate; 1.70 tok/s (-38%) | reject |
| Expert file clustering | co-occurrence reorder | 0% | reject |
| Per-expert files | one file per expert | -15% | reject |
| LZ4 experts | 209 -> 175 GB | 3.55 vs 4.36 tok/s (-13%) | reject |
| LZFSE / FS transparent compression | other codecs | too slow / ~2x slower | reject |
| BLAS delta-net (CPU) | scale/gemv/rank-1 | CPU attention 0.78 -> 0.28 ms/layer | keep |
| GPU delta-net (full chain in S1) | move recurrence to GPU | 3.90 vs 4.15 ms/layer, same output | keep (default) |
| GPU delta-net state (early attempt) | state on GPU | memory pressure | reject (then redone) |
| GPU combine + norm in S3 | fused combine kernel | deferred wait 0.83 -> 0 ms, absorbed by S1 wait | keep (pipeline) |
| GPU RoPE + fused full-attn submission | fewer syncs | 5.53 -> 4.76 ms/layer on full-attn layers; ~+2% | keep |
| GPU attention scores at short context | GPU softmax attention | slower for short ctx | reject below 32 tokens |
| Remove S2->CPU->S3 bounce | GPU-side handoff | 4.07 vs 3.70 ms/layer | reject |
| GPU softmax + top-K | routing on GPU | 3.96 vs 3.84 ms/layer | reject |
| Fused GPU final norm + LM head | tail fusion | 4.01 vs 3.93 ms/layer | reject |
| FMA dequant kernel | q*(s*x)+(b*x) | 3.90 -> 4.36 tok/s (4-bit) | keep |
| LUT dequant kernel | 16-entry table | -2% | reject |
| 128-bit vector loads | uint4 loads | -3% | reject |
| Bit-extract intrinsic | explicit intrinsic | 0% | reject |
| Single encoder per expert (cluster L2) | locality | -2% | reject |
| GPU-private compressed buffers | blit then matvec | matvec -13.5%, pipeline -20% | reject |
| Spin-poll GPU wait | busy wait | -23% | reject |
| Completion handler / unretained cmd buffers | sync style | neutral | reject |
| 2-bit experts | requantise 4->2 bit | 5.55-5.74 tok/s sustained, 7.05 peak; RMSE <=0.003; breaks JSON/tool calls | keep as speed mode, not production |
| 4-bit production | back from 2-bit | 3.50 -> 3.90 (trust-OS) -> 4.36 (FMA) tok/s | keep |
| Prefill discarding intermediate readback | skip CPU combine | TTFT 5,587 -> 2,615 ms | keep |
| Session state reuse | keep KV + recurrent state | prefill only new turn | keep |
| Native BPE tokenizer | replace Python | startup 3,500 -> 180 ms | keep |
| MTP speculation | 2-token verify | ~1.75x I/O for ~1.7 tokens; break-even (analysis, weights missing) | reject |
| Draft-model speculation (35B-A3B) | external draft | 33% acceptance; 4.5x slower | reject |
| 2 MB superpages | large pages | not supported by OS | n/a |
