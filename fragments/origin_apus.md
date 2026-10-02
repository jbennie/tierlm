# Fragments: apus (GLM-5.3-Flash C11 engine with NVMe expert streaming)

Source project: apus-glm5.3-flash, licence PolyForm Noncommercial 1.0.0 (some
design and code adapted from colibri, Apache-2.0). Read only; everything below
is pseudocode written fresh, with our own names. No source, comments or
project-specific identifiers are reproduced. Numbers are the project's own
stated measurements (MacBook Pro M1 Pro, 32 GB unified memory, unless noted).

Model facts used throughout (from the project's docs): 45 layers (34 KDA linear
attention + 11 DSA sparse attention at layers i%4==3), hidden 4096, 288 routed
experts top-8 + 1 shared, MoE on layers 3..44, dense MLP on 0..2, an mHC
residual of 4 streams, one NextN/MTP layer, FP8-E4M3 weights with F32 scales on
128x128 blocks, 305.8 GiB checkpoint over 62 shards, no RoPE anywhere.

---

## F-format-01: Byte-identical repack into safetensors shards plus a manifest
Origin: [colibri] [monolite] [slipstream] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (slipstream), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: apus (PolyForm-NC) tools/convert.py, tests/m1/README.md, tests/m1/test_3_schema.py, tests/m1/test_5_index_realism.py -- read only, not copied
- Topic: format
- Purpose: turn the vendor checkpoint into a container the engine can stream, without ever changing a single weight byte.
- Inputs / outputs: input = HF shard directory + index + config; output = ordinary safetensors shards in two groups (main, mtp), a standard index, and a manifest JSON {format version, model type, config hash, header reserve size, main layer count, slab member order, shard groups, tensor map name -> (shard, absolute file offset, nbytes, dtype, shape), expert slab records, strip statistics}.
- Pseudocode:
      for each source tensor:
          if name under vision prefix: count into stripped stats; skip
          new_name = drop the language-model prefix (head weight keeps its name)
          if new_name not in model-type whitelist: abort("naming drift")
          group = "mtp" if layer_index >= n_main_layers else "main"
      for each FP8 weight W[O,K]:
          require a sibling F32 scale tensor in the SAME source shard
          require scale.shape == [ceil(O/128), ceil(K/128)]
          scan payload bytes; if any byte in {0x7F, 0xFF}: abort("E4M3 NaN code")
      orphan or mis-shaped scales -> abort
      mtp-only tensors (embedding-concat projection, two input norms, head norm) only in mtp group;
          mtp layer has no hyper-connection tensors
      copy payload bytes verbatim; record absolute offsets in the manifest
      also emit a standard name->shard index so a reference loader can open the container directly
- Notes: "fail loudly, never guess" is the converter's whole policy. The naming convention was chosen to equal the numpy oracle's so tests can cross-load the container. Stated figures: 76,108 tensors (347 vision stripped), 37,338 FP8/scale pairs, none split across shards, total 328,326,771,576 B = 305.78 GiB. The project caught its own earlier arithmetic slip (a slab size that implied 16 B per F32 scale) by deriving sizes from config and validating them against the real index.
- Variants seen elsewhere: monolite writes a custom fixed-stride experts file with 4 KiB alignment; apus deliberately stays on plain safetensors so standard tools can still read the container.

## F-format-02: Coalesced six-member expert slabs and split-expert deferral
Origin: [colibri] [monolite] [slipstream] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (slipstream), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: apus (PolyForm-NC) tools/convert.py, tests/m1/test_2_coalescing.py, tests/m1/README.md, docs/ARCHITECTURE.md §6 -- read only, not copied
- Topic: format
- Purpose: make one routed expert = one contiguous byte range = one pread.
- Inputs / outputs: per expert the six tensors {gate weight, gate scale, up weight, up scale, down weight, down scale}; output one adjacent run inside one output shard plus a slab record (layer, expert, shard, file offset, nbytes).
- Pseudocode:
      MEMBER_ORDER = [gate.w, gate.s, up.w, up.s, down.w, down.s]
      on converting a source shard:
          group tensors by (layer, expert)
          for each expert key in deterministic order:
              if all six members present: append them back to back in MEMBER_ORDER; record slab
              else: put key + present members into pending set (persisted in state)
          for each pending key whose members are now all available (across shards):
              append its members contiguously at the open output shard tail; record slab
      before appending a slab: if open shard size + slab > target (~5 GiB): seal, open next
- Notes: real slab = 3 x 2048x4096 B FP8 + 3 x 16x32x4 B scales = 25,171,968 B (24.0059 MiB); 12,384 routed experts (42 main MoE layers + MTP layer, x288) = 290.32 GiB. Exactly one expert in the real checkpoint (an MTP expert) has its up pair in a different source shard, which is why deferral exists. Output shards stay within target + one slab. Tests assert members are adjacent in the header in pinned order and that manifest offsets match header offsets.
- Variants seen elsewhere: colibri requires gate/up/down to be byte-adjacent in a shard and falls back to three reads when not; apus moves bytes at conversion so the fallback never happens. Scales are inside the slab rather than in a separate small-tensor region.

## F-format-03: Crash-resumable, disk-bounded download + convert pipeline
Origin: [colibri] [slipstream] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (slipstream), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) tools/convert.py, tools/download.py, tests/m1/test_4_resume.py, tests/m1/test_6_download_driver.py -- read only, not copied
- Topic: storage
- Purpose: convert a 306 GiB checkpoint on a machine that cannot hold source and output side by side, surviving kills at any instant with byte-identical output.
- Inputs / outputs: remote repo (or a local directory standing in for it); output container + a converter state file + a driver state file.
- Pseudocode:
      RESERVE = 16 MiB                                  # fixed header region per output shard
      open_shard():
          write 8-byte header length = RESERVE, placeholder JSON padded with spaces
      append(tensor):
          write payload at tail; record entry; atomically rewrite state (tmp + rename)
      seal(shard):
          serialize real JSON header; assert fits RESERVE; pad with spaces; overwrite header region in place
          record (size, sha256 of header) in state
      resume():
          for sealed shards: verify size + header hash, else fail
          for open shard: truncate everything past last committed byte (torn write)
          skip finished input shards; re-resolve pending slabs idempotently
      driver loop over source shards, ONE at a time:
          download with HTTP range resume and unlimited backoff (socket timeout so stalls raise)
          convert shard; byte-verify every output tensor against source
          if shard referenced by a pending slab: hold it; else delete it
          after each shard: sweep held shards, delete those no longer referenced AND verified
- Notes: the space-padded header trick keeps every shard a valid safetensors file at every moment and avoids copying data on seal. Output is a pure function of the input set, so one-shot, resumed and shard-by-shard runs are byte-identical (tested with crashes mid-tensor, at seal, mid-slab-resolution). Peak disk ~ container + one ~5 GiB source shard (+ held shards); stated requirement ~315 GB free.
- Variants seen elsewhere: most converters assume the source is fully present and write in one pass.

## F-io-01: Lazy header-only shard reader and positioned-read shims
Origin: [colibri] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: apus (PolyForm-NC) c/st.h, c/compat.h -- read only, not copied
- Topic: io
- Purpose: open hundreds of GiB of shards without touching payloads; read only what is resolved; safe concurrent positioned reads on all three OSes.
- Inputs / outputs: shard path -> table of (name, dtype, shape, offset, nbytes); resolve(name) -> pointer to a buffer filled by one positioned read.
- Pseudocode:
      open(path): read 8-byte LE length, read JSON header, build table; payload pointers = null
      resolve(name): if null: buf = alloc(nbytes); positioned_read(fd, buf, nbytes, data_start + off); keep until close
      positioned_read on POSIX: pread
      positioned_read on Windows:
          handle opened overlapped-capable; loop in chunks <= 1 GiB:
              per-call OVERLAPPED with 64-bit offset split into two 32-bit halves
              issue read; if pending: wait synchronously for the result; stop at EOF
      uncached streaming:
          macOS: set no-cache flag on the fd so expert reads bypass the page cache
          Linux: none engaged (direct I/O would need aligned buffers/offsets the slab path does not guarantee)
          optional "don't need" advice after a read where supported
- Notes: expert payloads are never read through this table in tiered mode (the expert store owns them), so expert bytes never enter RAM through dense loading. The materialized dense buffers stay pinned until close, which lets the GPU backend wrap them zero-copy. Windows needs paired aligned alloc/free because its aligned allocator cannot be released with free().
- Variants seen elsewhere: monolite and slipstream use O_DIRECT with 4-16 KiB aligned records; apus chooses the macOS no-cache flag and accepts page-cached reads on Linux.

## F-cache-01: Expert store with per-layer LRU, per-forward working set and dequant-on-fill
Origin: [colibri] [ds4] [pulsar] [monolite] [slipstream] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar, slipstream), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gcache.h, tests/m6g/README.md -- read only, not copied
- Topic: cache
- Purpose: bounded RAM cache of routed experts behind a single accessor, returning identical bytes on hit and miss.
- Inputs / outputs: resolve(layer, expert) -> three BF16 views (gate [I,D], up [I,D], down [D,I]) valid until that layer's end-of-block call.
- Pseudocode:
      payload_bytes = 3 * I * D * 2                      # 48 MiB at real scale (vs 24 MiB FP8 slab)
      slots_per_layer = budget / payload_bytes / n_sparse_layers   # sparse layers only
      per layer: lru[slots_per_layer], working_set (list), each slot {expert, state EMPTY|LOADING|READY, payload, freq, last}
      resolve(l, e):
          s = find in working_set or lru
          if found: wait_ready(s); s.freq++; s.last = tick(); count hit if it was lru-resident else miss
          else: w = new working-set entry; submit_fill(w, demand=true); wait_ready(w); count miss
          return views into payload
      io worker fill(job):
          raw = staging buffer; ONE pread of whole slab
          carve six members by fixed order; dequant gate/up/down FP8 -> BF16 into payload
          lock; if job.generation == slot.generation: slot.payload = payload; READY; signal
                else: recycle buffers (straggler from a recycled slot)
      layer_end(l):
          for w in working_set:
              if not READY: keep (in-flight prefetch), continue
              v = lru victim: an empty slot if any, else coldest non-LOADING
              if v.last <= w.last: move w into v (evict v's payload to free list)
              else: drop w's payload (LRU full of warmer entries)
          generation++; run pressure guard
- Notes: misses never go straight into the LRU; they live in the working set until end-of-block, so views handed to the running MoE sublayer are never evicted under it. Dequant happens on the I/O worker, not the compute thread. The compute thread does no I/O in pool mode; it blocks just-in-time on a per-slot condition variable. Because dequant is deterministic, tests assert eager arena == big cache == 1-slot cache == cache+prefetch digests; the 1-slot run forced 41 evictions and still matched. Stated sizing: ~10-14 GB holds ~210-290 of 12,384 experts (1.7-2.4%); the slot count was fixed to derive from the 42 sparse layers (before, 16 GiB bought 7 slots/layer instead of 8).
- Variants seen elsewhere: the apus base engine (DeepSeek-V4) also had pins, usage history and re-pinning; this port dropped them. Caching dequantized BF16 doubles bytes per slot versus caching FP8 and dequantizing per use; apus trades RAM for decode-time compute.

## F-cache-02: Demand vs speculative load classes with admission control
Origin: [colibri] [ds4] [pulsar] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gcache.h, tests/m6g/README.md, docs/USAGE.md -- read only, not copied
- Topic: prefetch
- Purpose: let prefetch use only idle disk capacity; demand loads must never queue behind guesses.
- Inputs / outputs: hint(layer, expert, class) -> submit or drop; worker queue pop.
- Pseudocode:
      admit_speculative():
          if any demand-class load is in flight or queued: return false
          if speculative_queue_len >= cap (default 2 * io_threads): return false
          if pressure_bytes + payload_bytes + slab_bytes > guard_budget: return false
          return true
      hint(l, e, demand):
          s = lookup(l, e)
          if s exists and EMPTY (dropped by guard): if speculative and not admit: drop; else resubmit
          if s exists and LOADING and demand: mark hot (boost queued job)
          if not exists: if speculative and not admit: drop (count); else add working-set entry, submit
      worker pop: first job whose slot is hot; otherwise FIFO
      any resolve that has to block on a LOADING slot marks it hot
- Notes: dropping a speculative hint never changes bytes returned by any resolve, so the policy can be aggressive. A speculative working-set entry has last-use 0, so promotion can only displace empty or never-used slots: prefetch cannot evict demand-used experts. The fix was motivated by a measured hint storm that starved demand fills 5:1. In synchronous test mode the queue rules are vacuous and only the memory rule remains, keeping counters deterministic.
- Variants seen elsewhere: the base engine delivered hints through a single-producer ring and a dedicated pilot thread; apus found hint submission cheap enough to do inline.

## F-cache-03: Memory-pressure guard on the right metric, plus buffer recycling
Origin: [colibri] [apus]

Licence-of-origin: Apache-2.0 (colibri), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gcache.h, c/compat.h, docs/USAGE.md -- read only, not copied
- Topic: cache
- Purpose: keep the process under a hard memory ceiling without stalling the forward pass.
- Inputs / outputs: guard budget (default 26 GiB on a 32 GB box); drops payloads in place.
- Pseudocode:
      pressure_bytes():
          macOS: physical footprint from task VM info (counts compressed pages)
          Linux: resident pages from /proc statm; elsewhere: peak RSS fallback
      guard():                                   # only at layer boundaries
          p = pressure_bytes(); if p <= budget: return
          collect all READY lru slots; sort by last-use ascending
          free payloads (real free, not recycle) until freed >= p - budget
          slot.state = EMPTY but keep expert id and freq (identity retained)
      free lists: up to N payload buffers and M raw staging slabs reused across fills
- Notes: the project measured that plain resident size under-reports under macOS compression: a 32.5 GB footprint read below 26 GiB resident, so the old guard never fired while the machine thrashed. Recycling exists because a 25-48 MiB mmap/munmap plus zero-fill faults per fill was major kernel time. Recommended 32 GB config is a 2 GiB expert cache: larger caches "buy almost nothing" because the guard trims them and compressor churn costs more. Stated real run: ~7 s load, ~76 s prefill for 26 tokens, ~11-12 s/token decode (~0.085 tok/s), peak footprint ~22.3 GiB, ~7.9 GiB of expert reads per token cold.
- Variants seen elsewhere: other engines measure RSS; apus specifically uses the footprint metric the OS reclaimer answers to.

## F-prefetch-01: Router-lookahead prefetch (one layer ahead, reusing the real router math)
Origin: [colibri] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gpilot.h, tests/m6g/README.md, tools/measure_glocality.py, tests/m8g/m8c_spec_verbose.err -- read only, not copied
- Topic: prefetch
- Purpose: predict layer L+1's experts at layer L's post-attention point and warm the cache while layer L's MoE runs.
- Inputs / outputs: source layer's post-attention 4-stream hidden [4*D] BF16; target layer's mHC map weights, ffn norm, router weights, router bias -> top-N expert ids -> speculative hints.
- Pseudocode:
      on post_attention(layer L, h4):              # runs on compute thread via a hook
          T = L + 1; if T has no router: return
          for each token t (decode: s==1; prefill: all s):
              maps = hyper_connection_maps(h4[t], weights of T's ffn site)   # same code path as the real forward
              x = collapse(h4[t], maps.pre); x = weighted_rmsnorm(x, T.ffn_norm)
              scores = sigmoid(router_T @ x) + bias_T
              pred = stable_topk(scores, N)        # N = 12 by default for both phases
              decode: remember pred for recall accounting; hint each
              prefill: union into bitset
          prefill: hint each unique expert once
      on routed(layer, actual top-8): recall += |pred ∩ actual|
- Notes: prediction applies target-layer weights to the source-layer state, a heuristic whose recall is measured, not assumed. No prefetch thread: overlap comes from the store's I/O pool. On synthetic clustered fixtures: recall 1.000 (384/384) and demand loads 0 vs 70 without prefetch; random-weight control 0.901, which the project flags as an artifact of the constant selection bias. A real-container run log shows recall 1852/2688 (~0.69) at N=12, with 415 hits vs 13,457 misses and 9,463 speculative hints dropped, which shows the 2 GiB cache barely holds anything across tokens. A companion analysis tool reports recall curves for N in {6,8,12,16,24}, temporal reuse at lags 1-8, cross-layer coupling, Zipf fit, and an LRU simulation per slots/layer to size the cache.
- Variants seen elsewhere: base engine had the same dL=1 design with a ring buffer and dedicated consumer thread; others prefetch from previous-token routing (temporal) rather than next-layer router math.

## F-planner-01: Tiered MoE call sequence and token-chunked prefill
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gmodel.h, tests/m6g/README.md, docs/USAGE.md -- read only, not copied
- Topic: planner
- Purpose: touch only routed experts, overlap their loads, and bound the live expert set during prefill.
- Inputs / outputs: s tokens at one MoE layer -> MoE output; chunk size C (default 8; 0 = whole prompt).
- Pseudocode:
      for chunk in split(tokens, C):                 # decode is one chunk of 1
          sel = router pre-pass for every token in chunk (or forced ids in teacher-forced tests)
          union = unique(sel)
          for e in union: hint(layer, e, demand)     # whole union queued at once
          for e in sorted(union): views[e] = resolve(layer, e)
          experts not in union: null pointers (never dereferenced)
          run MoE forward for chunk with views
          layer_end(layer)                            # promotion + guard between chunks
- Notes: chunking is numerics-neutral because MoE is per-token independent and a resolve returns the same bytes regardless of cache state; tests pin chunked tiered == unchunked eager. The unchunked prefill held the whole prompt's union live, ~10 GiB of 48 MiB payloads per layer at s=26, producing a measured 41.5 GB peak; chunking fixed it.
- Variants seen elsewhere: engines that cache FP8 and dequantize per use avoid the payload doubling but pay dequant per token.

## F-quant-01: Exact FP8-E4M3 x F32 block-scale dequant to BF16 (scalar / NEON / AVX2)
Origin: [colibri] [ktransformers] [apus]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/fp8blk.h, c/x86.h, c/num.h, tests/m3g/README.md -- read only, not copied
- Topic: quant
- Purpose: dequantize W[O,K] FP8 codes with a 128x128 scale grid into BF16, bit-identically on every ISA.
- Inputs / outputs: codes u8[O*K], scales f32[ceil(O/128)*ceil(K/128)] row-major over the block grid -> bf16[O*K].
- Pseudocode:
      nkb = ceil(K/128)
      for o in rows:
          srow = scales + (o / 128) * nkb
          for kb in 0..nkb-1:                       # last block may be partial
              s = srow[kb]
              for k in block: out[o,k] = bf16_rne(decode_e4m3(code[o,k]) * s)    # ONE fp32 multiply
      SIMD decode of 16 codes, exact:
          h = ((c & 0x80) << 8) | ((c & 0x7F) << 7)  # place sign/exp/mantissa as a half-precision bit pattern
          f = float(half(h)) * 256                   # exact for all 256 codes, including subnormals
      bf16_rne in integer lanes:
          u = bits(f); if |u| > inf_bits: out = u >> 16          # NaN passthrough
          else out = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
      threaded: partition output rows across lanes (elementwise, so identical at any thread count)
- Notes: scales are plain F32, not power-of-two. The half-precision placement trick works because every E4M3 value is exactly representable in FP16 scaled by 2^-8. The two NaN codes decode as +-480 in C (the numpy oracle says NaN); pinned as a documented divergence since the converter refuses them. On x86 there are both an F16C and an integer-only expand, each proven on all 256 codes; AVX2 workers are per-function target attributes behind a runtime CPU check, so the binary runs on baseline x86-64. Multi-threaded dequant measured 5-7x single-thread.
- Variants seen elsewhere: the base engine used UE8M0 power-of-two scales and FP4; lookup-table decode is the common alternative.

## F-kernel-01: BF16 GEMV/GEMM with a pinned sequential reduction order
Origin: [colibri] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/bf16.h, c/x86.h, c/pool.h, tests/m3g/README.md -- read only, not copied
- Topic: kernel
- Purpose: matmul that is bitwise equal across scalar, NEON, AVX2, any thread count and the numpy oracle.
- Inputs / outputs: W bf16[O,K], x bf16[M,K] -> y bf16[M,O]; fp32 scratch for widened x.
- Pseudocode:
      reference definition (per output):
          acc = +0.0f
          for k ascending: acc = acc + (f32(W[o,k]) * f32(x[k]))     # mul then add, two roundings, never fused
          y[o] = bf16_rne(acc)
      SIMD version, same bits:
          widen x once (shift left 16, exact)
          for groups of 8 rows (then 4, then 1):
              for k in chunks of 32:
                  stage products for each row: vector multiply only, store to per-row buffer
                  for i in 0..31: acc_r += stage_r[i]  for r in group   # 8 independent chains, in order
              scalar tail for k % 32
      GEMM: process 4 activation rows per widened weight chunk; each output keeps its own ordered chain
      threads: contiguous output-row blocks per lane; caller lane is lane 0
- Notes: ILP comes only from interleaving independent rows, never from splitting one dot. Compiled with contraction disabled so the compiler cannot fuse. Every GEMM row equals the GEMV result (M-independence), which is what later makes batched verification safe. Stated bench: multi-threaded decode GEMV 24-64 GB/s, approximately the per-process read roofline, so decode GEMVs are memory-bound and the strict order costs little. A roofline that only widens and does the ordered adds (no multiply) is used as the ceiling for any bitwise-safe kernel.
- Variants seen elsewhere: BLAS/Accelerate and tree reductions are faster but change bits; apus defers those as a separately gated "reorder class".

## F-kernel-02: Optional GPU offload as a fail-soft hook table with zero-copy buffers
Origin: [ds4] [apus] [flashmoe]

Licence-of-origin: MIT (ds4), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: apus (PolyForm-NC) c/backend_gmetal.h, c/backend_gmetal.mm, c/bf16.h, tests/m7b/README.md, docs/USAGE.md -- read only, not copied
- Topic: kernel
- Purpose: add a GPU path without forking numerics or memory ownership.
- Inputs / outputs: hook table {bf16_gemm, fp8_block_linear}; each returns "handled" or "unsupported".
- Pseudocode:
      global hooks = all null; CPU-only build links weak stubs returning "unsupported"
      cpu_matmul_mt(...):
          if hooks.gemm and size >= floor and M <= max_rows and hooks.gemm(...) == handled: return
          run pinned CPU kernel
      fused fp8 shader: decode e4m3 * scale -> bf16 -> same sequential accumulate, no materialized weight
      buffers: wrap host memory without copying (unified memory)
          registered class: model-owned dense weights, wrapped once at open, released at close
          ephemeral class: cache payloads, activations, outputs wrapped per op and released before return
- Notes: because nothing ephemeral is held across ops, the cache can free payloads at any layer boundary with no invalidation protocol. Shaders reproduce the sequential two-rounding order; the only measured difference is subnormal flush on the GPU (|delta| = 1.7e-39 on all-subnormal rows), documented as unreachable from real data. Stated numbers: fused FP8 DSA linears x2.7-x3.8 faster (q_b 16.7 -> 6.19 ms, o_proj 43.7 -> 11.5 ms); bf16 expert GEMVs x0.85-0.96 (slower); tiny-model decode 173 tok/s CPU vs 24 tok/s GPU because ~50 us dispatch overhead x ~40 ops/token dominates. At real scale per-op wrapping of ~230 dense ranges per token caused a 4x loss, fixed by registered wraps. Defaults: bf16 offload off, fused FP8 on, prefill (M>1) stays CPU.
- Variants seen elsewhere: most engines keep separate GPU weight copies; apus requires zero extra RSS.

## F-testing-01: Bitwise-exactness discipline and gate design
Origin: [colibri] [ds4] [pulsar] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) tests/m0/README.md, tests/m4g/README.md, tests/m3g/README.md, tests/smoke/README.md, Makefile, tools/oracle.py -- read only, not copied
- Topic: testing
- Purpose: make "insufficient memory costs speed, never quality" provable.
- Inputs / outputs: a numpy oracle ported from the reference model file; synthetic tiny-config fixtures; digests of every output.
- Pseudocode:
      build: contraction off; no fast-math; scalar anchor first; SIMD ports must equal anchor
      oracle: two modes (f64 truth, f32 dtype-faithful with bf16 rounding exactly where reference casts)
          pin the pure-PyTorch fallback paths of the reference (never hub kernels)
      replicate reduction orders exactly:
          contiguous-axis sums: numpy pairwise order (n<8 sequential from a[0]; <=128 eight-accumulator tree; else recursive halving)
          strided-axis sums: sequential from +0.0
          matmuls: +0.0 start, ascending k, two roundings
          sigmoid: stable form e = exp(-|x|); x>=0 ? 1/(1+e) : e/(1+e)
      host transcendental probe at test start: compare libm expf vs stored numpy exp on 1024 points
          clean -> BITWISE tier (memcmp everything); else TOLERANCE tier (rel 1e-5, selections still exact)
      fixture generation pins numpy to its baseline exp kernel on x86 (disable AVX2/AVX512 features)
      fixtures redrawn until every top-k boundary gap > 1e-4 (and indexer gap > 0.05 in cross-engine runs)
      run every gate at 1, 4, 8 threads and diff full stdout; also diff -O2 vs sanitizer builds
      external anchor: oracle vs the real reference framework on a tiny checkpoint (rel 2e-2, argmax)
      real-model: per-layer FNV digests of C vs oracle on the real container; bisect to first divergence
- Notes: measured: numpy f32 exp equals Apple libm bitwise (0/500,007), differs on x86 SIMD (~40% of values by 1 ulp) but equals gcc libm once pinned (0/500,000). Real-container result: prefill logits plus 184/184 teacher-forced decode digests identical across all 45 layers. Important lesson recorded by the project: the self-built oracle and C engine shared a kv_b split bug (head-major instead of per-head interleaved) and agreed bitwise; only the external reference anchor caught it (rel error 0.74, argmax 6/16 before fix; 16/16 after). Each milestone has a hard gate (tokenizer, kernels, sublayers, full forward, tiering, server, GPU, MTP) and CI runs the battery on Linux, macOS and Windows.
- Variants seen elsewhere: most engines gate on tolerance (perplexity or logit RMS); apus gates bitwise and treats any reorder as a named, measured class.

## F-attention-01: KDA linear attention - conv, gates and the recurrent decode step
Origin: [colibri] [ds4] [pulsar] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: apus (PolyForm-NC) c/gkda.h, docs/ARCHITECTURE.md §3.2, tests/m4h/README.md -- read only, not copied
- Topic: attention
- Purpose: per-token Kimi-style gated delta-rule update with O(1) state.
- Inputs / outputs: x bf16[D]; per layer state: conv window bf16[C*(k-1)] (C = 3*64*128, k = 4), S f32[64,128,128] -> o bf16[64*128].
- Pseudocode:
      q,k,v = bf16 matmuls 4096 -> 8192 each; mixed = concat
      for channel c: acc = sum_i w[c,i] * window_i (ascending, fp32); out = bf16(acc * sigmoid(acc))
          shift window (holds last k-1 PRE-conv bf16 inputs)
      fg = bf16(Fb @ bf16(Fa @ x))                       # low-rank 4096 -> 128 -> 8192
      g = -5 * sigmoid(exp(A_log[h]) * (f32(fg) + dt_bias))   # fp32, never rounded; decay exp(g) in (e^-5, 1)
      beta = bf16(sigmoid(bf16(Wb @ x)))                 # per head
      q,k: to fp32, then l2-normalize (eps 1e-6, literal division); q *= 128^-0.5
      per head:
          S[i,j] *= exp(g[i])
          kv[j] = sum_i S[i,j] * k[i]           # sequential
          d[j]  = (v[j] - kv[j]) * beta
          S[i,j] += k[i] * d[j]                 # delta first, then outer product
          o[j]  = sum_i S[i,j] * q[i]
      o = bf16(o)                                # once
      o = per-head RMS norm (fp32, eps 1e-5) * w * sigmoid(gate_lowrank(x)); one bf16 rounding
      out = Wo @ o
- Notes: the project lists six bit-visible fixes vs its donor implementation: keep l2norm output fp32, numpy pairwise sum for the norm, stable sigmoid, o_norm eps 1e-5 not 1e-6, round beta to bf16, and compute the delta before the outer product. State is ~4 MiB/layer, ~136 MiB over 34 layers, independent of context length.
- Variants seen elsewhere: the donor (Ling bf16 engine) had recurrent only and kept beta fp32.

## F-attention-02: KDA chunked prefill and the two-ordering contract
Origin: [apus]

Licence-of-origin: PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gkda.h, docs/ARCHITECTURE.md §4, tests/m0/README.md, tests/m8g/README.md -- read only, not copied
- Topic: attention
- Purpose: fast prefill matching the reference's chunked algorithm bitwise, while decode matches the reference's recurrent step bitwise.
- Inputs / outputs: q,k,v bf16[s,H*D], g f32[s,H*D], beta bf16[s,H], S f32 in/out -> out bf16[s,H*D].
- Pseudocode:
      CH = 64; pad sequence to multiple of CH with zeros (padding extends cumulative decay flat)
      per chunk, per head:
          gc = running sum of g within chunk (sequential)
          decay(i,j,d) = exp(gc[i,d] - gc[j,d]) only for j <= i      # skip masked entries entirely
          A[i,j] = -sum_d (k_beta[i,d] * k[j,d]) * decay(i,j,d) for j < i
          for i ascending: A[i,:i] += A[i,:i] @ A[:i,:i]               # in-place triangular solve
          A += I
          u  = A @ v_beta;  w = A @ (k_beta * exp(gc))
          inter = (q * exp(gc)) @ S
          intra[i,j] = sum_d (q[i,d] * k[j,d]) * decay(i,j,d) for j <= i
          v_new = u - w @ S
          out = inter + intra @ v_new
          S[m,n] = S[m,n] * exp(gc_last[m]) + sum_j (k[j,m] * exp(gc_last[m] - gc[j,m])) * v_new[j,n]
      out = bf16(out)
      contract: prefill path ALWAYS chunked, decode path ALWAYS recurrent, chosen by an explicit flag, never by s
- Notes: the two orderings are mathematically equal but differ in final bits and can flip sampled tokens; the project pins each to its own goldens and makes no cross-phase bitwise claim. Measured on the tiny fixture: f64 max delta 4.55e-15, f32 delta 0 at that size. Consequence for speculative verification: a multi-token verify batch cannot use the chunked path; it must step the recurrence token by token (see F-mtp-02).
- Variants seen elsewhere: engines that use one kernel for both phases do not get reference bit-equality in both.

## F-attention-03: DSA layers - MLA with no RoPE, additive-mask attention, full-fidelity KV
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gdsa.h, docs/ARCHITECTURE.md §3.3 and §7, tests/m6g/README.md -- read only, not copied
- Topic: attention
- Purpose: the 11 exact-attention layers; selection from the indexer applied as a mask.
- Inputs / outputs: x bf16[D]; caches K,V per head [H][cap*256] bf16; selection ids [width] -> out bf16[D].
- Pseudocode:
      qr = rmsnorm_w(fp8_linear(Wqa, x))            # 4096 -> 1536
      q  = fp8_linear(Wqb, qr) as [64, 256]
      kr = rmsnorm_w(fp8_linear(Wkva, x))           # 4096 -> 512
      kv = bf16_linear(Wkvb, kr); per head h: k_h = kv[h][:256], v_h = kv[h][256:]   # per-head interleaved split
      append k,v to caches
      rmsnorm_w(x): fp32 mean, bf16 round, THEN * weight, round again
      scores over ALL cached positions:
          sc = bf16(q @ k^T); sc = bf16(sc * 256^-0.5)
          sc += (position in deduped selection ? 0 : float_min)
          p = bf16(softmax_fp32(sc)); o = bf16(p @ v)
      out = fp8_linear(Wo, o)
- Notes: masked lanes contribute exactly +0, which the project notes makes a future gather-only version bitwise-safe in principle. KV policy pinned: no eviction, no quantization, caller sizes capacity, forward fails loudly past it; exact footprint function provided. The expanded per-head cache costs ~64.5 KB/token/layer = ~0.71 MB/token over 11 layers (~23 GB at 32K, ~700 GB at 1M), so practical context on 32 GB is ~10-20K tokens; latent caching (512-dim, ~1.5 KB/token/layer) is the deferred path. No attention sink.
- Variants seen elsewhere: the base DeepSeek-V4 engine had rotary, a sink, de-rotation and Hadamard; the GLM variant has none.

## F-attention-04: Lightning indexer with 4-token pooled keys and an always-kept tail
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gdsa.h, docs/ARCHITECTURE.md §3.3, tests/m0/README.md -- read only, not copied
- Topic: attention
- Purpose: pick ~2048 positions per query from a long context cheaply, in fp32 with no score quantization.
- Inputs / outputs: q latent bf16[1536], x bf16[D]; token-major index caches key[cap,128], gate[cap,128] -> selection ids int[2048 + 3] (-1 = invalid).
- Pseudocode:
      iq = bf16(Wq_idx @ qr) as [32 heads, 128]
      ik = layernorm(bf16(Wk_idx @ x), w, b, eps 1e-6)         # one bf16 rounding
      ig = bf16(Wgate_idx @ x)
      append ik, ig to caches
      rebuild ALL pools from the full cache every call (decode included):
          pool p = tokens 4p..4p+3; only full pools scored
          probs = bf16(softmax_over_4(ig[member] + pos_bias[slot]))   # per channel, sequential over 4
          pool_key[p] = bf16(sum_member probs * ik[member])
      for head h: sc[h,p] = relu((iq[h] @ pool_key[p]) * 128^-0.5)   # fp32, relu after scale
      wh = bf16(Wproj @ x) * 32^-0.5
      score[p] = sum_h wh[h] * sc[h,p]
      valid iff pool's LAST token <= query position, else float_min
      pick top-512 pools (stable, ties to lower index); expand each to 4 token ids
      append current incomplete pool's tokens (count = (pos+1) % 4), pad -1
- Notes: no Hadamard rotation and no FP8/FP4 score quantization, unlike the base engine's indexer; a fresh port rather than a parameter tweak. Every DSA layer runs its own indexer in this checkpoint (cross-layer sharing exists in the reference but is unused). The full pool rebuild per call matters for exact speculative verification: a one-shot batched call would rebuild pools including not-yet-written tokens, so verification runs the per-token call sequence.
- Variants seen elsewhere: DeepSeek-style indexers score raw tokens (no pooling) with quantized scores.

## F-norm-01: mHC hyper-connection residual with Sinkhorn-20 mixing
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gmhc.h, tests/m4g/README.md, docs/ARCHITECTURE.md §3.4 -- read only, not copied
- Topic: norm
- Purpose: keep 4 residual streams per token and mix them with learned, doubly-stochastic-ish matrices at each sublayer.
- Inputs / outputs: streams bf16[4,D]; per site fn bf16[24, 4D], base f32[24], scale f32[3] -> collapsed input bf16[D]; after sublayer: new streams bf16[4,D].
- Pseudocode:
      maps(x4):
          flat = x4 flattened; flat *= 1/sqrt(mean(flat^2) + 1e-5)       # unweighted, fp32, BEFORE the matmul
          mix = fn @ flat                                                # fp32, no rounding
          pre[j]  = sigmoid(mix[j]*scale0 + base[j]) + 1e-6
          post[j] = 2 * sigmoid(mix[4+j]*scale1 + base[4+j])
          comb[j,k] = mix[8+4j+k]*scale2 + base[8+4j+k]
          sinkhorn(comb): row softmax then +eps per element; column-normalize (eps in denominator);
                          repeat 19x { row-normalize; column-normalize }        # ends on columns
      collapse: y = bf16(sum_j pre[j] * x4[j])
      sublayer input = weighted_rmsnorm(y)
      expand(F = sublayer output, R = old streams):
          pb = bf16(post); cb = bf16(comb)
          new[j] = bf16( bf16(pb[j] * F) + bf16(sum_k cb[k,j] * R[k]) )   # note transpose
      embedding replicated 4x at input; final head = bf16(mean over 4 streams), then norm, lm_head
- Notes: two sites per layer (attention, MLP). Applying the norm before the fn matmul is mathematically equal to scaling after but differs in fp32 bits; the base engine did the latter. Comb ends column-stochastic to ~1e-6 with rows off; the project warns not to "fix" this. The final head is an unweighted mean, unlike DeepSeek-V4's gated head. Stream-axis sums of length 4 start from the first product (numpy pairwise) while matmuls start from +0, a difference that only matters for signed zeros and is replicated.
- Variants seen elsewhere: base DeepSeek-V4 engine: rsqrt after matmul with eps 1e-6, sigmoid-gated head.

## F-routing-01: Sigmoid router with selection-only bias and clamped SwiGLU experts
Origin: [colibri] [ds4] [pulsar] [ktransformers] [apus]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gmoe.h, tests/m4g/README.md -- read only, not copied
- Topic: routing
- Purpose: bit-faithful MoE: which experts, with what weight, summed in what order.
- Inputs / outputs: x bf16[D]; router W bf16[288,D], bias f32[288] -> ids[8], weights[8]; experts -> y bf16[D].
- Pseudocode:
      logit[e] = dot(W[e], x) in fp32 (sequential, no rounding)
      score[e] = sigmoid(logit[e]); biased[e] = score[e] + bias[e]
      ids = stable_topk(biased, 8)                  # ties to LOWER expert index
      w = score[ids]                                # unbiased scores
      w = w / (pairwise_sum(w) + 1e-20) * 2.5
      expert(x): g = bf16(Wg@x); u = bf16(Wu@x)
                 g = min(g, 10); u = clamp(u, -10, 10)
                 h = bf16(g * sigmoid(g)); h = bf16(h * u); return bf16(Wd@h)
      y = 0
      for e in ids sorted ASCENDING by expert index:
          y = bf16(y + bf16(expert_e(x) * w_e))     # rounding per add
      y = bf16(y + shared_expert(x))                # shared last
- Notes: group-limited routing is a no-op at one group and deliberately not implemented. Gate is clamped above only, up both sides, both before the activation. Accumulating in ascending expert order (not routing-rank order) mirrors the reference's eager loop; it is also what makes tiered resolve order and chunking irrelevant. Batched forward is per-token GEMVs because matmul rows are M-independent.
- Variants seen elsewhere: many engines sum in top-k rank order in fp32 and round once; that is a different number.

## F-mtp-01: MTP (NextN) draft head and the empirical input pin
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gmtp.h, tests/m8g/README.md, tools/mtp_pin.py, tests/m8g/spec_sweep.sh -- read only, not copied
- Topic: mtp
- Purpose: cheap draft tokens from the checkpoint's extra layer, with its unspecified input semantics resolved by measurement.
- Inputs / outputs: previous hidden h bf16[D], token id -> draft logits; feeds its own output hidden back for chaining.
- Pseudocode:
      e = rmsnorm_w(embed[tok], enorm); hh = rmsnorm_w(h_src, hnorm)
      x = bf16_linear(concat(e, hh), Weh)            # BF16 [D, 2D]; embedding half FIRST
      x = bf16(x + DSA(rmsnorm_w(x, in_norm)))        # plain residual, no hyper-connections
      x = bf16(x + MoE(rmsnorm_w(x, post_norm)))      # includes a shared expert; experts served by same store
      out = rmsnorm_w(x, head_norm); logits = bf16_linear(out, shared lm_head)
      draft = argmax(logits); chain: next h = out, next tok = draft
      h_src choice and pairing lag were pinned empirically:
          candidates: mean of 4 streams, post-final-norm row, each single stream; lag 0 or 1
          sweep on the real container: run greedy main decode, replay draft chains, measure accept
          winner: post-final-norm row, lag 1 ((h_{p-1}, tok_p) predicts tok_{p+1})
- Notes: the reference framework drops this layer at load, so semantics came from another serving stack and were then verified by acceptance. Stated: 85.3% draft accept, 2.53 tokens/batch at depth 3, 1.47x faster than non-speculative. An earlier pin was measured while the kv_b split bug existed and was discarded; a log from that era shows 0/8 accepted. The compiled-in pin is checked against fixture manifests so a mismatch fails loudly.
- Variants seen elsewhere: EAGLE-style heads use lag-1 bookkeeping; the base engine used lag 0.

## F-mtp-02: Speculative decoding that is bitwise identical, including sampled decoding
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/gmtp.h, c/gmodel.h, c/sample.h, tests/m8g/README.md, tests/m7a/README.md -- read only, not copied
- Topic: mtp
- Purpose: exact output equivalence to one-token-at-a-time decoding for the same seed, greedy and sampled.
- Inputs / outputs: held next token x_q, drafts d2..dD; main and MTP states -> emitted tokens.
- Pseudocode:
      step:
          snapshot main (pos + every KDA conv/recurrent state copy) and MTP (pos + cache count)
          R, H = decode_batch([x_q, d2..dD])        # per-token interleaved: bitwise == D single decodes
          emit x_q
          for j in 1..D-1:
              x = pick(R[j-1])                       # argmax, or sample with ONE rng uniform
              emit x
              if d_{j+1} != x: break
          if all matched: emit bonus pick(R[D-1])
          if partial: restore snapshot; re-feed accepted prefix in one decode_batch (rows reused)
          MTP: restore; replay true (H[j], tok[j]) pairs batched; snapshot; chain next drafts
      decode_batch:
          KDA: batched projections, then conv and recurrence stepped token by token
          DSA: loop tokens; each appends kv + index rows, rebuilds pools, attends at count base+t+1
          MoE, norms, hyper-connection maps: per-token independent -> batched (I/O amortized)
      rollback of DSA = rewind count only (rows past count are dead and rewritten by re-feed)
      expert cache: never rolled back (content is numerics-neutral)
      sampler: splitmix64; temperature softmax fp32; top-p keeps token i iff mass before i <= p
               (stable sort, lower index first); draw = first CDF > u, else last kept
- Notes: acceptance is "draft equals the main model's own pick" rather than rejection sampling, so no draft probabilities are needed and the RNG stream advances exactly once per emitted token; drafts consume none. KDA snapshot is ~134 MiB at real scale. Tests force draft patterns (always-right, always-wrong, mixed) since random-weight models accept ~0, and assert stream and post-run state digests equal non-speculative decoding, including through a 1-slot tiered cache. Real-container runs at depth 2 and 3 report identical streams. Default on with depth 3; can be slower at low acceptance.
- Variants seen elsewhere: standard speculative sampling (accept with prob min(1, p/q)) preserves distribution but not the per-seed stream.

## F-tokenizer-01: Byte-level BPE with a hand-written split matcher and probed Unicode tables
Origin: [colibri] [ds4] [pulsar] [monolite] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: apus (PolyForm-NC) c/tok.h, c/uni_tables.h, tests/m2/README.md, tests/m2/gen_uni_tables.py -- read only, not copied
- Topic: tokenizer
- Purpose: reproduce the reference tokenizer exactly in libc-only C, with no regex library and no Unicode database.
- Inputs / outputs: tokenizer JSON (vocab 154,820 + 36 added tokens; merges as pairs) ; UTF-8 bytes -> ids, ids -> bytes.
- Pseudocode:
      added tokens: scan leftmost-longest literal matches first (when special recognition on)
      pre-split the rest with an ordered alternation, leftmost-first, greedy:
          case-insensitive contractions | optional non-letter/digit/newline prefix + letters
          | 1-3 digits | optional ASCII space + symbols + trailing newlines
          | whitespace ending in newline | whitespace not followed by non-space | whitespace
      letter/digit/space classes = range tables GENERATED by probing the reference tokenizer:
          wrap candidate codepoint between a digit and a letter, encode, classify from piece boundaries
      each piece -> byte-to-unicode alphabet -> BPE by merge rank (sentinel = max u32)
      decode: map alphabet chars back to bytes; pass bytes through verbatim
- Notes: case folding includes U+017F folding to 's'. The probe produced 143 digit ranges, 675 letter ranges and 8 whitespace ranges, eliminating Unicode-version drift. Exhaustive gate: 1,491,186 probe records (every codepoint to 0x2FFFF, then step 17, six probe strings each) with 0 mismatches. Post-processor adds nothing. Decoder differs from reference only on invalid UTF-8 (reference substitutes U+FFFD).
- Variants seen elsewhere: most C engines link a regex engine or use a precomputed Unicode DB.

## F-tokenizer-02: Chat template encoder validated byte-exact against jinja2 goldens
Origin: [colibri] [ds4] [pulsar] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/encoding.h, tests/m2/README.md, tests/m2/gen_golden.py -- read only, not copied
- Topic: tokenizer
- Purpose: render GLM chat prompts in C identically to the reference template, including thinking and tool calls.
- Inputs / outputs: messages JSON, tools JSON (top-level), reasoning effort, clear-thinking flag -> prompt bytes + ids.
- Pseudocode:
      out = "[gMASK]<sop><|system|>Reasoning Effort: " + (effort=="low"?"Low":effort=="high"?"High":"Max")
      if tools: system block with each tool's JSON (unwrap {"function":...}, skip deferred tools,
                drop two keys, python-style separators, non-ASCII kept)
      for message:
          user: "<|user|>" + content (not stripped)
          assistant: "<|assistant|><think>" + reasoning + "</think>" + strip(content) + tool calls
              reasoning from field, else extracted from content around think tags
              clear_thinking: drop reasoning for assistant turns before the last user turn
          tool calls: "<tool_call>name" + per arg "<arg_key>k</arg_key><arg_value>v</arg_value>" + "</tool_call>"
              arguments must be an object; strings verbatim, others JSON-dumped
          consecutive tool results -> one "<|observation|>" block, each wrapped in response tags,
              reordered to match preceding call ids when ids are complete and unique
      generation prompt: "<|assistant|><think>"
- Notes: goldens are rendered with jinja2 in the same sandboxed environment and filters as the HF stack; 21 conformance cases byte-compared on text and ids, 3 error cases must fail where the template raises. Python whitespace semantics for strip are replicated. EOS set is three ids (end-of-text, user, observation).
- Variants seen elsewhere: engines usually embed a mini-jinja interpreter; apus ports the one template and proves it.

## F-server-01: OpenAI gateway driving a single engine over NDJSON stdio
Origin: [colibri] [ds4] [pulsar] [monolite] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: apus (PolyForm-NC) tools/server.py, tests/m7a/README.md, tests/m7a/gen_fixtures.py -- read only, not copied
- Topic: server
- Purpose: keep the C engine socket-free; put HTTP, SSE, auth and GLM response parsing in a stdlib-only Python process.
- Inputs / outputs: engine requests {id, cmd: encode|generate, messages|text|ids, tools, sampling, stop}; events {encoded | prompt | token | done | error} one JSON per line.
- Pseudocode:
      gateway: one engine subprocess; global lock -> requests processed one at a time in arrival order
      generate: engine allocates fresh KV sized prompt + max_tokens (default cap 65,536 when unset)
          per token: emit token event; check EOS set and stop strings on assembled text
          stop string: truncate at match start, emit partial piece if before match, finish "stop_string"
      streaming splitter for reasoning:
          before end-of-think seen: emit all but the last (len(end tag)-1) chars as reasoning; hold the rest
          after: forward everything as content (raw tool-call markup)
      non-stream parse: cut at EOS; split at first end-of-think; parse leading tool-call blocks;
          arg values: json-parse if possible else string; malformed markup -> plain content
      errors: schema/encoding failures -> 400, engine failures -> 500/503; optional bearer key
- Notes: rationale: no socket code in C to audit, trivially testable pipes, one process = one model. No KV reuse across turns (every request re-prefills). Test fixtures are "parrot" containers: all layer weights zero (sublayers vanish, the 4-stream residual passes through since Sinkhorn of zeros is uniform and post = 1), and a scripted head so greedy decode follows a fixed Markov chain with top-2 margin ~1824, giving a known stream on any host. Speculative bursts arrive as ordinary consecutive token events, so the protocol did not change.
- Variants seen elsewhere: colibri uses the same gateway-drives-engine split; llama.cpp-style servers embed HTTP in C.

## F-config-01: Portability layer and ISA dispatch without global flags
Origin: [colibri] [pulsar] [monolite] [ktransformers] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite, ktransformers), MIT (pulsar), PolyForm-NC-1.0.0 (apus)
- Source: apus (PolyForm-NC) c/compat.h, c/x86.h, c/pool.h, Makefile, .github/workflows/ci.yml -- read only, not copied
- Topic: config
- Purpose: one C11 tree that builds and passes the same bitwise battery on macOS/ARM, Linux/x86-64 and Windows/MinGW.
- Inputs / outputs: platform shims (cpu count, aligned alloc/free, read-only open, file size, fsync, atomic replace-rename, mkdir -p, positioned read, memory pressure, no-cache fd, env knobs).
- Pseudocode:
      rename on Windows: move-file with replace-existing flag (plain rename fails if target exists)
      x86 kernels: each SIMD worker has a per-function target attribute; plain entry checks cpuid once
          (cached), env switch forces scalar; counter records that the SIMD path ran (tests assert it)
      no FMA anywhere on x86 (scalar anchor is mul+add)
      ARM: NEON compiled in unconditionally under the ARM predicate
      thread pool: lazy spawn, lanes = performance cores (env override, clamp 32), caller is lane 0;
          contiguous row blocks per lane; one job at a time; grow-only thread-local bump scratch, LIFO marks
      build: C11, -O2, warnings on, contraction off; header-only single-TU modules (impl macro in one TU)
      Linux/Windows: auto-vectorization disabled (works around an emulator miscompile; numerics unaffected)
- Notes: no BLAS dependency; libc + pthreads (winpthreads on MinGW). CI runs the battery on all three OSes, minus sanitizers and GPU on Windows. Sanitizer gates are UBSan-only on macOS because ASan hung there. The project argues disabling auto-vectorization is numerics-neutral since FP reductions are never reassociated without fast-math.
- Variants seen elsewhere: colibri's compat header is the ancestor of this pattern.

---

## Notable ideas not seen elsewhere

- Caching dequantized BF16 expert payloads (dequant on the I/O worker at fill time) so hit, miss and eager paths return byte-identical weights, then gating eager == big cache == 1-slot cache == cache+prefetch by digest.
- Speculative prefetch admission control with three drop rules (demand in flight, backlog cap, near memory ceiling), plus a promotion rule that makes it impossible for prefetch to evict demand-used experts.
- Memory guard on macOS physical footprint (counts compressed pages) because resident size under-reported a 32.5 GB footprint as below 26 GiB.
- Router-lookahead prefetch that reuses the exact gated sublayer code (hyper-connection maps, norm, router) with the target layer's weights on the source layer's state, with recall measured on real weights (~0.69 at top-12).
- Token-chunked prefill MoE to bound the live expert union (fixed a 41.5 GB prefill peak), justified as bitwise-safe by per-token MoE independence.
- Converter with a fixed space-padded safetensors header reserve sealed in place, atomic per-tensor state, torn-tail truncation, and a download driver that holds source shards still referenced by a cross-shard expert slab.
- Exact SIMD E4M3 decode via half-precision bit placement times 256, and integer-lane RNE narrowing, proven on all 256 codes.
- Bitwise SIMD GEMV by staging products and keeping strictly ordered per-row add chains (ILP only across rows), measured at the memory roofline.
- Host-transcendental probe deciding a bitwise vs tolerance test tier at runtime, plus pinning numpy to its baseline exp kernel to make goldens host-independent on x86.
- Replicating numpy's pairwise-summation tree in C to bit-match a numpy oracle.
- Two pinned KDA orderings (chunked prefill vs recurrent decode) with no cross-phase claim; verification batches therefore step the recurrence per token.
- Speculative decoding exact per seed even when sampling: accept iff draft equals the main model's own sampled pick, one RNG draw per emitted token, drafts draw none.
- Rollback by copying only recurrent state and rewinding the attention cache count, never rolling back the expert cache.
- Choosing the MTP hidden source and pairing lag by an acceptance sweep on the real model (85.3% accept, 2.53 tok/batch, 1.47x).
- Lesson made explicit: a self-written oracle and the engine shared a layout bug and matched bitwise; only an external reference anchor caught it.
- Unicode class tables generated by probing the target tokenizer itself rather than from a Unicode database; 1.49M-probe exhaustive gate.
- "Parrot" test containers: zero layer weights plus a scripted head give a known, large-margin token stream through the whole real pipeline on any host.
- GPU offload as a weak-symbol hook table with per-op fail-soft and two zero-copy buffer persistence classes, so the expert cache can free memory at any layer boundary without GPU invalidation.
