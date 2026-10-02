# Fragments: MoE expert storage on NVMe, RAM expert cache, I/O paths, prefetch

Topic tag: `store` (layout / sidecars), `cache` (RAM residency), `io` (read paths),
`pf` (prefetch). All pseudocode below is written fresh for this project after reading
the cited sources; names are ours. Numbers quoted are the projects' own stated
measurements on their own hardware, not ours.

Licences of sources: colibri (Apache-2.0), ds4 (MIT), pulsar (MIT, Rust+CUDA),
monolite (Apache-2.0), slipstream (MIT; new code under engine/), ktransformers
kt-kernel (Apache-2.0).

---

## F-store-01: One expert = one aligned, contiguous record
Origin: [colibri] [monolite] [slipstream] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (slipstream), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: monolite (Apache-2.0), src/experts_file.h, src/pager.c, convert/convert_mnl.py, docs/mnl_format.md -- read only, not copied
- Purpose: make a cache miss exactly one positioned read of one aligned block, so O_DIRECT and slot buffers line up with no repacking at load.
- Inputs / outputs: converter takes per-layer stacked expert tensors (gate+up fused, down, their scales and biases); emits a dense-weights file plus a separate experts file. Reader yields `view{w1_q, w1_scale, w1_bias, w2_q, w2_scale, w2_bias}` pointers into one slot buffer.
- Pseudocode:
      ALIGN_IO  = 4096
      ALIGN_SUB = 64                       # SIMD-friendly inner alignment
      # --- writer ---
      sub = {}; cur = 0
      for part in [w1_q, w1_scale, w1_bias, w2_q, w2_scale, w2_bias]:
          cur = round_up(cur, ALIGN_SUB); sub[part] = (cur, size(part)); cur += size(part)
      stride = round_up(cur, ALIGN_IO)
      assert stride <= 256 MiB             # reader rejects bigger as corrupt
      header = {magic, version, n_layers, n_experts, shape dims, stride, data_start, sub}
      data_start = round_up(sizeof(header), ALIGN_IO)
      write header; pad to data_start
      for layer in layers: for e in experts:
          write parts at their sub offsets; zero-pad to stride
      # --- reader ---
      offset(layer, e) = data_start + (layer*n_experts + e) * stride
      on open: check magic/version; stride != 0, stride % ALIGN_IO == 0, stride <= cap;
               data_start aligned and >= header; every sub range lies inside stride;
               file_size >= data_start + n_layers*n_experts*stride (overflow-checked)
- Notes: order is (layer, expert) so a layer's experts are near each other on disk. Inner 64 B alignment lets kernels load directly from the slot. Header is read with an aligned bounce buffer if the fd is O_DIRECT. The whole design assumes every expert of a model has the same byte size (one stride); mixed-precision experts would need per-record sizes (see F-store-02). The project states the 120B model's experts are ~59 GiB and each layer costs four ~13 MB reads.
- Variants seen elsewhere:
  - slipstream: same idea but 16 KiB alignment, per-record size and CRC in a directory (F-store-02).
  - colibri: keeps safetensors but requires each expert's gate/up/down tensors to be byte-adjacent in one shard, so the three can be fetched with one coalesced read; falls back to three reads when not adjacent. Per-row scales live in separate small "scale" tensors read buffered.
  - ds4 / pulsar / kt-kernel: no repack; experts are slices of GGUF stacked tensors (F-store-04), so one expert = three separate reads (gate, up, down) at `base + e*bytes`.

## F-store-02: Sidecar container with JSON header block and trailing CRC directory
Origin: [colibri] [monolite] [slipstream] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (slipstream), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: slipstream (MIT), engine/src/peregrine_pgrn.c, engine/src/peregrine_pgrn.h, engine/tools/pgrn-convert/pgrn-convert.cpp -- read only, not copied
- Purpose: a streamable expert-only file next to the original model, self-validating and bound to the exact source model.
- Inputs / outputs: file = [16 KiB header block][expert records, each 16 KiB aligned][directory]. Lookup (layer, expert) -> {offset, nbytes, crc, precision, flags, heat}.
- Pseudocode:
      BLOCK = 16384
      header block: 8-byte magic | u32 version | u32 json_len | canonical JSON (fits in block)
         JSON keys: source model sha256, expert_count, experts_per_layer,
                    directory offset, per-layer tensor table [layer, (type, bytes) x 3 roles]
      record(layer, e) = concat(gate_bytes, up_bytes, down_bytes); padded to BLOCK
      directory entry (26 bytes LE): layer u16, expert u16, precision u8, flags u8,
                                     heat f32, offset u64, nbytes u32, crc32 u32
      open(path, expected_hash):
          parse fixed header; json_len <= BLOCK-16; reject NUL in JSON
          extract each key exactly once (reject duplicates), dir_off % BLOCK == 0
          if expected_hash given and != json hash: refuse
          read directory; for each entry: offset >= BLOCK, offset % BLOCK == 0,
              offset+nbytes <= dir_off, precision in range
          sort by (layer, expert); reject duplicates
          check count == layers * experts_per_layer and each nbytes == sum of 3 role sizes
      find(layer, e): binary search in sorted directory
      read(entry, dst): full positioned read (retry EINTR/short), then crc32(dst) == entry.crc else error
- Notes: directory is written last, so a truncated file has no valid directory (that is why resume needs a journal, F-store-03). Verifying CRC on every read costs CPU but makes multi-disk mirrors "parity safe" (a wrong disk can never silently serve bad bytes). Heat field in the directory lets a converter ship a prior. Reader never mmaps. The 26-byte record is unaligned; decode with explicit little-endian helpers.
- Variants seen elsewhere: monolite has a fixed binary header and fixed stride, no CRC. colibri relies on safetensors headers and verifies mirrors by size + header-byte equality (F-io-20), plus SHA-256 only in its offline mirror planner.

## F-store-03: Resumable conversion with an fsync-ordered journal
Origin: [colibri] [slipstream] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (slipstream), PolyForm-NC-1.0.0 (apus)
- Source: slipstream (MIT), engine/tools/pgrn-convert/pgrn-convert.cpp; colibri (Apache-2.0), c/tools/convert_fp8_to_int4.py, c/tools/convert_fmt4_to_fmt2.py -- read only, not copied
- Purpose: survive Ctrl-C / power loss during a multi-hour convert without rewriting completed work and without trusting bytes that never hit disk.
- Inputs / outputs: source model shards -> `out.partial` + `out.journal`; on success atomic rename to `out` and delete journal.
- Pseudocode:
      journal header (fixed size, own CRC over its bytes):
          magic, version, source sha256, source size, expert_count, n_layers,
          padded payload size, crc32 over per-layer record sizes (geometry fingerprint)
      if resume:
          check header == freshly computed header (else refuse: different source/geometry)
          n_done = (journal_size - header) / DIR_ENTRY
          re-read the n_done records from .partial and verify each CRC from journal
          truncate .partial and journal to that boundary
      for each layer, in chunks sized so buffer <= ~256 MiB:
          parallel preads of chunk's experts into buffer slots (worker pool, atomic index)
          zero the padding tail of every slot
          compute crc per record; build directory entries
          write chunk payload sequentially
          fsync(payload)           # data durable first
          append chunk's entries to journal; fsync(journal)   # journal can only lag
          on cancel signal: stop at chunk boundary, exit "resumable"
      write directory; write header block at offset 0; fsync
      optional verify pass: crc sweep over every record via directory
      rename partial -> final; unlink journal
- Notes: because offsets are a pure function of geometry, a resumed prefix is bit-identical to an uninterrupted run; the stored CRCs prove it rather than assume it. Project reports: 22.8 GB source converted in about 2.25 min on an M3 Pro; two runs each interrupted twice and resumed gave byte-identical output. Hashing the whole source (for the header identity) dominated time until hardware SHA was used.
- Variants seen elsewhere: colibri keeps a per-outdir JSON progress manifest mapping input shard -> output shard name (or "empty"), plus the full conversion parameters; a resume with different parameters on the same outdir is refused (prevents mixing containers); manifest is written to a temp file then atomically replaced after each shard. Simpler: shard-granular, no CRC replay.

## F-store-04: Stacked-tensor expert addressing and triple grouping
Origin: [ds4] [pulsar] [ktransformers] [flashmoe]

Licence-of-origin: Apache-2.0 (ktransformers), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: pulsar (MIT), crates/stream/src/lib.rs, crates/engine/src/lib.rs; ds4 (MIT), ds4_cuda.cu (stream expert cache); ktransformers kt-kernel (Apache-2.0), operators/llamafile/moe.hpp -- read only, not copied
- Purpose: use an unmodified GGUF where each layer has three stacked tensors [n_expert, rows, cols] and treat one expert as three slabs.
- Inputs / outputs: (layer, e) -> three (file_offset, len) reads; a cache key that is unique per slab.
- Pseudocode:
      for each MoE layer L, role r in {gate, up, down}:
          slab_bytes[L][r] = row_bytes(type, cols) * rows
          slab_off(L, r, e) = data_start + tensor_off[L][r] + e * slab_bytes[L][r]
      cache key := absolute file offset of the slab (unique across layer/role/expert)
      expert group := the three keys of one (L, e); admit / evict as a unit where possible
- Notes: keying by file offset makes the cache model-agnostic and sharding-agnostic (a split model becomes one virtual address space, F-io-21). pulsar measured that letting single slabs evict members of warm-loaded triples dropped slab hit rate from 72% to 53%; it therefore only replaces whole triples. ds4 additionally validates that a cached entry's up/down offsets still match before trusting a gate-offset hit. Three reads per expert cost more IOPS than F-store-01 but need no conversion.
- Variants seen elsewhere: kt-kernel keeps everything in RAM; with one NUMA node it aliases the mmapped GGUF directly (no anonymous copy, quoted ~138 GB saved) and issues a WILLNEED advice over the three stacked tensors to warm the page cache; with several NUMA nodes it copies a per-node slice of the intermediate dimension of every expert (tensor-parallel split across nodes).

## F-store-05: Persistent routing-usage history (warm-start sidecar)
Origin: [colibri] [ds4] [pulsar] [monolite]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar)
- Source: colibri (Apache-2.0), c/route_trace.h, docs/tuning.md; pulsar (MIT), crates/engine/src/lib.rs (warm census save/load); monolite (Apache-2.0), src/pager.c (count save/load); ds4 (MIT), ds4.c (hotlist writer), ds4_streaming_hotlist*.inc -- read only, not copied
- Purpose: remember which (layer, expert) pairs the user's workload routes to, so the next process starts with the right experts resident.
- Inputs / outputs: per-(layer, expert) u32 counters; file next to the model.
- Pseudocode (colibri-style text history):
      file = lines "layer expert count", sparse (non-zero only)
      two header lines use NEGATIVE layer ids so old readers skip them:
          (-1, n_layers, n_experts)
          (-2, format_version, engine_id_hash)       # hash in the unsigned 3rd column
      save():
          if read-only mode: return success
          counters *= decay_factor (round half up; default 1.0 = no decay)
          if all zero: write empty file (keeps "empty means no history" semantics)
          write to path.tmp; rename over path (atomic)
      load(path, trusted):
          dims mismatch -> refuse (always)
          version newer than reader -> refuse (always)
          engine id mismatch -> refuse unless path was given explicitly by the user (then warn once)
          accept a record only if layer has a counter row (dense layers dropped) and expert in range
- Notes: decay 0.99 per save gives a half-life of ~69 turns; colibri measured that on a history of 18.2M selections one turn (~38k) moved the ranking by only 0.2%, i.e. an undecayed history stops tracking workload changes. Placement-only data: a wrong history only costs speed, never correctness, which is why a user override exists. A benchmark mode must not write history (it would skew the profile being measured).
- Variants seen elsewhere:
  - pulsar `.warm`: binary array of (offset u64, len u64, count u64) records sorted by count; merge rule on save = per-slab max of THIS run's heat (current minus the seed it was loaded with) and the stored value. Rationale: overwriting let one short run poison the census (measured: resident-tier hits halved, decode 8.2 -> 5.8 tok/s); summing ossifies it. Delete the file to reset.
  - monolite: u32 count + dense u32 array [layers*experts]; GPU-served hits must be fed back into counts, otherwise the next run's GPU set starves itself of the experts that were working.
  - ds4: text hotlist with a comment header (model, dims, totals) and lines "layer expert hits weight", sorted by hits per byte; plus compiled-in default hotlists per model for a cold first run. pulsar also seeds from a built-in hotlist when no census exists.

## F-cache-06: Fixed slot pool with in-flight states and leases
Origin: [colibri] [ds4] [pulsar] [monolite] [slipstream] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar, slipstream), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (expert slot struct, victim choice, promotion swap), c/expert_store.h; monolite (Apache-2.0), src/pager.c; slipstream (MIT), engine/src/peregrine_stream.h -- read only, not copied
- Purpose: bounded RAM for experts with safe concurrent loads, consumers and evictors.
- Inputs / outputs: N slots of `slot_bytes` (allocated once, aligned for O_DIRECT); per-(layer,expert) table -> slot or -1.
- Pseudocode:
      slot = {key or EMPTY or RESERVED(key), buf, last_use, refcount, loading, owner_layer, speculative}
      table[layer][e] = {slot_idx or -1, loading flag, use_count, state in COLD/WARM/PINNED}
      lookup(key):
          lock
          if table[key].loading: wait on cond until done        # dedupe with prefetch
          if resident: refcount++; last_use = ++clock; unlock; return view
          s = choose_victim(); detach old key; mark RESERVED(key), loading
          unlock; read into s.buf (no lock held); lock
          if ok: publish (table -> s, state WARM) else: release reservation, slot back to EMPTY with last_use = 0
          broadcast; unlock
      release(view): refcount--          # caller must release exactly once
      choose_victim():
          skip slots that are loading, RESERVED, refcount>0, owned by current layer, or PINNED
          prefer EMPTY slot that still owns a buffer
          prefer EMPTY slot without buffer only while live_buffers < cap (else it is growth, not eviction)
          else smallest last_use
- Notes: a failed speculative load must reset the slot to "first victim" (last_use 0); colibri found that leaving it marked in-flight leaked one ~19 MB slot per failed read forever. Never compact slots while readers may hold pointers into them; evict in place. Publishing only after a full, verified read means a short read can never become a hit (slipstream/colibri both enforce this). colibri admits a demand-loaded expert into the cache by swapping the working-set buffer with the victim's buffer (pointer swap, no copy) and must re-size a reused buffer when expert widths differ (int8 MTP expert vs int4 routed) to avoid all slots silently growing to the widest width.
- Variants seen elsewhere: monolite marks slots "in use for layer L" and clears all of them at end of layer instead of refcounts; minimum 8 slots so one layer's top-k always fits. ds4 CUDA path uses a per-request "stamp" and treats any slot stamped this request as protected.

## F-cache-07: CLOCK-LRU-K eviction (and the HOT/WARM extension)
Origin: [slipstream]

Licence-of-origin: MIT (slipstream)
- Source: slipstream (MIT), engine/src/peregrine_cache.c, engine/src/peregrine_tier.c, BENCHMARKS.md, README.md -- read only, not copied
- Purpose: O(1) amortised replacement that blends recency and bounded frequency.
- Inputs / outputs: capacity C, saturation K (default 4); per-slot small counter and pin flag.
- Pseudocode:
      on hit(slot): ctr[slot] = min(ctr[slot]+1, K)
      on insert(key) after the bytes are fully loaded:
          if used < C: take next free slot, ctr=1
          else if all slots pinned: fail (caller streams through staging)
          else loop:
              if pinned[hand]: advance hand
              elif ctr[hand] > 0: ctr[hand]--; advance hand
              else: victim = hand; ctr=1; advance; return
      pin(slot) for every slot the current graph references; unpin_all at next batch start
      # HOT/WARM extension (optional)
      every access: epoch++, score = sat_inc(score)
      WARM slot with score >= promote_hits (3) becomes HOT if hot pool < hot_cap,
           else swaps with the coldest HOT past its cooldown if strictly colder
      HOT slot idle >= demote_idle epochs (64) -> WARM
      eviction scans only WARM slots with the clock; HOT only if no WARM victim
      newly published slot may start with score = 1 + heat*(promote_hits-1) from a prior
- Notes: lookup is split from insert so that the I/O completes before residency changes. Measured: reserving HOT capacity cost 1-6% tok/s at 4-10 GiB caches (shrinks the general working set); pure CLOCK-K (hot share 0) was best. Cache hit rates reported 69% / 78% / 83% / 87% at 4 / 6 / 8 / 10 GiB on a 118B model.
- Variants seen elsewhere: colibri plain LRU per layer plus LFRU for the pinned tier (F-cache-09); ds4 hotness-LFU (F-cache-08); pulsar sampled LFU (F-cache-08 variant); monolite plain LRU.

## F-cache-08: Route-hotness LFU with periodic halving and hit protection
Origin: [colibri] [ds4] [pulsar]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar)
- Source: ds4 (MIT), ds4_metal.m (stream expert cache prune / hotness), ds4_cuda.cu (selected-cache begin load); pulsar (MIT), crates/engine/src/lib.rs (host store) -- read only, not copied
- Purpose: keep experts that the router keeps selecting, even if they missed last time.
- Inputs / outputs: hot[layer][expert] u32; resident entries with last_use; this step's selected ids.
- Pseudocode:
      on each router decision (hit OR miss): hot[L][e] = sat_add(hot[L][e], 1)
      every 16 decode tokens (counted at the first routed layer): hot >>= 1 for all
      resolve(layer, wanted_ids):
          uniq = dedupe(wanted_ids)
          stamp = ++clock
          pass 1: for each id in uniq that is resident: mark protected (last_use = stamp)
          pass 2: for each missing id: victim = argmin over unprotected, not-in-flight
                    entries of (hot, last_use); load into victim
      prune(layer) when over cap: repeatedly evict argmin(hot, last_use) among
                  valid, not-in-flight, not-protected
- Notes: counting misses matters: a hit-count LFU penalises experts that are repeatedly selected but evicted before a second hit. Two-pass resolve ("protect every hit first") prevents a miss from evicting an expert that a later token in the same batch hits. ds4 reports, on a 128 GB M5 Max with automatic cache sizing, generation of 11.9 / 19.3 tok/s (after initial / continued prefill) for a 145 GiB DeepSeek Flash with an 86.2 GiB expert cache and the updated eviction priorities, and 11.9 / 14.9 tok/s for a 178 GiB GLM 5.3 Q4 (workload references, not guarantees). ds4's GPU path also splits a layer into "resident experts now, missing experts after" only when >= 3 experts miss; with 1-2 misses a single pass was faster.
- Variants seen elsewhere: pulsar host cache (tens of thousands of slabs) evicts the minimum (freq, tick) among a strided sample of 64 eligible entries instead of a full scan (full scan cost seconds; first-64 was iteration-order biased); entries in the current want-list and those being read by the CPU lane are excluded.

## F-cache-09: LFRU score, hysteretic promotion and live re-pinning
Origin: [colibri] [ds4] [pulsar]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar)
- Source: colibri (Apache-2.0), c/tier.h, c/colibri.c (re-pin pass), docs/tuning.md -- read only, not copied
- Purpose: decide when a non-resident expert deserves a pinned/hot slot without ping-pong.
- Inputs / outputs: heat[e] (decaying session counter), last[e] (access clock), pinned set per layer.
- Pseudocode:
      score(e) = (heat[e] << 8) | max(0, 255 - (clock - last[e]))   # frequency dominates, recency breaks ties
      should_promote(hot, cold) = hot > cold + cold/4 + 4             # computed in 64-bit, no wrap
      repin_pass(at safe point between turns):
          for each layer: coldest_pinned = argmin score; hottest_unpinned = argmax score
              if score(hot) > score(cold) + score(cold)/4 + (4 << 8): candidate(gain)
          keep the top 4 candidates by gain; swap each (read new expert, drop old)
          heat >>= 1 (session heat decays; the persistent history does not)
- Notes: recency contributes at most 255 while one frequency unit is worth 256, so a merely recent expert cannot displace a genuinely hotter one. Saturated counters must stay saturated (widen before adding margin). Four swaps per pass ~ 80 MB of reads at ~20 MB/expert. Re-pin runs only at turn boundaries where no MoE is in flight. Exposed as "balanced" policy (re-pin every 64 tokens) in colibri.

## F-cache-10: Cache partitioning: per layer vs global
Origin: [colibri] [ds4] [pulsar] [monolite] [slipstream] [ktransformers] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite, ktransformers), MIT (ds4, pulsar, slipstream), PolyForm-NC-1.0.0 (apus)
- Source: slipstream (MIT), engine/src/peregrine_runtime.cpp; ds4 (MIT), ds4_metal.m (effective cap); colibri (Apache-2.0), c/colibri.c (per-layer cache rows) -- read only, not copied
- Purpose: decide whether layers compete for slots.
- Inputs / outputs: total slots C, layer list, optional per-layer weights file.
- Pseudocode (weighted per-layer split):
      if weights file valid and covers every layer:
          raw[i] = C * w[i] / sum(w); size[i] = max(1, floor(raw[i])); frac[i] = raw[i] - size[i]
          while sum(size) < C: give 1 to argmax frac (then frac = -1)
          while sum(size) > C: take 1 from the largest (never below 1)
      else: equal split, remainder to the first layers
      each layer gets its own CLOCK-K instance over its own arena slice
- Notes: slipstream chose per-layer partitions so cross-layer eviction is impossible and behaviour is predictable; weights come from measured per-layer working-set width (a 6 GiB cache on Qwen3.6 reported 72.1% hit rate with weighted partition). ds4 went the other way: a single global policy because a per-layer cap made effective cache size depend on depth rather than bytes. colibri uses a uniform per-layer slot cap (plus one extra row for the MTP layer) that auto-grows to fill the RAM budget.
- Variants seen elsewhere: kt-kernel chooses GPU-resident experts by a GLOBAL top-N over the (layers x experts) frequency table, which naturally gives hot layers more slots.

## F-cache-11: Warm-start pinning from history under a budget
Origin: [colibri] [ds4] [pulsar] [monolite]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar)
- Source: monolite (Apache-2.0), src/pager.c; colibri (Apache-2.0), c/colibri.c (pin load, autopin reserve), docs/tuning.md; pulsar (MIT), crates/engine/src/lib.rs (warm load); ds4 (MIT), ds4.c hotlist seeding -- read only, not copied
- Purpose: start a run with the historically hottest experts resident and immune to eviction, without starving the streaming cache.
- Inputs / outputs: counts[layer][expert]; pin budget; returns pinned set.
- Pseudocode:
      ranked = sort (layer, e) by count desc, drop zero counts
      # do not let pins eat the streaming cache
      lru_reserve = min(requested_lru_slots, floor(expert_budget / bytes_per_slot)) * bytes_per_slot
      pin_bytes = min(planned_pin_bytes, expert_budget - lru_reserve)        # automatic mode only
      for (L, e) in ranked while pinned_bytes + size(L,e) <= pin_bytes:      # price each by its REAL width
          read expert (parallel, across all drives); mark PINNED
      # explicit user pin request bypasses the reserve rule
- Notes: monolite caps pins at half the slot count and pins top 64 by default. colibri prices each candidate by its actual byte width (mixed int4/int8 rows) - pricing all at the widest bought ~half as many experts. pulsar fills the fastest tier with complete triples first (slot count floored to a multiple of 3), the next ranks into the host cache, and seeds touch counters so admission thresholds start informed. colibri states the learning cache makes it "faster the more you use it" but also that history can overfit.
- Variants seen elsewhere: ds4 seeds from compiled-in per-model hotlists; can reserve whole prefix layers instead of selected experts.

## F-cache-12: RAM admission planner (memory guard sizing)
Origin: [colibri] [ds4] [slipstream]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, slipstream)
- Source: slipstream (MIT), engine/src/peregrine_admission.c/.h, engine/src/peregrine_runtime.cpp; colibri (Apache-2.0), c/colibri.c (RAM cap planning); ds4 (MIT), ds4_ssd.c, docs/SSD_STREAMING.md -- read only, not copied
- Purpose: pick the expert-cache size from measured memory, refusing configurations that would swap or OOM.
- Inputs / outputs: total RAM, currently available RAM, model bytes, expert bytes, KV bytes, overhead, staging, min headroom, requested cache. Output: status OK/WARN/REFUSE, mode RESIDENT/STREAMING, cache bytes, reason code.
- Pseudocode:
      if total <= headroom: REFUSE(headroom exceeds RAM)
      ceiling = total - headroom
      mandatory = (model - experts) + kv + overhead + staging          # checked u64 adds
      if mandatory > ceiling: REFUSE(model exceeds ceiling)
      room = ceiling - mandatory
      cache = requested == 0 ? min(experts, room)
                             : min(requested, experts)  and REFUSE(too large) if > room
      mode = cache == experts ? RESIDENT : STREAMING
      if available unknown: REFUSE(fail closed)
      need_now = (resident - already_allocated) + max(headroom, 1 GiB)
      status = available >= need_now ? OK : WARN(reserve at risk)
      after KV/context allocation: re-check available >= headroom
- Notes: each refusal carries a distinct reason so the user knows which knob to turn. Staging buffers (io_width x slot per layer) are counted explicitly.
- Variants seen elsewhere:
  - colibri: auto budget = 88% of MemAvailable sampled before load; an explicit budget larger than MemAvailable is clamped unless overcommit is forced; reserve explicitly: working-set buffers (64 x widest expert, or budget+4 when expert count is capped), KV at max context, attention scratch (tiled cap), ~1.2 GB activations, and 2.5 GB left to the kernel page cache because starving it dropped buffered reads from ~800 to ~180 MB/s.
  - ds4: auto = recommended bytes x 80% (allowed 50..95) minus non-routed weights, divided by per-expert bytes; on GPU keeps 8 GiB free and on allocation failure backs off capacity by x3/4 until it fits or equals the minimum needed.

## F-cache-13: Runtime RSS guard
Origin: [colibri] [apus]

Licence-of-origin: Apache-2.0 (colibri), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (RSS guard, issue #403 notes) -- read only, not copied
- Purpose: the planned budget is an estimate; enforce it on measured RSS.
- Inputs / outputs: budget (resolved by F-cache-12 or explicit), current RSS.
- Pseudocode:
      every 16 emitted tokens at a safe point (no MoE in flight):
          if rss <= budget*1.02 + 0.3 GB: return
          need = rss - budget
          for pass in 1..8 while freed < need:
              for each layer: under the cache lock pick LRU among published, buffered, not-busy slots
                  hide it from lookups; free its buffer IN PLACE (slot stays, key EMPTY, last_use 0)
              per-layer cap -= 1 (min 2) so the cache does not regrow
          malloc_trim
- Notes: freeing must happen while still holding the lock the prefetch worker uses, otherwise the worker can pick the half-freed slot and read into freed memory. colibri recorded a projected 74.4 GB budget overshot to 115.6 GB real on long generations (three OOM kills) before this guard. Large buffers are mmap-backed by malloc, so free returns pages immediately.

## F-io-14: Blocking pread worker pool overlapped with compute
Origin: [colibri] [ds4] [monolite] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: monolite (Apache-2.0), src/pager.c; colibri (Apache-2.0), c/colibri.c (async load pipe) -- read only, not copied
- Purpose: overlap expert reads with the matmuls of experts already available, without io_uring.
- Inputs / outputs: list of (slot, layer, expert) misses for the current layer; per-slot ready flags.
- Pseudocode:
      # dispatch (main thread)
      gen++; publish jobs[0..n) and reset ready[]; release-store cursor = (gen << 8) | 0
      wake workers
      # worker
      loop: c = acquire-load cursor; if index(c) >= njobs or gen(c) stale: sleep
            if CAS(cursor, c, c+1): job = jobs[index(c)]  # only read job after winning CAS
                read_full(job); release-store ready[job] = 1
      # consumer (main thread)
      for each expert q in routing order: wait ready[q] (spin-yield or condvar); matmul(q)
- Notes: the generation tag in the cursor makes stale workers fail their CAS, so no extra drain barrier is needed. Defaults: monolite 4 reader threads (1..8, "threads = NVMe queue depth"), queue of 64 requests; colibri 8 I/O workers. monolite argues io_uring adds nothing for a few ~13 MB reads per layer (transfer-bound, not syscall-bound). colibri measured the async pool cut disk service time by 18%; a condvar wait instead of spin costs ~5 us against 0.5-3 ms reads and avoids stealing cycles from the OpenMP team. Full-read loop must retry EINTR and short reads; a short read is an error, never a hit.

## F-io-15: io_uring batched expert reads
Origin: [colibri] [pulsar]

Licence-of-origin: Apache-2.0 (colibri), MIT (pulsar)
- Source: colibri (Apache-2.0), c/uring.h, c/colibri.c (uring batch), docs/ENVIRONMENT.md; pulsar (MIT), crates/stream/src/lib.rs, README.md -- read only, not copied
- Purpose: submit all of a layer's misses in one syscall and let the kernel own concurrency.
- Inputs / outputs: list of reads (fd, buf, len, off); completions mapped back by user_data.
- Pseudocode (raw-syscall variant, one owning thread per ring):
      setup ring with E entries (512); mmap SQ/CQ (single mmap when supported); MAP_POPULATE
      register max io-worker count = min(io_threads, 64) for both bounded and unbounded pools
      add_read(fd, buf, len, off, tag):
          if tail - head >= entries: EAGAIN
          sqe = {op=READ, fd, off, addr=buf, len, user_data=tag, flags=FORCE_ASYNC}
          release-store tail
      submit_batch(): enter(to_submit = tail-head, min_complete = 0); reap
      wait_for(load): while not done(load): reap; enter(0, min_complete=1)
      reap(): for each CQE: req = table[user_data-1]
                if res < 0 or short and req came from a mirror: synchronous fallback read from primary
                account bytes per drive; when all parts of a load done -> mark load done
      finalize(load): wait_for(load); publish slot; drop page-cache only for the fd actually read
- Notes: FORCE_ASYNC matters: cold buffered regular-file reads can otherwise execute inline inside the submit call and serialise the submitter, killing overlap. Limit 64 loads / 512 requests per batch. A separate ring is used for speculative prefetch loads. Mirror routing must also apply on the ring path (colibri had a bug where the ring always used drive 0, leaving mirrors idle).
- Variants seen elsewhere: pulsar (via a Rust crate) keeps QD reads in flight (32 for demand misses, 16 for the prefetch ring, ring size 2xQD), allocates one aligned bracket buffer per read (offset rounded down, length rounded up to 4096, +256 B tail slack for kernels that over-read), and invokes a callback per completion so the consumer's upload/compute overlaps remaining reads; buffers move into the cache on completion (no copy). Measured: io_uring path at parity with a C reference (~4.8 GB/s); a PCIe 5.0 x4 drive sustained ~10.7 GB/s of real expert slabs, flat from QD 4 to 128; a ~7-read batch is latency bound (~6 GB/s) vs ~11 GB/s once 8+ reads are in flight.

## F-io-16: O_DIRECT and page-cache hints
Origin: [colibri] [ds4] [pulsar] [monolite] [slipstream] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar, slipstream), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/st.h, c/colibri.c, README.md, docs/tuning.md; monolite (Apache-2.0), src/pager.c; slipstream (MIT), engine/src/peregrine_io.h, BENCHMARKS.md; ds4 (MIT), ds4.c -- read only, not copied
- Purpose: stop the kernel keeping a second, uncontrolled copy of expert bytes and stop readahead waste.
- Inputs / outputs: per-shard twin fds (buffered + direct); aligned buffers.
- Pseudocode:
      open each shard twice: buffered fd, and O_DIRECT fd (macOS: F_NOCACHE; else -1)
      direct_read(off, len, dst):            # dst 4096-aligned, sized for the bracket
          base = off & ~4095; n = round_up(off - base + len, 4096)
          pread(direct_fd, dst, n, base); payload starts at dst + (off - base)
      if direct fd missing or read fails: buffered read fallback
      buffered fds for random expert access: fadvise(RANDOM) (no readahead)
      optionally after each buffered expert read: fadvise(DONTNEED) on that range
      small scale tensors stay buffered (their readahead remains useful)
- Notes: measured wins are drive dependent. colibri: +65% decode from O_DIRECT alone on a Strix Halo; +34% with the async pool on a Blackwell/Windows box; iobench 4.25 -> 9.69 GB/s on a GB10; but neutral/negative on QLC, DRAM-less or virtualised disks, so it is a flag not a default there. slipstream: using the OS page cache instead of a bounded arena was ~75% faster but caused 1000+ swap-outs; rejected for violating the memory invariant. A Linux pitfall noted by slipstream: defining a fake F_NOCACHE on Linux issues an unknown fcntl that silently fails, leaving fully buffered reads with readahead.
- Variants seen elsewhere: colibri default keeps buffered reads and lets the page cache act as a free L2 for hot experts, which is why its RAM planner reserves 2.5 GB for it; ds4 applies RANDOM+NOREUSE hints for disk-only tables.

## F-io-17: Batch-union and read coalescing
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (moe batch union), README.md; ds4 (MIT), ds4_cuda.cu (prefetch copy merge); pulsar (MIT), crates/engine/src/lib.rs (prefetcher coalesce) -- read only, not copied
- Purpose: read each needed expert once per layer step and turn many small reads into few large ones.
- Inputs / outputs: routing ids [tokens x k]; returns unique misses sorted by file offset.
- Pseudocode:
      uniq = []; seen = bitset(n_experts)
      for t in tokens: for id in topk[t]: if not seen[id]: seen[id]=1; uniq.push(id)
      hits, misses = partition(uniq, resident)
      issue hits to compute first; enqueue misses
      sort misses by file offset
      merge consecutive reads when src ranges abut AND destination slots abut
      # speculative queue coalescing (prefetch thread)
      first = recv(); batch = first                  # never truncate the first request
      for more in try_recv_all(): if len(batch) < 192: batch += more else drop the rest
      sort batch by offset; dedupe by offset; fetch
- Notes: pulsar found an unbounded speculative backlog harmful (it queued ~177 GiB of reads during prefill that then drained into decode); bounded coalescing with discard of stale requests fixed it. Sorting by offset reads faster than routing order. colibri reads one ~19 MB coalesced region per expert when gate/up/down are adjacent.

## F-io-18: Multi-SSD replica routing weighted by measured bandwidth
Origin: [colibri] [slipstream]

Licence-of-origin: Apache-2.0 (colibri), MIT (slipstream)
- Source: colibri (Apache-2.0), c/colibri.c (mirror setup, replica hash, probe), c/st.h (mirror add), docs/multidisk.md; slipstream (MIT), engine/src/peregrine_pgrn.c (stripe calibrate / pick fd), BENCHMARKS.md -- read only, not copied
- Purpose: N full (or partial) read-only copies on N drives; spread expert reads by drive speed with no RAID.
- Inputs / outputs: primary dir + list of mirror dirs; per-drive weights (measured or user override).
- Pseudocode:
      accept mirror file i only if size == primary size and its header bytes == primary header bytes
          (else that shard stays primary-only: partial mirrors allowed)
      weight[d] = user list if valid else probe(d) else 1
      probe(d): 8 parallel O_DIRECT reads of 19 MB at deterministic spread offsets of the largest shard;
                GB/s = bytes / wall      (warn: buffered probe measures page cache)
      buckets: cut[d] = round(256 * cumsum(weight)[d] / total), each drive >= 1 bucket, last = 256
      drive_for(layer, e):
          h = mix32(layer * C1 xor e * C2); b = h & 255
          return first d with b < cut[d]                 # deterministic
      read(layer, e): d = drive_for(...); if shard absent on d: d = primary
          on error/short read from mirror: log once, re-read from primary
          account bytes and reads per drive
- Notes: determinism is required: prefetch hints and the demand read must hit the same drive/page cache, and in buffered mode an expert must never be cached twice. Measured: two independent NVMe drives took 0.80 -> 1.10 tok/s (+37.5%) with O_DIRECT, ~16% buffered; adding a 10x slower SATA drive with EQUAL shares dropped 0.900 -> 0.666 tok/s, after bandwidth weighting it was neutral (1.030 vs 1.006). "More drive bandwidth != more tok/s": compute and residency dominate once hit rate is high.
- Variants seen elsewhere: slipstream supports one mirror, hashes the 16 KiB block index into 1024 buckets with a primary share = sp/(sp+sm) clamped to [2%, 98%] from 6 probe reads of one record size per disk; measured tok/s-negative when the second disk shares a bus (internal NVMe + slow USB), shipped opt-in for capacity.

## F-io-19: Weighted intra-expert striping across replicas
Origin: [colibri] [slipstream]

Licence-of-origin: Apache-2.0 (colibri), MIT (slipstream)
- Source: colibri (Apache-2.0), c/colibri.c (stripe plan / striped pread), c/tests/test_mirror_stripe_split.c -- read only, not copied
- Purpose: cut the latency of ONE large cold expert read by reading disjoint pieces from every drive in parallel.
- Inputs / outputs: aligned (base, len) of one expert; replicas holding the shard with their weights.
- Pseudocode:
      if not direct mode or replicas < 2 or len < 4 MiB: return NOT_APPLICABLE
      order replicas starting at the hash-routed one (keeps first-piece load spread)
      for i in 0..n-1:
          k = (routed + i) % n
          size[i] = (i == n-1) ? len - acc : round_up(len * w[k] / sum(w), 4096)
          clamp so acc + size <= len; off[i] = acc; acc += size
      skip zero-size pieces (track the real owning drive per piece for accounting)
      run piece 0 on caller thread, others on helper threads; join
      any piece short -> whole attempt fails -> single-drive read fallback
- Notes: only with O_DIRECT (no page cache, so no double caching). colibri numbers for a 19 MB expert on 990 Pro (5.69 GB/s) + SN850P (5.03) + MX500 SATA (0.44): equal 2-way join 2.0 ms; equal 3-way 12.6 ms; weighted 3-way 1.9 ms. Thread create per read is acceptable at ~19 MB; a persistent per-drive worker would be cleaner.

## F-io-20: Capacity layouts across drives and partial mirrors
Origin: [colibri] [pulsar]

Licence-of-origin: Apache-2.0 (colibri), MIT (pulsar)
- Source: pulsar (MIT), crates/stream/src/lib.rs (split fetcher), README.md; colibri (Apache-2.0), c/colibri.c (model dirs split), c/tools/mirror_plan.py, docs/multidisk.md -- read only, not copied
- Purpose: put DIFFERENT shards on different drives (capacity), or copy only the hot subset to a small extra drive.
- Inputs / outputs: shard list with virtual base offsets; usage history.
- Pseudocode:
      virtual file = concat of shards; shard i covers [base_i, base_{i+1})
      route(voff): i = upper_bound(bases, voff) - 1; return (fd_i, voff - base_i)   # tensors never straddle shards
      partial mirror plan:
          rank shards by summed usage of the routed experts they contain
          copy best shards to the mirror until free-space reserve would be violated
          copy via temp file, verify SHA-256, never delete an existing mirror shard,
          publish a receipt file atomically only after all selected shards verified
- Notes: pulsar: shard-per-drive gives parallelism "for free" but its decode only pulled 1.4-2.1 GB/s from disk because tiers absorbed most lookups, so bandwidth is rarely the limit on one modern NVMe. Capacity split and mirror replicas are orthogonal and can be combined.

## F-pf-21: Router lookahead prefetch (next layer's router on current hidden state)
Origin: [colibri] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (router-lookahead prefetch and its loader), docs/tuning.md, README.md, CHANGELOG.md; pulsar (MIT), crates/engine/src/lib.rs (prefetcher, decode loop), docs/DESIGN-expert-store.md; monolite (Apache-2.0), src/model.c, src/pager.c -- read only, not copied
- Purpose: start disk reads for layer L+1 while layer L computes.
- Inputs / outputs: post-attention hidden x of layer L (decode, 1 token); router weights of L+1; prefetch queue.
- Pseudocode:
      at layer L after attention, before L's expert matmuls:
          h = x
          if two_step and layer L has shared expert:      # cheap correction, no disk
              h = x + shared_expert_L(norm_L(x))
          logits = router_{L+1}(norm_{L+1}(h)); apply L+1's scoring + bias
          cand = topk(logits, k_pf)                       # k_pf 8 (6 when doing real loads)
          for e in cand: if not resident(L+1, e) and not reserved: push (L+1, e) to ring (4096 entries)
      prefetch worker: pop; if L+1 <= layer main thread already entered: drop
          reserve slot (F-pf-22 rules); read; publish
      score: of experts actually routed at L+1, fraction predicted at L (cheap to log)
- Notes / measured: colibri: L+1 router on L's post-attention state recalls 71.6% of the true top-8 vs 41.3% for "same experts as previous token"; making prefetch real loads (not WILLNEED hints) added +11 pp hit rate on a big-cache host; the shared-expert two-step adds ~+2.3-3% recall; on a Strix Halo decode went 1.10 -> 1.83 tok/s with O_DIRECT + async pool + real-load lookahead + two-step. Hint-only lookahead can be net negative on disk-saturated hosts. pulsar: described as a measured +21% cross-layer gain; it computes the next router on the same normalised input, sends predicted slab reads over a channel to a background thread with its own ring, and absorbs completed slabs into the host LFU at the next resolve (ownership moves, no shared locking). monolite: default lookahead off; when on, prefetches up to N (<=16) next-layer experts ranked by historical counts (not router), only when the CPU had no experts to run this layer.
- Variants seen elsewhere: slipstream uses static tables instead of the router (F-pf-23) and measured them as losses.

## F-pf-22: Safety rules for speculative loads
Origin: [colibri] [ds4] [pulsar] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (lookahead loader handshake, eviction guard); ds4 (MIT), ds4_cuda.cu (prefetch finish); monolite (Apache-2.0), src/pager.c (relaxed prefetch) -- read only, not copied
- Purpose: speculation may waste bandwidth but must never corrupt, block, or evict the warm set.
- Pseudocode:
      speculative writer may only touch cache rows of layers > layer the main thread has entered
      main thread entering layer L: set entered = L; wait until inflight[L] == 0
      reserve(L, e) under lock: dedupe vs resident AND vs other reservations;
          choose victim; mark RESERVED(e) with last_use = MAX (never a victim while loading)
      eviction guard: if victim has >= 2 demand accesses and
          lfru(victim) + lfru(victim)/4 + (4<<8) > lfru(candidate): drop the speculation
      on success: publish, last_use = fresh tick
      on failure: unreserve, last_use = 0 (first victim), log one line, continue (non-fatal)
      speculatively loaded entries enter with LOWEST priority (ds4: used = 1) so only a real
          router hit promotes them
- Notes: colibri found its first guard (protect any warm victim) dropped ~100% of speculations once the cache filled, collapsing real-load mode to hint-only; narrowing to ">= 2 accesses and clearly hotter" fixed it. monolite distinguishes "current layer" prefetch (pinned to layer until consumed) from "relaxed" lookahead (evictable immediately) and counts usefulness separately.

## F-pf-23: Co-activation coupling tables
Origin: [colibri] [slipstream]

Licence-of-origin: Apache-2.0 (colibri), MIT (slipstream)
- Source: colibri (Apache-2.0), c/colibri.c (coupling prefetch), c/tools/route_pairs.py; slipstream (MIT), engine/src/peregrine_predict.c/.h, engine/src/peregrine_runtime.cpp, BENCHMARKS.md, README.md -- read only, not copied
- Purpose: predict layer L+1 (or L+2) experts from the ids layer L just chose, with a table lookup instead of a router matmul.
- Inputs / outputs: offline table T[L][dL][e] -> up to M (16) (successor, weight); fired ids at L.
- Pseudocode:
      build offline from routing traces: count (L,e) -> (L+dL, e') co-occurrences per position; keep top M
      predict(L, fired):
          score[*] = 0
          for e in fired: for (s, w) in T[L][1][e]: score[s] += w
          take top K (8) by score (ties by id); skip resident; enqueue as speculative
      online variant: counts[L][e][e'] (u16, saturating), updated from live routing; enabled only
          if L*E*E counters fit a small budget (<= 16 MB)
      binary format idea: magic, version, layer count; per layer sorted expert blocks with
          (successor u16, weight u16) lists; strictly ascending ids validated on load
- Notes / measured: colibri reports median co-activation lift 1.8x over independence (p99 40x) and that the structure transfers across workloads. slipstream: coupled table +14-20 pp recall over a per-layer marginal hot set in trace replay, yet end-to-end the online predictor was -8% at a 6 GiB cache and a naive coupled prefetch was 5-15% slower (cache thrash on a fetch-bound path). Lesson: recall gains do not pay unless reads are cheap relative to the evictions they cause.

## F-pf-24: Whole-next-layer prefetch for prefill (coverage-gated)
Origin: [ds4] [pulsar]

Licence-of-origin: MIT (ds4, pulsar)
- Source: pulsar (MIT), crates/engine/src/lib.rs (prefill layer pipeline); ds4 (MIT), ds4_cuda.cu (stream prefetch), ds4.c (prefill layer pread) -- read only, not copied
- Purpose: during prefill a chunk touches nearly every expert, so fetch all of layer L+1 while L computes.
- Pseudocode:
      p_untouched = (1 - k/E) ^ tokens_in_chunk
      if 1 - p_untouched >= 0.75 and not a short verify batch:
          enqueue every non-resident slab of layer L+1 to the background fetcher
      ds4 GPU form: when slots >= experts(L) + experts(L+1): reserve victims (LRU, not in L or L+1),
          read L+1 in file order via a 2-buffer pinned staging ring on a helper thread,
          publish at next layer start with lowest priority; cancel on any error
- Notes: pulsar measured an ungated version (any chunk > 8 tokens) issued 177 GiB of speculative reads across 75 layers into a ~20 GB cache: GLM-5.2 fell to 1.43 tok/s vs 2.63 with prefetch off; the coverage gate (12-token prompt reaches only ~32% of experts) fixed it.

## F-pf-25: Cache-aware routing (lossy, opt-in)
Origin: [colibri]

Licence-of-origin: Apache-2.0 (colibri)
- Source: colibri (Apache-2.0), docs/CACHE_ROUTE.md, c/colibri.c (residency probe) -- read only, not copied
- Purpose: trade routing fidelity for hits by substituting resident experts for low-ranked misses.
- Pseudocode:
      ranked = sort experts by router score
      chosen = ranked[0:J]                               # J = 2 always kept
      for slot in J..K-1:
          pick highest-ranked expert within top M (12) (or within cumulative mass P) that is resident
              (tiers searched fastest first) and not chosen; else next ranked
      optionally scale substituted gate mass by alpha before renormalising
      log swap %, overlap with true top-K, KL of gate mass
- Notes: changes model outputs; default OFF; must pass quality gates. Complementary to prefetch, which never changes expert ids.

---

## Recommended synthesis (CPU-first C engine, Linux, 4 dedicated NVMe, 128 GB RAM)

1. **Container:** convert once into our own sidecar: one record per (layer, expert) holding gate|up|down|scales contiguously, 4 KiB-aligned records with 64 B inner sub-alignment (F-store-01), per-record size + CRC32 + precision in a trailing directory and a source-model SHA-256 bound into a header block (F-store-02). Allow per-record sizes so mixed-precision layers work. Keep GGUF-slice addressing (F-store-04) only as a no-convert fallback.
2. **Converter:** chunked parallel preads, fsync payload before appending directory entries to a CRC-headed journal, CRC-verified resume, `.partial` + atomic rename, optional final CRC sweep (F-store-03); record conversion parameters and refuse mixing.
3. **Placement on 4 drives:** full byte-identical replicas on all four (validated by size + header + model hash), not a capacity split, unless the model does not fit one drive; then shard-split + partial hot mirrors (F-io-20). Route each expert deterministically by hash into 256 buckets weighted by a startup O_DIRECT probe (F-io-18), fall back to another replica on any error, and stripe large cold reads (>= 4 MiB) across all four in bandwidth-proportional 4 KiB-aligned pieces (F-io-19). Verify CRC on mirror reads only (cheap insurance).
4. **I/O engine:** O_DIRECT by default on dedicated NVMe (F-io-16) with buffered fallback and `fadvise(RANDOM)`; no reliance on the page cache. One io_uring per drive (or one ring with per-drive tags) using forced-async SQEs, registered fixed buffers over the slot arena, ~32-64 reads in flight per drive (pulsar shows a single drive saturates from QD 8 up), per-completion callbacks so matmuls start as soon as each expert lands (F-io-15). Keep a pread worker pool (F-io-14) as the portable fallback.
5. **Batch-union + coalescing:** dedupe routed experts across the batch, compute hits first, sort misses by offset, merge abutting reads; bound speculative queues (cap ~192 reads, discard stale) (F-io-17).
6. **Slot pool:** fixed arena of equal-size aligned slots allocated at start (huge-page friendly, NUMA-bound per socket if multi-socket), lease/refcount + RESERVED/loading states, publish only after a full verified read, failed loads reset to first-victim (F-cache-06).
7. **Eviction:** global (not per-layer) CLOCK-LRU-K (K=4) over the RAM tier (F-cache-07), with ds4-style two-pass "protect all hits of this step first" and route-hotness that counts misses and halves every ~16 tokens as the victim tie-breaker (F-cache-08). Skip a reserved HOT sub-tier (measured negative); provide per-layer weighted partitioning only as an experiment (F-cache-10).
8. **Pinned hot set:** at start pin from history ranked by count, priced at true record size, capped so at least the configured streaming reserve survives (<= ~50% of the expert budget by default) (F-cache-11); live re-pin at turn boundaries with LFRU score and 25%+4 hysteresis, max 4 swaps per pass (F-cache-09).
9. **History sidecar:** self-identifying text or binary file (dims, version, engine/model hash), atomic tmp+rename save, decay ~0.99 per save, merge rule = max of per-run deltas to resist poisoning, read-only mode for benchmarks; include counts for experts served from any faster tier (F-store-05).
10. **RAM sizing for 128 GB:** admission planner with explicit line items (dense weights, KV at max context, scratch, staging = in-flight reads x slot, ~1-2 GB activations) and an OS headroom of max(4 GB, 3%); auto expert cache = remainder; fail closed if MemAvailable unreadable (F-cache-12); plus the measured-RSS guard every 16 tokens that drops LRU slots in place and lowers the cap (F-cache-13). Since we use O_DIRECT, no page-cache reserve beyond headroom is needed.
11. **Prefetch:** router lookahead for decode (next layer's router on the current post-attention state, optional shared-expert two-step), top-6..8 predictions, real loads into reserved slots under the safety rules of F-pf-22 (layer-ownership barrier, narrowed eviction guard, lowest-priority publish) (F-pf-21). Gate it on measured recall and on available drive headroom; make it per-run switchable since it can lose on saturated disks.
12. **Prefill:** coverage-gated whole-next-layer prefetch (>= 75% expected expert coverage) (F-pf-24); otherwise batch-union only.
13. **Coupling tables / cache-aware routing:** ship as off-by-default experiments with telemetry (F-pf-23, F-pf-25); evidence says coupling recall gains did not translate into tok/s on fetch-bound hosts, and cache-aware routing changes outputs.
14. **Telemetry from day one:** per-drive bytes/reads, hit/miss per tier and per layer, prefetch issued/useful/dropped, wait time vs read time, predicted-vs-actual recall; every policy above has a measured case where it lost, so all must be A/B-able.
