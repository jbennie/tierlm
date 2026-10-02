# Consensus: which design patterns the reference projects converge on

This file groups the fragments in this folder into **patterns** (one idea, however many
fragments describe it), records which of the eight reference projects each pattern
appears in, and turns that count into a verdict for the library.

## Method

- **Evidence is convergence.** A pattern that several independently written engines arrived
  at is more likely to be right than one project's idea. The count of projects in which a
  pattern appears is the main evidence; measurements quoted in the fragments are the second.
- **Ideas, not code.** Every fragment is pseudocode written after reading a project. Algorithms,
  layouts and measured lessons are distilled; no source, comments or identifiers are copied
  (see FORMAT.md). `Licence-of-origin` is provenance for review, not permission to copy:
  patterns whose only origins are apus (PolyForm-NC-1.0.0) or flashmoe (no licence file) are
  re-derived from first principles or public specifications and should be reviewed with that
  in mind.
- **Origins.** A project counts when it implements the pattern in its own code, or ran and
  measured it (rejected patterns count the projects that measured them). Each fragment's
  `Origin:` line is the union of the origins of the patterns it is a primary member of.
  Origins were taken from each fragment's Source and "Variants seen elsewhere" notes and
  extended by grep checks of local clones of the reference projects (for example: ds4
  and pulsar also implement DeepSeek V4 mHC, hash routing and MTP; pulsar has prefix reuse and
  MTP; ds4 uses O_DIRECT on CUDA/ROCm and F_NOCACHE on macOS; colibri and flashmoe ship C
  OpenAI-compatible servers; ds4 has a thinking-budget controller).
- **Verdict rule.** `adopt` if Count >= 3, or if the pattern is the only correct way
  (model-defined maths, a file-format definition, or a correctness requirement - stated in
  the cell); `adopt-tested` if Count = 2 (adopt behind a measurement gate); `experimental` if
  Count = 1 (off by default, A/B only); `reject` if a project measured it negative (this
  overrides the count).
- **Lineage (†).** Some projects are not independent. Count is the literal number of tags,
  as on the fragments' Origin lines; where it includes a lineage-linked pair the Notes cell
  starts with † and gives the independent count. Treat a † `adopt` whose independent count is
  below 3 as weaker evidence.

| Linked | To | Basis |
|---|---|---|
| apus | colibri | apus's NOTICE states its tiered expert store (slab store, bounded RAM cache with eviction pins, RSS guard), the router-lookahead prefetch pilot, its compat shims and the gateway-drives-engine split follow colibri's design and adapt colibri code. |
| slipstream | colibri | slipstream's README credits colibri for its disk-benchmarking method, route-trace and expert-coupling analysis and RAM-admission idea; it keeps a colibri reference contract. |
| monolite | ds4 | monolite credits ds4 for the narrow, model-specific engine philosophy only; it states no ds4 code is included. Not treated as a pattern-level link. |
| ds4 / pulsar / ktransformers / slipstream | GGML | GGUF block formats come from the public GGML specification; agreement on those layouts is expected, not independent evidence. slipstream is a llama.cpp fork: only its new engine/ code counts as slipstream. |

### Fragment ids

Ids in the four multi-project files are unique and are used as-is. The single-project files
reuse short ids (for example `F-io-01` exists in both), so in this file their fragments are
written `apus/F-...` (origin_apus.md) and `fmoe/F-...` (origin_flashmoe.md). In the Fragments
column, ids before `also:` are primary members (they set the fragment's Origin line); ids after
`also:` mention or measure the pattern without being about it.

### Project tags and licences

| Tag | Licence |
|---|---|
| colibri | Apache-2.0 |
| ds4 | MIT |
| pulsar | MIT |
| monolite | Apache-2.0 |
| slipstream | MIT |
| ktransformers | Apache-2.0 |
| apus | PolyForm-NC-1.0.0 |
| flashmoe | no-licence |

### Totals

172 fragments, 145 patterns: 92 adopt, 22 adopt-tested, 21 experimental, 10 reject
(P-128..P-145, the Distributed section, added 7 adopt, 4 adopt-tested and 7 experimental).

Of the 92 adopt verdicts, 27 are model-, format- or correctness-defined (the maths is fixed by the
checkpoint or file format, or delegation must not change output; 8 of those have Count < 3),
and 8 carry a † lineage note. The remaining
design choices with Count >= 3 and no lineage caveat are the strongest consensus in this library.

## Storage / format

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-01 | One expert = one contiguous record, so a cache miss is one positioned read (offline repack or byte-adjacent slab) | F-store-01, F-store-02, apus/F-format-02, fmoe/F-storage-02 | [colibri] [monolite] [slipstream] [apus] [flashmoe] | 5 | adopt | † apus's slab store follows colibri's design; independent count 4. Alignment differs: 4 KiB records + 64 B sub-alignment (monolite), 16 KiB (slipstream), none beyond 128 KiB-multiple sizes (flashmoe), safetensors adjacency (colibri, apus). colibri falls back to three reads when members are not adjacent; apus moves bytes at conversion so the fallback never happens. flashmoe measured one file per expert at -15% (keep one container or one file per layer). |
| P-02 | No-repack addressing: expert = slice of a stacked [E, rows, cols] tensor at base + e*stride; cache keyed by absolute file offset | F-store-04, fmoe/F-storage-01 | [ds4] [pulsar] [ktransformers] [flashmoe] | 4 | adopt | Costs three reads per expert (more IOPS than the record layout). pulsar: letting single slabs evict members of warm-loaded gate/up/down triples dropped slab hit rate 72% -> 53%, so admit/evict the triple as a unit (pulsar; ds4 checks sibling offsets before trusting a hit). flashmoe uses the arithmetic only to drive its repacker. Keep as the no-convert fallback. |
| P-03 | Self-describing, validated container: magic/version/geometry header or manifest; every offset and size bounds- and overflow-checked on open; optional per-record CRC and source-model hash binding | F-store-01, F-store-02, apus/F-format-01; also: fmoe/F-storage-02 | [monolite] [slipstream] [apus] | 3 | adopt | CRC-on-every-read and SHA-256 source binding are slipstream-only (CPU cost; makes mirrors parity-safe). colibri checks mirrors by size + header bytes; apus binds a config hash in its manifest; flashmoe relies on constants compiled into the engine (no validation). apus/colibri keep plain safetensors so standard tools still read the container; monolite/slipstream/flashmoe use custom binaries; nothing measured favours either. |
| P-04 | Crash-resumable conversion: durable progress record (CRC journal / per-shard manifest / per-tensor state), payload fsync before progress, refuse resume on changed parameters or geometry, .partial + atomic rename | F-store-03, apus/F-format-03; also: fmoe/F-storage-02 | [colibri] [slipstream] [apus] | 3 | adopt | slipstream (two runs interrupted twice) and apus (crashes mid-tensor, at seal, mid-slab) both measured byte-identical output vs one-shot runs. Disk-bounded variants: apus converts one downloaded shard at a time (peak ~ container + one 5 GiB shard); flashmoe repacks layer subsets when disk is short. |
| P-05 | Converter fails loudly and verifies bytes: pre-check names, shapes, sizes, NaN codes and free disk; byte-compare or CRC-sweep the output against the source | apus/F-format-01, fmoe/F-storage-02; also: F-store-03, apus/F-format-03 | [colibri] [slipstream] [apus] [flashmoe] | 4 | adopt | colibri's fmt4->fmt2 converter asserts byte/scale counts against a shape table and aborts on anything unclassified (grep-confirmed in tools). flashmoe spot-verifies four experts per layer; apus verifies every tensor; slipstream has an optional full CRC sweep. apus caught its own slab-size arithmetic slip this way. |
| P-06 | Dense (non-expert) weights kept resident in their own file/blob (mmap is fine there), pointers resolved once per layer; experts in separate streamable file(s) | fmoe/F-format-01; also: F-store-01, F-store-02, F-store-04, F-FMT-02 | [ds4] [monolite] [slipstream] [ktransformers] [flashmoe] | 5 | adopt | flashmoe: mmap is right for the resident blob but 5x slower for cold experts (see mmap-cold reject). ds4 maps the whole GGUF privately for the CPU path; a shared mapping on the CPU path caused macOS VM accounting panics. ktransformers aliases the mmapped GGUF on one NUMA node (~138 GB saved). slipstream's reader never mmaps the expert sidecar. |
| P-07 | Persistent per-(layer, expert) routing-usage history next to the model for warm starts: self-identifying (dims, version, engine/model id), atomic save, read-only mode for benchmarks; shipped hotlists as cold-start prior | F-store-05; also: F-cache-11, F-store-02 | [colibri] [ds4] [pulsar] [monolite] | 4 | adopt | Merge rule is contested (see contradictions): overwrite let one short run poison the census (pulsar: resident-tier hits halved, decode 8.2 -> 5.8 tok/s); undecayed sums ossify (colibri: one turn moved the ranking 0.2%). Use decay ~0.99/save (colibri) or max of this run's delta vs stored (pulsar). ds4/pulsar ship compiled-in hotlists; slipstream's directory carries a converter-supplied heat prior. apus dropped history in its port. |
| P-08 | Atomic write-temp-then-rename (plus fsync where durability matters) for every state, sidecar, probe-cache and status file | also: F-store-03, F-store-05, F-HW-03, F-SRV-03, apus/F-format-03, apus/F-config-01 | [colibri] [pulsar] [monolite] [slipstream] [apus] | 5 | adopt | Cross-cutting; no fragment is primarily about it. Windows needs a replace-existing move because plain rename fails when the target exists (apus). |
| P-09 | Compressing or re-laying-out expert files (LZ4 / LZFSE / filesystem compression, co-occurrence reordering, one file per expert, compressed RAM tier) | fmoe/F-storage-03 | [flashmoe] | 1 | reject | flashmoe: LZ4 3.55 vs 4.36 tok/s (-13%) because most reads are warm and decompression (0.68 ms/layer) dominates; LZFSE too slow; FS compression ~2x slower; per-expert files -15%; clustering 0%. Entropy coding may pay only when reads are mostly cold. |

## Cache

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-10 | Bounded application-level RAM expert cache: fixed pool of equal, aligned slots allocated once; buffers recycled or pointer-swapped, never allocated per fill | F-cache-06, apus/F-cache-01; also: apus/F-cache-03 | [colibri] [ds4] [pulsar] [monolite] [slipstream] [apus] | 6 | adopt | † apus's bounded cache is adapted from colibri; independent count 5. Contradicted by flashmoe's no-cache result on unified memory (see contradictions). apus recycles payload/staging buffers because an mmap/munmap + zero-fill per 25-48 MiB fill was major kernel time. |
| P-11 | No application expert cache: read the K experts into fixed scratch every layer and let the OS page cache keep hot experts ("trust the OS") | fmoe/F-cache-01; also: F-io-16 | [flashmoe] | 1 | experimental | flashmoe +38% (4.11 -> 5.74 tok/s at 2-bit, 48 GB unified memory, ~71% page-cache hit); wired GPU buffers had shrunk the page cache and driven the compressor. Against: slipstream measured the page-cache path ~75% faster but with 1000+ swap-outs and rejected it; colibri keeps an app cache and reserves 2.5 GB of page cache as a free L2. |
| P-12 | Concurrency-safe slot lifecycle: reserve under lock (dedupe vs resident and in-flight), read without the lock, publish only after a full verified read, never evict anything in use this step (refcount/lease, per-layer in-use mark, request stamp, protect-all-hits-first, working set until layer end, graph pins), failed load resets the slot to first victim | F-cache-06, apus/F-cache-01; also: F-cache-07, F-cache-08, F-pf-22 | [colibri] [ds4] [pulsar] [monolite] [slipstream] [apus] | 6 | adopt | † apus follows colibri's slot design; independent count 5. colibri: a failed speculative load left marked in-flight leaked one ~19 MB slot forever; never compact slots while pointers are out. apus gates eager == big cache == 1-slot cache by output digest (41 forced evictions, identical). |
| P-13 | CLOCK-LRU-K replacement (K = 4 saturating counters, pins, lookup split from insert) | F-cache-07 | [slipstream] | 1 | experimental | slipstream hit rates 69/78/83/87% at 4/6/8/10 GiB on a 118B model. No other project ran it. |
| P-14 | Reserved HOT sub-tier (promotion/demotion) inside the RAM cache | F-cache-07 | [slipstream] | 1 | reject | slipstream: -1% to -6% tok/s at 4-10 GiB caches (it shrinks the general working set); pure CLOCK-K was best. |
| P-15 | Frequency-aware eviction that counts every router selection (hit or miss), decays periodically (halve every ~16 tokens), recency only as tie-break | F-cache-08, F-cache-09 | [colibri] [ds4] [pulsar] | 3 | adopt | A hit-count LFU penalises experts evicted before their second hit. pulsar evicts the min over a strided sample of 64 (full scan cost seconds; first-64 was order-biased). colibri LFRU = freq<<8 \| recency(<=255). Plain LRU is used by monolite, apus and colibri's per-layer rows with no head-to-head; flashmoe observed the OS (recency + frequency) beating strict LRU. |
| P-16 | Live re-pinning at turn boundaries using LFRU score with hysteresis (gain > 25% + 4 units), at most 4 swaps per pass | F-cache-09 | [colibri] | 1 | experimental | apus's base engine re-pinned; the apus port dropped it. No measured gain recorded. |
| P-17 | Per-layer cache partitions (uniform, or weighted by measured per-layer working-set width) | F-cache-10, apus/F-cache-01 | [colibri] [slipstream] [apus] | 3 | adopt | † apus's per-layer LRU follows colibri; independent count 2. slipstream 72.1% hit at 6 GiB on Qwen3.6 with a weighted split, chosen for predictability. Contradicted by the global-cache pattern (ds4). apus: derive slots from sparse layers only (a bug gave 7 instead of 8 slots/layer). |
| P-18 | One global expert cache across layers (bytes and frequency decide which layer gets slots) | F-cache-10 | [ds4] [pulsar] [monolite] [ktransformers] | 4 | adopt | ds4 moved to one global policy because a per-layer cap made effective size depend on depth; ktransformers picks GPU-resident experts by a global top-N over (layer x expert) counts; monolite has one LRU pool (grep-confirmed victim scan); pulsar's host LFU spans all slabs. |
| P-19 | Warm-start pinned set: pin the historically hottest experts at load, priced at true byte width, capped so a streaming reserve survives; explicit user pins bypass the cap | F-cache-11; also: F-store-05 | [colibri] [ds4] [pulsar] [monolite] | 4 | adopt | colibri: pricing every candidate at the widest width bought ~half as many experts. monolite caps pins at half the slots; pulsar fills the fastest tier with whole triples. Mixed-width records (int8 MTP vs int4 routed in colibri; per-record precision in slipstream) need width-aware pricing and buffer re-sizing. |
| P-20 | RAM admission planner: expert cache = (total - headroom) - explicit mandatory line items (dense, KV at max context, staging = in-flight x slot, scratch, activations); auto mode; fail closed when availability is unknown; one typed, actionable refusal per reason | F-cache-12, F-HW-05 | [colibri] [ds4] [slipstream] | 3 | adopt | † slipstream credits colibri for the RAM-admission idea; independent count 2. Defaults disagree: colibri 88% of MemAvailable plus 2.5 GB page-cache reserve (starving it cut buffered reads 800 -> 180 MB/s); ds4 80% of recommended, GPU keeps 8 GiB and backs off x3/4 on allocation failure; slipstream headroom clamp(total/8, 512 MiB, 3 GiB), macOS 3 GiB (1.5 GiB stalled below 0.1 tok/s); monolite's plan suggests 60%. Typed refusal reasons are slipstream's (a single generic message left users stuck). |
| P-21 | Runtime memory guard: at safe points every N tokens compare measured RSS / physical footprint with the budget; free LRU payloads in place under the cache lock and lower the cap | F-cache-13, apus/F-cache-03 | [colibri] [apus] | 2 | adopt-tested | † apus's guard is adapted from colibri; independent count 1. colibri: a planned 74.4 GB budget overshot to 115.6 GB real (three OOM kills) before the guard. apus: on macOS use physical footprint; resident size read below 26 GiB while the footprint was 32.5 GB, so an RSS guard never fired. flashmoe sidesteps it with a static resident set. |
| P-22 | Cache dequantised (BF16) expert payloads, dequantising on the I/O worker at fill time | apus/F-cache-01 | [apus] | 1 | experimental | Doubles bytes per slot (48 vs 24 MiB). Buys identical bytes on hit/miss/eager paths and keeps dequant off the compute thread. Contradicts keep-packed (colibri measured unpacked int8 experts at 34 ms/token vs a 9 ms packed floor). |

## I/O

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-23 | Persistent blocking positioned-read worker pool (about one thread per NVMe queue slot) overlapped with compute; the consumer waits per expert as each lands | F-io-14, fmoe/F-io-01; also: apus/F-cache-01, F-pf-24 | [colibri] [ds4] [monolite] [apus] [flashmoe] | 5 | adopt | flashmoe: 4-way parallel pread ~9x faster than sequential; persistent pool 3.81 vs 3.97 ms/layer (4.28 vs 4.09 tok/s). colibri: pool cut disk service time 18%; condvar wait (~5 us) instead of spinning avoids stealing OpenMP cycles. flashmoe found little to overlap (routing is the last step before I/O) and that overlap with GPU work hurts on unified memory. |
| P-24 | io_uring batched expert reads (forced-async SQEs, per-completion callback, ~32-64 in flight, separate ring for speculation) | F-io-15 | [colibri] [pulsar] | 2 | adopt-tested | pulsar: parity with a C reference (~4.8 GB/s); a PCIe 5.0 drive sustained ~10.7 GB/s flat from QD 4 to 128; a ~7-read batch is latency-bound (~6 vs ~11 GB/s at 8+ in flight). Without FORCE_ASYNC cold buffered reads run inline and serialise the submitter (colibri). monolite argues no gain for four ~13 MB reads per layer (argument, not measurement). flashmoe (macOS): POSIX aio -7%, dispatch_io -70% vs threaded pread. |
| P-25 | Cache-bypass expert reads as a switch: O_DIRECT twin fd with 4 KiB-aligned bracket and buffered tail, F_NOCACHE on macOS, buffered fallback on any failure | F-io-16, F-FMT-02; also: F-io-15, apus/F-io-01 | [colibri] [ds4] [pulsar] [monolite] [slipstream] [apus] | 6 | adopt | colibri: +65% decode (Strix Halo), +34% (Blackwell/Windows), microbench 4.25 -> 9.69 GB/s; neutral or negative on QLC, DRAM-less and virtualised disks, so default off and decided by tune. flashmoe: F_NOCACHE +3% only while an app cache existed, -1% vs trust-the-OS. apus does not use O_DIRECT on Linux (slab offsets not aligned). Pitfall: a fake F_NOCACHE define on Linux silently fails (slipstream). |
| P-26 | Page-cache advice on buffered expert fds: fadvise RANDOM (no readahead) / NOREUSE, DONTNEED after each read; keep small scale tensors buffered | F-io-16; also: apus/F-io-01 | [colibri] [ds4] [apus] | 3 | adopt | flashmoe found MADV_RANDOM on a mapping harmful (each read split into ~5.7 x 512 KB ops) and readahead toggles 0%: different API, same warning to measure. |
| P-27 | Advisory read-ahead hints (WILLNEED / F_RDADVISE) instead of real speculative loads | fmoe/F-io-03; also: F-pf-21, fmoe/F-prefetch-01 | [colibri] [flashmoe] | 2 | reject | flashmoe: -8% issued immediately, -4% one token ahead (65-80% of advised experts unused), net 0% during GPU compute. colibri: hint-only lookahead can be net negative on disk-saturated hosts; turning hints into real loads added +11 pp hit rate. ktransformers' one-time WILLNEED over stacked tensors at load is warming, not prediction. |
| P-28 | Demand-paging cold experts through mmap + page faults | fmoe/F-io-03 | [flashmoe] | 1 | reject | flashmoe: 0.56 tok/s, 5x slower (~240 16 KB faults per expert). |
| P-29 | Batch-union: dedupe routed experts across all tokens of a step or prefill chunk, run hits first, sort misses by file offset, coalesce abutting reads; chunk prefill to bound the live union | F-io-17, apus/F-planner-01 | [colibri] [ds4] [pulsar] [apus] | 4 | adopt | apus: unchunked prefill held ~10 GiB of payloads per layer, a 41.5 GB peak; token chunks of 8 fixed it and are numerics-neutral. ds4: with only 1-2 misses a single pass beat splitting resident/missing. colibri reads one ~19 MB coalesced region per expert. |
| P-30 | Full-read discipline: loop over EINTR and short reads, cap one read at ~1 GiB, a short read is an error and never becomes a hit | F-FMT-02; also: F-io-14, F-store-02, apus/F-io-01, fmoe/F-io-01 | [colibri] [monolite] [slipstream] [apus] [flashmoe] | 5 | adopt | flashmoe instead marks the slot invalid and gives that expert weight 0 (output changes silently). Windows needs ReadFile with explicit 64-bit offsets (CRT read is racy; text mode corrupts 0x0A). |
| P-31 | Multi-SSD read-only replicas: accept a mirror only if size + header match, route each expert deterministically by hash into bandwidth-weighted buckets from a startup O_DIRECT probe, fall back to primary on error; optionally stripe one large cold read across replicas | F-io-18, F-io-19 | [colibri] [slipstream] | 2 | adopt-tested | † slipstream's disk-benchmarking method comes from colibri; independent count 1. colibri: two NVMe 0.80 -> 1.10 tok/s (+37.5%, O_DIRECT), ~16% buffered; equal shares including a 10x slower SATA drive 0.900 -> 0.666 tok/s, weighted ~neutral; weighted 3-way striping 1.9 ms vs equal 12.6 ms (striping colibri-only). slipstream: negative when the second disk shares a bus; shipped opt-in for capacity. |
| P-32 | Capacity layouts across drives: shards on different drives as one virtual address space; partial mirrors of the hottest shards (verified copy, atomic receipt) | F-io-20 | [colibri] [pulsar] | 2 | adopt-tested | pulsar: decode pulled only 1.4-2.1 GB/s from disk because tiers absorbed most lookups, so bandwidth is rarely the limit on one modern NVMe. |

## Prefetch

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-33 | Router-lookahead prefetch: at layer L after attention, run layer L+1's norm + router on L's state and issue the top-N (6-12) non-resident experts as real loads into reserved slots | F-pf-21, apus/F-prefetch-01 | [colibri] [pulsar] [apus] | 3 | adopt | † apus's prefetch pilot is adapted from colibri; independent count 2. colibri: 71.6% recall of the true top-8 (vs 41.3% for previous-token); Strix Halo 1.10 -> 1.83 tok/s with O_DIRECT + pool + lookahead + shared-expert two-step (two-step +2.3-3% recall, colibri-only). pulsar: +21%. apus: recall ~0.69 at N=12 on the real model but its 2 GiB cache kept little (415 hits vs 13,457 misses). monolite's lookahead ranks next-layer candidates by history (not counted; off by default). Contradicted by flashmoe (cross-layer correlation ~0). |
| P-34 | Speculative loads never hurt demand: layer-ownership barrier, reservations deduped and never victims while loading, narrowed eviction guard, lowest-priority publish, bounded backlog that drops stale/excess work and drops speculation while demand loads are queued or memory is near the ceiling; failures are non-fatal | F-pf-22, apus/F-cache-02; also: F-io-17 | [colibri] [ds4] [pulsar] [monolite] [apus] | 5 | adopt | † apus's loader follows colibri's pilot; independent count 4. colibri's first guard (protect any warm victim) dropped ~100% of speculations once the cache filled; pulsar's unbounded backlog queued ~177 GiB of reads during prefill; apus's hint storm starved demand fills 5:1. |
| P-35 | Co-activation coupling tables (offline or online) predicting next-layer experts from the ids just chosen | F-pf-23 | [colibri] [slipstream] | 2 | reject | † slipstream's coupling analysis follows colibri; independent count 1. colibri: median lift 1.8x over independence (p99 40x), transfers across workloads (recall only). slipstream: +14-20 pp recall in trace replay, but the online predictor was -8% at 6 GiB and naive coupled prefetch -5% to -15% end to end. Keep only as an off-by-default experiment with telemetry. |
| P-36 | Coverage-gated whole-next-layer prefetch during prefill (enqueue all of L+1 only when expected expert coverage >= 75%) | F-pf-24 | [ds4] [pulsar] | 2 | adopt-tested | pulsar: the ungated version (any chunk > 8 tokens) issued 177 GiB of speculative reads; GLM-5.2 fell to 1.43 tok/s vs 2.63 with prefetch off; the gate fixed it. |
| P-37 | Cache-aware (lossy) routing: substitute resident experts for low-ranked misses | F-pf-25 | [colibri] | 1 | experimental | Changes model outputs; off by default; must pass quality gates. slipstream lists it only as a later A/B candidate. |
| P-38 | Temporal prediction: prefetch the experts the previous token chose at this layer | fmoe/F-prefetch-01; also: F-pf-21 | [colibri] [flashmoe] | 2 | reject | flashmoe: 25.6% hit; 3.18 vs 3.90 tok/s (-18%); a layer only speeds up if all K reads hit. colibri measured 41.3% recall for previous-token reuse, well below lookahead. |
| P-39 | Learned or early routing predictors: per-layer MLP predictor, same-layer router on the pre-attention state, two-pass routing of all layers first | fmoe/F-prefetch-02 | [flashmoe] | 1 | reject | flashmoe: MLP 31% accurate; early routing 53% accurate but 1.70 tok/s (-38%); two-pass ~6 tok/s with wrong (repetitive) output. |

## Quantisation formats

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-40 | GGUF block formats at byte-exact layouts (Q8_0, Q4_K/Q5_K/Q6_K with the 12-byte 6-bit scale/min packing, Q8_K activation blocks) and their integer dot kernels | F-quant-01, F-quant-02, F-quant-03, F-quant-04, F-quant-05, F-quant-06, F-quant-08 | [ds4] [pulsar] [ktransformers] | 3 | adopt (format-defined) | All three follow the public GGML definitions (shared lineage, so agreement is expected). Q8_K activation scale: ggml convention d = 1/iscale may be negative (ds4) vs positive d = amax/127 clamped to +-127 (pulsar, ktransformers); prefer positive d (safe for the sign trick). |
| P-41 | IQ2_XXS codebook format (8-bit grid index, 7-bit signs + parity, 4-bit sub-scale) | F-quant-10 | [ds4] [pulsar] | 2 | adopt (format-defined; only if IQ2_XXS files are supported) | pulsar: AVX-512 VNNI slower than AVX2 on Zen 5 (2.1 vs 4.8 GB/s single-thread). ds4 precomputes a signed grid table for NEON. |
| P-42 | Quantise each activation vector once per token and reuse it for every consumer (Q/K/V, gate/up, all experts of a layer) | F-quant-01, F-quant-20; also: F-quant-19 | [colibri] [ds4] [pulsar] [ktransformers] | 4 | adopt | colibri measured ~7 us per 4096-wide scalar quantisation x ~150 GEMVs/token before hoisting. |
| P-43 | SIMD kernels bit-identical to the scalar reference: exact integer inner sums per block, one float update per block, blocks accumulated in a fixed ascending order | F-quant-07; also: F-quant-08, F-quant-21, apus/F-kernel-01 | [colibri] [ds4] [pulsar] [apus] | 4 | adopt | pulsar asserts bitwise SIMD == scalar in a test; apus achieves the same for float BF16 GEMV by fixed sequential order (see drift rules). |
| P-44 | Fold offsets and mins out through activation group sums; keep weights as the unsigned operand of u8 x s8 dot units | F-quant-05, F-quant-06; also: F-quant-07, F-quant-09, F-quant-17 | [colibri] [ds4] [pulsar] [ktransformers] | 4 | adopt | Q6_K's -32 and K-quant mins become one integer term per group; colibri int4 offset 8 folded via per-64 activation sums; ktransformers Q8_0 uses +128 activation bias with precomputed per-row compensation. |
| P-45 | Signed x signed int8 on unsigned-operand units via abs/sign emulation; activations clamped to [-127, 127] | F-quant-11; also: F-quant-01, F-quant-18 | [colibri] [pulsar] [ktransformers] | 3 | adopt | -128 does not negate in int8; weights may use the full range. Integer sums are associative, so 2-4 accumulators change nothing but latency. |
| P-46 | Repack weights at load into multi-row interleaved tiles (8/16/32 rows, VNNI k-quad order) so one activation broadcast feeds many rows and no horizontal reduction is needed; repack keeps the source byte size | F-quant-09; also: F-quant-13, F-quant-15, F-quant-17, F-quant-18 | [colibri] [ktransformers] | 2 | adopt-tested | ktransformers' AVX-512 path is a port of llama.cpp's AMX backend; colibri uses rows8 (FP8, AVX2), rows16 (MXFP4) and 1x4 tiles (int4). |
| P-47 | FP8 E4M3 weights with 128x128 block scales: exact decode (bit trick, VBMI permute, half-precision x 256, or table) with an explicit NaN policy; scale dtype (f32 vs UE8M0) detected by byte size | F-quant-12, F-quant-13, apus/F-quant-01 | [colibri] [ktransformers] [apus] | 3 | adopt (model-defined for FP8 checkpoints) | NaN codes 0x7F/0xFF: colibri propagates NaN, apus decodes +-480 in C and makes the converter refuse such codes. Cross-block accumulation in double (colibri reference) vs one fp32 multiply per element into BF16 (apus dequant). |
| P-48 | MXFP4 (E2M1 + E8M0): normalise the two nibble orders to one internal order at load; document E8M0 edge cases (0, 255) | F-quant-14, F-quant-15 | [colibri] [ds4] [pulsar] [monolite] [ktransformers] | 5 | adopt (format-defined) | Kernels differ (AVX2 integer with per-32 activation quant, AVX-512 float LUT, AVX-512 BF16 permute); colibri's per-32 integer activation path adds ~0.4%/group noise and is opt-in. pulsar reads gpt-oss MXFP4 experts (grep-confirmed). |
| P-49 | Group-wise 4-bit expert weights with group size 64 (symmetric planar int4 + f32 scale, or affine with bf16 scale and bias) | F-quant-16, F-quant-17; also: fmoe/F-kernel-01 | [colibri] [flashmoe] | 2 | adopt-tested | colibri's planar nibble order makes AND/shift yield contiguous elements (zero shuffles); flashmoe's (MLX-style) packs 8 nibbles LSB-first per u32 with a bias. |
| P-50 | Keep 4/5/6-bit weights packed in memory; unpack only in registers (never expand to int8/bf16 at load) | F-quant-17; also: F-quant-09, fmoe/F-kernel-01 | [colibri] [ds4] [pulsar] [ktransformers] [flashmoe] | 5 | adopt | colibri: Qwen3.6 MoE at full residency cost 34 ms/token vs a DRAM floor of 17 ms for int8-unpacked and 9 ms for packed int4. Contradicted by apus's BF16 payload cache (chosen for exact identity, at 2x bytes). |
| P-51 | int4 stored as value x 16 in nibbles so signed int8 dot units need no sign extension (AMX/AVX-512 tile order) | F-quant-18; also: F-quant-09 | [ktransformers] | 1 | experimental | Needs activations in [-127, 127] for the emulated signed dot; no AMX on Zen 5. |
| P-52 | Offline requantisation of routed experts to fewer bits to cut bytes read per token (FP8 -> int4, 4-bit -> 2-bit), gated on quality | fmoe/F-quant-01; also: F-store-03 | [colibri] [flashmoe] | 2 | adopt-tested | flashmoe 4 -> 2 bit: per-layer I/O 2.6 -> 1.5 ms, RMSE 0.0012-0.0028, but it broke JSON/tool calling, so 4-bit stays production. colibri converts FP8 checkpoints to int4 containers. |
| P-53 | 2-bit min/max affine experts (no clipping search, bf16 truncation) as the production format | fmoe/F-quant-01 | [flashmoe] | 1 | reject | flashmoe: 5.55-5.74 tok/s sustained vs 4.36 at 4-bit, but emitted backslash-escaped names instead of quoted strings in tool calls; per-weight RMSE and short prompts did not catch it. |

## Kernels

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-54 | Fused expert FFN per layer: gate and up over the same rows in one pass, SwiGLU in registers, quantise h for down in place per whole activation block, two parallel phases (gate/up, down), tokens grouped by expert | F-quant-19, F-quant-20, fmoe/F-ffn-01 | [colibri] [ktransformers] [flashmoe] | 3 | adopt | flashmoe keeps two encoders per expert (gate+up, then SwiGLU+down); one encoder per expert was -2% (less cross-expert parallelism). colibri sums contributions in rank order so output is schedule-independent. |
| P-55 | CPU feature dispatch: probe once (CPUID + XGETBV / hwcaps), separate translation units or per-function target attributes per ISA tier, env downgrade, never execute an unsupported instruction, startup self-test against scalar; choose per format by measurement, not by vector width | F-quant-22, apus/F-config-01; also: F-HW-02, F-quant-10, F-quant-17 | [colibri] [pulsar] [monolite] [ktransformers] [apus] | 5 | adopt | pulsar keeps AVX2 for IQ2_XXS because AVX-512 measured slower; AMX (colibri, ktransformers) needs an OS permission call and is Intel-only. ktransformers builds several extension variants and picks at import; colibri chooses at compile time with optional self-tested fast paths. |
| P-56 | Determinism rules: control float contraction globally (explicit fma or none, never compiler-chosen), fixed lane mapping and reduction trees, ILP only across independent output rows, no batch-size-dependent numerics | F-quant-21, apus/F-kernel-01; also: F-fwd-24, apus/F-config-01 | [colibri] [pulsar] [apus] | 3 | adopt | colibri: GCC contracted the 4-accumulator FP8 form differently (~1 ulp, flips argmax) and ignored pragmas; only the global flag works. apus forbids FMA entirely (mul then add) to bit-match numpy; its ordered BF16 GEMV still reached 24-64 GB/s, the memory roofline. colibri's exact-dot oracle (exact products binned by exponent, round once) is colibri-only. |
| P-57 | ARM NEON paths: sdot with 4 accumulators, widen-multiply fallback, i8mm smmla 2x2 tiles for batch >= 2, table gathers | F-quant-23; also: apus/F-config-01 | [colibri] [ds4] [apus] | 3 | adopt | ds4's Q4_K NEON path unpacks nibbles through a stack buffer; vector AND/shift unpack is strictly better. |
| P-58 | Zero-copy host buffers wrapped for the GPU (unified memory): long-lived weights registered once, per-op buffers released before return; GPU paths optional and fail-soft | apus/F-kernel-02, fmoe/F-io-02; also: fmoe/F-format-01, F-FMT-02 | [ds4] [apus] [flashmoe] | 3 | adopt | apus: per-op wrapping of ~230 dense ranges per token cost 4x until registered; GPU bf16 expert GEMVs were 0.85-0.96x CPU and a tiny model ran 24 vs 173 tok/s CPU (dispatch overhead), while fused FP8 DSA linears were 2.7-3.8x faster. flashmoe: 2 MiB-aligned destination buffers 3.6x faster warm copies (~5% end to end). |
| P-59 | GPU affine 4-bit dequant matvec: one SIMD group per output row, cooperative activation copy (barrier before any early exit), dequant folded into one FMA per weight q*(s*x) + (b*x) | fmoe/F-kernel-01, fmoe/F-kernel-02 | [flashmoe] | 1 | experimental | flashmoe: FMA form 3.90 -> 4.36 tok/s at 4-bit; GPU utilisation 2.4% (data-delivery bound). Bias folding per group (s*dot(q,x) + b*sum(x)) is the block-level analogue of the fold-offsets pattern and is untested there. |
| P-60 | Apple-GPU kernel and sync ideas measured negative: LUT dequant, 128-bit weight loads, one encoder per expert, GPU-private compressed copies, spin-polling waits, routing/top-K on GPU, fused final norm + LM head | fmoe/F-kernel-03; also: fmoe/F-planner-01 | [flashmoe] | 1 | reject | flashmoe: -2%, -3%, -2%, -20% overall, -23%, 3.96 vs 3.84 ms/layer, 4.01 vs 3.93 ms/layer. Compute is ~41% of the 4-bit layer and bandwidth-bound, so kernel wins are capped. |

## Attention

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-61 | Scalar causal GQA reference attention (scale after the dot, max-subtracted softmax, consecutive q heads share a kv head), one path for prefill and decode, score scratch sized to the context | F-fwd-05; also: fmoe/F-attention-02 | [colibri] [ds4] [pulsar] [monolite] [flashmoe] | 5 | adopt (model-defined) | colibri batches all rows through projections but warns that letting batch size change the kernel family is not bit-identical; monolite loops one token at a time. |
| P-62 | Per-family attention details: QK-norm after projection and before RoPE (full-width for OLMoE, per-head for Qwen), sigmoid output gate from a split q projection (Qwen3.5/3.6) | F-fwd-06, F-fwd-07, fmoe/F-attention-02 | [colibri] [ds4] [pulsar] [flashmoe] | 4 | adopt (model-defined) | Key the output-gate split on tensor shape, not the config flag (colibri). flashmoe: GPU attention below 32 tokens was slower; fusing RoPE + q de-interleave + k norm gave 5.53 -> 4.76 ms/layer on those layers. |
| P-63 | Attention sinks (unscaled per-head logit in the denominator) and sliding-window layers | F-fwd-08 | [colibri] [ds4] [pulsar] [monolite] | 4 | adopt (model-defined) | gpt-oss: window 128 on even layers; GLM-5.3-Flash DSA layers have no sink (apus). |
| P-64 | MLA with a compressed latent KV row (512 + 64 = 576 floats) cached after norm/rotation, decode-time weight absorption, expand path for large prefill | F-fwd-09, F-fwd-10 | [colibri] [ds4] [pulsar] [ktransformers] | 4 | adopt | colibri: 576 vs 32,768 floats per token per layer (57x) on GLM-5.2. Contradicted by apus, which caches expanded per-head K/V for exactness. |
| P-65 | Expanded per-head BF16 K/V cache for MLA/DSA layers (no latent caching, no KV quantisation, no eviction), selection applied as an additive mask | apus/F-attention-03 | [apus] | 1 | experimental | apus: ~64.5 KB/token/layer (~0.71 MB/token over 11 layers), practical context 10-20K on 32 GB; latent caching (~1.5 KB/token/layer) deferred. |
| P-66 | DSA lightning indexer: per-query top-k past positions (ties in position order), shared-layer reuse; selecting everything must equal dense attention | F-fwd-11, apus/F-attention-03, apus/F-attention-04 | [colibri] [ds4] [pulsar] [apus] | 4 | adopt (model-defined) | GLM-5.3-Flash variant (apus) scores 4-token pooled keys in fp32 with an always-kept tail and no Hadamard/FP4 emulation; DeepSeek-style indexers need FP4 + Hadamard emulation for selection parity (pulsar notes). |
| P-67 | Delta-rule linear attention (Gated DeltaNet per-head decay, KDA per-channel decay): conv ring, L2-norm with eps inside the sqrt, decay applied before the memory read, rank-1 write, gated output norm | F-fwd-12, F-fwd-13, apus/F-attention-01, fmoe/F-attention-01 | [colibri] [ds4] [pulsar] [apus] [flashmoe] | 5 | adopt (model-defined) | Value head h uses key head h / (Hv/Hk), not h mod Hk. Per-thread scratch is required when heads run in parallel (colibri). flashmoe: BLAS step 0.78 -> 0.28 ms/layer; whole chain on GPU 3.90 vs 4.15 ms/layer, same output. apus lists six bit-visible fixes vs its donor (fp32 l2norm output, pairwise sums, stable sigmoid, eps 1e-5, bf16 beta, delta before outer product). |
| P-68 | Chunked linear-attention prefill with its own pinned ordering; decode stays recurrent; the path is chosen by an explicit flag, never by sequence length; verify batches step the recurrence per token | apus/F-attention-02 | [apus] | 1 | experimental | Mathematically equal orderings differ in final bits and can flip sampled tokens. colibri, pulsar v1 and flashmoe prefill sequentially; slipstream has only a Python chunked prototype in its benchmarks. |
| P-69 | DeepSeek V4 attention: one shared 512-wide K=V row, sliding window + compressed rows (+ indexer), per-head sink, inverse RoPE on the output, grouped low-rank output projection | F-fwd-20 | [colibri] [ds4] [pulsar] | 3 | adopt (model-defined) | Compressor pools R positions by softmax-weighted gates; pulsar keeps compressor steps per token inside a prefill chunk. |
| P-70 | Emulate the reference's bf16 rounding points (after projections, norms, probabilities, outputs) when greedy/bitwise parity is the goal | apus/F-norm-01; also: F-fwd-20, apus/F-attention-03, apus/F-routing-01, apus/F-attention-01 | [colibri] [apus] | 2 | adopt-tested | colibri does this for DeepSeek V4 greedy parity; apus does it everywhere (bf16 residual streams, per-add bf16 rounding in the MoE sum). Engines targeting tolerance keep f32 throughout. |

## Routing / FFN

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-71 | Softmax router: softmax over all E experts, then top-K, with a per-family renormalise flag (OLMoE no, Qwen3-MoE yes, Qwen3.5/3.6 always); NaN-safe selection with ties to the lowest index | F-fwd-14, fmoe/F-routing-01 | [colibri] [ds4] [pulsar] [flashmoe] | 4 | adopt (model-defined) | Guard the all-NaN case (colibri saw expert index -1 after corrupt tiles). flashmoe: routing on the GPU was slower than on the CPU. |
| P-72 | gpt-oss router: biased logits, top-K by raw logit, softmax over the K winners only | F-fwd-15 | [pulsar] [monolite] | 2 | adopt (model-defined) | Mathematically equal to softmax-then-renormalise but evaluated in a different order; reproduce the reference order for bit-closeness. pulsar supports gpt-oss (grep-confirmed). |
| P-73 | Sigmoid router: bias added for selection only, unbiased scores as weights, optional group limit (top-2 group score), renormalise (+1e-20) then multiply by the routed scale | F-fwd-16, apus/F-routing-01 | [colibri] [ds4] [pulsar] [ktransformers] [apus] | 5 | adopt (model-defined) | Three easy mistakes: weighting by the biased value, scaling before renormalising, adding a group step when n_group = 1 (GLM). Stable top-k with ties to the lower index (apus). |
| P-74 | DeepSeek V4 router: sqrt(softplus) scores, hash-table routing by token id on early layers, normalised weights x routed scale | F-fwd-17 | [colibri] [ds4] [pulsar] | 3 | adopt (model-defined) | Token ids must be plumbed to the FFN. pulsar floors the normaliser (~6e-5); colibri errors on a non-positive sum. |
| P-75 | SwiGLU expert FFN with family variants (clamped gate/up, gpt-oss alpha form with interleaved gate/up and biases) plus shared expert (optional per-token sigmoid gate), shared added after the routed sum | F-fwd-18, fmoe/F-ffn-01; also: F-quant-20, apus/F-routing-01 | [colibri] [ds4] [pulsar] [monolite] [ktransformers] [apus] [flashmoe] | 7 | adopt (model-defined) | None of the engines apply the routed scale to the shared expert. pulsar notes the V4 reference multiplies the router weight before the down projection (same maths, different rounding). |
| P-76 | Fixed, documented expert accumulation order, independent of thread scheduling, cache state and resolve order | also: F-fwd-18, F-quant-19, apus/F-routing-01, fmoe/F-ffn-01 | [colibri] [apus] [flashmoe] | 3 | adopt | Which order is contested: colibri sums in routing-rank order in f32 then the shared expert; apus sums in ascending expert index with a bf16 rounding per add (mirrors its reference's eager loop, and makes tiered resolve order and chunking irrelevant); flashmoe binds a fixed 8 slots in its combine kernel. |
| P-77 | A missing or failed expert contributes zero; do not renormalise the survivors | also: F-fwd-18, fmoe/F-ffn-01 | [colibri] [flashmoe] | 2 | adopt-tested | colibri measured renormalising survivors as "catastrophically worse" on OLMoE. flashmoe zero-weights an expert whose read failed instead of aborting. |
| P-78 | Run fewer routed experts per token than the model default (K 10 -> 4) | fmoe/F-routing-01 | [flashmoe] | 1 | experimental | flashmoe: K=6 1.91, K=4 3.15 tok/s with "no observable degradation" (qualitative only, no perplexity); K <= 3 and per-depth K reductions collapse output. Re-measure with a real quality harness before relying on it. |

## Norms / RoPE / residual

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-79 | Decoder skeleton: pre-norm residual stream kept in f32, final norm + LM head on the last row only unless all rows are needed (verify, scoring) | F-fwd-01; also: fmoe/F-planner-01 | [colibri] [ds4] [pulsar] [monolite] [apus] [flashmoe] | 6 | adopt (model-defined) | Norms, router weights and bias stay f32 (colibri). apus keeps bf16 residual streams to bit-match its reference (see bf16 rounding points). |
| P-80 | RMSNorm family (plain, 1 + w for Qwen3.5/3.6, gated x silu(z), weightless) with eps added to the mean inside the sqrt and read from config | F-fwd-02; also: apus/F-attention-01, fmoe/F-ffn-01 | [colibri] [ds4] [pulsar] [monolite] [apus] [flashmoe] | 6 | adopt (model-defined) | Using plain w for Qwen3.5/3.6 gives plausible but wrong text (colibri). Sum of squares: double (colibri) vs numpy-pairwise f32 (apus, to match its oracle). |
| P-81 | RoPE layouts (half-split, interleaved, interleaved-in/half-out, partial rotary, inverse) with inv_freq cached per model | F-fwd-03; also: fmoe/F-attention-02 | [colibri] [ds4] [pulsar] [monolite] [ktransformers] [flashmoe] | 6 | adopt (model-defined) | colibri: recomputing pow() per position was a real prefill cost. mRoPE collapses to plain RoPE for text-only input (pulsar). GLM-5.3-Flash has no RoPE at all (apus). |
| P-82 | YaRN: ramp over the pair index between correction dims, optional truncation, mscale applied where the model family applies it (cos/sin vs softmax scale) | F-fwd-04 | [colibri] [ds4] [pulsar] [monolite] [ktransformers] | 5 | adopt (model-defined) | monolite's ramp over wavelength produced coherent text and was found only by the token oracle. |
| P-83 | Manifold-constrained hyper-connections: n residual streams, weightless norm over all streams, pre/post sigmoid gates, Sinkhorn-normalised mixing (20 iterations), learned or mean merge before the final norm | F-fwd-19, apus/F-norm-01 | [colibri] [ds4] [pulsar] [apus] | 4 | adopt (model-defined) | Numerically fragile: colibri saw GCC reduction rounding change when scratch moved from heap to stack; apus applies the norm before the matmul (bit-different from its base engine) and warns the comb matrix ends column-stochastic, not row-stochastic. |

## MTP / speculation

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-84 | MTP/NextN draft head: concat [embedding(t+1), normed hidden(t)], projection, one full layer with its own KV slot, shared LM head; weights at int8 or better | F-fwd-21, apus/F-mtp-01; also: fmoe/F-mtp-01 | [colibri] [ds4] [pulsar] [apus] | 4 | adopt | colibri: int4 MTP heads collapsed acceptance to 0-4%. colibri feeds the post-final-norm hidden (vLLM convention); apus picked the same source by an acceptance sweep on the real model (85.3% accept, 2.53 tokens/batch, 1.47x). Wrong concat order or norm placement only lowers acceptance. |
| P-85 | Speculative loop: draft, one batched verify over 1 + g rows, commit the accepted span, feed verified pairs back to the MTP layer; attention KV past n is simply overwritten; recurrent state is snapshotted and restored | F-fwd-22, apus/F-mtp-02 | [colibri] [ds4] [pulsar] [apus] | 4 | adopt | A rejected draft costs a full MoE row; colibri measured break-even near 50% acceptance and pauses a draft source below 70% (MTP) / 50% (corpus) over 24 proposals, re-arming after 256 tokens. pulsar disables its prefix cache while speculative modes rewrite KV. apus never rolls back the expert cache (content is numerics-neutral). |
| P-86 | Acceptance rules: greedy exact match; exact sampling with a point-mass drafter (accept with p[d], resample with d banned); or draft equals the main model's own pick | F-fwd-23; also: apus/F-mtp-02 | [colibri] [ds4] [apus] | 3 | adopt | Variants disagree on goals: colibri preserves the distribution; ds4's opportunistic mode biases toward determinism; apus makes the output stream identical per seed (one RNG draw per emitted token, drafts draw none). |
| P-87 | Draft, verify and absorb compute the same function: no kernel-family choice that depends on batch size while speculation is live (GEMM rows equal GEMV results) | F-fwd-24; also: apus/F-kernel-01, apus/F-attention-02 | [colibri] [apus] | 2 | adopt (correctness requirement for speculative == non-speculative output) | colibri: an integer kernel chosen for S >= 2 flipped near-tie argmaxes and broke greedy equivalence; the same integer kernel for GLM prefill cost ~12% perplexity. ds4 documents that batched verify is not byte-identical to S=1 decode without pinning. |
| P-88 | External draft-model speculation on an SSD-streamed MoE (verification touches the union of experts over all drafted tokens) | fmoe/F-mtp-01 | [flashmoe] | 1 | reject | flashmoe: external 35B-A3B draft at 33% acceptance was 4.5x slower; its MTP analysis (weights absent) predicted break-even. apus measured MTP 1.47x faster at 85% acceptance on a disk-tiered container: acceptance and expert overlap decide it. |

## KV cache

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-89 | Position-addressed KV cache (head-major or position-major rows, latent rows for MLA, fixed state for linear attention, separate MTP slot) so keeping a prefix means resetting a length; optional quantised rows (fp8 + scale) with an f32 oracle path | F-fwd-25; also: F-fwd-09 | [colibri] [ds4] [pulsar] [monolite] [apus] [flashmoe] | 6 | adopt | Quantised KV: colibri (fp8 / polar codecs), ds4 (fp8 for V4), pulsar (fp8/fp4 QAT sims); apus pins no quantisation for exactness. |
| P-90 | Prefix reuse across turns: record ids where they enter the cache, reuse the matching prefix, prefill only the suffix, always recompute at least one token; recurrent state allows only exact extension; non-token inputs taint the state | F-fwd-26, fmoe/F-planner-02; also: F-SRV-03 | [colibri] [ds4] [pulsar] [monolite] [flashmoe] | 5 | adopt | colibri: DeepSeek V4 second turn reusing 82% of the prompt took 61 s instead of 320 s. apus re-prefills every request (design simplicity, no measurement). |
| P-91 | KV persistence on disk: append-only records with a geometry header, count written last after fsync, sanitise on load (colibri); or text-prefix-hash keyed checkpoint files with a budget (ds4) | F-fwd-27 | [colibri] [ds4] | 2 | adopt-tested | Writing the count last makes the file crash-safe without a journal. Cache files contain prompt text: private directory. |
| P-92 | Chunked prefill is just incremental suffix prefill over position-addressed KV and must equal single-shot prefill | F-fwd-30; also: apus/F-planner-01 | [colibri] [ds4] [pulsar] [apus] | 4 | adopt | colibri: 1,683 tokens at chunk 256 gave byte-identical greedy output. Exceptions: MTP absorb pairs across chunks, V4 compressor state, recurrent layers (sequential within the chunk). |
| P-93 | Prefill of non-final prompt tokens without reading back or combining expert outputs on the CPU (less sync, same I/O) | fmoe/F-planner-02 | [flashmoe] | 1 | experimental | flashmoe: TTFT 5,587 -> 2,615 ms (-53%). The paper says expert I/O is skipped; the code still reads every expert (intermediate tokens feed later layers), so the gain is fewer syncs. |

## Sampling

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-94 | Sampler: NaN-safe greedy, temperature softmax with sum in double, top-p, min-p in logit space, seeded RNG, argmax fallback when the sum is not finite; min_p = 0 equals plain temperature exactly | F-fwd-28; also: apus/F-mtp-02 | [colibri] [ds4] [pulsar] [monolite] [apus] | 5 | adopt | Apply penalties/grammar masks before temperature and top-p/min-p after (colibri). top-p by heap (colibri) vs stable sort, ties to lower index (apus). |
| P-95 | Thinking-token budget: force the end-of-think token when the budget runs out (hard) or when it is near the top of the distribution close to the limit (soft) | fmoe/F-sampling-01 | [ds4] [flashmoe] | 2 | adopt-tested | ds4's eval runner has soft and hard think-close limits (grep-confirmed); flashmoe forces end-think at a fixed budget (default 2048). |

## Tokenizer

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-96 | Byte-level BPE core: GPT-2 byte-to-codepoint tables, merge loop over contiguous-slice symbols using merges-list ranks, whole-piece shortcut, single-byte fallback | F-TOK-01, F-TOK-02, apus/F-tokenizer-01 | [colibri] [ds4] [pulsar] [apus] [flashmoe] | 5 | adopt (tokenizer-defined) | Decode: unknown codepoints should be emitted as UTF-8 (pulsar) rather than dropped. flashmoe's native tokenizer (byte tables + merge hash, grep-confirmed) cut startup 3,500 -> 180 ms vs Python. Contiguous-slice merges avoid per-merge allocation (colibri) vs a new string per merge (ds4). |
| P-97 | tiktoken-style rank BPE for vocabularies without a merges list: merge the adjacent pair whose concatenation has the lowest id, recomputing only the neighbours | F-TOK-03 | [colibri] [monolite] | 2 | adopt (tokenizer-defined) | monolite exports o200k as raw bytes (no byte map); colibri switches to this mode automatically when tokenizer.json has an empty merges list. |
| P-98 | Hand-written pre-tokeniser per family (ordered rules, first match wins, regex backtracking order reproduced) driven by generated Unicode class-range tables; no regex library | F-TOK-04, F-TOK-05, apus/F-tokenizer-01 | [colibri] [ds4] [pulsar] [monolite] [apus] | 5 | adopt | colibri: the wrong family is silent (cl100k rules on a GPT-2 model matched 1560/1708). Tables from the Unicode database (colibri, monolite) vs probed from the reference tokenizer itself (apus: 1.49M probes, 0 mismatches). |
| P-99 | Special/added tokens matched first as atomic units (longest at earliest position); user text always encoded with specials disabled; separate added vs control flags | F-TOK-06; also: apus/F-tokenizer-01 | [colibri] [ds4] [pulsar] [apus] | 4 | adopt | Otherwise a user can inject role markers. Validate added-token ids on load (colibri). |
| P-100 | Chat templates hard-coded per family in the engine (marker ids pushed directly or text rendered with escaping), not a template engine | F-TOK-07, apus/F-tokenizer-02 | [colibri] [ds4] [pulsar] [monolite] [apus] | 5 | adopt | Validated against the model's Jinja template by goldens in apus (21 cases byte-exact), colibri (template test harnesses) and ds4 (release QA). pulsar also has a Jinja path (next row). |
| P-101 | Execute the model's own Jinja chat template with a mini-Jinja subset (fallback for unknown families) | F-TOK-08 | [pulsar] | 1 | experimental | Large surface; renders user content as text, so specials inside user content are matched on re-tokenisation unless escaped. |
| P-102 | Streaming detokenisation that holds back incomplete UTF-8 tails and the last (max_stop_len - 1) bytes / partial end-of-think tags until they resolve | F-TOK-09; also: apus/F-server-01 | [colibri] [ds4] [monolite] [apus] | 4 | adopt | ds4: clients replace split UTF-8 with U+FFFD and send it back next turn, breaking KV prefix matching. colibri's C SSE path drains UTF-8 the same way (grep-confirmed). |
| P-103 | One tokenizer object built from either GGUF metadata or tokenizer.json, with pre-tokeniser family selected from metadata | F-FMT-04 | [colibri] [ds4] [pulsar] | 3 | adopt | Unknown family: refuse with a clear message (recommended; colibri measured silent mismatch) vs fall back to a default (pulsar). |

## File readers

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-104 | Hardened safetensors indexing without reading tensor data: header cap, overflow-checked shapes, byte-span cross-check for float dtypes, index.json for duplicates, lazy per-tensor positioned reads | F-FMT-01, apus/F-io-01; also: fmoe/F-storage-01 | [colibri] [apus] [flashmoe] | 3 | adopt | colibri caps the header at 512 MiB; missing-tensor errors should name the probably-missing shard (final norm and head live in the last shard). |
| P-105 | Bounds-checked GGUF cursor: lazy typed metadata, pre-allocation sanity on counts, recursion cap for nested arrays, alignment from metadata (default 32), block-size table, absolute offsets checked against file size | F-FMT-03 | [ds4] [pulsar] | 2 | adopt (format-defined) | ktransformers uses the official gguf Python reader rather than its own parser (not counted). |
| P-106 | Small strict JSON parser for C: arena-allocated tree (linked blocks, never realloc), depth cap, full escape and surrogate handling, hand-validated numbers, typed getters with fallbacks, matching escaper | F-JSON-01 | [colibri] [ds4] [monolite] [apus] | 4 | adopt | colibri bugs designed out: a fixed 64 KiB string buffer truncated tokenizer strings; a realloc'ing arena invalidated pointers. ds4's server walks requests with a cursor instead of building a tree. |

## Planner / config

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-107 | One shared memory snapshot: MemAvailable (never MemFree) min'd with cgroup v2/v1 limits plus inactive file pages, macOS free+inactive, Windows min(physical, commit); explicit unknown state | F-HW-01; also: apus/F-cache-03 | [colibri] [ds4] [pulsar] [monolite] [slipstream] | 5 | adopt | colibri: on Windows physical-only budgeting allocated until the commit charge was refused; three engines with three definitions disagreed until consolidated. |
| P-108 | Hardware inventory: CPU features with XGETBV confirmation, physical cores from topology, GPUs marked placement-qualified only when free-memory figures are trustworthy | F-HW-02 | [colibri] [pulsar] [monolite] | 3 | adopt | colibri warns loudly when physical-core detection falls back to 1 (silently halves throughput); unified-memory GPUs must share one budget with RAM. |
| P-109 | Disk bandwidth probe that cannot measure RAM: cache-bypass reads of cold windows only (mincore), sparse/small/resident vetoes, result cached per device id | F-HW-03; also: F-io-18 | [colibri] [slipstream] | 2 | adopt-tested | † slipstream's disk-benchmarking method comes from colibri; independent count 1. colibri: warm buffered reads reported 23-97 GB/s vs a real 14 GB/s; cache bypass alone still served a resident shard at 22.1 GB/s. Block size must match real expert reads. |
| P-110 | Assign roles and split work by measured bandwidth (proportional shares rounded to whole quant blocks and clamped), never by device name or free memory alone | F-HW-04; also: F-io-18 | [colibri] [pulsar] [slipstream] | 3 | adopt | † slipstream's per-disk share method follows colibri; independent count 2. pulsar: an even split ran at the slow card's pace (275/288 GB/s vs fast card idle 61%); truncating instead of rounding to a block cost 47.8 -> 46.2 tok/s; a 64 MB probe fit in L2 and read ~390 GB/s on every card; free-memory selection picked the slower card. |
| P-111 | Throughput/hit-rate prediction from a coverage curve of recorded per-expert counts and a bytes / bandwidth time model, calibrated on measured points | F-HW-06; also: apus/F-prefetch-01, fmoe/F-config-01 | [colibri] [monolite] [apus] [flashmoe] | 4 | adopt | monolite: t = RAM bytes / 25.2 GB/s + 0.5703 s/GB x disk bytes fit three points within 3.5%; another model's curve under-predicted by 19-32%, own counts within 16%. apus simulates LRU per slots/layer to size its cache; flashmoe derives I/O-bound ceilings the same way. |
| P-112 | doctor / plan / tune commands sharing the probing code; tune measures only hardware-dependent, quality-preserving knobs and saves a (model, machine) profile | F-HW-07 | [colibri] [monolite] | 2 | adopt-tested | colibri: tune once recommended knobs another engine ignored; only recommend what the selected engine reads. |
| P-113 | Config precedence flag > env (one prefix) > tune profile > plan > platform default > compiled default; strict numeric parsing; unknown flags are errors; print the effective config with sources | F-CFG-01 | [colibri] [ds4] [monolite] | 3 | adopt | colibri: a TEMP env var collided with the Windows temp dir; a launcher-set affinity variable pinned every thread to one core. |
| P-114 | Portability layer of small shims (aligned alloc/free pairs, replace-rename, positioned read with 64-bit offsets, file size, fsync, memory pressure, no-cache fd) | apus/F-config-01; also: F-FMT-02, apus/F-io-01 | [colibri] [apus] | 2 | adopt-tested | † apus's compat header is adapted from colibri's; independent count 1. apus runs the same bitwise battery on macOS, Linux and Windows CI (ASan hung on macOS; auto-vectorisation disabled on Linux/Windows after an emulator miscompile). |
| P-115 | Per-layer three-submission GPU pipeline with the expert submission left in flight and the next layer's input norm fused into it; serial GPU -> SSD -> GPU on unified memory | fmoe/F-planner-01 | [flashmoe] | 1 | experimental | flashmoe: expert I/O 56% of the 4-bit layer; moving the combine to the GPU removed a 0.83 ms CPU wait that reappeared in the next sync. On unified memory SSD DMA and GPU share the memory controller, so overlap slowed the GPU. |
| P-116 | Static memory plan: weights, scratch, slots, KV and recurrent state allocated once at start; nothing allocated in the decode loop | fmoe/F-config-01; also: F-cache-06, apus/F-cache-03 | [colibri] [monolite] [slipstream] [apus] [flashmoe] | 5 | adopt | flashmoe: a fixed ~6 GB resident set is what makes trust-the-OS safe. apus keeps grow-only thread-local scratch and buffer free lists. |

## Server

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-117 | Minimal HTTP/1.1 reader: bounded headers/body, strict Content-Length, chunked refused, Connection: close, TCP_NODELAY, localhost bind, signals without restart semantics | F-SRV-01 | [colibri] [ds4] [monolite] [flashmoe] | 4 | adopt | monolite: with restart semantics accept() never returned EINTR and only SIGKILL stopped the server. Nobody uses keep-alive. |
| P-118 | OpenAI chat-completions shape with SSE streaming (role chunk, content deltas, finish chunk, optional usage chunk, [DONE]), OpenAI error objects, reasoning/tool calls routed to their own fields | F-SRV-02, apus/F-server-01 | [colibri] [ds4] [pulsar] [monolite] [apus] [flashmoe] | 6 | adopt | monolite: a fixed-size SSE buffer truncated chunks into malformed JSON and clients dropped the stream. ds4 counts only truly reused tokens as cached_tokens. |
| P-119 | One model, one KV: serialise generation through a single worker (bounded queue, 429/503 when full), poll for client disconnect and cancel between tokens | F-SRV-03; also: apus/F-server-01 | [ds4] [monolite] [apus] | 3 | adopt | monolite argues interleaving conversations halves each cache; ds4 optionally runs several KV slots chosen by longest reusable prefix. |
| P-120 | Socket-free C engine driven by a small gateway process over line-delimited JSON on stdio | apus/F-server-01 | [colibri] [apus] | 2 | adopt-tested | † apus follows colibri's gateway-drives-engine split; independent count 1. No socket code in C to audit; one process = one model. colibri also ships a C server path. |

## Testing

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-121 | External reference oracle: teacher forcing against the reference framework (argmax per position, allowance, top-5 and gap dump), free-run and perplexity checks, per-layer hidden-state dumps, finite-logit check | F-fwd-29, apus/F-testing-01 | [colibri] [ds4] [pulsar] [monolite] [apus] | 5 | adopt | apus: its self-built numpy oracle shared a kv_b split bug with the engine and agreed bitwise; only the external anchor caught it (rel err 0.74, argmax 6/16 -> 16/16). A gap ~1e-4 is a near-tie, a large gap a bug (colibri). |
| P-122 | Keep the scalar CPU reference path permanently; every fast backend or kernel is accepted only if it matches it | F-fwd-29, apus/F-testing-01; also: F-quant-21 | [colibri] [pulsar] [monolite] [apus] | 4 | adopt | monolite's oracle found a CUDA race, a double-accumulated layer and the YaRN ramp bug, all producing plausible text. |
| P-123 | Exactness identities as tests on tiny synthetic fixtures: select-all == dense, chunked == single-shot, union MoE == per-token, saved/loaded KV == uninterrupted, any cache size (down to 1 slot) == eager, equal output at 1/4/8 threads | F-fwd-29, apus/F-testing-01; also: apus/F-cache-01, F-fwd-11, F-fwd-30 | [colibri] [apus] | 2 | adopt-tested | apus also ships "parrot" containers (zero layer weights + scripted head) giving a known large-margin token stream on any host, and redraws fixtures until every top-k gap exceeds 1e-4. |
| P-124 | Bitwise vs tolerance test tier chosen at runtime by probing the host's transcendental functions; goldens generated with numpy pinned to baseline kernels | apus/F-testing-01 | [apus] | 1 | experimental | apus: numpy f32 exp equals Apple libm bitwise but differs on x86 SIMD for ~40% of values until pinned. |
| P-125 | Measure-everything discipline: per-phase and per-tier telemetry (hits, prefetch issued/useful/dropped, recall, wait vs read time), every optimisation A/B'd, logged and kept only if faster with unchanged output | fmoe/F-testing-01; also: F-pf-21, F-pf-22, apus/F-prefetch-01 | [colibri] [monolite] [slipstream] [apus] [flashmoe] | 5 | adopt | flashmoe: 90+ experiments, ~42% discarded; several early best numbers were invalidated because they depended on a warm app cache later removed. Every storage/prefetch policy in this library has a measured case where it lost. |
| P-126 | Structured-output (JSON / tool-call) quality gate for any lossy change (low-bit formats, reduced K, routing changes) | fmoe/F-testing-01; also: fmoe/F-quant-01 | [ds4] [flashmoe] | 2 | adopt-tested | flashmoe's 2-bit regression passed RMSE and short prompts and failed tool calls. ds4 tests tool calls in its integrated evals. |
| P-127 | Tokenizer and chat-template parity gates: token-for-token against the reference tokenizer on a corpus (code, CJK, numbers, CRLF), renderer byte-exact against the reference Jinja template | apus/F-tokenizer-02; also: F-TOK-04, apus/F-tokenizer-01 | [colibri] [ds4] [apus] | 3 | adopt | apus: 1,491,186 tokenizer probes with 0 mismatches and 21 template cases byte-compared; colibri ships per-family template harnesses; ds4's release QA compares rendering with chat_template.jinja byte for byte. |

## Distributed

From fragments/distributed.md (F-dist-01..20). Design discussion: docs/design/cluster.md and docs/design/swarm.md.

| Pattern id | Pattern | Fragments (ids) | Origins (tags) | Count | Verdict | Notes |
|---|---|---|---|---|---|---|
| P-128 | Identity handshake (magic, version, model fingerprint, exchange schedule) that rejects mismatched peers before inference | F-dist-01 | [colibri] [ds4] [slipstream] | 3 | adopt | Correctness requirement as well: ds4 derives RDMA slots from the schedule. |
| P-129 | Typed frames with fixed records; every size, count, flag and cross-field rule validated before allocation; error frames instead of silent drops | F-dist-02 | [colibri] [ds4] [slipstream] | 3 | adopt | slipstream caps frames at 4 MiB and keeps a replay cache; colibri froze its codec behind byte-exact transcripts. |
| P-130 | Expert-parallel workers fed one batch-union request per layer per worker | F-dist-03 | [colibri] | 1 | experimental | Sequential per-worker visits and fail-stop in the first version; ds4 reaches the same split as an expert-owning TP pair. |
| P-131 | Token-exact parity gate for delegated work, physical runs per transport | F-dist-04 | [colibri] [ds4] | 2 | adopt (correctness requirement: delegation must not change output) | colibri caught a worker-only bug in a rotated weight format with it. |
| P-132 | Two-rank lockstep TP: mirrored session commands, fixed exchange schedule, f32 partials, agreed cancellation | F-dist-05, F-dist-06, F-dist-17 | [ds4] | 1 | experimental | Two Sparks over RoCE: 21.9 tok/s single session (V4.1 Flash Q2). |
| P-133 | Symmetric TCP exchange that interleaves send and receive with a progress-extended deadline and a desync header | F-dist-07 | [ds4] | 1 | experimental | Avoids buffer-size deadlocks on large symmetric payloads. |
| P-134 | Optional verbs transport: dlopen, device by link address, full-size warm-up, bounded QP retries, TCP fallback | F-dist-08 | [ds4] | 1 | experimental | Apple TB RDMA: UC only, no one-sided writes, 16 KiB max message. |
| P-135 | Bounded waits everywhere, liveness from the control socket, invalidate affected sessions, rebuild by transcript replay | F-dist-09 | [ds4] | 1 | adopt (correctness requirement: no hangs, no silent state loss) | colibri fail-stops instead. |
| P-136 | Contiguous layer-range pipeline with stage-to-stage forwarding and per-hop telemetry; capacity, not decode speed | F-dist-10 | [ds4] [pulsar] | 2 | adopt-tested | ds4: no decode overlap for one stream. |
| P-137 | Rolling token-prefix hash on every remote work item and snapshot | F-dist-11 | [ds4] | 1 | adopt (correctness requirement: remote KV must be on the coordinator's timeline) | |
| P-138 | Explicit wire precision for activations (f32 default; 16/8-bit opt-in and validated); partial sums f32 | F-dist-12 | [ds4] [pulsar] | 2 | adopt-tested | pulsar: BF16 + P2P hops 11.15 -> 10.12 s prefill. |
| P-139 | Several prefill chunks in flight across stages (window ~ stages + 2) | F-dist-13 | [ds4] [pulsar] | 2 | adopt-tested | pulsar: -20% (7.4k) and -12% (50k) prefill, bit-identical. |
| P-140 | Count hops before FLOPs; split by measured speed, rounded to quant blocks | F-dist-14; also: F-HW-04 | [ds4] [pulsar] | 2 | adopt-tested | pulsar: 2.79x faster GEMM moved TP prefill 11.15 -> 11.14 s; dropping a slow-link card: prefill faster, decode 34.3 -> 31.8 tok/s. |
| P-141 | FFN split across memory domains with partial outputs summed in fixed order | F-dist-15 | [ds4] [pulsar] [ktransformers] | 3 | adopt | Split by intermediate slices (kt-kernel, pulsar) or by expert ownership (ds4). |
| P-142 | Vocabulary-split output head with explicit missing-half failure | F-dist-16 | [ds4] | 1 | experimental | ~253 KiB per row on the wire for a 129k vocabulary. |
| P-143 | Topology-neutral distributed snapshots streamed in bounded chunks | F-dist-18 | [ds4] | 1 | experimental | |
| P-144 | Registry with heartbeats and staleness; static lists still work; workers dial with retry | F-dist-19 | [colibri] [ds4] [slipstream] | 3 | adopt | |
| P-145 | Scored whole-request routing with a quadratic load penalty; explicit trust boundary | F-dist-20 | [slipstream] | 1 | experimental | |

## Contradictions

Patterns where projects disagree and there are measurements on each side.

### C-01: Where hot experts live: application RAM cache vs the OS page cache (P-10, P-11)

- **For:** colibri, ds4, pulsar, monolite, slipstream and apus all ship an app cache. slipstream measured the page-cache path ~75% faster but with 1000+ swap-outs and rejected it for breaking its memory invariant; colibri measured buffered reads falling 800 -> 180 MB/s when the page cache was starved.
- **Against:** flashmoe: no app cache 5.74 tok/s vs GPU LRU 4.80 vs LRU + F_NOCACHE 4.11 (2-bit, +38%); every app-cache size it tried (500-3000 GPU entries, 500/2581 malloc entries) lost; compressor activity fell from 60-130k decompressions/s to ~0.
- **Reading:** Platform-specific: on unified memory, wired GPU buffers shrink the page cache and wake the compressor. On a CPU box with dedicated RAM, the measurements favour an app cache plus cache-bypass reads; keep trust-the-OS as an A/B mode.

### C-02: Cache-bypass reads (O_DIRECT / F_NOCACHE) (P-25)

- **For:** colibri: +65% decode on a Strix Halo, +34% on a Blackwell/Windows box, 4.25 -> 9.69 GB/s microbench on a GB10.
- **Against:** colibri: neutral or negative on QLC, DRAM-less and virtualised disks. flashmoe: F_NOCACHE +3% only while an app cache existed and -1% vs trust-the-OS.
- **Reading:** Drive- and platform-dependent: ship as a switch decided by a tune measurement, default off.

### C-03: Async read mechanism: io_uring vs threaded pread (P-24, P-23)

- **For:** pulsar: io_uring at parity with a C reference (~4.8 GB/s) and ~10.7 GB/s on a PCIe 5.0 drive at QD >= 8; a ~7-read batch is latency-bound (~6 GB/s).
- **Against:** flashmoe (macOS): threaded pread beat POSIX aio (-7%) and dispatch_io (-70%). monolite argues io_uring adds nothing for four ~13 MB reads per layer (argument only). colibri's measured -18% service-time gain came from its pread pool.
- **Reading:** Keep the pread pool as the portable baseline; adopt io_uring where many small reads must be in flight (fine-grained experts, deep QD).

### C-04: Cross-layer prediction (router lookahead) (P-33, P-39)

- **For:** colibri: 71.6% recall of the true top-8, Strix Halo 1.10 -> 1.83 tok/s with the full stack; pulsar: +21%; apus: ~0.69 recall at N = 12 on the real GLM-5.3-Flash.
- **Against:** flashmoe (Qwen3.5-397B, 512 experts, K = 4): cross-layer correlation ~0; same-layer early routing on the pre-attention state 53% accurate but -38%; MLP predictor 31%.
- **Reading:** Model- and K-dependent, and flashmoe tested a different predictor (this layer's router on the pre-attention state, not the next layer's router on the post-attention state). Measure recall on the target model before enabling.

### C-05: Overlapping expert I/O with compute (P-23, P-33, P-36, P-115)

- **For:** colibri (pool + lookahead), pulsar (+21% cross-layer), apus and monolite overlap reads with compute on CPU/discrete-GPU hosts.
- **Against:** flashmoe on unified memory: advisory prefetch during GPU compute cut expert I/O 31% but raised the following GPU wait 73% (net 0%); serial GPU -> SSD -> GPU treated as optimal.
- **Reading:** Overlap pays when I/O and compute do not share the memory controller; check on the target host.

### C-06: Co-activation coupling tables (P-35)

- **For:** colibri: median co-activation lift 1.8x over independence (p99 40x), structure transfers across workloads.
- **Against:** slipstream: +14-20 pp recall in trace replay, but -8% end to end for the online predictor at 6 GiB and -5% to -15% for naive coupled prefetch; flashmoe: co-occurrence file clustering 0%.
- **Reading:** Recall gains do not pay unless reads are cheap relative to the evictions they cause.

### C-07: Cache partition scope: per-layer vs global (P-17, P-18)

- **For:** slipstream: weighted per-layer partitions reached 72.1% hit at 6 GiB and make cross-layer eviction impossible.
- **Against:** ds4: per-layer caps made effective cache size depend on model depth, so it moved to one global policy (11.9 / 19.3 tok/s reported with the updated priorities); ktransformers' global top-N gives hot layers more slots.
- **Reading:** No head-to-head on one model; default global, keep weighted per-layer as an experiment.

### C-08: What the cache holds: packed bytes vs dequantised BF16 (P-22, P-50)

- **For:** apus: BF16 payloads make hit, miss and eager paths byte-identical and move dequant off the compute thread.
- **Against:** colibri: with int8-unpacked experts Qwen3.6 cost 34 ms/token against DRAM floors of 17 ms (int8) and 9 ms (packed int4). apus's BF16 slots are twice the FP8 slab size, and it recommends only a 2 GiB cache on 32 GB because larger caches are trimmed by its guard.
- **Reading:** Cache packed bytes; dequantise in registers.

### C-09: Routing-history merge rule (P-07)

- **For:** Overwrite (latest run wins) is simplest.
- **Against:** pulsar: one short run poisoned the census (resident-tier hits halved, 8.2 -> 5.8 tok/s). colibri: an undecayed sum stops tracking (one turn moved the ranking 0.2% on 18.2M selections).
- **Reading:** Decay ~0.99 per save (colibri) or per-slab max of this run's delta and the stored value (pulsar).

### C-10: Speculative decoding on a streamed MoE (P-85, P-88, P-84)

- **For:** apus: MTP at depth 3, 85.3% acceptance, 2.53 tokens/batch, 1.47x faster on a disk-tiered container, stream identical per seed.
- **Against:** flashmoe: external draft model 33% acceptance, 4.5x slower; MTP analysis ~1.75x expert I/O for ~1.7 accepted tokens (break-even). colibri: a rejected draft costs a full MoE row, break-even near 50% acceptance.
- **Reading:** Acceptance rate and expert overlap between drafted tokens decide it; gate on measured acceptance with an adaptive pause.

### C-11: More drives (P-31, P-32)

- **For:** colibri: two NVMe replicas 0.80 -> 1.10 tok/s (+37.5%) with O_DIRECT.
- **Against:** slipstream: tok/s negative when the second disk shares a bus; colibri: an equal-share slow SATA mirror 0.900 -> 0.666 tok/s; pulsar: decode pulled only 1.4-2.1 GB/s from disk once tiers absorbed lookups.
- **Reading:** Weight shares by a probe and expect gains only while disk is the bottleneck.

### C-12: Float contraction / FMA policy (P-56, P-59)

- **For:** colibri writes explicit fma in source (GCC-chosen contraction drifted ~1 ulp and flipped argmax); flashmoe's FMA-folded dequant gave +12% with outputs reported unchanged.
- **Against:** apus forbids FMA (mul then add, contraction off) and still reached the memory roofline (24-64 GB/s), because it bit-matches a numpy oracle.
- **Reading:** All agree the compiler must not decide; which form depends on the reference you bit-match.

### C-13: Expert accumulation order (P-76)

- **For:** colibri: routing-rank order in f32, shared expert last; output bit-identical across batched/union paths.
- **Against:** apus: ascending expert index with a bf16 rounding per add, measured bitwise against its reference over 184/184 decode digests.
- **Reading:** The order is reference-defined; make it a per-model parameter.

### C-14: MLA KV representation (P-64, P-65)

- **For:** colibri: 576-float latent rows, 57x smaller than expanded K/V on GLM-5.2.
- **Against:** apus: expanded per-head BF16 K/V for exactness (~0.71 MB/token over 11 layers; practical context 10-20K tokens on 32 GB).
- **Reading:** Latent caching is the default; expanded rows only as an exactness oracle.

### C-15: Vector width (P-55, P-46, P-41)

- **For:** ktransformers builds AVX-512 / AMX tile kernels for K-quants and FP8 (claims ~493 GB/s effective, hardware unspecified).
- **Against:** pulsar: AVX-512 VNNI 2.1 vs AVX2 4.8 GB/s for IQ2_XXS on Zen 5; flashmoe: 128-bit GPU loads -3%.
- **Reading:** Format-dependent: benchmark each (format, ISA) pair.

## Negative results ledger

The failed or net-negative experiments recorded in the fragment files, with each project's own
numbers on its own hardware.

| # | Experiment | Project | Measured result | Pattern | Fragment |
|---|---|---|---|---|---|
| N-01 | LZ4-compressed expert records | flashmoe | 3.55 vs 4.36 tok/s (-13%); cold reads 15-24% faster in isolation | P-09 | fmoe/F-storage-03 |
| N-02 | LZFSE codec; filesystem transparent compression | flashmoe | 2.6 GB/s (too slow); ~2x slower | P-09 | fmoe/F-storage-03 |
| N-03 | One file per expert | flashmoe | -15% (metadata) | P-09 | fmoe/F-storage-03 |
| N-04 | Co-occurrence reordering of experts on disk | flashmoe | 0% | P-09 | fmoe/F-storage-03 |
| N-05 | Whole-source hashing for container identity | slipstream | dominated convert time until hardware SHA | P-04 | F-store-03 |
| N-06 | Single slabs allowed to evict members of warm triples | pulsar | slab hit rate 72% -> 53% | P-02 | F-store-04 |
| N-07 | Overwrite routing history on save | pulsar | resident-tier hits halved; 8.2 -> 5.8 tok/s | P-07 | F-store-05 |
| N-08 | Undecayed routing history | colibri | one turn moved the ranking 0.2% (stops tracking) | P-07 | F-store-05 |
| N-09 | Reserved HOT sub-tier in the RAM cache | slipstream | -1% to -6% tok/s at 4-10 GiB | P-14 | F-cache-07 |
| N-10 | Full-scan LFU eviction over tens of thousands of slabs | pulsar | seconds per scan; first-64 sample order-biased | P-15 | F-cache-08 |
| N-11 | Splitting a layer into resident-then-missing passes with 1-2 misses | ds4 | single pass faster | P-15 | F-cache-08 |
| N-12 | Pricing pinned experts at the widest width | colibri | ~half as many experts pinned | P-19 | F-cache-11 |
| N-13 | Starving the page cache under buffered reads | colibri | 800 -> 180 MB/s | P-20 | F-cache-12 |
| N-14 | Planned RAM budget with no runtime guard | colibri | 74.4 GB planned, 115.6 GB real; three OOM kills | P-21 | F-cache-13 |
| N-15 | Memory guard on resident size (macOS) | apus | 32.5 GB footprint read below 26 GiB; guard never fired while thrashing | P-21 | apus/F-cache-03 |
| N-16 | mmap/munmap + zero-fill per 25-48 MiB fill | apus | major kernel time (fixed by buffer recycling) | P-10 | apus/F-cache-03 |
| N-17 | Expert cache above 2 GiB on a 32 GB machine | apus | buys almost nothing (guard trims, compressor churn) | P-22 | apus/F-cache-03 |
| N-18 | OS page cache instead of a bounded arena | slipstream | ~75% faster but 1000+ swap-outs; rejected | P-11 | F-io-16 |
| N-19 | GPU-buffer LRU expert cache, 1000 / 2500 / 3000 entries | flashmoe | 2.24 / 2.24 / 1.99 tok/s; final GPU LRU 4.80 vs no cache 5.74 | P-10 | fmoe/F-cache-01 |
| N-20 | Malloc zero-copy expert cache, 2581 / 500 entries | flashmoe | 2.10 tok/s (52% hit) / 2.77 tok/s (37% hit) | P-10 | fmoe/F-cache-01 |
| N-21 | Tiered fds (cache bypass on first read only) | flashmoe | marginal; same memory pressure | P-25 | fmoe/F-cache-01 |
| N-22 | O_DIRECT on QLC, DRAM-less or virtualised disks | colibri | neutral to negative | P-25 | F-io-16 |
| N-23 | F_NOCACHE vs trust-the-OS | flashmoe | -1% (5.68 vs 5.74 tok/s) | P-25 | fmoe/F-io-03 |
| N-24 | mmap + memcpy for cold expert reads | flashmoe | 0.56 tok/s (5x slower) | P-28 | fmoe/F-io-03 |
| N-25 | dispatch_io channel reads | flashmoe | -70% | P-24 | fmoe/F-io-03 |
| N-26 | POSIX aio | flashmoe | ~-7% / parity | P-24 | fmoe/F-io-03 |
| N-27 | F_RDADVISE immediate / one token ahead | flashmoe | -8% / -4% (65-80% of advised experts unused) | P-27 | fmoe/F-io-03 |
| N-28 | MADV_RANDOM on the expert mapping | flashmoe | harmful (reads split into ~5.7 x 512 KB ops) | P-26 | fmoe/F-io-03 |
| N-29 | F_RDAHEAD toggle; MADV_SEQUENTIAL | flashmoe | 0% | P-26 | fmoe/F-io-03 |
| N-30 | C pread extension called from Python/MLX | flashmoe | 18.8 GB/s alone, 2.6 GB/s under the interpreter lock | P-23 | fmoe/F-io-01 |
| N-31 | Equal shares including a 10x slower SATA mirror | colibri | 0.900 -> 0.666 tok/s | P-31 | F-io-18 |
| N-32 | Second disk on a shared bus (internal NVMe + slow USB) | slipstream | tok/s negative; shipped opt-in for capacity | P-31 | F-io-18 |
| N-33 | Equal 3-way striping including a SATA drive | colibri | 12.6 ms vs 1.9 ms weighted for a 19 MB expert | P-31 | F-io-19 |
| N-34 | Spin-polling instead of blocking GPU waits | flashmoe | -23% (CPU heat steals GPU budget) | P-60 | fmoe/F-planner-01 |
| N-35 | Unbounded speculative read backlog | pulsar | ~177 GiB queued during prefill, drained into decode | P-34 | F-io-17 |
| N-36 | Ungated whole-next-layer prefetch in prefill | pulsar | 1.43 vs 2.63 tok/s with prefetch off (GLM-5.2) | P-36 | F-pf-24 |
| N-37 | Hint-only lookahead on disk-saturated hosts | colibri | net negative; real loads +11 pp hit rate | P-27 | F-pf-21 |
| N-38 | Eviction guard protecting any warm victim | colibri | dropped ~100% of speculations once the cache filled | P-34 | F-pf-22 |
| N-39 | Unthrottled speculative hints | apus | starved demand fills 5:1 | P-34 | apus/F-cache-02 |
| N-40 | Online coupling predictor | slipstream | -8% at a 6 GiB cache | P-35 | F-pf-23 |
| N-41 | Naive coupled prefetch | slipstream | -5% to -15% | P-35 | F-pf-23 |
| N-42 | Temporal prediction (previous token's experts) | flashmoe | 25.6% hit; 3.18 vs 3.90 tok/s (-18%) | P-38 | fmoe/F-prefetch-01 |
| N-43 | Advisory prefetch during GPU compute | flashmoe | expert I/O -31%, next GPU wait +73%, net 0% | P-27 | fmoe/F-prefetch-01 |
| N-44 | Per-layer MLP routing predictor | flashmoe | 31% accuracy | P-39 | fmoe/F-prefetch-02 |
| N-45 | Early routing on the pre-attention hidden state | flashmoe | 53% accurate; 1.70 tok/s (-38%) | P-39 | fmoe/F-prefetch-02 |
| N-46 | Two-pass routing (route all layers, then I/O + compute) | flashmoe | ~6 tok/s but wrong, repetitive output | P-39 | fmoe/F-prefetch-02 |
| N-47 | AVX-512 VNNI kernel for IQ2_XXS on Zen 5 | pulsar | 2.1 vs 4.8 GB/s single-thread (AVX2 wins) | P-41 | F-quant-10 |
| N-48 | Unpacking int4 experts to int8 at load | colibri | 34 ms/token vs 17 ms (int8 floor) / 9 ms (packed int4 floor) | P-50 | F-quant-17 |
| N-49 | Letting GCC contract the 4-accumulator FP8 kernel | colibri | ~1 ulp drift flips argmax; pragmas ignored | P-56 | F-quant-13 |
| N-50 | Per-32 integer activation quantisation for MXFP4 | colibri | ~0.4% noise per group; kept opt-in | P-48 | F-quant-15 |
| N-51 | Integer (quantised-activation) kernel for GLM prefill projections | colibri | ~12% perplexity cost | P-87 | F-fwd-24 |
| N-52 | 2-bit requantised experts in production | flashmoe | 5.55-5.74 vs 4.36 tok/s but broke JSON / tool calls | P-53 | fmoe/F-quant-01 |
| N-53 | LUT dequant on the GPU | flashmoe | -2% | P-60 | fmoe/F-kernel-03 |
| N-54 | 128-bit vector weight loads | flashmoe | -3% (register pressure) | P-60 | fmoe/F-kernel-03 |
| N-55 | Explicit bit-field-extract intrinsic | flashmoe | 0% | P-60 | fmoe/F-kernel-03 |
| N-56 | One GPU encoder per whole expert | flashmoe | -2% (less cross-expert parallelism) | P-60 | fmoe/F-kernel-03 |
| N-57 | Copy experts to GPU-private compressed memory | flashmoe | matvec -13.5% but -20% overall | P-60 | fmoe/F-kernel-03 |
| N-58 | Softmax + top-K on the GPU | flashmoe | 3.96 vs 3.84 ms/layer | P-60 | fmoe/F-planner-01 |
| N-59 | Removing the GPU -> CPU -> GPU routing bounce | flashmoe | 4.07 vs 3.70 ms/layer | P-60 | fmoe/F-planner-01 |
| N-60 | Fused GPU final norm + LM head | flashmoe | 4.01 vs 3.93 ms/layer | P-60 | fmoe/F-planner-01 |
| N-61 | GPU attention scores at short context | flashmoe | slower below 32 tokens | P-62 | fmoe/F-attention-02 |
| N-62 | GPU BF16 expert GEMVs | apus | 0.85-0.96x of CPU | P-58 | apus/F-kernel-02 |
| N-63 | GPU dispatch for a tiny model | apus | 24 vs 173 tok/s on CPU (~50 us x ~40 ops/token) | P-58 | apus/F-kernel-02 |
| N-64 | Per-op GPU wrapping of ~230 dense ranges per token | apus | 4x loss (fixed by registered wraps) | P-58 | apus/F-kernel-02 |
| N-65 | Renormalising surviving experts when one is dropped | colibri | "catastrophically worse" on OLMoE | P-77 | F-fwd-18 |
| N-66 | int4 MTP head | colibri | acceptance 0-4% | P-84 | F-fwd-21 |
| N-67 | MTP pin chosen while a kv_b layout bug existed | apus | 0/8 drafts accepted | P-84 | apus/F-mtp-01 |
| N-68 | Batch-size-dependent kernel family during speculation | colibri | greedy output diverged from non-speculative; acceptance stats meaningless | P-87 | F-fwd-24 |
| N-69 | External draft model (35B-A3B) on a streamed 397B MoE | flashmoe | 33% acceptance; 4.5x slower | P-88 | fmoe/F-mtp-01 |
| N-70 | MTP speculation on a streamed MoE (analysis; weights absent) | flashmoe | ~1.75x expert I/O for ~1.7 tokens: break-even | P-88 | fmoe/F-mtp-01 |
| N-71 | K = 3 or K = 2 routed experts | flashmoe | output collapses to end-of-sequence | P-78 | fmoe/F-routing-01 |
| N-72 | Variable K by depth (0 or 2-3 in middle layers) | flashmoe | 10 tok/s with blank / collapsed output | P-78 | fmoe/F-routing-01 |
| N-73 | Unchunked prefill MoE union | apus | 41.5 GB peak | P-29 | apus/F-planner-01 |
| N-74 | Wrong pre-tokeniser family (cl100k rules on a GPT-2 model) | colibri | 1560/1708 cases matched; silent | P-98 | F-TOK-04 |
| N-75 | Windows budget from physical availability only | colibri | commit charge refused while RAM was free | P-107 | F-HW-01 |
| N-76 | 64 MB GPU bandwidth probe | pulsar | fit in L2; ~390 GB/s on every card | P-110 | F-HW-04 |
| N-77 | Even FFN split across unequal GPUs | pulsar | ran at the slow card's pace (fast card idle 61%) | P-110 | F-HW-04 |
| N-78 | Truncating (not rounding) the split to a quant block | pulsar | 47.8 -> 46.2 tok/s on identical cards | P-110 | F-HW-04 |
| N-79 | Choosing a GPU by free memory | pulsar | picked the slower card over a 69 MiB difference | P-110 | F-HW-04 |
| N-80 | Buffered / warm disk probe | colibri | 23-97 GB/s reported vs 14 GB/s real; resident shard 22.1 GB/s under bypass | P-109 | F-HW-03 |
| N-81 | Extrapolating another model's coverage curve | monolite | under-predicted by 19-32% (own counts: within 16%) | P-111 | F-HW-06 |
| N-82 | One generic admission refusal message | slipstream | users could not tell which knob to change | P-20 | F-HW-05 |
| N-83 | macOS 1.5 GiB headroom | slipstream | <0.1 tok/s from GPU residency thrash, zero swap-outs | P-20 | F-HW-05 |
| N-84 | Quality checks on a handful of prompts | flashmoe | missed the 2-bit JSON regression | P-126 | fmoe/F-testing-01 |
| N-85 | Speed numbers taken with a warm app cache later removed | flashmoe | 5.29 / 4.20 / 4.80 tok/s invalidated | P-125 | fmoe/F-testing-01 |

### Silent-correctness traps

Bugs that produced plausible output rather than a crash; each is a reason for a test in the
Testing table.

| # | Trap | Project | What happened | Fragment |
|---|---|---|---|---|
| T-01 | YaRN ramp computed over wavelength instead of pair index | monolite | coherent text; first divergence from transformers; found only by the token oracle | F-fwd-04 |
| T-02 | Plain w instead of (1 + w) RMSNorm on Qwen3.5/3.6 | colibri | plausible but wrong text | F-fwd-02 |
| T-03 | Shared scratch across parallel KDA heads | colibri | plausible-but-wrong tokens; only teacher forcing caught it | F-fwd-13 |
| T-04 | kv_b split head-major instead of per-head interleaved | apus | self-built oracle agreed bitwise; external anchor rel err 0.74, argmax 6/16 | apus/F-testing-01 |
| T-05 | CUDA race; a double-accumulated layer | monolite | plausible text; caught by the scalar oracle | F-fwd-29 |
| T-06 | All-NaN router scores | colibri | selected expert index -1 | F-fwd-14 |
| T-07 | Failed speculative load left marked in flight | colibri | leaked one ~19 MB slot per failure | F-cache-06 |
| T-08 | Reused buffers not re-sized across expert widths | colibri | every slot silently grew to the widest width | F-cache-06 |
| T-09 | Slot count derived from all layers instead of sparse layers | apus | 7 instead of 8 slots/layer at 16 GiB | apus/F-cache-01 |
| T-10 | io_uring path ignoring mirror routing | colibri | ring always read drive 0; mirrors idle | F-io-15 |
| T-11 | Fake F_NOCACHE define on Linux | slipstream | unknown fcntl silently failed; fully buffered reads with readahead | F-io-16 |
| T-12 | GPU matvec bounds check before the shared-memory barrier | flashmoe | early return left the shared activation copy partly filled | fmoe/F-kernel-01 |
| T-13 | Heap -> stack scratch in the mHC block | colibri | changed GCC's reduction rounding | F-fwd-19 |
| T-14 | Fixed 4096-entry attention score scratch | colibri | early version used a fixed stack array of 4096 scores; scratch must be sized to the context capacity | F-fwd-05 |
| T-15 | Fixed-size SSE chunk buffer | monolite | truncated chunks = malformed JSON; clients dropped the stream | F-SRV-02 |
| T-16 | signal() with restart semantics | monolite | accept() never returned EINTR; only SIGKILL stopped the server | F-SRV-01 |
| T-17 | Fixed 64 KiB JSON string buffer; realloc'ing arena | colibri | truncated tokenizer strings; dangling string pointers | F-JSON-01 |
| T-18 | Sending split UTF-8 sequences | ds4 | clients send U+FFFD back, breaking KV prefix matching | F-TOK-09 |
| T-19 | TEMP env var for temperature | colibri | collided with the Windows temp directory | F-CFG-01 |
| T-20 | Launcher-set thread-affinity env var | colibri | pinned every thread to one core | F-CFG-01 |
| T-21 | Physical-core detection silently falling back to 1 | colibri | halves throughput | F-HW-02 |
| T-22 | Slab-size arithmetic slip (16 B per F32 scale) | apus | caught by deriving sizes from config and checking the real index | apus/F-format-01 |

## Note on the files' own summaries

The "Recommended synthesis", "Notable ideas not seen elsewhere" and "Experiment ledger"
sections at the end of the fragment files predate this scoring and are left unchanged. Where
they disagree with the counts above, the counts win. Examples: apus's "not seen elsewhere" list
includes router-lookahead prefetch and a memory guard, whose base mechanisms appear in colibri
(apus adapted them); only the refinements (reusing the exact gated sublayer code for prediction,
guarding on physical footprint) are apus's own. The storage synthesis's "global CLOCK-LRU-K"
combines a slipstream-only policy (experimental here) with the global scope that ds4, pulsar,
monolite and ktransformers use.
