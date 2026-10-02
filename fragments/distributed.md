# Fragments: running one model across several machines (wire protocols, splits, failure handling)

Topic tag: `dist` (anything that crosses a process or machine boundary during inference:
handshakes, framing, expert / tensor / pipeline splits, state consistency, failure
semantics, discovery). All pseudocode below is written fresh after reading the cited
sources; names are ours. Numbers are each project's own measurements on its own
hardware, not ours.

Licences of sources: colibri (Apache-2.0), ds4 (MIT), pulsar (MIT, Rust+CUDA),
slipstream (MIT; its own Rust crates and engine code, not the llama.cpp base),
ktransformers kt-kernel (Apache-2.0). apus and flash-moe were searched and contain no
cross-machine inference code (apus has a local HTTP gateway in front of its C engine;
flash-moe is single-machine), so neither appears in an Origin line here. monolite is
single-machine too.

What the reference trees contain, in one paragraph: ds4 has the only mature multi-machine
code (a two-rank tensor-parallel pair over RDMA or TCP, and an N-stage layer pipeline over
TCP, about 11.6k lines between them). colibri has a small expert-parallel mode (routed FFNs
computed by disk-backed workers over persistent TCP) plus an HTTP registry. pulsar splits
work across GPUs inside one process by measured bandwidth, which carries the same lessons
about hop stalls. ktransformers splits each expert across NUMA nodes inside one box.
slipstream routes whole requests to remote peers over an authenticated mesh; it does not
split a model.

---

## F-dist-01: Peer identity handshake that refuses mismatched pairs before any work
Origin: [colibri] [ds4] [slipstream]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, slipstream)
- Source: ds4 (MIT), ds4_tp.c, ds4_tp.h, ds4_distributed.c (hello records and their checks); colibri (Apache-2.0), c/colibri.c (cluster request header); slipstream (MIT), docs/plans/2026-08-02-public-community-mesh-design.md, crates/p2p-core -- read only, not copied
- Purpose: make a wrongly paired cluster fail at connect time with a readable error, never mid-generation with plausible garbage.
- Inputs / outputs: each side's identity record -> accept (continue) or reject (both sides close with an error naming the first mismatching field).
- Pseudocode:
      identity = {
          magic (4 bytes; also detects byte-order mismatch),
          protocol_version,
          role (leader | worker | stage),
          model fingerprint: file size in bytes, model family id, n_layer, d_model, n_vocab,
                             quant bits / profile id,
          runtime shape: context size, layer range owned, owns output head?,
          per-token exchange schedule (how many exchange points per token and their order),
          transport capability (verbs device usable? yes/no)
      }
      on connect:
          send(mine); receive(theirs) with timeout
          if theirs.magic != MAGIC                      -> reject "not a peer / byte order"
          if theirs.protocol_version != mine            -> reject "update all peers together"
          for f in fingerprint and schedule fields: if theirs.f != mine.f -> reject naming f
          if both claim the leader role                 -> reject
          if both offered verbs and both succeed setup  -> data plane = RDMA else TCP
- Notes: ds4 bumps one protocol integer whenever any frame changes and documents "update all peers together" (its protocol is at version 14). Its QA forces an identity mismatch on purpose and requires both ranks to refuse before inference, and reflects a leader hello back to check that two leaders are refused. colibri's expert workers re-check an 8-byte magic, a version, the layer index, the model width and the expert FFN width on every request, and close the connection on any mismatch. The exchange schedule is part of identity in ds4 because RDMA receive slots are derived from it (F-dist-06): two peers that disagree on it would write partial sums into the wrong slots without any error.
- Variants seen elsewhere:
  - slipstream: the hello is a signed, short-lived capability advertisement (node key, encryption key, protocol versions, engine and model ids, hardware claims, expiry, nonce); signatures prove continuity of a pseudonymous identity only, and hardware/speed claims stay untrusted until observed locally.
  - colibri: no separate hello; each request carries magic + version + shape and the worker validates it.

## F-dist-02: Typed frames with fixed little records and bounded, pre-validated sizes
Origin: [colibri] [ds4] [slipstream]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, slipstream)
- Source: ds4 (MIT), ds4_distributed.c (frame header, record conversions, payload checks); colibri (Apache-2.0), c/colibri.c (cluster wire), c/tests/test_cluster_protocol.c; slipstream (MIT), crates/p2p-net/src/frame.rs, crates/p2p-net/src/replay.rs -- read only, not copied
- Purpose: a wire format that a C program can parse without trusting the peer.
- Inputs / outputs: byte stream -> sequence of (type, fixed record, variable tail) or a typed error.
- Pseudocode:
      frame  = header{magic u32, type u32, payload_len u32} + payload
      payload = fixed record (all u32 fields; u64 values split into hi/lo halves)
                + variable tails whose lengths are fields of the fixed record
      read_frame:
          read header; if magic wrong -> transport error
          if payload_len > cap_for(type) -> error   (never allocate first)
          read payload into a bounded buffer
          decode fixed record (convert byte order); then check:
              sum of declared tail lengths == payload_len - sizeof(fixed)
              every count <= its cap (rows <= 65536, items <= 64, tokens <= ctx - pos0, ...)
              flags & ~known_flags == 0
              cross-field rules (e.g. "stage 0 must not receive a hidden state",
                                 "logits requested only from the stage that owns the head")
          on any failure: reply ERROR{request id, short text} and keep the link if framing is intact
      unknown type: read and discard payload_len bytes, reply ERROR
- Notes: ds4's work record validates every size, flag, layer range, context bound and stage rule before touching the engine, and answers with an error frame rather than dropping the link. colibri's worker bounds item count (1..64) and rows per item (1..65536) and closes on violation; a socketpair unit test freezes the request/response layout with no model needed. Fixed u32 fields in network order cost a conversion per field but make records trivially portable.
- Variants seen elsewhere:
  - slipstream: u32 big-endian length prefix, 4 MiB cap per frame, zero-length frames rejected; a process-wide bounded replay cache rejects a captured request replayed on another connection.
  - colibri's gateway codec (between its Python server and the C engines) was migrated across engines behind byte-exact wire-transcript freezes, after duplicated framing code silently lost Windows binary mode in sibling engines.

## F-dist-03: Expert-parallel workers fed one batch-union request per layer
Origin: [colibri]

Licence-of-origin: Apache-2.0 (colibri)
- Source: colibri (Apache-2.0), c/colibri.c (cluster connect, per-layer batch send, worker loop), c/cluster.py, README.md (local cluster mode), CHANGELOG.md -- read only, not copied
- Purpose: let other machines' RAM and cores compute routed experts while the coordinator keeps tokens, routing, attention, shared experts and KV.
- Inputs / outputs: per MoE layer: hidden rows x[S][D], routing (ids[S][K], weights[S][K]), the unique expert list of the batch -> out[S][D] += sum over remote experts of weight * expert(x).
- Pseudocode:
      coordinator, per MoE layer:
          uniq = unique expert ids routed by any row in this batch
          for each worker w (fixed order):
              mine = [e in uniq if (e + layer) mod n_workers == w]       # static, layer-rotated shard
              if mine empty: continue
              request = header(layer, D, ffn_width, n_items)
                        + for e in mine: (e, n_rows_using_e, those rows of x as f32)
              send request on w's persistent socket
              read reply: header(status, n_items) + for each item: (e, n_rows, y rows as f32)
              check e and n_rows echo the request
              for each row r of item e: out[row] += routing_weight(row, e) * y[r]
          shared expert and attention stay local
      worker:
          load model config; open expert files (may be spread over several drives)
          per layer: one-slot cache of the last expert used
          loop: read request; validate; for each item: load expert if not cached;
                compute gate/up/act/down on its rows; stream y back
- Notes: the README's point is that a whole layer's union goes as one request per worker, so a token pays one round trip per layer per worker, not one per expert. Things this first version does not do (and a successor should): workers are visited one after another (send, wait, next), so worker latencies add instead of overlapping; any worker failure ends the coordinator process; activations travel as f32 in both directions; shard ownership is a fixed formula, not a heat- or capacity-aware map; workers keep only one expert per layer in RAM. The mode currently supports one model family; dense-layer sharding is listed as future work.
- Variants seen elsewhere:
  - ds4: the two-rank "tensor parallel" mode is in effect expert sharding (each rank holds one contiguous half of the routed experts) with the per-layer partial outputs summed through an exchange (F-dist-05/06), rather than a request/response to a server that holds no attention state.

## F-dist-04: Token-exact parity gate for any delegated computation
Origin: [colibri] [ds4]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4)
- Source: colibri (Apache-2.0), c/tests/test_cluster_sharding.py, c/tests/test_cluster.py; ds4 (MIT), ds4_tp.h (hidden-state hash check), QA_BEFORE_RELEASES.md (network TP sections) -- read only, not copied
- Purpose: prove that moving work to another machine changes speed only.
- Inputs / outputs: tiny generated model fixtures (several weight formats), the same binary on both sides -> pass/fail.
- Pseudocode:
      pin numeric configuration on all processes (same build, same kernel choices,
          drafting off, same thread-count-independent reductions)
      run A: single process, teacher-forced over a fixed token sequence -> per-position signature
      run B: same, with routed experts delegated to a local worker process
      require signature(A) == signature(B) and zero oracle mismatches in both runs
      repeat for every weight format whose worker path differs (e.g. formats that need a
          pre-rotation of the input before the expert matmul)
      lockstep pairs (debug mode): after each token both ranks hash their hidden state and
          exchange hashes; any difference aborts with the position
- Notes: colibri's gate exists because a weight format with an input rotation computed wrong on the worker path while the local path was right; the parity test caught it. ds4 requires physical two-machine runs over both transports, two- and four-session runs, and checks that each rank's partial matches its scalar reference exactly for small batches ("not merely produce a close combined sum"). For speculative decoding this composes with the draft == verify rule already in the library (P-87).
- Variants seen elsewhere: ds4 also scores fixed quality fixtures through physical TP and compares to the unsplit reference after changes to split attention.

## F-dist-05: Two-rank tensor parallel by lockstep command mirroring
Origin: [ds4]

Licence-of-origin: MIT (ds4)
- Source: ds4 (MIT), ds4_tp.h, ds4_tp.c, docs/DISTRIBUTED.md -- read only, not copied
- Purpose: run one decode stream on two machines that each hold half of the routed experts, with both executing the identical sequence of graph calls.
- Inputs / outputs: leader session API calls -> mirrored command frames on a control socket; worker replays them on its engine.
- Pseudocode:
      leader wraps every session operation:
          create/destroy session, sync(prompt tokens [+ images]), eval(token, seq),
          rewind(pos), invalidate, eval_batch(items), mixed prefill+decode batch,
          draft verify (see F-dist-17), model-specific draft step
          -> send the matching command frame, then run the same operation locally
      worker: loop { cmd = receive(); run the same engine call; ack where the leader needs it }
      inside each operation both ranks hit the same exchange points in the same order
          (per layer: after attention partial, after FFN partial; F-dist-06)
      cancellation: both ranks call an agreement step at fixed prefill boundaries,
          exchange {seq, point, progress, requested?}; cancelled = mine OR theirs;
          a rank never decides to cancel on its own while inside an exchange
      output head: optionally split by vocabulary (F-dist-16)
- Notes: unsupported combinations are refused at option-validation time (no SSD streaming, no layer pipeline, no CPU backend, some speculation modes) rather than discovered at run time. Disk-cache restore under TP rebuilds the saved prefix on both ranks (a prefill) instead of restoring one side. With five or more ready sessions the server batches decode across both GPUs; smaller groups run one after another because that measured faster. Measured: two DGX Sparks over RoCE averaged 21.9 tok/s for one session (V4.1 Flash Q2, 1K prompt, 2,048 teacher-forced tokens, no speculation); eight sessions about 28 tok/s aggregate; 32K prefill about 400 tok/s.
- Variants seen elsewhere: none in the reference set over a network; pulsar and ktransformers split inside one process (F-dist-15).

## F-dist-06: Partial-sum exchange buffer with a fixed per-token schedule
Origin: [ds4]

Licence-of-origin: MIT (ds4)
- Source: ds4 (MIT), ds4_tp.h (buffer layout notes), ds4_tp.c (receive window, exchange, batch and bulk paths) -- read only, not copied
- Purpose: exchange one D-wide partial vector per exchange point with microsecond overhead and no per-message control traffic.
- Inputs / outputs: one registered buffer per rank; per exchange point (layer, kind) a local partial -> the peer's partial landed in a known slot.
- Pseudocode:
      buffer (registered once with the NIC):
          out[n_slots][D] f32   written by local kernels
          in [n_slots][D] f32   written by the peer's sends
          arrival flags[n_slots]
          a small token slot (leader -> worker)
          batch rows for verify blocks: [n_layer][max_rows][D] out and in
      n_slots = n_layer * exchange_kinds; the slot for sequence number s is a pure function of s
          (start + ((s-1) mod per_token) * step, or an explicit bitmask of which layers fire)
      decode:
          once per window: post R receives (R = 16) for seq .. seq+R-1, then a barrier with the peer
              (unreliable connections drop a send that arrives before its receive is posted)
          exchange(seq): check slot(seq) matches the layer/kind the engine is at (else abort: order broke)
                         post send of out[slot] (split into <= 16 KiB pieces if needed)
                         poll completions until the receive for seq completed
                         post the receive for seq + R
      verify block (speculation, rows <= 8): one bulk message of rows partials per layer,
          receives posted one layer ahead, no per-exchange control traffic
      long prefill: bulk symmetric exchange over RDMA staging slots; TCP fallback in 2 MiB rounds
- Notes: partials are f32 and never quantised on the wire (vec bytes = D * 4). On Apple's Thunderbolt RDMA the project found only unreliable-connected queue pairs, one-sided writes accepted but never executed, messages above 16 KiB not delivered, and completions reported for unsignalled sends; so the data plane is two-sided send/receive with every send accounted for. On Linux RoCE it uses reliable-connected pairs (timeout and retry counts set). An optional profile prints mean post cost and peer-wait per exchange kind.
- Variants seen elsewhere: none in the reference set.

## F-dist-07: Full-duplex non-blocking TCP exchange with a desync header and a progress-extended deadline
Origin: [ds4]

Licence-of-origin: MIT (ds4)
- Source: ds4 (MIT), ds4_tp.c (TCP exchange path, socket options) -- read only, not copied
- Purpose: the TCP fallback for symmetric exchanges that cannot deadlock when both sides send large payloads at once.
- Inputs / outputs: (header, out bytes) to send, (peer header, in bytes) to receive -> both complete or a typed failure.
- Pseudocode:
      header = {magic, layer u16, kind u16, seq u64}      # lets a desynchronised pair fail loudly
      set TCP_NODELAY; separate sockets: control (blocking, framed commands) and data (exchanges)
      exchange:
          tx = [header, out]; rx = [peer_header, in]
          deadline = now + T
          while tx or rx not finished:
              progress = false
              if tx: n = non-blocking scatter send; advance; progress |= n > 0
              if rx: n = non-blocking scatter receive; advance; progress |= n > 0
              if peer closed: fail
              if progress: deadline = now + T; continue          # slow but moving is fine
              if now > deadline: fail (timeout)
              wait for readable/writable with the remaining time
          check peer_header == expected (magic, layer, kind, seq) else fail
- Notes: sending first and then receiving would deadlock once both payloads exceed the kernel socket buffers; interleaving both directions in one loop avoids it regardless of buffer sizes. On macOS the send could block despite a non-blocking flag, so the socket is switched to non-blocking mode around the exchange and restored afterwards. The exchange deadline default is 750 ms (F-dist-09).
- Variants seen elsewhere: ds4's pipeline mode instead uses large socket buffers (128 MiB default), a 60 s send timeout and no receive timeout on control sockets, because those sockets may sit idle while a snapshot moves on another connection.

## F-dist-08: Optional verbs transport: loaded at run time, picked by address, warmed up, retried, with TCP fallback
Origin: [ds4]

Licence-of-origin: MIT (ds4)
- Source: ds4 (MIT), ds4_tp.c (verbs loader, device and GID selection, queue pair setup, warm-up, retry), docs/DISTRIBUTED.md -- read only, not copied
- Purpose: use RDMA when it truly works and fall back cleanly when it does not, without making the build depend on an RDMA stack.
- Inputs / outputs: peer address, optional device/GID override -> an established RDMA data plane or the TCP data plane.
- Pseudocode:
      at start: dlopen the verbs library; resolve only setup entry points
          (post/poll are inline wrappers over the device context, so they need no lookup)
      if load fails or no device is active: transport = TCP
      pick device: the one whose port has a GID mapped from the local address on the direct link
          (refuse management-network addresses); allow explicit device/GID override when ambiguous
      exchange endpoint info over the control socket: {buffer address, remote key, queue pair number,
          packet sequence number, MTU, LID, GID, link layer, reliable?}
      bring queue pair to ready-to-send; register the exchange buffer
      warm-up: both sides post a receive and send one FULL-SIZE message; report success to each
          other over the control socket (tiny messages can pass on a direction that drops real ones)
      if warm-up fails: destroy and recreate CQ/QP, retry (at most 3 attempts), else fail the setup
- Notes: the docs insist that "ping works" is not RDMA evidence: check the verbs device is active and has the right address-mapped GID. On Linux the same code drives RoCEv2 (DGX Spark pairs) with host staging, not GPUDirect. Do not probe a waiting leader with ad-hoc tools; it may take the probe for a worker handshake.
- Variants seen elsewhere: none in the reference set.

## F-dist-09: Failure semantics: bounded waits, liveness from the control socket, invalidate and replay
Origin: [ds4]

Licence-of-origin: MIT (ds4)
- Source: ds4 (MIT), ds4_tp.c (exchange deadline, peer-closed probe), ds4_distributed.c (route invalidation, transcript replay), QA_BEFORE_RELEASES.md (disconnect and SIGSTOP tests) -- read only, not copied
- Purpose: a dead or stalled peer turns into a prompt, clean error and a recoverable state, never a hang or silent corruption.
- Inputs / outputs: any wait on a peer -> success, or failure that invalidates exactly the affected sessions.
- Pseudocode:
      every wait on the peer has a deadline:
          per-exchange: default 750 ms (well below the GPU watchdog), env-overridable for diagnosis
          connection/dial: 300 s total while the peer loads its model (dial retries are normal)
      while polling RDMA completions: every ~16k polls, peek the control socket;
          a closed control socket means the peer is gone (unreliable QPs never report it)
      on failure: mark the transport failed; fail the current operation;
          invalidate every session that touched the peer
      pipeline mode: a lost stage invalidates the route; later requests wait for a complete route;
          the coordinator rebuilds remote state by replaying the saved token prefix (F-dist-11)
- Notes: QA kills the worker during a batch over TCP and RDMA, and separately SIGSTOPs it at a marker and resumes it after the leader returns: both runs must fail the exchange, invalidate the sessions and return control with no GPU watchdog error. A worker pointed at an unused port with a 1 s timeout must return a connection error instead of retrying for ever.
- Variants seen elsewhere:
  - colibri: fail-stop; a worker error during a layer prints the worker and layer and exits the coordinator.
  - slipstream: request expiry, cancellation, per-protocol timeouts, capped exponential reconnect backoff; local inference keeps working when the mesh is gone.

## F-dist-10: Layer-range pipeline: registration, route plan, stage-to-stage forwarding, per-hop telemetry
Origin: [ds4] [pulsar]

Licence-of-origin: MIT (ds4, pulsar)
- Source: ds4 (MIT), ds4_distributed.c (hello with layer range, route plan, work and result records, forwarder), docs/DISTRIBUTED.md; pulsar (MIT), crates/engine/src/lib.rs (per-layer device split filled fastest-first) -- read only, not copied
- Purpose: pool several machines' memory by giving each a contiguous range of layers.
- Inputs / outputs: workers announce (layer_start, layer_end, owns head?, listen port); a token span enters stage 0 and logits leave the last stage.
- Pseudocode:
      worker hello: {model id, quant, layer range, owns output?, ctx, n_layers, listen port, name}
      coordinator: route = ordered non-overlapping ranges covering 0..n_layers-1 (+ head);
                   ready only when coverage is complete
      work record: {model, session, request id, prefix hash, result hash, pos0, n_tokens,
                    layer range, flags: has-input-hidden | want-logits | reset | ack-only,
                    sizes of token / hidden / route tails}
                   + tokens + optional input hidden (wire precision per F-dist-12) + route blob
      stage: validate (F-dist-02); run its layers; if not last: forward {work, new hidden}
             directly to the next stage named in the route blob; else return logits or hidden upstream
      result record: {request id, result hash, status, kind (ack | hidden | logits),
                      telemetry count/bytes, payload bytes/bits}
      telemetry per hop: {layer range, route index, pos0, n_tokens, eval us,
                          downstream wait us, forward-send us, input bytes, output bytes}
- Notes: ds4 says plainly that one generation stream gets no overlap: each token must traverse the whole route before the next can be sampled, so pipeline mode is for capacity and long-prefill throughput, not decode speed. Workers read the next job while evaluating (receive depth 2) and keep up to 4 forwards in flight. Startup is expensive because each side makes its slice resident. Snapshot files stay topology-neutral (F-dist-18).
- Variants seen elsewhere: pulsar (one process, several GPUs) fills the fastest card first with as many layers as fit, then spreads the rest over every secondary fastest-first, because layers run one after another within a token so filling fast devices first minimises the sum of bytes/bandwidth; a list override exists because a weight-only plan cannot see context length and ran out of KV memory at long contexts.

## F-dist-11: Token-prefix hash chain to keep remote state on the right timeline
Origin: [ds4]

Licence-of-origin: MIT (ds4)
- Source: ds4 (MIT), ds4_distributed.c (token hash, work validation, rebuild from transcript) -- read only, not copied
- Purpose: detect, before computing, that a remote stage's KV holds a different token history than the coordinator thinks.
- Inputs / outputs: committed tokens -> 64-bit rolling hash; each work item carries (hash of prefix before the span, hash after the span).
- Pseudocode:
      h(empty) = FNV-1a offset basis; h(prefix + t) = fold the 4 bytes of t into h with FNV-1a
      coordinator: work.prefix_hash = h(tokens[0..pos0)); work.result_hash = h(tokens[0..pos0+n))
      stage: require h_update(work.prefix_hash, span tokens) == work.result_hash  (record integrity)
             (and its own session timeline must match the prefix it holds)
             reply carries result_hash; coordinator checks it equals what it sent
      on mismatch or after a route change: replay the full transcript as a prefill to rebuild
          every remote stage, then continue
- Notes: cheap (a few multiplies per token) and catches the bugs that otherwise show up as fluent but wrong text: a stage that missed a rewind, a reconnect that reused a session id, two callers sharing a session. Snapshot requests carry the same token hash and count.
- Variants seen elsewhere: none in the reference set.

## F-dist-12: Activation precision on the wire
Origin: [ds4] [pulsar]

Licence-of-origin: MIT (ds4, pulsar)
- Source: ds4 (MIT), ds4_distributed.c (activation encode/decode), ds4_tp.h, docs/DISTRIBUTED.md; pulsar (MIT), crates/kernels/src/lib.rs (default-on hop levers), docs/prefill-chunk-pipeline-design.md -- read only, not copied
- Purpose: trade link bytes for numerical exactness, explicitly and per mode.
- Inputs / outputs: f32 hidden rows -> wire bytes at 32, 16 or 8 bits per value -> f32 rows.
- Pseudocode:
      bits = 32 (default) | 16 | 8
      32: send the f32 bytes as is (receiver may use the wire buffer directly)
      16: IEEE half with round-to-nearest-even; overflow -> inf; NaN kept as NaN
      8:  E4M3-style byte: saturate at the format max; tiny values to the subnormal grid
      encode/decode in bounded chunks (<= 1M values) to cap scratch memory
      record bits in the work/result record; receiver checks size == values * bits/8
- Notes: ds4 keeps pipeline activations at 32 bits by default and documents 16/8 as changing numerical precision on the wire (validate output when changing them); its tensor-parallel partial sums are always f32. pulsar's in-process TP hops default to BF16 plus peer-to-peer copies; together they measured 11.15 -> 10.12 s on a 7.4k-token prefill.
- Variants seen elsewhere: colibri's expert workers move f32 rows both ways.

## F-dist-13: Chunk-windowed pipelined prefill (decode gets no such overlap)
Origin: [ds4] [pulsar]

Licence-of-origin: MIT (ds4, pulsar)
- Source: ds4 (MIT), ds4_distributed.c (prefill sender/reader threads, window and chunk sizing); pulsar (MIT), docs/prefill-chunk-pipeline-design.md -- read only, not copied
- Purpose: keep every stage busy during long prompts by having several chunks in flight.
- Inputs / outputs: prompt of N tokens -> chunks of C tokens, W in flight -> final logits.
- Pseudocode:
      C = session prefill capacity (overridable, never above it)
      W = override or (number of remote stages + 2), clamped to [2, 8]; hard max 64; W <= #chunks
      only pipeline when the prompt is longer than one chunk and the last stage owns the head
      sender thread: while chunks remain and in_flight < W: send chunk; in_flight++
      reader thread: receive results in order; in_flight--; report progress; wake sender
      cancel: stop sending, drain or drop in-flight results, invalidate the session
- Notes: pulsar's profile of 3-GPU tensor-parallel prefill showed the GPUs 49/45/10% busy with the idle time made of thousands of 3.5 ms per-layer hop stalls, and "nothing inside one chunk can fill a stall: layer i+1 depends on layer i; the filler has to be another chunk". Two lanes of chunks cut a 7.4k prefill from 10.11 s to 8.08-8.16 s (-20%) and a 50k prefill by 12%, with bit-identical ids.
- Variants seen elsewhere: none beyond these two.

## F-dist-14: Hop stalls, not FLOPs, bound split execution; balance splits by measured speed
Origin: [ds4] [pulsar]

Licence-of-origin: MIT (ds4, pulsar)
- Source: pulsar (MIT), docs/prefill-chunk-pipeline-design.md, crates/kernels/src/lib.rs (link probe and device choice), crates/engine/src/lib.rs (bandwidth-proportional FFN split); ds4 (MIT), docs/DISTRIBUTED.md, docs/SERVER.md -- read only, not copied
- Purpose: decide how many parties to split across, and in what proportions, from measurements.
- Inputs / outputs: per-party measured bandwidth (link and memory) -> number of parties, split fractions.
- Pseudocode:
      probe each party (best of 3 timed transfers, buffer larger than any cache); never trust labels
      primary = party with the fastest link to the data source
      split a barrier-synchronised layer in proportion to measured speed:
          frac_a = bw_a / (bw_a + bw_b); round the cut to the nearest quant block; clamp to [1/4, 3/4]
      add a party only if its compute share saves more than the extra per-layer hop costs:
          gain(n+1) = T_compute/n - T_compute/(n+1) - layers_per_token * hop_cost
- Notes: pulsar: CUDA's own device order and lspci both hid a card training at PCIe x1 (0.8 GB/s vs 28.8 GB/s), so the primary is chosen by an H2D probe; truncating instead of rounding the split cost 47.8 -> 46.2 tok/s on identical cards; dropping the third (x4-linked) card made a long prefill faster (9.33 s vs 10.11 s) but decode slower (34.3 -> 31.8 tok/s): fewer hops help latency-bound phases, more memory bandwidth helps decode. A 2.79x faster GEMM moved the TP prefill wall from 11.15 to 11.14 s: kernel work does not help when the wall is hops. ds4 documents that TP's gain "depends on the model, link, and comparison setup", and runs small session groups in order because that measured faster than batching across ranks.
- Variants seen elsewhere: colibri weights multi-drive reads by probed per-drive speed (same measure-first rule; F-HW-04).

## F-dist-15: FFN tensor split with per-partition partial outputs and a sum merge
Origin: [ds4] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (ktransformers), MIT (ds4, pulsar)
- Source: ktransformers kt-kernel (Apache-2.0), operators/moe-tp.hpp; pulsar (MIT), crates/engine/src/lib.rs (FFN split across GPUs); ds4 (MIT), ds4_tp.h -- read only, not copied
- Purpose: let several memory domains read disjoint slices of every expert at once, then add the partial outputs.
- Inputs / outputs: x[S][D], routing; P partitions -> y[S][D] = sum over partitions of partial_p.
- Pseudocode:
      split the expert intermediate dimension I into P slices (whole quant blocks per slice;
          remainder blocks to the first slices); each partition holds gate/up rows and down
          columns of its slice for every expert, in memory local to that partition
      forward: in parallel for p in partitions:
                  partial_p = down_p( act( gate_p(x) ) * up_p(x) ) for the routed experts
               y = sum_p partial_p        # fixed order of p
- Notes: SwiGLU is elementwise over I, so slicing I needs no exchange until the final sum: one reduction per layer. In ktransformers the partitions are NUMA nodes of one machine with one thread pool each; in pulsar they are GPUs with BF16 hops; in ds4 they are two machines, but there the split is by expert ownership (each rank holds half the experts whole) and the per-layer exchange carries the f32 partial sum. Summing in a fixed partition order keeps results reproducible.
- Variants seen elsewhere: slicing by expert (ds4, colibri) moves whole experts and gives uneven load when routing is skewed; slicing I gives even load but every partition touches every routed expert.

## F-dist-16: Vocabulary-split output head
Origin: [ds4]

Licence-of-origin: MIT (ds4)
- Source: ds4 (MIT), ds4_tp.h, ds4_tp.c (worker logits return path), QA_BEFORE_RELEASES.md (head checks) -- read only, not copied
- Purpose: halve the output-head weight bytes per rank.
- Inputs / outputs: final hidden h (replicated) -> each rank computes logits for half the vocabulary; worker ships its half to the leader.
- Pseudocode:
      rank 0 owns rows [0, V/2), rank 1 owns rows [V/2, V)   (V must be even)
      after every eval / sync: worker sends its V/2 f32 logits to the leader on the control socket
      leader: logits = concat(own half, received half); sample as usual
      a missing half is an error, never "use the stale buffer"
- Notes: the cost is V/2 * 4 bytes per evaluated row (about 253 KiB for a 129,280-entry vocabulary). ds4's QA requires both compact halves to match the full head's rows exactly, including batch rows and untouched output tails.
- Variants seen elsewhere: none in the reference set. (Our note, not from any project: for greedy decoding each rank could send only its local (max value, argmax); for exact sampling, (max, log-sum-exp, top-k) per rank is enough to pick the half with probability proportional to its mass and then sample inside it. See docs/design/cluster.md.)

## F-dist-17: Speculative verify across ranks with three commit modes
Origin: [ds4]

Licence-of-origin: MIT (ds4)
- Source: ds4 (MIT), ds4_tp.h (verify announce and commit), ds4_tp.c (verify-block window) -- read only, not copied
- Purpose: run a batched draft verification on a split model and leave both ranks with the same committed state.
- Inputs / outputs: draft tokens d[0..n) -> both ranks verify rows [next] + d; leader decides commit -> both apply it.
- Pseudocode:
      leader: send VERIFY{session, drafts}; both ranks open a verify window (receives for
              rows x layers posted ahead) and run the batched verify with one exchange per layer
      leader: choose commit:
          full        -> keep all verified rows
          prefix(m)   -> keep the first m rows; restore the matching verifier prefix on both ranks
          rollback(r) -> restore the pre-verify frontier on both, then replay r tokens
      leader: send COMMIT{mode, count}; worker blocks until it arrives, applies it
      close the window: every posted receive consumed and every send reaped, else fail
- Notes: the speculative block size is small (rows <= 8; drafts <= 5) so one bulk exchange per layer carries all rows. Recurrent/linear-attention state needs the explicit restore paths; attention KV beyond the commit point is simply overwritten later (P-85).
- Variants seen elsewhere: none in the reference set over a network.

## F-dist-18: Topology-neutral KV snapshots streamed in chunks
Origin: [ds4]

Licence-of-origin: MIT (ds4)
- Source: ds4 (MIT), ds4_distributed.c (snapshot save/load, begin/chunk/done records), ds4_distributed.h -- read only, not copied
- Purpose: save and restore a distributed session in the same file format as a single-machine session.
- Inputs / outputs: session -> one payload; payload -> state on whatever route is currently registered.
- Pseudocode:
      save: for each remote stage: send SAVE_REQ{session, token hash, token count, layer range}
            stage replies BEGIN{status, payload size, tokens, message} + CHUNK{<= 8 MiB}... + DONE{status}
            coordinator splices each stage's layer tensors into the normal payload layout
      load: parse the normal payload; validate its KV layout against the live model;
            cut it by the current route's layer ranges; stream each stage its shard the same way
      snapshots move on a separate data connection so control sockets stay free
- Notes: this is why control sockets in pipeline mode have no receive timeout (F-dist-07 variant). Because the format does not encode the topology, a session saved on two machines can be restored on three, or on one.
- Variants seen elsewhere: ds4's TP mode instead rebuilds a restored prefix on both ranks by prefill (F-dist-05).

## F-dist-19: Registration and discovery with heartbeats and staleness
Origin: [colibri] [ds4] [slipstream]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, slipstream)
- Source: colibri (Apache-2.0), c/cluster.py, c/coli (worker registration thread, discovery), README.md; ds4 (MIT), ds4_tp.c, ds4_distributed.c (dial with retry, accept thread, worker monitor); slipstream (MIT), docs/plans/2026-08-02-public-community-mesh-design.md, crates/p2p-net -- read only, not copied
- Purpose: find workers and forget dead ones without manual bookkeeping.
- Inputs / outputs: node records {id, host, port, role, protocol version} -> the live set of workers per role.
- Pseudocode:
      registry (optional; static host:port lists keep working without it):
          register(node) -> stamp last_seen, protocol version; roles: expert | dense | coordinator
          heartbeat(id)  -> refresh last_seen (worker thread every few seconds)
          topology()     -> nodes with now - last_seen <= stale_after (30 s), sorted by role, id
      worker start: register, start heartbeat thread, listen
      coordinator start: discover expert endpoints (or use the static list); connect each;
          refuse to start if none are reachable
      leader/worker pairs without a registry: worker dials the leader and retries while the
          leader is still loading its model; the leader accepts exactly the expected peers
- Notes: colibri's registry is HTTP/1.0 JSON with a Host-header guard (loopback plus the bind address unless hosts are explicitly allowed), and a bug where a missing default turned into the literal string "None" on the worker command line shows that launcher glue needs the same tests as the engine. ds4 monitors worker sockets from an accept thread and drops a worker's route entries when its socket closes.
- Variants seen elsewhere: slipstream uses libp2p (QUIC, Noise, mDNS, Kademlia, NAT traversal, relays) for a public mesh and plain configured peers for private ones.

## F-dist-20: Scored peer selection for whole requests, and the trust boundary
Origin: [slipstream]

Licence-of-origin: MIT (slipstream)
- Source: slipstream (MIT), crates/p2p-router/src/score.rs, crates/p2p-net/src/frame.rs, crates/p2p-net/src/replay.rs, docs/plans/2026-08-02-public-community-mesh-design.md, README.md -- read only, not copied
- Purpose: send a whole request to the best node when work is not split inside a token (multi-user serving across boxes).
- Inputs / outputs: peer snapshots {capability, load, rtt, reputation, is_local} + request -> chosen peer or none.
- Pseudocode:
      hard gates: enough RAM/VRAM; advertises the requested model; load < 1.0
      score = big bonus if local and capable
            + backend fit + hardware headroom + speed estimate * w1 + reputation * w2
            + stickiness bonus for an ongoing coding session
            - rtt/2 (capped) - price (capped)
            - load * a - load^2 * b          # quadratic: shed hot peers before they saturate
      pick max score; requests classed local-only never leave the machine
- Notes: the quadratic load term is there because a linear one let hot peers keep winning near saturation (their own swarm benchmark). Trust: requests are sealed end to end, but the chosen worker must decrypt them to run inference, which the design states plainly. ds4 and colibri instead assume a trusted network: no authentication or encryption, "use trusted machines and a trusted network".
- Variants seen elsewhere: none in the reference set.

---

## Lessons for tierlm (synthesis of this file, our wording)

1. Only ds4 has production-grade cross-machine code; its rules are the ones to copy in spirit:
   identity before work (F-dist-01), every wait bounded (F-dist-09), state checked by
   prefix hash (F-dist-11), output identical to single-machine and tested physically over
   each transport (F-dist-04).
2. Nothing in the reference set does the arrangement the user wants (predictors on other
   machines feeding the master's prefetcher and drafter). The closest pieces are colibri's
   in-box router lookahead (P-33), the speculative-load safety rules (P-34), and the
   rejected predictors (P-35, P-38, P-39, P-88): recall gains do not pay unless the reads
   they trigger are cheap relative to the evictions they cause. A remote predictor inherits
   all of that.
3. Per-layer round trips are the cost to count. pulsar measured per-layer hop stalls as the
   wall for split prefill; colibri's expert workers pay one round trip per layer per worker
   (sequentially). Any design whose critical path has L round trips per token needs a link
   whose round trip is small next to token_time / L.
4. Keep the transport swappable and the verbs library optional (F-dist-08). RDMA quirks are
   provider-specific and only show up on real hardware.
5. Fail-stop (colibri) is acceptable for a first prototype; tierlm's master always holds the
   whole model on its own NVMe, so every remote role can and should degrade to local work.

## CONSENSUS.md rows (merged into its "Distributed" section)

Merged as the "Distributed" section of CONSENSUS.md; numbering continues after P-127. Counts follow README rules.

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
