# Forward pass fragments: transformer MoE numerics

Scope: everything between "token ids in" and "next token out" for MoE decoders.
OLMoE and Qwen3-MoE style are described in full detail first; MLA/DSA (GLM,
DeepSeek V3 lineage), hybrid linear attention and DeepSeek V4 follow as deltas.
All pseudocode below is written from scratch after reading the listed files;
naming is ours. Abbreviations: D = hidden size, H = query heads, Hkv = kv
heads, hd = head dim, E = routed experts, K = experts per token, T = keys
visible to a query, V = vocab.

Licences of the sources consulted: colibri (Apache-2.0), monolite
(Apache-2.0), ds4 (MIT), pulsar (MIT).

---

## F-fwd-01: Decoder skeleton, embedding, final norm, lm_head
Origin: [colibri] [ds4] [pulsar] [monolite] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/olmoe.c (step / layer loop / generate); monolite (Apache-2.0), src/model.c (scalar forward) -- read only, not copied
- Purpose: fix the order of operations around the layer stack.
- Inputs / outputs: ids[S] (int), start position p0 -> logits[V] (float32) for the last row (or [S,V] when every row is needed: verify, scoring, teacher forcing).
- Pseudocode:
      h[s] = embedding_table[ids[s]]            # plain row gather, no sqrt(D) scaling for these families
      for layer in layers:
          a   = rmsnorm(h, layer.attn_norm_w)
          h   = h + attention(a, positions p0..p0+S-1)       # residual add in float32
          b   = rmsnorm(h, layer.ffn_norm_w)                # "post_attention_layernorm" in HF naming
          h   = h + moe_or_dense_ffn(b)
      last = rmsnorm(h[S-1], final_norm_w)
      logits = lm_head @ last                               # separate matrix (OLMoE, gpt-oss, GLM); some small models tie it to the embedding
- Notes:
  - Pre-norm residual everywhere in the referenced families; the residual stream itself is never normalised in place.
  - When only the next token is needed, apply final norm + lm_head to the last row only (lm_head is often the single biggest dense matmul: V*D).
  - For prompt scoring / teacher forcing / speculative verify, run the head on every row; row s predicts token s+1.
  - Keep the residual stream, norms, router and lm_head input in float32 even when weights are int8/int4/fp8. colibri keeps norms, router weights and router bias in f32 explicitly ("small and sensitive").
  - monolite stores the embedding as bf16 and widens to f32 on gather.
- Variants seen elsewhere: DeepSeek V4 (colibri c/deepseek_v4.c, pulsar docs/deepseek4-port-notes.md) replaces the single residual with hc_mult parallel streams (see F-fwd-19); the embedding is broadcast into every stream and a learned merge collapses streams before the final norm. Qwen3.6 alternates linear-attention and full-attention layers inside the same skeleton (F-fwd-12).

## F-fwd-02: RMSNorm and its family of variants
Origin: [colibri] [ds4] [pulsar] [monolite] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/olmoe.c, c/qwen36.c, c/colibri.c (GLM), c/hyper_connections.h -- read only, not copied
- Purpose: exact normalisation semantics per family.
- Inputs / outputs: x[n] float32, weight w[n] -> y[n].
- Pseudocode:
      sumsq = sum_i (double)x[i]^2                          # accumulate in double (or pairwise f32)
      r     = 1 / sqrt(float(sumsq / n) + eps)              # eps INSIDE the sqrt, added to the MEAN
      plain:          y[i] = x[i] * r * w[i]                # OLMoE, Qwen3-MoE, GLM, gpt-oss, DeepSeek
      offset-weight:  y[i] = x[i] * r * (1 + w[i])          # Qwen3.5/3.6 ("zero-centred" weights)
      gated (linear-attn output): y[i] = x[i] * r * w[i] * silu(z[i])
      weightless:     y[i] = x[i] * r                        # mHC stream pre-norm (V4)
- Notes:
  - Typical eps: 1e-5 (OLMoE default, gpt-oss), 1e-6 (Qwen, DeepSeek); always read `rms_norm_eps` from config, fall back to 1e-5.
  - The (1+w) form is the most common silent bug when porting Qwen3.5/3.6: using plain w gives plausible but wrong text. Detect it per checkpoint, not per family name.
  - The gated variant multiplies by silu(z) (z in the numerator), not sigmoid(z); colibri calls this out as a trap.
  - The DSA indexer key path uses a true LayerNorm (mean and variance, weight + bias, eps 1e-6), not RMSNorm.
  - Double accumulation for the sum of squares is cheap and removes one source of backend drift; colibri does it in every engine.
- Variants seen elsewhere: monolite uses a straight f32 RMSNorm with eps 1e-5. DeepSeek V4 in colibri additionally rounds intermediate tensors to bf16 at the same points as the reference kernels (F-fwd-20).

## F-fwd-03: Rotary embedding layouts
Origin: [colibri] [ds4] [pulsar] [monolite] [ktransformers] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite, ktransformers), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/olmoe.c (half-split RoPE), c/qwen36.c (partial rotary), c/colibri.c (interleaved RoPE for MLA), c/deepseek_v4.c (interleaved + inverse); monolite (Apache-2.0), src/model.c -- read only, not copied
- Purpose: rotate q/k by absolute position; three layout conventions must be supported.
- Inputs / outputs: vector v[rot_dim] (a slice of a head), position p, theta -> v rotated in place.
- Pseudocode:
      inv_freq[j] = theta ^ (-2j / rot_dim)      for j in 0..rot_dim/2-1   # compute once per (theta, rot_dim)
      ang = p * inv_freq[j]
      half-split ("NeoX", HF rotate_half):  pair (v[j], v[j + rot_dim/2])
      interleaved (GPT-J / DeepSeek MLA):   pair (v[2j], v[2j+1])
      rotate pair (a, b): a' = a*cos - b*sin ; b' = b*cos + a*sin
      partial rotary: apply only to the first rot_dim of hd (Qwen3.6: rot_dim = 64 of hd = 256);
                      MLA / V4: the rope part is a separate trailing slice of the head
      inverse rotation: same with sin negated
- Notes:
  - colibri's MLA path reads interleaved pairs but writes the rotated result in half-split order (first halves then second halves). Because q and k get the identical permutation the dot product is unchanged; only matters if you compare intermediate tensors with HF.
  - Cache inv_freq per model; recompute cos/sin per position (or table them). colibri measured that recomputing pow() per position was a real prefill cost; caching inv_freq is bit-identical because the same float product is formed.
  - Qwen3.6 multimodal RoPE (mRoPE sections) collapses to plain half-split RoPE for text-only input because all three position ids are equal (pulsar docs/qwen36-port-notes.md).
  - Compute angles in float; for very long contexts compute p*inv_freq in double before cos/sin (monolite does the whole table in double).
- Variants seen elsewhere: DeepSeek V4 applies the INVERSE rotation to the rope slice of the attention output per head before the output projection (colibri c/deepseek_v4.c), because K = V there (F-fwd-20).

## F-fwd-04: YaRN frequency scaling (and monolite's ramp bug)
Origin: [colibri] [ds4] [pulsar] [monolite] [ktransformers]

Licence-of-origin: Apache-2.0 (colibri, monolite, ktransformers), MIT (ds4, pulsar)
- Source: monolite (Apache-2.0), src/model.c, src/params.h, README.md; colibri (Apache-2.0), c/deepseek_v4.c (yarn bounds / frequency) -- read only, not copied
- Purpose: stretch RoPE to factor x the original context.
- Inputs / outputs: theta, rot_dim, factor, original_ctx, beta_fast (32), beta_slow (1), truncate flag -> inv_freq[rot_dim/2], attention scaling m.
- Pseudocode:
      corr_dim(beta) = rot_dim * ln(original_ctx / (beta * 2*pi)) / (2 * ln(theta))
      lo = corr_dim(beta_fast) ; hi = corr_dim(beta_slow)
      if truncate: lo = floor(lo) ; hi = ceil(hi)            # DeepSeek V4 style; gpt-oss has truncate = false
      lo = max(lo, 0) ; hi = min(hi, rot_dim - 1) ; if hi == lo: hi += 0.001
      for j in pairs:
          base_f  = theta ^ (-2j / rot_dim)
          ramp    = clamp((j - lo) / (hi - lo), 0, 1)        # ramp over the PAIR INDEX j
          keep    = 1 - ramp                                  # 1 = keep original (extrapolate), 0 = interpolate
          inv_freq[j] = (base_f / factor) * (1 - keep) + base_f * keep
      m = 0.1 * ln(factor) + 1                                # "mscale"
      gpt-oss: multiply cos and sin by m (equivalently scale q.k by m^2)
- Notes:
  - The documented monolite bug: the interpolation ramp was first computed over the wavelength instead of over the dimension (pair) index bounded by the correction dims. It produced coherent text with no crash and was the first divergence from transformers; it was found only by the token-level oracle. Lesson: the ramp variable is j, and lo/hi are dimension indices, not wavelengths.
  - monolite verifies its inv_freq table against HF to 1e-7 and builds it in double once.
  - Where m is applied differs by family: gpt-oss scales cos/sin; DeepSeek V3-style MLA conventionally folds m^2 into the softmax scale. colibri's V4 code applies YaRN only on compressed-attention layers (with their own rope theta) and plain RoPE on the others; it does not scale cos/sin. Treat "where does mscale go" as a per-model config switch and test it with the oracle.
- Variants seen elsewhere: OLMoE and Qwen3-MoE at native context use no scaling (theta 10000 / 1e6 respectively).

## F-fwd-05: GQA causal attention core (decode and prefill)
Origin: [colibri] [ds4] [pulsar] [monolite] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/olmoe.c (attention), c/qwen36.c (attention) -- read only, not copied
- Purpose: reference attention that every faster kernel must reproduce.
- Inputs / outputs: q[S,H,hd], new k/v rows [S,Hkv,hd]; KV cache per layer; -> ctx[S,H*hd].
- Pseudocode:
      write k, v for positions p0..p0+S-1 into cache[layer][kvh][pos][:]
      scale = 1 / sqrt(hd)
      for each head h, row s (parallel over both):
          kvh = h / (H / Hkv)                                 # integer division: consecutive q heads share a kv head
          qpos = p0 + s
          for t in 0..qpos: score[t] = dot(q[s,h], K[kvh][t]) * scale       # causal: t <= qpos
          softmax over score[0..qpos] (subtract max, exp, sum in float, divide)
          ctx[s,h] = sum_t score[t] * V[kvh][t]
      out = W_o @ ctx
- Notes:
  - Multiply by the scale after the dot, before the max-subtract. Use a running-max/online softmax in optimised kernels but keep this scalar version as the oracle.
  - Score scratch must be sized to the context capacity, per thread (an early colibri version used a fixed stack array of 4096).
  - Prefill and decode share the same code path; prefill is just S > 1 with the causal bound per row.
  - OLMoE has Hkv = H (plain MHA); Qwen3-MoE and gpt-oss are GQA (gpt-oss 64 q heads over 8 kv heads).
- Variants seen elsewhere: monolite loops one token at a time even for the prompt; colibri batches all S rows through projections (bit-identical) but warns that letting batch size change the matmul kernel family is NOT bit-identical (see F-fwd-24).

## F-fwd-06: QK-norm placement (OLMoE vs Qwen)
Origin: [colibri] [ds4] [pulsar] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/olmoe.c, c/qwen36.c -- read only, not copied
- Purpose: normalise q and k before RoPE.
- Inputs / outputs: q, k after projection -> normalised q, k.
- Pseudocode:
      OLMoE:   q_row = rmsnorm(q_row over ALL H*hd values, q_norm_w[D]) ; same for k with k_norm_w
      Qwen3/3.6: per head: q_h = rmsnorm(q_h over hd, q_norm_w[hd]) ; k_h likewise with k_norm_w[hd]
      then RoPE, then write k to the cache (cache holds normalised + rotated keys)
- Notes:
  - Order is projection -> norm -> RoPE in both. Never normalise after RoPE.
  - The OLMoE variant normalises across the full projected width (one weight vector of length H*hd), which is easy to get wrong when coming from Qwen code.
  - Same eps as the layer norms.
- Variants seen elsewhere: GLM MLA has no qk-norm but normalises the q latent (q_a) and kv latent (kv_a) with RMSNorm instead (F-fwd-09).

## F-fwd-07: Output-gated attention (Qwen3.5/3.6)
Origin: [colibri] [ds4] [pulsar] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/qwen36.c (attention); pulsar (MIT), docs/qwen36-port-notes.md -- read only, not copied
- Purpose: an elementwise sigmoid gate on the attention context.
- Inputs / outputs: q projection of width H*2*hd -> query and gate halves.
- Pseudocode:
      qfull = W_q @ x                         # per head: [query (hd) | gate (hd)]
      query_h = qfull[h][0:hd] ; gate_h = qfull[h][hd:2hd]
      ... per-head qk-norm, partial RoPE, GQA attention as F-fwd-05 ...
      ctx_h = ctx_h * sigmoid(gate_h)          # elementwise, before W_o
      out = W_o @ concat(ctx_h)
- Notes: HF always splits the q projection this way whenever the per-head q width exceeds the kv head dim, regardless of the config flag; colibri keys on the shape rather than the flag.

## F-fwd-08: Attention sinks and sliding windows
Origin: [colibri] [ds4] [pulsar] [monolite]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar)
- Source: monolite (Apache-2.0), src/model.c, src/params.h; colibri (Apache-2.0), c/deepseek_v4.c (sparse attention reference) -- read only, not copied
- Purpose: learned per-head "attend to nothing" logit; local window on alternate layers.
- Inputs / outputs: per-head sink logit s_h (f32), window W.
- Pseudocode:
      start = (layer is sliding and qpos >= W) ? qpos - W + 1 : 0
      logits = [s_h] ++ [dot(q,k_t)*scale for t in start..qpos]      # sink logit is NOT scaled
      p = softmax(logits)
      ctx = sum over real keys only of p[t] * v_t                     # sink mass is discarded
- Notes:
  - gpt-oss: window 128 on even layers, full causal on odd layers; sinks on every layer.
  - Equivalent single-key form used by V4 when only one key exists: weight = 1/(1 + exp(s_h - score)).
  - A sliding-window layer only needs W cache rows (ring buffer); keep full rows if you want simple prefix reuse.

## F-fwd-09: MLA projections and the 576-wide compressed KV row
Origin: [colibri] [ds4] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), MIT (ds4, pulsar)
- Source: colibri (Apache-2.0), c/colibri.c (attention_rows, header comment, config), c/kv_persist.h, README.md -- read only, not copied
- Purpose: Multi-head Latent Attention as used by GLM-5.x (and DeepSeek V3 lineage).
- Inputs / outputs: x[D] -> per-position cache row = latent[kv_lora] ++ k_rope[qk_rope]; queries [H, qk_nope + qk_rope].
- Pseudocode:
      q_lat  = rmsnorm(W_qa @ x, q_a_norm)                 # q_lora wide (GLM-5.2: 2048); also feeds the DSA indexer
      q      = W_qb @ q_lat  -> per head [q_nope (qk_nope) | q_rope (qk_rope)]
      rope(q_rope_h, pos) for every head                    # interleaved convention
      comp   = W_kva @ x -> [latent (kv_lora=512) | k_rope (qk_rope=64)]
      cache.latent[pos] = rmsnorm(latent, kv_a_norm)
      cache.krope[pos]  = rope(k_rope, pos)                 # ONE rope key shared by all heads
      expand path (prefill, large S): kv = W_kvb @ cache.latent[t] -> per head [k_nope (qk_nope) | v (v_head)]
      score_h(t) = (dot(q_nope_h, k_nope_h(t)) + dot(q_rope_h, cache.krope[t])) * scale
      scale = 1/sqrt(qk_nope + qk_rope)                     # (+ YaRN mscale^2 if the config scales rope)
- Notes:
  - The cache row is 512 + 64 = 576 floats per token per layer, versus H*(k+v) for plain MHA; colibri quotes 576 vs 32,768 floats (57x) on GLM-5.2 (H=64, qk_nope=192, v_head=256).
  - Store the latent AFTER its RMSNorm and the rope key AFTER rotation, so cached rows are directly usable by the absorbed path.
  - Quantised cache options seen: fp8-e4m3 row + per-row (or per-group) f32 scale; polar/angle codec with per-row radius. Keep an f32 path as the oracle.
- Variants seen elsewhere: DeepSeek V4 goes further: a single 512-wide row serves as both K and V for all heads (F-fwd-20).

## F-fwd-10: MLA weight absorption for decode
Origin: [colibri] [ds4] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), MIT (ds4, pulsar)
- Source: colibri (Apache-2.0), c/colibri.c (attention_rows CPU absorb core and its explanatory block) -- read only, not copied
- Purpose: avoid re-expanding every cached latent through W_kvb at each decode step.
- Inputs / outputs: q_nope_h, q_rope_h, cached latent rows L_t and rope keys R_t -> ctx_h[v_head].
- Pseudocode:
      split W_kvb rows per head into W_uk_h [qk_nope x kv_lora] and W_uv_h [v_head x kv_lora]
      q_abs_h = W_uk_h^T @ q_nope_h                          # kv_lora wide
      for t in visible keys: s[t] = (dot(q_abs_h, L_t) + dot(q_rope_h, R_t)) * scale
      a = softmax(s)
      c_lat_h = sum_t a[t] * L_t                             # kv_lora wide
      ctx_h   = W_uv_h @ c_lat_h                             # v_head wide
      out     = W_o @ concat_h(ctx_h)
- Notes:
  - Cost per step O(T * kv_lora) per head instead of O(T * H * (qk_nope + v_head)) for the expand path. colibri uses absorption for small S (decode, MTP verify; S <= 4 by default) and the expand path for large prefill batches.
  - Both paths are mathematically identical but not bit-identical; pick one deterministically by S and keep the choice stable between draft and verify (F-fwd-24).
  - colibri offers an "exact" variant for verify rows where the q_abs.L dot and the value mix accumulate integer products and round once, making near-ties resolve identically across hosts (about 0.6x speed on a tiny oracle model). Not available with quantised KV.

## F-fwd-11: DSA lightning indexer (brief)
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (DSA block in attention_rows, README validation note); pulsar (MIT), docs/deepseek4-port-notes.md -- read only, not copied
- Purpose: choose top-k past positions per query so attention cost stays bounded.
- Inputs / outputs: x, normalised q latent, per-layer index-key cache Ik[t][index_hd] -> selected positions (<= index_topk).
- Pseudocode:
      on "full" indexer layers, for each new position: k_idx = rope(layernorm(W_ik @ x, eps 1e-6), pos)   # rope on the leading qk_rope dims
      if visible_keys <= index_topk: no selection (dense attention)
      q_i[h] = rope(W_iq @ q_lat)  for nh indexer heads ; w = W_iw @ x   (one weight per indexer head)
      for each visible t: score[t] = (1/sqrt(nh)) * sum_h w[h] * relu( dot(q_i[h], Ik[t]) / sqrt(index_hd) )
      threshold = value of the k-th largest score (partial select, O(n))
      selected = positions with score > threshold in position order, then fill ties (== threshold) in position order
      "shared" indexer layers reuse the selection of the most recent full layer
- Notes:
  - Validation trick from colibri: force selection on with k >= T; the output must equal dense attention exactly. Selecting everything is the identity.
  - V4's indexer additionally applies Hadamard-128 + fp4 quant-dequant to indexer q and k (QAT parity) per pulsar's notes; selection parity needs that emulation.

## F-fwd-12: Gated DeltaNet recurrent decode (Qwen3.6 linear layers, brief)
Origin: [colibri] [ds4] [pulsar] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/qwen36.c (deltanet); pulsar (MIT), docs/qwen36-port-notes.md -- read only, not copied
- Purpose: O(1)-per-token linear attention layer with a matrix state.
- Inputs / outputs: x[D]; state S_h[kdim x vdim] per value head; conv ring [conv_dim x (k-1)] -> out[D].
- Pseudocode:
      qkv = W_qkv @ x ; z = W_z @ x ; b = W_b @ x ; a = W_a @ x
      beta_h = sigmoid(b_h) ; g_h = -exp(A_log_h) * softplus(a_h + dt_bias_h)     # g < 0, softplus stable: z>20 -> z
      c = silu(depthwise causal conv1d(qkv, ring, kernel k))   ; ring shifts in this token's qkv
      split c -> q (Hk*kdim), k (Hk*kdim), v (Hv*vdim)
      value head h uses key head h / (Hv/Hk)                   # repeat-interleave, NOT h mod Hk
      q_h = q_h / sqrt(|q_h|^2 + 1e-6) / sqrt(kdim) ; k_h = k_h / sqrt(|k_h|^2 + 1e-6)   # L2 norm, eps inside sqrt
      per head: S_h *= exp(g_h)
                m = S_h^T k_h ; d = (v_h - m) * beta_h ; S_h += k_h d^T
                o_h = S_h^T q_h
      o_h = rmsnorm(o_h, w_norm) * silu(z_h)     ; out = W_out @ concat(o_h)
- Notes:
  - The "memory" read m must use the decayed state, before the rank-1 write.
  - L2 norm, not RMS norm (they differ by sqrt(kdim) and L2 has no weight).
  - Prefix reuse with recurrent layers requires snapshotting state + conv ring; it cannot be rewound by position like KV.
  - Prefill can stay sequential (colibri, pulsar v1); chunked prefill is a later optimisation.

## F-fwd-13: KDA (per-channel decay delta rule) variant
Origin: [colibri] [ds4] [pulsar] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/delta_attention.h -- read only, not copied
- Purpose: same delta rule with a per-key-channel decay vector instead of a per-head scalar (Kimi, GLM-5.3-Flash).
- Pseudocode:
      conv + silu on q,k,v channels first ; L2-normalise q (scaled by kdim^-0.5) and k, eps inside sqrt
      for each key channel i: row_i(S) *= exp(gate[i])           # per-channel alpha
      m = S^T k ; S += k ((v - m) * beta)^T ; out = S^T q
- Notes: gate and beta arrive pre-resolved; the model file computes them. Per-thread scratch for m is required when heads run in parallel (sharing one buffer produced plausible-but-wrong tokens that only the teacher-forcing oracle caught).

## F-fwd-14: Router A -- softmax then top-k (OLMoE, Qwen3-MoE)
Origin: [colibri] [ds4] [pulsar] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/olmoe.c (moe, load_cfg), c/qwen36.c (moe); pulsar (MIT), docs/qwen36-port-notes.md -- read only, not copied
- Purpose: pick K experts and their mixing weights.
- Inputs / outputs: x_norm[D], W_router[E,D] -> idx[K], w[K].
- Pseudocode:
      logits = W_router @ x_norm                  (float32)
      p = softmax(logits over all E)
      idx = indices of the K largest p (ties -> lowest index; skip NaN; if nothing finite, fall back deterministically)
      w[k] = p[idx[k]]
      if renormalise: w /= sum(w)
- Notes:
  - OLMoE: `norm_topk_prob` defaults false -> raw softmax probabilities (weights sum < 1). Qwen3-MoE: true. Qwen3.5/3.6 in HF renormalises unconditionally. Read the flag; default per family.
  - Softmax over all E BEFORE selection (differs from gpt-oss, F-fwd-15).
  - Guard the "all scores NaN" case: a selection loop that starts at -1 and never improves will index expert -1 (colibri hardened this after corrupt-tile incidents).
- Variants seen elsewhere: Qwen3.6 also supports optional group-limited selection with group score = top-2 sum (same as F-fwd-16) and optional router bias added to logits before softmax.

## F-fwd-15: Router B -- top-k then softmax (gpt-oss)
Origin: [pulsar] [monolite]

Licence-of-origin: Apache-2.0 (monolite), MIT (pulsar)
- Source: monolite (Apache-2.0), src/model.c (expert selection) -- read only, not copied
- Pseudocode:
      logits = W_router @ x_norm + router_bias
      idx = top-K by raw logit ; w = softmax(logits[idx])        # softmax over the K winners only
- Notes: equivalent to "softmax then renormalise" mathematically but evaluated in a different order; reproduce the reference order for bit-closeness.

## F-fwd-16: Router C -- sigmoid scores, bias for selection only, group limit, routed scale (DeepSeek V3 / GLM)
Origin: [colibri] [ds4] [pulsar] [ktransformers] [apus]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (moe router block, config parsing), c/qwen36.c (group-limited selection) -- read only, not copied
- Pseudocode:
      logits = W_router @ x_norm
      score[e]  = sigmoid(logits[e])
      choice[e] = score[e] + correction_bias[e]               # bias ONLY affects which experts win
      if n_group > 1:
          group_score[g] = sum of the top-2 choice values inside group g
          keep the topk_group best groups; mask all others
      idx = top-K of choice among unmasked experts
      w[k] = score[idx[k]]                                    # UNBIASED score
      if norm_topk: w /= (sum(w) + 1e-20)
      w *= routed_scaling_factor                              # applied after renormalisation
- Notes:
  - Three easy mistakes: using the biased value as the weight; scaling before renormalising (renorm would cancel it); forgetting that GLM has n_group = 1 (no group step).
  - Router weights and bias stay f32.
- Variants seen elsewhere: none of the referenced engines apply the routed scale to the shared expert.

## F-fwd-17: Router D -- DeepSeek V4 (sqrt-softplus scores, hash-routed early layers)
Origin: [colibri] [ds4] [pulsar]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar)
- Source: colibri (Apache-2.0), c/deepseek_v4.c (route function, layer plan); pulsar (MIT), docs/deepseek4-port-notes.md -- read only, not copied
- Pseudocode:
      score[e]  = sqrt(softplus(logit[e]))      ; softplus(z) = max(z,0) + log1p(exp(-|z|))
      choice[e] = score[e] + bias[e]
      if layer < num_hash_layers: idx = hash_table[token_id][0..K-1]     # selection by token id, not activations
      else: idx = top-K of choice
      w[k] = score[idx[k]] / sum_k score[idx[k]] * routed_scale          # V4 Flash: K=6, scale 1.5
- Notes: hash layers mean token ids must be plumbed down to the FFN; weights are still activation-derived. pulsar reports a small floor on the normaliser (~6e-5); colibri errors out on a non-positive sum.

## F-fwd-18: Expert FFN, shared expert, combine order
Origin: [colibri] [ds4] [pulsar] [monolite] [ktransformers] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite, ktransformers), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/olmoe.c, c/qwen36.c (shared expert), c/deepseek_v4.c (clamped SwiGLU); monolite (Apache-2.0), src/model.c -- read only, not copied
- Purpose: per-expert MLP and how results are summed.
- Inputs / outputs: x_norm[D], idx[K], w[K] -> ffn_out[D].
- Pseudocode:
      out = 0
      for k in 0..K-1 (in router rank order):
          g = W_gate[e] @ x ; u = W_up[e] @ x
          act = silu(g) * u                                    # standard SwiGLU
          out += w[k] * (W_down[e] @ act)                      # weight applied AFTER down (colibri), fp32 accumulate
      shared (if present):
          s = W_sd @ (silu(W_sg @ x) * (W_su @ x))
          Qwen: s *= sigmoid(dot(w_shared_gate, x))            # per-token scalar gate
          out += s                                             # added after the routed sum, no routed_scale
- Notes:
  - Fix the accumulation order (routed in rank order, then shared) and keep it identical across CPU/GPU/batched paths; colibri's batched "union" path preserves that per-position order so outputs are bit-identical.
  - Variants of the activation:
      clamped (DeepSeek V4): g = min(g, L); u = clamp(u, -L, L); act = g*sigmoid(g)*u
      gpt-oss: gate/up interleaved even/odd in one matrix with biases; g = min(g, 7); u = clamp(u, -7, 7); act = (u + 1) * g * sigmoid(1.702 g)
  - Dropping an expert (zero contribution) is less harmful than renormalising the survivors (colibri measured renorm "catastrophically worse" on OLMoE).
- Variants seen elsewhere: pulsar notes that the V4 reference multiplies the router weight in before the down projection (equivalent in exact math; changes rounding).

## F-fwd-19: DeepSeek V4 delta -- manifold-constrained hyper-connections (mHC)
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/hyper_connections.h, c/deepseek_v4.c; pulsar (MIT), docs/deepseek4-port-notes.md -- read only, not copied
- Purpose: replace the single residual with n = hc_mult streams (V4: 4) mixed by a doubly stochastic matrix.
- Pseudocode:
      streams[n][D] initialised with the token embedding in every stream
      per block (attention and FFN separately, each with its own params):
          flat = weightless_rmsnorm(concat(streams))           # ONE norm over n*D
          mix  = W_hc @ flat                                   # (2+n)*n values
          pre[i]  = sigmoid(mix[i]*a0 + b[i]) + eps
          post[i] = 2 * sigmoid(mix[n+i]*a1 + b[n+i])
          C = row-softmax(mix[2n..]*a2 + b[2n..]) + eps ; then alternate column / row normalisation for hc_sinkhorn_iters (V4: 20) total iterations
          block_in = sum_i pre[i] * streams[i]                  # raw streams, not the normed flat vector
          y = block(block_in)
          streams'[d] = post[d] * y + sum_s C[d][s] * streams[s]
      before the final norm: a learned merge collapses n streams -> 1
- Notes: colibri found that even changing heap scratch to fixed-size stack arrays changed GCC's reduction rounding here; this is a numerically fragile site, keep a scalar reference.

## F-fwd-20: DeepSeek V4 delta -- attention
Origin: [colibri] [ds4] [pulsar]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar)
- Source: colibri (Apache-2.0), c/deepseek_v4.c (attention units, sparse attention reference, rope), docs/deepseek-v4.md; pulsar (MIT), docs/deepseek4-port-notes.md -- read only, not copied
- Purpose: summarise how V4 attention differs from MLA.
- Pseudocode:
      q = W_qb @ rmsnorm(W_qa @ x)     -> H heads x 512
      kv = rmsnorm(W_kv @ x) (512)     # ONE row: both key and value, shared by all heads (MQA-like)
      rope the last 64 dims of q heads and kv (interleaved); YaRN only on compressed layers (F-fwd-04)
      keys visible = sliding window of recent raw rows + compressed rows (one per R positions, R per layer)
                    (+ indexer top-k over compressed rows on layers that have it)
      p = softmax with per-head sink in the denominator ; out_h = sum p * kv_row
      inverse-rope the last 64 dims of out_h
      grouped output: 8 groups of 8 heads -> per-group low-rank (4096 -> 1024) -> concat 8192 -> W_ob -> D
- Notes:
  - Compressor: per position project x to a candidate row and a gate score (+ learned positional add by pos mod R); every R positions pool the buffered rows weighted by softmax of scores, RMSNorm, rope at the compressed position, quantise (fp8 for main, fp4+Hadamard for indexer) and append one compressed row.
  - colibri rounds to bf16 at the same points as the reference kernels (after projections, after norms, attention probabilities before the value GEMM, outputs). For V4, emulating these roundings matters for greedy parity.
  - Scale = 1/sqrt(512).

## F-fwd-21: MTP / NextN head structure
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (MTP draft and absorb functions, README); ds4 (MIT), docs/SPECULATIVE_DECODING.md -- read only, not copied
- Purpose: one extra transformer block that predicts token t+2 from (hidden at t, embedding of token t+1).
- Inputs / outputs: h_t (final-layer hidden, pre final-norm), token x_{t+1} -> draft logits for x_{t+2} and new hidden h'.
- Pseudocode:
      e = rmsnorm(embed(x_{t+1}), w_enorm)
      hh = rmsnorm(rmsnorm(h_t, final_norm), w_hnorm)          # colibri feeds h AFTER the model's final norm (vLLM convention)
      u = W_eh @ concat(e, hh)                                  # 2D -> D ; order is [embedding, hidden]
      u = mtp_block(u, position = t)                            # full attention+MoE layer with its own KV cache slot
      draft_logits = lm_head @ rmsnorm(u, w_mtp_norm)          # lm_head (and embedding) shared with the main model
      chain: next draft uses the drafted token and u as the new hidden (without re-applying final_norm)
- Notes:
  - The MTP layer's KV cache only needs rows from when drafting started; colibri tracks a per-layer "valid from" position for it and never prefills it.
  - colibri: the MTP head must be int8 or better; int4 heads collapsed acceptance to 0-4%.
  - Getting the concat order or the extra final-norm wrong does not crash; it only lowers acceptance. Measure acceptance against a known-good number.

## F-fwd-22: Speculative loop wiring (draft, batched verify, commit)
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (speculative decode loop, step/step_all, MTP absorb); ds4 (MIT), docs/SPECULATIVE_DECODING.md -- read only, not copied
- Pseudocode:
      state: committed tokens all[0..n], kv length n (all[n] already sampled, not yet forwarded), logits for position n
      loop:
          next = pick(logits) ; emit ; all[n] = next
          drafts[0..g-1] = draft_source(next, last_hidden)       # MTP, n-gram, grammar-forced span, corpus
          g = min(g, remaining budget, cache room)
          L = forward_all_rows([next] ++ drafts, start = n)        # S = 1+g rows; returns logits per row and hidden per row
          k = 0
          while k < g and accept(L[k], drafts[k]): emit drafts[k] ; k++
          feed the verified pairs (token at p+1, true hidden at p) for the accepted span through the MTP layer so its KV stays in sync
          last_hidden = hidden of row k ; n += 1 + k ; logits = L[k]
          KV rows beyond n are stale and simply overwritten next round (no explicit rollback for attention KV)
- Notes:
  - Recurrent layers (DeltaNet/KDA) are the exception: they need a pre-verify snapshot and restore on rejection (ds4 mentions restoring a pre-verify recurrent snapshot).
  - colibri pauses a draft source when acceptance over a recent window (24 proposals) falls below a threshold (70% for MTP, 50% for corpus drafts) for 256 tokens and then re-arms, instead of disabling forever. A rejected draft costs a full MoE row (each row activates its own experts); measured break-even near 50% acceptance for long drafts.
  - Chunked prefill is incompatible with MTP absorb across chunk boundaries unless pairs are carried over.

## F-fwd-23: Acceptance rules -- greedy vs sampling
Origin: [colibri] [ds4] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (verify loop), c/sample.h; ds4 (MIT), docs/SPECULATIVE_DECODING.md -- read only, not copied
- Pseudocode:
      greedy (T = 0): accept draft d at row k iff argmax(L[k]) == d
      exact sampling with a deterministic (greedy) drafter:
          p = target distribution after temperature/top-p/min-p at row k
          accept with probability p[d]
          on reject: sample from p with d removed and renormalised (this IS the residual max(p - q, 0) when q is a point mass)
      opportunistic (ds4 default at T>0): accept drafts that match the target's greedy choice; otherwise fall back to ordinary sampling (biased toward determinism, not distribution-preserving)
- Notes:
  - The "ban the rejected token on resample" detail is required for correctness; colibri carries the banned id into the next pick.
  - If the drafter itself samples (non point-mass q), use the general min(1, p/q) rule with the residual; neither reference does this.
  - Accepted rows keep the batched verifier's state; float reduction order differs from S=1 decode, so long greedy continuations need not be byte-identical to non-speculative decode (ds4 states this explicitly).

## F-fwd-24: The SPEC_PIN lesson -- draft and verify must compute the same function
Origin: [colibri] [apus]

Licence-of-origin: Apache-2.0 (colibri), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), README.md, docs/tuning.md, c/colibri.c (speculation gate) -- read only, not copied
- Purpose: avoid batch-size-dependent numerics corrupting greedy speculative output.
- Notes:
  - Failure mode: the matmul dispatcher chose a different kernel family by row count (e.g. integer-dot with quantised activations for S >= 2, exact dequant kernel for S = 1). Verify runs at S = 1+g, plain decode at S = 1, so near-tie argmaxes flipped between the two, acceptance statistics became meaningless and greedy output differed from non-speculative greedy.
  - Fix: while speculation is live, pin every forward (draft, verify, absorb) to the S=1 kernel family; restore normal gates for prefill afterwards. Default on in colibri.
  - Generalisation for our engine: kernel selection must never depend on batch size unless the alternatives are bit-identical; expose a "deterministic family" flag, and keep a scalar path. colibri also notes letting prefill projections use the faster integer kernel cost ~12% perplexity on GLM, so prefill stays on the exact kernel there.
  - For byte-exact reproducibility, disable drafting entirely; acceptance percentages are not comparable across versions under top-p.

## F-fwd-25: KV cache layouts
Origin: [colibri] [ds4] [pulsar] [monolite] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/olmoe.c, c/qwen36.c, c/colibri.c, c/kv_persist.h, c/kv_fp8.h; monolite (Apache-2.0), src/model.c; pulsar (MIT), docs/qwen36-port-notes.md -- read only, not copied
- Layouts observed:
      GQA (colibri): per layer K[kvh][pos][hd], V same; head-major so one head's keys are contiguous over time (good for decode dot loops)
      GQA (monolite): per layer K[pos][kvh*hd]; position-major (good for appends and prefix truncation)
      MLA: per layer latent[pos][kv_lora] + krope[pos][qk_rope] (576 floats), plus index keys [pos][index_hd] on DSA layers
      quantised rows: fp8-e4m3 bytes + f32 scale per row (or per group); decode-time dequant
      linear-attention layers: fixed state [heads][kdim][vdim] + conv ring, independent of length (Qwen3.6: ~2 MB per layer)
      V4: ring of the last W raw rows + growing compressed rows + compressor partial buffers
      MTP layer: its own slot (index n_layers) valid only from the first drafted position
- Notes: whatever the layout, address rows by absolute position so a prefix can be kept by just resetting the length.

## F-fwd-26: Prefix reuse across turns
Origin: [colibri] [ds4] [pulsar] [monolite] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/kv_prefix.h, c/olmoe.c, c/qwen36.c; ds4 (MIT), docs/SERVER.md -- read only, not copied
- Purpose: a chat client resends the whole transcript; only prefill the new suffix.
- Pseudocode:
      record fed_ids[pos] at the exact place tokens enter the cache (inside the forward, not from caller bookkeeping)
      on a new prompt P[0..n):
          reuse = len(fed) if (not tainted and len(fed) < n and P[0..len) == fed[0..len)) else 0
          if reuse == 0: clear all caches and recurrent state
          prefill P[reuse..n) starting at position reuse
      growing the cache by copy keeps the record; reallocating without copy clears it
      any input not described by token ids (audio/image embeddings) marks the state tainted -> never reuse
- Notes:
  - All-or-nothing match and at least one new token are required: rewinding to a shorter prefix is impossible for recurrent state, and zero new tokens leave no hidden state to sample from.
  - colibri measured a DeepSeek V4 second turn reusing 82% of the prompt: 61 s instead of 320 s.
  - For pure-attention models, partial rewind (longest common prefix) is possible by truncating the length; recurrent hybrids need snapshots.
- Variants seen elsewhere: ds4 first tries the live token prefix, then a rendered-text prefix from its disk store.

## F-fwd-27: Disk persistence of KV
Origin: [colibri] [ds4]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4)
- Source: colibri (Apache-2.0), c/kv_persist.h; ds4 (MIT), docs/SERVER.md, ds4_kvstore.h -- read only, not copied
- Pseudocode (colibri-style append log):
      file = magic(format version) + header{n_layers, kv_lora, qk_rope, index_hd, #index layers, vocab, n_records, codec tag}
      record per position = token id + for each layer (latent row, rope row [, scales]) + index rows
      after each turn: append records [saved..len) ; flush+fsync data ; THEN rewrite n_records ; fsync
      truncate = rewrite n_records only (never above what was physically written)
      load: reject on magic/geometry mismatch or if longer than the context ; sanitise quantised rows (NaN codes -> 0, bad scale -> inert row) ; restore ids + rows ; zero re-prefill
- Notes:
  - Writing the count last makes the file crash-safe without a journal.
  - An f32 file loaded under a quantised-KV setting is quantised in RAM and rewritten in the new format on the next save.
  - ds4 instead keys checkpoint files by a SHA of the rendered text prefix (the payload still carries exact tokens and state), with an LRU-ish budget, min-token thresholds, boundary trimming/alignment to avoid fragile token boundaries, and optional same-quant-only reuse. Cache files contain prompt text: private directory.

## F-fwd-28: Sampling (temperature, top-p, min-p)
Origin: [colibri] [ds4] [pulsar] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/sample.h; monolite (Apache-2.0), src/sampler.c -- read only, not copied
- Pseudocode:
      if T <= 0: argmax over finite logits (NaN skipped; first max wins)
      m = max finite logit
      p[i] = exp((l[i] - m) / T) for finite l[i], else 0 ; sum in double ; p /= sum
      if sum not finite or 0: fall back to argmax and warn once
      min-p (monolite): keep i iff l[i] >= m + T * ln(min_p)          # == p[i] >= min_p * p_max, done in logit space
      top-p (colibri): heapify indices by p (O(V)); pop until cumulative >= top_p; zero the rest; renormalise
      draw u in [0,1) (xorshift64* -> 53-bit double), walk the cumulative sum in id order
      optional banned id (speculative reject): treat its mass as removed, scale u by (1 - p_ban)
- Notes:
  - min_p = 0 must reproduce plain temperature sampling exactly (monolite keeps a separate exact branch for it).
  - Apply penalties/grammar masks to logits before temperature; top-p and min-p after.
  - Keep logprob computation separate: log p = (l - m) - log(sum exp(l - m)) in double.

## F-fwd-29: Correctness oracle -- teacher forcing against HF transformers
Origin: [colibri] [ds4] [pulsar] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/oracle.h, c/olmoe.c (teacher-forced NLL), c/colibri.c (forward over all rows, logit gap dump), README.md; monolite (Apache-2.0), README.md, src/model.c (per-layer dump); ds4 (MIT), docs/TESTING.md -- read only, not copied
- Purpose: decide whether the engine computes the model.
- Pseudocode:
      reference JSON (made offline with transformers in fp32/bf16): prompt_ids, full_ids, tf_pred (HF argmax at every position of full_ids)
      teacher forcing: one forward over full_ids; pred[s] = argmax(logits row s); compare with tf_pred[s]
      pass = mismatches <= allowance (default 0; allowance must leave at least one required match)
      on mismatch: print top-5 logits and top1-top2 gap -> a gap ~1e-4 is a float near-tie, a large gap is a real bug
      free-run check: greedy-generate from prompt_ids and compare with full_ids
      perplexity check: feed reference tokens one at a time through the decode path; NLL = -(l[ref] - m - log sum exp)
      per-layer dumps: residual after embedding and after each layer, compared with HF hidden_states
      all logits must be finite (reject NaN/Inf rows)
- Notes:
  - colibri OLMoE with int8 experts: 12.11 ppl vs HF bf16 12.25 on identical ids. GLM teacher forcing typically 30-32/32 on the tiny oracle; the misses are near-ties that change with toolchain.
  - Build the tiny oracle model (random small config of the same architecture) so the HF reference runs in seconds; use real checkpoints for perplexity only.
  - monolite rule: never delete the scalar CPU path; every fast backend is accepted only if it matches it. Its oracle found a CUDA race, a double-accumulated layer and the YaRN ramp bug, all producing plausible text.
  - Exactness tests worth automating: DSA select-all == dense; chunked prefill == single-shot; batched-union MoE == per-token MoE; KV saved/loaded == uninterrupted; prefix-reuse continuation == fresh prefill.
  - ds4: compare against continuations from the same checkpoint only; use token dumps to catch chat-template differences before blaming the graph.

## F-fwd-30: Chunked prefill equivalence
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/colibri.c (step); pulsar (MIT), docs/deepseek4-port-notes.md -- read only, not copied
- Pseudocode:
      while remaining > chunk: forward(ids[done..done+chunk), start = p0 + done) writing KV only ; done += chunk
      final slice forward produces the logits
- Notes: because KV rows are position-addressed, a chunk is just an incremental suffix prefill; colibri verified a 1,683-token prompt at chunk 256 gives byte-identical greedy output. Exceptions: MTP absorb pairs straddling a chunk boundary, V4 compressor state (pulsar keeps compressor steps per token inside a chunk), and recurrent layers (sequential within the chunk).

---

## Recommended synthesis

- Shared library `nn_core`: RMSNorm (plain, 1+w, gated-silu, weightless; double sum-of-squares; eps from config), LayerNorm, stable softmax, silu/sigmoid/softplus, L2-norm with eps inside the sqrt. One implementation, unit-tested against HF tensors.
- Shared `rope`: inverse-frequency table builder with YaRN (truncate flag, pair-index ramp, mscale returned separately), and three apply modes (half-split, interleaved, interleaved-in/half-out) plus partial rotary and inverse. Where mscale is applied (cos/sin vs softmax scale) is a per-model parameter, never hard-coded.
- Shared `attn_gqa`: scalar causal GQA core with optional sinks, sliding window, output gate hook; it is the oracle for every SIMD/GPU kernel.
- Shared `attn_mla`: projections into the 576-wide latent row, expand path and absorbed path behind one interface; choice made by a deterministic policy that the speculative loop can pin.
- Shared `router`: one function with explicit options {score: softmax | sigmoid | sqrt-softplus | topk-then-softmax; bias-for-selection; n_group/topk_group; renormalise + eps; routed_scale; forced/hash indices}. Per-model files only fill the option struct from config.
- Shared `expert_ffn`: SwiGLU with activation variants (plain, clamped, gpt-oss alpha form, interleaved gate/up) and a fixed accumulation order (routed in rank order, then shared, shared gate optional).
- Shared `linear_attn`: delta-rule step parameterised by decay shape (scalar per head vs per channel), conv ring, gated output norm; per-model code computes gate/beta.
- Shared `kv`: position-addressed cache with pluggable row codecs (f32, fp8+scale), a fed-token record for prefix reuse, taint flag, snapshot/restore for recurrent state, and the append-only crash-safe disk format (count written last, geometry header, sanitise on load).
- Shared `sampler`: greedy, temperature, top-p (heap), min-p in logit space, banned-id resample, logprob helper, seeded RNG; min_p=0 path exactly equals plain temperature.
- Shared `spec`: speculative loop (draft sources: MTP, n-gram, grammar), batched verify, greedy/exact/opportunistic acceptance, adaptive pause guard, kernel-family pin while drafts are live.
- Shared `oracle` test harness: reference JSON reader, teacher forcing with allowance, top-5/gap dump on mismatch, finite-logit check, PPL meter, layer-dump comparer, and the exactness identities listed in F-fwd-29.
- Per-model file (olmoe, qwen3moe, qwen36, glm, deepseek_v4, gpt-oss): config parsing with family defaults (norm_topk, eps, rope type), weight name mapping, layer-type schedule (dense vs MoE, linear vs full attention, sliding vs full, hash layers, indexer full vs shared), and wiring of the shared pieces.
- Per-model only: mHC stream plumbing and Sinkhorn (DeepSeek V4, GLM-5.3) can live in a small shared header but the wiring (init, per-block pre/post, final merge) stays in the model file; same for the V4 compressor and grouped output projection, and the bf16 rounding points emulation.
- Per-model only: MTP/NextN head construction (norm placement, concat order, weight precision >= int8), since conventions differ by checkpoint; the shared spec loop calls it through a draft/absorb interface.
- Rule for all layers: keep the scalar f32 reference path in the tree permanently, make no kernel choice depend on batch size unless proven bit-identical, and gate every optimisation on the teacher-forcing oracle.
