# Quantisation formats and CPU compute kernels — fragments

Scope: block layouts and dequant math (Q8_0, Q4_K, Q5_K, Q6_K, Q8_K, IQ2_XXS,
FP8 E4M3 with 128x128 scales, MXFP4/E8M0, colibri int4 gs64 planar), integer and
float dot kernels (scalar / AVX2 / AVX-512 VNNI+BF16 / NEON), fused expert FFN,
weight repacking for tiles, and CPU dispatch.

Conventions in this file:
- All multi-byte fields are little-endian. "fp16" = IEEE binary16, "bf16" = top
  16 bits of an IEEE binary32.
- "Super-block" = 256 weights (the K-quant unit). "Sub-block" = 16 or 32 weights
  sharing one small scale.
- Byte offsets are given as `[start..end]` inclusive, relative to the block.
- GGML/GGUF block formats are public de-facto standards; layouts below are
  described from the reading of ds4 (MIT, which carries GGML-derived code under
  MIT), pulsar (MIT) and ktransformers kt-kernel (Apache-2.0). Lookup tables
  (the IQ2 grid etc.) are NOT reproduced; obtain them from the GGUF/ggml
  specification (MIT) when implementing.

---

## F-quant-01: Q8_0 block (dense weights) and per-block activation quantisation
Origin: [colibri] [ds4] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), MIT (ds4, pulsar)
- Source: ds4 (MIT), `ds4.c` (Q8_0 row dot, activation quantiser, block size
  table); pulsar (MIT), `crates/quant/src/cpu_dot.rs` (q8_0 dequant) -- read only, not copied
- Purpose: 8-bit symmetric weights with one fp16 scale per 32 elements; the
  workhorse for dense projections.
- Inputs / outputs: weight row of K elements (K % 32 == 0) = K/32 blocks of 34
  bytes; activation f32[K] -> int8[K] + f32[K/32] scales.
- Layout (34 bytes per 32 weights, 8.5 bits/weight):
      [0..1]   fp16 d            block scale
      [2..33]  int8 q[32]        element i of the block = q[i]
      dequant: w[i] = fp16_to_f32(d) * q[i]
- Pseudocode (activation quantisation, one scale per 32):
      for each block b of 32 activations:
          amax = max |x[j]| over block
          sx[b] = amax / 127
          inv = (sx[b] != 0) ? 1/sx[b] : 0
          xq[j] = clamp(round_to_nearest_even(x[j] * inv), -127, 127)
          (pad tail lanes of a partial last block with 0)
- Pseudocode (scalar dot of one weight row with quantised activation):
      acc = 0.0f
      for each block b:
          isum = sum_{i<32} int32(q_b[i]) * int32(xq_b[i])      # exact int32
          acc += fp16_to_f32(d_b) * sx[b] * float(isum)
      return acc
- Notes:
  * Clamp activations to [-127, 127], not -128: the x86 "sign trick" (F-quant-11)
    negates activations, and -128 does not negate in int8.
  * ds4 keeps two float accumulators and processes two blocks per iteration for
    ILP; the per-block float scale (d * sx) is folded once per block, integer
    sum inside the block is exact.
  * Pre-quantise the activation ONCE per token per input vector and reuse it for
    every matrix that consumes the same input (Q/K/V, gate/up, all experts of a
    layer). Every reference project ended up hoisting this (colibri comments a
    measured ~7 us per 4096-wide scalar quantisation x ~150 GEMVs/token).
- Variants seen elsewhere: colibri `fused_simd.h` uses 16-element activation
  blocks for an int8-per-row weight path; colibri `idot.h` uses ONE scale for the
  whole activation row (amax/127) plus per-64 integer sums for its int4 path.

## F-quant-02: K-quant super-block and the 12-byte 6-bit scale/min packing
Origin: [ds4] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (ktransformers), MIT (ds4, pulsar)
- Source: ds4 (MIT), `ds4.c` (Q4_K/Q5_K structs, scale/min unpack helper,
  static size asserts); pulsar (MIT), `cpu_dot.rs` (scale-min mirror);
  ktransformers kt-kernel (Apache-2.0), `operators/amx/test/mmq.cpp` (word-wise
  unpack) -- read only, not copied
- Purpose: Q4_K and Q5_K share a super-block of 256 weights split into 8
  sub-blocks of 32, each with a 6-bit scale and a 6-bit min, packed into 12 bytes.
- Inputs / outputs: 12 bytes S[0..11] -> sc[0..7], mn[0..7] (each 0..63).
- Pseudocode:
      for j in 0..3:
          sc[j] = S[j]   & 0x3F
          mn[j] = S[j+4] & 0x3F
      for j in 4..7:
          sc[j] = (S[j+4] & 0x0F) | ((S[j-4] >> 6) << 4)
          mn[j] = (S[j+4] >> 4)   | ((S[j]   >> 6) << 4)
  In words: bytes 0..3 hold scales 0..3 in their low 6 bits and the top 2 bits of
  scales 4..7 in bits 6..7; bytes 4..7 do the same for mins 0..3 / mins 4..7;
  bytes 8..11 hold the low 4 bits of scale 4+i (low nibble) and min 4+i (high
  nibble).
- Notes:
  * A vector-friendly equivalent treats S as three little-endian u32 words and
    builds four u32 words whose 16 bytes are sc[0..7] followed by mn[0..7] with
    masks 0x3F3F3F3F / 0x0F0F0F0F / 0x03030303 (ktransformers does this when
    repacking). Do it once at load/repack time, not per dot.
  * Dequant for both formats: w = d * sc[s] * q - dmin * mn[s], where s is the
    sub-block (32 elements) index; d and dmin are fp16 super-scales.

## F-quant-03: Q4_K block (144 bytes / 256 weights, 4.5 bpw)
Origin: [ds4] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (ktransformers), MIT (ds4, pulsar)
- Source: ds4 (MIT), `ds4.c`; pulsar (MIT), `cpu_dot.rs` -- read only, not copied
- Purpose: asymmetric 4-bit with per-32 scale and min.
- Layout:
      [0..1]     fp16 d       (super-scale for scales)
      [2..3]     fp16 dmin    (super-scale for mins)
      [4..15]    12 bytes     packed 6-bit scales/mins (F-quant-02)
      [16..143]  128 bytes    4-bit quants
- Element mapping (chunk c = 0..3 covers elements 64c..64c+63, uses quant bytes
  Q[32c .. 32c+31]):
      element 64c + l       (l < 32) = Q[32c + l] & 0x0F      -> sub-block 2c
      element 64c + 32 + l  (l < 32) = Q[32c + l] >> 4        -> sub-block 2c+1
      value = d*sc[sub]*q - dmin*mn[sub],  q in 0..15
- Notes: the low/high nibble split is "planar per 64" (same idea as colibri's
  planar int4, F-quant-16): one AND gives 32 consecutive elements in order, one
  shift+AND gives the next 32. No byte interleave is ever needed.

## F-quant-04: Q5_K block (176 bytes / 256 weights, 5.5 bpw) and 5th-bit unpack
Origin: [ds4] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (ktransformers), MIT (ds4, pulsar)
- Source: ds4 (MIT), `ds4.c` (Q5_K struct, scalar Q5_K x Q8_K dot);
  ktransformers kt-kernel (Apache-2.0), `operators/amx/test/mmq.cpp` (Q5_K
  repack layout) -- read only, not copied
- Layout:
      [0..1]     fp16 d
      [2..3]     fp16 dmin
      [4..15]    12 bytes packed 6-bit scales/mins (F-quant-02)
      [16..47]   32 bytes  high-bit plane H[0..31]
      [48..175]  128 bytes low-nibble plane Q[0..127]
- Element mapping (chunk c = 0..3, l = 0..31):
      element 64c + l      : q = (Q[32c+l] & 0x0F) + 16 * bit(H[l], 2c)
      element 64c + 32 + l : q = (Q[32c+l] >> 4)   + 16 * bit(H[l], 2c+1)
      value = d*sc[sub]*q - dmin*mn[sub],  q in 0..31, sub = 2c or 2c+1
  i.e. H[l] carries, in bit k, the 5th bit of element 32k + l (k = sub-block
  index 0..7). The same 32 H bytes are re-read for every chunk with a mask that
  moves left by 2 bits per chunk.
- SIMD unpack (per 64-element chunk): load 32 Q bytes and the 32 H bytes once
  per super-block; lo = Q & 0x0F, hi = (Q >> 4) & 0x0F;
  hb_lo = ((H >> 2c) & 1) << 4, hb_hi = ((H >> (2c+1)) & 1) << 4;
  u_lo = lo | hb_lo, u_hi = hi | hb_hi (unsigned 0..31, ideal for vpdpbusd /
  vpmaddubsw where the weight is the unsigned operand).
  Alternative: compare-against-mask (AND H with a per-chunk bit mask, compare
  equal to the mask, AND with 16) avoids variable shifts.
- Notes: ds4's scalar reference accumulates in int64; int32 is provably enough
  (|q*x| <= 31*127, x32 per sub-block, x63 scale, x8 sub-blocks ~ 6.3e7).

## F-quant-05: Q6_K block (210 bytes / 256 weights, 6.5625 bpw) and 6-bit unpack
Origin: [colibri] [ds4] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), MIT (ds4, pulsar)
- Source: ds4 (MIT), `ds4.c` (Q6_K struct + scalar Q6_K x Q8_K and x f32 dots);
  ktransformers kt-kernel (Apache-2.0), `operators/amx/test/mmq.cpp` (128-wide
  unpack, repack, VNNI kernel) -- read only, not copied
- Purpose: symmetric 6-bit, 16 sub-blocks of 16 with signed 8-bit scales. Typical
  choice for sensitive expert/down tensors.
- Layout (note: super-scale is LAST):
      [0..127]    128 bytes  low-nibble plane L[0..127]
      [128..191]  64 bytes   2-bit plane      Hh[0..63]
      [192..207]  16 x int8  sub-block scales sc[0..15]
      [208..209]  fp16 d
- Element mapping: process two halves h = 0,1 of 128 elements; for half h use
  L' = L + 64h, H' = Hh + 32h. For l = 0..31:
      e0 = 128h + l        q = (L'[l]    & 0x0F) | (((H'[l] >> 0) & 3) << 4)
      e1 = 128h + l + 32   q = (L'[l+32] & 0x0F) | (((H'[l] >> 2) & 3) << 4)
      e2 = 128h + l + 64   q = (L'[l]    >> 4)   | (((H'[l] >> 4) & 3) << 4)
      e3 = 128h + l + 96   q = (L'[l+32] >> 4)   | (((H'[l] >> 6) & 3) << 4)
      value(e) = d * sc[e / 16] * (q - 32)        q in 0..63
  (scale index is simply element_index / 16 across the whole super-block.)
- SIMD unpack per 128-element half (32-byte vectors): A = L'[0..31],
  B = L'[32..63], C = H'[0..31];
      u0 = (A & 0xF) | ((C      & 3) << 4)   -> elements  0..31
      u1 = (B & 0xF) | (((C>>2) & 3) << 4)   -> elements 32..63
      u2 = (A >> 4)  | (((C>>4) & 3) << 4)   -> elements 64..95
      u3 = (B >> 4)  | (((C>>6) & 3) << 4)   -> elements 96..127
  (16-bit shifts are fine: cross-byte bits are removed by the masks.)
- Notes: keep u unsigned (0..63) for the integer dot and fold the -32 out with
  the activation group sums: sum(sc*(u-32)*x) = sum(sc*u*x) - 32*sum_g sc_g*S_g,
  where S_g is the int sum of the 16 activation int8s of group g (F-quant-06).
  This removes a per-element subtract and keeps the unsigned operand of
  vpdpbusd/vpmaddubsw legal.

## F-quant-06: Q8_K activation block (292 bytes / 256) and its quantiser
Origin: [colibri] [ds4] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), MIT (ds4, pulsar)
- Source: ds4 (MIT), `ds4.c` (Q8_K quantiser used before K-quant/IQ2 expert
  dots); pulsar (MIT), `cpu_dot.rs`; ktransformers kt-kernel (Apache-2.0),
  `operators/amx/test/mmq.cpp` (AVX-512 quantiser computing sums with VNNI)
  -- read only, not copied
- Purpose: the activation format paired with every K-quant / IQ weight; carries
  per-16 group sums so mins/offsets fold out exactly.
- Layout:
      [0..3]      f32 d               (activation scale; NOTE f32, not fp16)
      [4..259]    int8 q[256]
      [260..291]  int16 S[16]         S[g] = sum of q[16g .. 16g+15]
- Pseudocode (ggml-compatible convention, as in ds4):
      find the element with the largest |x|; call its signed value m
      if |m| == 0: d = 0, q = 0, S = 0; continue
      iscale = -127 / m
      q[j] = clamp(round_nearest(iscale * x[j]), -128, 127)
      S[g] = sum of q in group g
      d = 1 / iscale            (may be negative: the extreme maps to -127)
- Variant (pulsar, ktransformers AVX-512): d = amax/127, q = round(x/d) clamped
  to [-127, 127] (positive d). Both are valid; the product d_w*d_x*isum is what
  matters. Prefer the positive-d, [-127,127] variant in a new engine (safe for
  the sign trick).
- Notes:
  * Q4_K/Q5_K need sums per 32 = S[2s] + S[2s+1]; Q6_K uses S[g] directly
    (16-element groups); Q2_K/Q3_K use S per 16.
  * AVX-512 trick (ktransformers): after storing the 256 int8 values, transpose
    so each 32-bit lane holds 4 bytes of one group, then vpdpbusd with an
    all-ones unsigned vector produces 16 group sums in one register; narrow to
    int16 with a saturating/plain down-convert.
  * Sums are exact integers, so any SIMD path is bit-identical to scalar.

## F-quant-07: scalar reference for K-quant x Q8_K dot (the numerics contract)
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: ds4 (MIT), `ds4.c` (Q4_K/Q5_K/Q6_K x Q8_K scalar paths); pulsar
  (MIT), `cpu_dot.rs` (scalar + bitwise-equality test) -- read only, not copied
- Inputs / outputs: weight row (nb super-blocks), Q8_K row (nb blocks) -> f32.
- Pseudocode (Q4_K/Q5_K):
      total = 0.0f
      for each super-block i:
          dA = f32(w.d) * x.d ; dM = f32(w.dmin) * x.d
          isum = 0 ; msum = 0                        # int32
          for sub s in 0..7:
              msum += mn[s] * (x.S[2s] + x.S[2s+1])
              isum += sc[s] * sum_{e in sub s} q_w[e] * x.q[e]
          total += dA * isum - dM * msum
- Pseudocode (Q6_K):
      for each super-block i:
          isum = 0
          for group g in 0..15:
              isum += sc[g] * ( sum_{e in g} u[e]*x.q[e]  -  32 * x.S[g] )
          total += f32(w.d) * x.d * isum
- Notes:
  * Everything inside a super-block is exact integer math; only ONE float
    multiply-add per super-block (two for the min term). That is why SIMD
    paths can be made bit-identical to this reference regardless of lane order
    (pulsar asserts bitwise equality in a test).
  * The f32 accumulation across super-blocks must happen in a fixed order (block
    0,1,2,...) in both scalar and SIMD. Do the horizontal reduction of the
    integer vector, not of floats, then one scalar float update per block.
- Variants seen elsewhere: ds4 also has "K-quant x f32" scalar dots (dequant to
  float, multiply by f32 activation) used where activation quantisation is not
  acceptable; slower, no Q8_K needed.

## F-quant-08: AVX2 K-quant dot (vpmaddubsw / vpmaddwd), Q4_K and Q2_K/IQ shapes
Origin: [ds4] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (ktransformers), MIT (ds4, pulsar)
- Source: pulsar (MIT), `crates/quant/src/cpu_dot.rs` (AVX2 module: q4_K, q2_K,
  q3_K, iq2_xxs, iq2_xs, iq3_xxs) -- read only, not copied
- Purpose: integer dot of one weight row with a Q8_K activation on Haswell+/Zen.
- Pseudocode (Q4_K, per super-block):
      accI = int32x8 zero ; msum = 0
      for chunk c in 0..3:
          (scA, mA) = scale/min of sub 2c ; (scB, mB) = of sub 2c+1
          W = load 32 bytes of quants for chunk c
          lo = W & 0x0F ; hi = (W >> 4) & 0x0F               # unsigned 0..15
          XA = load x.q[64c .. 64c+31] ; XB = load x.q[64c+32 .. 64c+63]
          pA = pairwise_add_i16_to_i32( u8xi8_pair_mul_add(lo, XA) )   # maddubs then madd with 1s
          pB = same for (hi, XB)
          accI += pA * broadcast(scA) + pB * broadcast(scB)   # 32-bit mullo
          msum += mA*(S[4c]+S[4c+1]) + mB*(S[4c+2]+S[4c+3])
      isum = horizontal_add_i32(accI)
      total += f32(d)*x.d*isum - f32(dmin)*x.d*msum
- Notes:
  * Saturation budget of vpmaddubsw (int16 result of two u8*i8 products):
    Q4_K 2*15*127 = 3810; Q5_K 2*31*127 = 7874; Q6_K (unsigned 0..63)
    2*63*127 = 16002; IQ2 grid magnitude <= 43: 2*43*127 = 10922; all < 32767.
    int8 x int8 via the sign trick: 2*128*127 = 32512 (just fits).
  * Q2_K: one 32-byte load per 128 elements; shifts of 0/2/4/6 with mask 3 give
    four 32-element vectors; the two 16-element scales of a 32-vector are
    applied by vpmaddwd with a vector holding scale A in the low 128-bit lane and
    scale B in the high lane (saves a mullo).
  * pulsar keeps SIMD bit-identical to scalar by doing integer horizontal adds
    and the same float update order.

## F-quant-09: AVX-512 VNNI K-quant GEMV with 16-row VNNI-transposed repack
Origin: [colibri] [ktransformers]

Licence-of-origin: Apache-2.0 (colibri, ktransformers)
- Source: ktransformers kt-kernel (Apache-2.0),
  `operators/amx/test/mmq.cpp` (packing for Q4_0/Q8_0/Q4_K/Q5_K/Q6_K/IQ4_XS and
  the VNNI micro-kernels; this file is a port of llama.cpp's AMX backend, MIT)
  -- read only, not copied
- Purpose: decode-time GEMV that produces 16 output rows per zmm without any
  horizontal reduction; best fit for Zen 5's full-width AVX-512.
- Repacked tile for one super-block column of 16 rows (Q6_K example):
      quants  : for each pair of 4-element "k-quads" interleaved across the 16 rows:
                each 64-byte vector = 16 rows x 4 consecutive k (one 32-bit lane
                per row), low nibbles = k-quad j, high nibbles = k-quad j+1
      hbits   : the 2-bit planes repacked the same way, four k-quads per byte
      scales  : int8 [16 groups][16 rows]
      d       : fp16 [16 rows]
  (Q4_K/Q5_K: scales [8][16] u8, mins stored as [4][16 rows][2] int8 so they can
  feed a 16-bit pairwise dot; d and dmin fp16 [16].)
- Pseudocode (one super-block, 16 rows, one token):
      xq = 256 int8 activation (4 zmm), xS = 16 group sums, xd = activation scale
      comp = 32 * widen_i16_to_i32(xS)          # Q6_K offset compensation, per group
      accI = int32x16 zero
      for group g in 0..15:                     # 16 elements = 4 k-quads
          vsum = 0
          for each k-quad pair (j, j+1) in g:
              a0 = broadcast 32-bit lane j   of xq (vpermd with constant index)
              a1 = broadcast 32-bit lane j+1 of xq
              B = load 64 packed bytes ; H = matching high-bit vector
              w0 = (B & 0xF) + ((H & mask_j)   shifted into bits 4..5)
              w1 = ((B>>4) & 0xF) + ((H & mask_j1) shifted into bits 4..5)
              vsum = dpbusd(vsum, w0, a0) ; vsum = dpbusd(vsum, w1, a1)
          vsum -= broadcast(comp[g])
          accI += vsum * widen_i8_to_i32(scales[g][0..15])
      out16 = fma(float(accI), f32(d[0..15]) * xd, out16)
- Q4_K mins: broadcast 2 int16 group-sums per lane and use vpdpwssd against the
  [k/2][row][2] int8->int16 mins to get 16 min-terms; subtract with fnmadd using
  dmin*xd.
- Notes:
  * Weight is the unsigned operand of vpdpbusd (u8 x s8 -> s32, 4 products per
    lane, NO int16 intermediate, so no saturation concerns).
  * One activation broadcast feeds 16 rows; weights stream linearly; no
    horizontal adds; output written as a 16-float vector. This is the layout to
    repack Q5_K/Q6_K experts into at load time.
  * Q8_0 variant (both operands signed) uses an activation-side bias: the
    activation bytes are flipped to unsigned by adding 128 (XOR 0x80), and at
    repack time a per-row compensation 128 * sum(weight bytes of the block) is
    precomputed with the same dpbusd (all-0x80 vector against the weights);
    sum((a+128)*w) - comp = sum(a*w), exact.

## F-quant-10: IQ2_XXS (66 bytes / 256, 2.06 bpw) and its AVX2 kernel
Origin: [ds4] [pulsar]

Licence-of-origin: MIT (ds4, pulsar)
- Source: ds4 (MIT), `ds4.c` (block, f32 dot, NEON dot with signed-grid table);
  pulsar (MIT), `crates/quant/src/cpu_dot.rs` (scalar + AVX2, measurement note)
  -- read only, not copied
- Layout:
      [0..1]   fp16 d
      [2..65]  8 groups x 8 bytes; group g at offset 2+8g:
               word A (u32, bytes 0..3): four 8-bit grid indices i0..i3
               word B (u32, bytes 4..7): bits 0..27 = four 7-bit sign fields,
                                         bits 28..31 = 4-bit sub-scale s
- Dequant of group g (32 elements = 4 x 8):
      for k in 0..3:
          mag[0..7] = GRID[byte k of A]          # 256-entry table of 8 magnitudes,
                                                 # each magnitude in {8, 25, 43}
          s7 = (B >> 7k) & 0x7F
          signs8 = s7 | (parity(s7) << 7)        # 8th sign makes #negatives even
          for i in 0..7: v = mag[i] * (bit i of signs8 ? -1 : +1)
          w[32g + 8k + i] = d * (2s + 1) * v / 8
- AVX2 kernel (per 32-group): build a 32-byte vector from four 8-byte grid rows
  (unsigned magnitudes), build a 32-byte +-1 vector from a 128-entry table of
  precomputed sign patterns, apply signs to the ACTIVATION (sign_epi8), then
  maddubs(grid, signed_x) + madd(ones) -> int32x8, multiply by (2s+1), integer
  accumulate across the super-block, finally total += d * x.d * isum / 8.
- Notes:
  * pulsar measured AVX-512 VNNI SLOWER than AVX2 for IQ2_XXS on a Ryzen 9 9900X
    (Zen 5): 2.1 vs 4.8 GB/s single-thread, because assembling a zmm from 8
    scattered table loads plus the scalar sign chain stalls; the aggregate is
    RAM-bound at 12 threads anyway. Lesson: table-driven codebook formats do not
    automatically benefit from 512-bit width.
  * ds4 precomputes a signed grid table [256 grid][128 sign] of int8x8 so the
    NEON inner loop is load + sdot only (64 KiB table, L2 resident).

## F-quant-11: int8 x int8 dot primitives: sign trick, VNNI, NEON
Origin: [colibri] [pulsar] [ktransformers]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), MIT (pulsar)
- Source: colibri (Apache-2.0), `c/idot.h` (int8 x int8 and int4 x int8 dots
  for AVX-512 VNNI / AVX-VNNI / AVX2 / NEON dotprod / NEON / VSX);
  ktransformers kt-kernel (Apache-2.0), `operators/amx/la/amx_quantization.hpp`
  (signed x signed emulation on vpdpbusd) -- read only, not copied
- Purpose: one exact int32 dot for Q8_0 rows, int8 experts, and as a building
  block for everything else.
- Pseudocode:
      AVX2:        p16 = maddubs(abs(w), sign(x by w)) ; acc32 += madd(p16, ones16)
      AVX-512 VNNI: negmask = sign bits of w
                   xs = x with lanes negated where negmask
                   acc = dpbusd(acc, abs(w), xs)            # 64 bytes / instruction
      AVX-VNNI(256): same as above with 4 independent accumulators per 64 bytes
                   (colibri: one accumulator is latency-bound, ~5 cycles chain)
      NEON+dotprod: acc = sdot(acc, w, x), 4 accumulators per 64 bytes
      NEON plain:  widen-multiply low/high halves to int16, pairwise-add-accumulate to int32
      NEON i8mm:   smmla on 2 weight rows x 2 activation rows -> 2x2 int32 tile per
                   16-byte step (batch>=2 only)
- Notes:
  * abs(-128) stays -128 in int8 but is read as 128 by the unsigned operand, so
    weights may use the full range; activations must stay in [-127, 127].
  * Integer sums are associative: any accumulator count or lane order gives the
    same bits. Use 2-4 accumulators to break the dependency chain.
  * Signed x signed is not native on Zen 5 (no AVX-VNNI-INT8 assumed); emulate
    with the abs/sign trick above, or keep weights unsigned with an offset that
    is folded out via activation sums (preferred, F-quant-05/16).

## F-quant-12: FP8 E4M3FN decoding (scalar, AVX2 bit trick, AVX-512 VBMI to bf16)
Origin: [colibri] [ktransformers] [apus]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), `c/quant.h` (decode table + AVX2 8-lane decoder,
  NaN policy), `c/deepseek_v4.c` (exact scalar decode/encode); ktransformers
  kt-kernel (Apache-2.0), `operators/amx/la/amx_raw_kernels.hpp` (VBMI 2-table
  byte permute to bf16), `operators/avx2/fp8_dequant.hpp` (gather from 256-float
  table) -- read only, not copied
- Format: 1 sign, 4 exponent (bias 7), 3 mantissa. exp==0: subnormal
  mant * 2^-9. exp in 1..15: (1 + mant/8) * 2^(exp-7). exp==15 && mant==7: NaN
  (0x7F, 0xFF). No infinities. Max finite 448.
- Pseudocode (AVX2, 8 lanes, no table):
      b = zero-extend 8 bytes to int32x8
      bits = ((b & 0x80) << 24) | ((b & 0x7F) << 20)
      v = reinterpret_f32(bits) * 2^120        # rebias exponent 7 -> 127, also
                                               # exact for subnormals
      v = select(NaN, v, (b & 0x7F) == 0x7F)   # otherwise NaN would become +-480
- Pseudocode (AVX-512 VBMI -> bf16, 64 lanes):
      hiByte = two-table byte permute (128-entry table split across 2 zmm) indexed
               by the low 7 bits, giving bf16 bits 8..14
      hiByte |= (fp8 & 0x80)                   # sign into bf16 bit 15
      loByte = same permute with a second table pair (bf16 bits 0..7)
      bf16 lanes = byte-interleave(loByte, hiByte)   # unpacklo/unpackhi -> 2 zmm
- Notes:
  * The VBMI path makes FP8 -> bf16 essentially free and feeds vdpbf16ps
    (F-quant-13). Zen 5 has VBMI and AVX512_BF16.
  * Decide a NaN policy explicitly; colibri propagates NaN and relies on a
    downstream sampler guard.
  * Encoding (needed for dynamic activation FP8): round-to-nearest-even on the
    3-bit mantissa; values < 2^-6 handled as subnormal by scaling by 512 and
    rounding; saturate at 448.

## F-quant-13: FP8 weights with 128x128 block scales (DeepSeek-style)
Origin: [colibri] [ktransformers] [apus]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), `c/quant.h` (block-scaled matmul, two kernels
  chosen by compiler for bit-exactness), `c/fp8_format.h`, `c/deepseek_v4.c`
  (UE8M0-scale variant, dynamic activation qdq, 8-row interleaved repack);
  ktransformers kt-kernel (Apache-2.0), `operators/amx/fp8-moe.hpp`,
  `operators/amx/la/amx_raw_kernels.hpp` (bf16 dot kernel) -- read only, not copied
- Layout:
      weight : O x I bytes, row-major, one E4M3 byte per element
      scale  : ceil(O/128) x ceil(I/128) values, row-major; w = e4m3(b) * scale[o/128][i/128]
               (MULTIPLY; checkpoints name it an inverse scale)
               scale dtype: f32 (DeepSeek V3 / GLM FP8) or 1-byte UE8M0 = 2^(e-127)
               (DeepSeek V4); detect by byte size, never guess
- Pseudocode (reference, one row):
      acc64 = 0.0 (double)
      for each 128-column block bi:
          part = 0.0f
          for i in block: part += e4m3(w[i]) * x[i]        # float inside the block
          acc64 += double(part) * scale[o/128][bi]
      y = float(acc64)
- Pseudocode (AVX-512 BF16, ktransformers style, 32 rows x M tokens):
      for each 128-wide K group:
          c[m][0..31] = 0 (two zmm per token)
          for each pair of k values:
              decode 64 fp8 bytes (32 rows x 2 k) -> two bf16 zmm (F-quant-12)
              a = broadcast the bf16 pair (x[k], x[k+1]) of token m
              c[m] = vdpbf16ps(c[m], a, wbf16)    # f32 accumulate, pairs of k
          out[m][rows] += c[m] * scale[row_block][kgroup]   # one scale for all 32 rows
- Pseudocode (AVX2, colibri "rows8" repack):
      repack: for each tile of 8 rows: bytes ordered [column][row 0..7]
      per tile: sum8 = 0 ; for each 128-col block: s = scale
                 for each column c: v8 = decode(8 bytes) ; sum8 += (x[c]*v8)*s
- DeepSeek V4 native FP8 path (colibri): the ACTIVATION is also quantised
  dynamically per 128 elements to E4M3 with a UE8M0 scale
  (exponent = ceil(log2(max(amax,1e-4)/448)), clamp to +-127), then dequantised
  back to f32 before the dot; this simulates the official kernel's numerics.
- Notes:
  * Block structure (128 rows share a scale) means N_STEP = 32 rows never
    straddles a scale boundary when tiles are 32-aligned.
  * vdpbf16ps rounds the activation to bf16; keep a pure-f32 reference for
    validation. ktransformers' comment claims ~493 GB/s effective for its
    4-k-unrolled variant (hardware unspecified).
  * colibri found GCC contracts the 4-accumulator form differently from the
    1-accumulator form (~1 ulp drift, flips argmax) and that GCC ignores
    pragma/attribute requests to disable contraction; only the global flag
    works. It ships the fast form only under clang.

## F-quant-14: MXFP4 (E2M1 + E8M0) layouts — GGML block vs native split tensors
Origin: [colibri] [ds4] [pulsar] [monolite] [ktransformers]

Licence-of-origin: Apache-2.0 (colibri, monolite, ktransformers), MIT (ds4, pulsar)
- Source: ds4 (MIT), `ds4.c` (17-byte block, E8M0 decode, f32 dot);
  monolite (Apache-2.0), `src/kernels_avx512.c`, `docs/mnl_format.md`
  (gpt-oss native layout); colibri (Apache-2.0), `c/quant.h`,
  `c/deepseek_v4.c` (separate scale tensor, E8M0 decode variants) -- read only, not copied
- E2M1 nibble: bit 3 = sign, bits 1..2 = exponent, bit 0 = mantissa;
  magnitudes for codes 0..7: 0, 0.5, 1, 1.5, 2, 3, 4, 6; codes 8..15 negative.
  Doubled magnitudes are exact integers 0,1,2,3,4,6,8,12 (fits int8).
- E8M0 scale byte e: 2^(e-127); e = 255 is NaN per OCP. Edge handling differs
  across projects (e=0 -> 2^-127 subnormal in ds4/colibri-native, -> +0 with a
  plain `e << 23` bit cast; e=255 -> NaN / +inf / 0). Pick one and document it;
  real checkpoints never use 0 or 255.
- Layout A, GGML block (17 bytes / 32 weights):
      [0]      uint8 e (E8M0)
      [1..16]  16 bytes; element j (j<16) = low nibble of byte j,
               element j+16 = high nibble of byte j      ("split halves")
- Layout B, native checkpoint (gpt-oss/monolite, DeepSeek-V4/colibri):
      blocks tensor [rows][cols/32][16 bytes], scales tensor [rows][cols/32] uint8
      element 2i = low nibble of byte i, element 2i+1 = high nibble ("interleaved")
- Dequant (both): w = e2m1(nib) * 2^(e-127)  ==  (doubled_int(nib)) * 2^(e-128)
- Notes: the two nibble orders are NOT interchangeable; convert to one internal
  order at load time (prefer split halves: one AND gives 16 elements in order,
  one shift gives the next 16, matching activation memory without permutes).

## F-quant-15: MXFP4 dot kernels (AVX2 integer, AVX-512 float LUT, AVX-512 BF16)
Origin: [colibri] [ds4] [pulsar] [monolite] [ktransformers]

Licence-of-origin: Apache-2.0 (colibri, monolite, ktransformers), MIT (ds4, pulsar)
- Source: colibri (Apache-2.0), `c/quant.h` (float path and int8 "IDOT" path
  with per-32 activation quant), `c/deepseek_v4.c` (16-row interleaved repack);
  monolite (Apache-2.0), `src/kernels_avx512.c` (register LUT + activation
  de-interleave); ktransformers kt-kernel (Apache-2.0), `operators/amx/fp4-moe.hpp`
  (E2M1 -> bf16 via word permute, activation permuted once) -- read only, not copied
- Pseudocode (AVX2 integer, colibri):
      x is quantised per 32 to int8 with scale sx[g] = amax/127 (clamp +-127)
      per group g of 32:
          lo = bytes & 0xF ; hi = (bytes >> 4) & 0xF
          w = 16-entry byte shuffle (doubled signed ints) of re-interleaved nibbles
          p = maddubs(abs(w), sign(x by w)) ; i32 = madd(p, ones)     # |w|<=12: no saturation
          acc = fma(float(i32), broadcast(2^(e-127) * 0.5 * sx[g]), acc)
- Pseudocode (AVX-512 float, monolite, interleaved layout):
      once per call: split x into even-index and odd-index arrays
      per row, per group: 16 bytes -> widen to 16 x int32
          wl = vpermps(lut16f, bytes & 0xF) ; wh = vpermps(lut16f, bytes >> 4)
          s = 2^(e-127) ; acc += wl*(x_even*s) + wh*(x_odd*s)
- Pseudocode (AVX-512 BF16, ktransformers):
      once per token: permute each 32-element bf16 activation group into the
      "all low nibbles, then all high nibbles" order the decoder produces
      per group: idx = widen nibbles to 16-bit ; w_bf16 = vpermw(lut16_bf16, idx)
          part = vdpbf16ps(0, act_group, w_bf16) ; acc += part * 2^(e-127)
      scales may be stored as raw exponent bytes after validating that every
      source scale is a pure power of two (sign 0, mantissa 0)
- Pseudocode (colibri rows16 repack for hot experts): interleave 16 rows so each
  16-byte load holds byte c of rows 0..15; per column decode 16 values with a
  register LUT, gather 16 row scales once per 32 columns, accumulate a 16-wide
  output vector (no horizontal reduction).
- Notes: per-32 integer activation quant adds ~0.4%/group noise (colibri
  comment); keep it opt-in and A/B against the float path.

## F-quant-16: colibri int4 gs64 planar layout (and repack from pair order)
Origin: [colibri] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), no-licence (flashmoe)
- Source: colibri (Apache-2.0), `c/expert_ffn.h` (layout, repack, scalar refs),
  `c/idot.h` (planarise helper, planar dots), `c/quant.h` (pair-order packer,
  grouped kernels) -- read only, not copied
- Layout (row of I weights, I % 64 == 0):
      I/64 blocks of 32 bytes; in block b:
          byte k (k = 0..31): low nibble = element 64b + k, high nibble = element 64b + 32 + k
          nibbles store v + 8 (unsigned 0..15), v in [-8, 7]
      scales: f32 [O][I/64], one per (row, block); w = scale * (u - 8)
      quantiser: per 64 group, symmetric absmax, s = amax/7, v = clamp(round(x/s), -8, 7)
- Repack from "pair order" (byte j = element 2j low, 2j+1 high, two's-complement
  4-bit): for each block, element k comes from byte k/2 nibble k&1, element k+32
  from byte 16 + k/2; signed nibble -> unsigned by XOR 8. AVX2: split nibbles of
  the two 16-byte halves, byte-interleave lo/hi to restore element order, XOR 8,
  then OR(lo_part, hi_part << 4).
- Notes: planar order means AND 0x0F yields elements 0..31 contiguous and
  (>> 4) & 0x0F yields 32..63, matching a straight 32-byte activation load —
  zero unpack shuffles. Same property as Q4_K's chunk layout.

## F-quant-17: planar int4 dot kernels (f32 mode and int8/VNNI mode) + AMX tiles
Origin: [colibri] [ds4] [pulsar] [ktransformers] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: colibri (Apache-2.0), `c/expert_ffn.h`, `c/idot.h` (AVX-512 VNNI one
  zmm per block, 1x4 row-tile, AMX int8 tile kernel with OS permission request)
  -- read only, not copied
- Pseudocode (f32 mode, defines the exact numerics):
      acc[8] = 0
      for block g:
          lane[8] = 0
          for i in 0..63: lane[i mod 8] = fma(float(u_i - 8), x[64g+i], lane[i mod 8])
          for l in 0..7: acc[l] = fma(lane[l], scale[g], acc[l])
      return ((acc0+acc4)+(acc2+acc6)) + ((acc1+acc5)+(acc3+acc7))
- Pseudocode (i8 mode, unsigned weights, offset folded out):
      xq = int8(x / sx) with one sx per row ; xsum[g] = int sum of xq over block g
      for block g:
          d = dpbusd(0, u_lo32, xq[64g..+31]) ; d = dpbusd(d, u_hi32, xq[64g+32..+63])
          acc8 = fma(float(d), scale[g], acc8)
      corr = sequential fma over g of scale[g]*xsum[g]
      y = (hsum8(acc8) - 8*corr) * sx
      AVX-512: the whole 64-element block is one zmm (low nibbles in lanes 0..31,
      high in 32..63) -> ONE dpbusd per block, then integer reduce per group.
      Row tiling: 1 weight block x 4 activation rows per load (prefill / batch).
- Pseudocode (AMX int8, Sapphire Rapids+, not Zen 5): per output tile of 16 rows
  and per 64-wide group, unpack int4 once to SIGNED int8 in VNNI order
  [k/4][row][4] into a 1 KiB staging buffer; for each 16-token tile: zero C,
  tile-dot (signed x signed), store, then per lane fma(int32_as_float, scale,
  acc). Unpack cost amortised over all token tiles. Needs an OS permission call
  for tile state before first use (Linux arch_prctl request; Windows XState opt-in).
- Notes:
  * colibri deliberately keeps NO 512-bit f32 path for this kernel: folding 16
    lanes onto the reference's 8 changes fma order. Integer paths are exact and
    can widen freely.
  * colibri's measured motivation: Qwen3.6 MoE at full residency cost 34 ms/token
    vs a DRAM floor of 17 ms for int8-unpacked experts and 9 ms for the packed
    int4 bytes — never unpack int4 to int8 at load time.

## F-quant-18: ktransformers int4 "value x 16" packing and AMX/AVX-512 tile order
Origin: [ktransformers]

Licence-of-origin: Apache-2.0 (ktransformers)
- Source: ktransformers kt-kernel (Apache-2.0), `operators/amx/la/amx_buffers.hpp`
  (int4 B-buffer packer, k-group variants), `operators/amx/la/amx_kernels.hpp`
  (224 int4 kernel: AMX and AVX-512 fallback), `operators/amx/la/amx_quantization.hpp`
  -- read only, not copied
- Purpose: int4 weights consumable by signed int8 dot units without sign
  extension, in a layout matching 16x64 AMX B tiles.
- Packing:
      per output row: scale = amax / 112   (112 = 7 * 16)
      q8 = saturate_int8(round(w / scale))         # multiples of 16 after rounding
      round each q8 to nearest multiple of 16 (|q|+8, clear low 4 bits, restore sign)
      two consecutive K_STEP=64 slices share bytes: slice A keeps its value in the
      HIGH nibble (q8 as-is, i.e. v*16), slice B is shifted right by 4 into the LOW nibble
      after packing each 16-row x 64-byte slab, transpose 16x16 32-bit words so the
      slab is in VNNI order [k/4][row][4]
      blocking: N_BLOCK = 128 rows per thread task, K_BLOCK = 3584, N_STEP = 32, M_STEP = 32
- Compute:
      high-nibble operand = bytes & 0xF0          -> int8 equal to v*16 (sign correct)
      low-nibble operand  = (bytes & 0x0F) << 4   -> int8 equal to v*16
      AMX: 2x2 tile grid (2 A tiles of 16 tokens, 2 B tiles of 16 rows) per slice
      AVX-512 fallback: for each token and each k-quad: broadcast 4 activation bytes,
          signed-x-signed dot (emulated: dpbusd(abs(a), sign(b by a))) against 32 rows
      epilogue: y = int32 * act_scale[token] * row_scale[row]  (row_scale already /16-aware)
- Notes: avoids per-element subtract/sign-extend at the cost of one AND (+shift)
  per 64 bytes; requires activations in [-127,127] for the emulation. On Zen 5
  (no AMX) only the AVX-512 path applies.

## F-quant-19: fused routed-expert FFN for one layer (colibri)
Origin: [colibri] [ktransformers] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), no-licence (flashmoe)
- Source: colibri (Apache-2.0), `c/expert_ffn.h` (layer runner over planar int4
  gs64 experts, f32 and i8 modes) -- read only, not copied
- Inputs / outputs: x[S][H] f32, routing idx/val [S][K], expert weights
  gate/up [F][H], down [H][F] planar int4 + scales -> out[S][H].
- Pseudocode:
      if int mode: quantise each token row once (row scale + per-64 sums)
      group (token, slot) pairs by expert id (first-seen order, linked chain)
      chunk sizes: aim for ~4*threads work items; rows per chunk multiple of 8
      PHASE 1 (parallel, dynamic): item = (expert e, gate/up row chunk)
          for every token routed to e:
              for r in chunk: g[r] = dot(gate row r, x) ; u[r] = dot(up row r, x)   # same pass
      PHASE 1b (parallel per pair): h = silu(g) * u ; if int mode: quantise h (scale + per-64 sums)
      PHASE 2 (parallel, dynamic): item = (expert e, down row chunk)
          for every token routed to e: contrib[pair][r] = dot(down row r, h_pair)
      PHASE 3 (parallel per token): out[s] = sum over k in rank order of val[s][k]*contrib[s][k]
- Notes:
  * Two parallel regions per layer instead of 3*K; a prompt's tokens that hit
    the same expert reuse its weights from cache within a chunk.
  * Summing contributions in rank order (k = 0..K-1) keeps output independent of
    thread scheduling.
  * silu written as g/(1+exp(-g)).

## F-quant-20: ktransformers MoE flow: per-expert batching, SwiGLU epilogue, requant for down
Origin: [colibri] [ds4] [pulsar] [ktransformers] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, ktransformers), MIT (ds4, pulsar), no-licence (flashmoe)
- Source: ktransformers kt-kernel (Apache-2.0), `operators/amx/moe_base.hpp`,
  `operators/amx/la/amx.hpp` (vectorised activation with clamps),
  `operators/llamafile/moe.hpp` (GGML-typed experts: quantise to the weight's
  activation type, act, requantise per chunk) -- read only, not copied
- Pseudocode (GGML-typed experts, decode, one token):
      xa = quantise(x, activation_type_of(gate))   # Q8_K for K-quants, Q8_0 for Q8_0/Q4_0
      (reuse xa for up if its activation type matches, else quantise again)
      parallel over (active expert, row block of m_block rows):
          g_blk = gemv(gate rows, xa) ; u_blk = gemv(up rows, xa)
          h_blk = act(g_blk, u_blk)                       # f32
          if m_block is a multiple of the down activation block size:
              quantise h_blk directly into its slot of the down-input buffer
      if not: barrier, then quantise each full h row
      parallel over (active expert, output row block): y_e = gemv(down rows, h_q)
      weighted sum over experts
- act(g, u) variants (AVX-512, 16 lanes):
      standard SwiGLU:     g * sigmoid(g) * u, with -g clamped to <= 88 before exp
      clamped (DeepSeek V4 style): g = min(g, L) ; u = clamp(u, -L, L) before the above
      alpha variant (gpt-oss style): sigmoid(alpha*g) with symmetric clamps on g and u
- Notes:
  * Key enabler: Q8_K/Q8_0 scales are per-block, so a thread owning whole
    blocks of the intermediate vector can quantise its slice locally — the down
    input is ready without an extra pass or barrier (only the down GEMV itself
    needs all of h).
  * AMX path (prefill) instead copies each token's input into per-expert
    contiguous buffers, converts them to the kernel's A format once per expert,
    runs gate and up as separate tasks (2 x row-blocks x experts), applies act
    in bf16->f32->bf16, then converts h into the down A-buffer.
  * Small batches (< 10 tokens) run setup loops inline instead of on the pool.

## F-quant-21: avoiding numerical drift across ISAs and compilers
Origin: [colibri] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), `c/gsgemv.h`, `c/expert_ffn.h`, `c/quant.h`,
  `c/exact_dot.h`; pulsar (MIT), `cpu_dot.rs` (bitwise SIMD==scalar test);
  ds4 (MIT), `ds4.c` (int64 scalar accumulators) -- read only, not copied
- Rules distilled:
      1. Make every inner block an exact integer sum (int8/int4/K-quant x int8).
         Integer addition is associative: SIMD width, unroll and accumulator
         count cannot change the result.
      2. Apply float scales once per block, then accumulate across blocks in a
         FIXED order (ascending block index) per output element.
      3. Where a float multiply-add is semantically intended, write an explicit
         fused multiply-add in source; never let -ffp-contract decide (colibri
         found the same source giving different bits by flag and by -march).
      4. For f32-activation paths, fix lane assignment (element i -> lane i mod 8)
         and a fixed reduction tree; do not widen to 16 lanes if 8-lane
         reference results must be reproduced.
      5. Long reductions over many large-magnitude blocks (FP8 128-blocks):
         accumulate the cross-block sum in double, float inside a block.
      6. Interleave multiple output rows (4-row tiles) instead of splitting one
         row's accumulator — same per-row op sequence, better latency hiding.
      7. Keep an order-independent oracle: colibri's exact dot forms each product
         exactly (integer mantissas), bins by exponent in 128-bit integers and
         rounds once; used only in a verify mode to adjudicate near-ties.
- Notes: quantising activations (Q8_K, per-32 int8 for MXFP4, bf16 rounding in
  vdpbf16ps) is a deliberate numerics change; gate it behind a mode switch and
  compare against a float reference, not against itself.

## F-quant-22: CPU feature dispatch
Origin: [colibri] [pulsar] [monolite] [ktransformers] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite, ktransformers), MIT (pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), `c/idot.h` (compile-time kernel selection and
  name string, AMX arming), `c/quant.h` + `c/colibri.c` (runtime self-test and
  env gates); ktransformers kt-kernel (Apache-2.0), `python/_cpu_detect.py`
  (multi-variant extension loader), `operators/avx2/rawint4_avxvnni-moe.hpp`
  (constructor-time feature check), `CMakeLists.txt`; pulsar (MIT), `cpu_dot.rs`
  (per-call feature detection); monolite (Apache-2.0), `src/monolite.c`
  (doctor report) -- read only, not copied
- Patterns seen:
      colibri: kernels chosen at COMPILE time by an ordered chain
               (AVX-512 VNNI > AVX-VNNI > AVX2 > NEON i8mm > NEON dotprod > NEON > VSX > scalar),
               exposing the chosen name for logs; optional fast paths additionally
               gated by a startup self-test against the scalar reference and by
               env vars; AMX enabled only after an OS permission syscall succeeds.
      ktransformers: build the extension several times (amx, avx512_bf16,
               avx512_vbmi, avx512_vnni, avx512_base, avx2); at import read CPU
               flags, pick the first variant whose required flag set is complete
               (accepting underscore/no-underscore flag spellings), allow an env
               override, fall back down the chain if a variant fails to load.
               Individual operators re-check a feature in their constructor and
               refuse with a clear error.
      pulsar:  runtime detection per call with scalar fallback; chose AVX2 over
               AVX-512 for a codebook format after measuring.
- Pseudocode (recommended for a C engine, see synthesis):
      at startup: probe CPUID leaves 1, 7.0, 7.1 and XGETBV (OS must enable
                  zmm/opmask state: XCR0 bits 1,2,5,6,7)
      level = best of {AVX512_VNNI_BF16, AVX512_VNNI, AVX2_FMA, SCALAR} whose bits all present
      allow override via env (downgrade only)
      fill a function-pointer table: per (weight format, kernel kind) -> impl at level
      run a micro self-test of each selected kernel against scalar on fixed data;
      on mismatch beyond contract, drop one level and log
- Notes: compile each ISA tier as a separate translation unit with its own
  target flags (or target attributes) so the binary runs on any x86-64 and
  never executes an instruction the CPU lacks.

## F-quant-23: NEON strategies (portable reference for ARM targets)
Origin: [colibri] [ds4] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4), PolyForm-NC-1.0.0 (apus)
- Source: ds4 (MIT), `ds4.c` (Q8_0, Q4_K, IQ2_XXS, Q2_K with sdot; widen-multiply
  fallback); colibri (Apache-2.0), `c/idot.h`, `c/deepseek_v4.c` (i8mm smmla
  2x2 tiles; 4-register table lookup to gather float E2M1 values) -- read only, not copied
- Pseudocode:
      dotprod available: per 16 bytes acc = sdot(acc, w, x); 4 accumulators; final add-across
      no dotprod: low/high halves widen-multiply to int16, then pairwise add-accumulate to int32
      Q4_K: unpack the 32 nibbles of a sub-block to bytes, two sdot with the 32 activations,
            multiply the int32 by the 6-bit scale (scalar), mins via group sums
      i8mm: smmla consumes 2x8 x 8x2 int8 -> 2x2 int32; pair 2 weight rows with 2 tokens
      FP4 float gather: 64-byte E2M1 float table in 4 q-registers; a byte-table
            lookup expands each nibble code into 4 byte offsets -> one tbl4 gathers 4 floats
- Notes: ds4's Q4_K NEON path unpacks nibbles through a stack buffer (scalar);
  a vector AND/shift unpack is strictly better.

---

## Recommended synthesis (Zen 5: AVX-512 VNNI/BF16, AVX2 and scalar fallbacks)

1. Formats to support natively, all at GGUF-compatible byte layouts: Q8_0 (dense),
   Q4_K/Q5_K/Q6_K (experts), Q8_K (activation, internal only), IQ2_XXS (optional),
   MXFP4 (normalise both nibble orders to the "split halves" order at load),
   FP8 E4M3 with 128x128 scales (f32 or UE8M0 scale, detected by size), and
   colibri-style planar int4 gs64 as our own compact expert format.
2. One scalar reference per (weight format x activation format), written as
   exact integer inner sums + one float update per block in ascending order. All
   SIMD kernels must match it bit-for-bit (integer paths) or within a stated ulp
   bound (bf16 paths); enforce in unit tests with random blocks and edge values
   (all-zero block, max magnitudes, NaN FP8 bytes, e8m0 0/255).
3. Activations: quantise once per (token, input vector) and share across all
   consumers. Q8_K (positive d = amax/127, clamp +-127, int16 per-16 sums) for
   K-quants/IQ; Q8_0 per-32 for Q8_0 weights; bf16 for FP8/MXFP4 BF16 paths.
   Hoist out of per-expert loops.
4. Zen 5 primary GEMV for K-quant experts: repack at load into 16-row
   VNNI-transposed tiles per super-block (F-quant-09): keep the nibble plane and
   the high-bit plane(s) at their original size (Q5_K 1 bit, Q6_K 2 bits per
   weight), permuted into the tile order, and merge them in registers (AND/shift/OR)
   into unsigned bytes; unpacked 6-bit scales as int8 [groups][16 rows], fp16
   d/dmin [16]. Broadcast activation quads, vpdpbusd, fold Q6_K's -32 and the
   K-quant mins through activation group sums, one fma per super-block into a
   16-float output vector.
5. Dense Q8_0: same 16-row transposed tile on AVX-512 (weights signed: use the
   activation +128 bias with a per-row precomputed compensation, or abs/sign);
   on AVX2 use maddubs/madd with the sign trick, 4 accumulators.
6. FP8: on Zen 5 decode E4M3 -> bf16 with two VBMI byte permutes + sign OR and
   accumulate with vdpbf16ps over 32-row tiles, one scale per 128-row x 128-col
   block applied per K-group; AVX2 fallback uses the shift-and-multiply-by-2^120
   decoder with NaN blend and f32 FMA; scalar reference accumulates per block in
   float and across blocks in double.
7. MXFP4: AVX-512 BF16 path decodes nibbles to bf16 with one 16-bit permute from
   a 16-entry table and permutes the activation group once per token to match;
   integer alternative (doubled E2M1 ints via byte shuffle + per-32 int8
   activation) on AVX2. Store E8M0 as raw bytes; fold the 0.5 un-doubling into
   the group scale.
8. Fused expert FFN: per active expert, process gate and up for the same row
   chunk in one task (interleave gate/up rows in the repacked tile stream so one
   pass yields both), apply SwiGLU (with optional clamp/alpha variants) in
   registers, and quantise h for down in-place when the chunk is a whole number
   of activation blocks (256 for Q8_K) — no extra pass. Down projection after a
   per-layer barrier; sum expert contributions in routing-rank order.
9. Scheduling: two parallel regions per MoE layer (gate/up, down) over
   (expert, row-chunk) items with dynamic scheduling, ~4 items per thread, chunk
   rows multiple of 16; tokens routed to the same expert in a batch iterate inside
   the chunk to reuse weights from cache.
10. Never unpack 4/5/6-bit weights to int8 in memory (colibri measured ~2x DRAM
    cost); unpack in registers only. Keep repacked layouts the same byte size as
    the source format (repack is a permutation plus scale reshuffle).
11. Drift policy: explicit fma in source for every intended fused op, fixed lane
    mapping and reduction trees, no reliance on -ffp-contract; integer exactness
    everywhere possible; an opt-in exact-dot oracle for validation.
12. Dispatch: separate translation units per tier (AVX-512 VNNI+BF16+VBMI,
    AVX2+FMA, scalar; NEON dotprod/i8mm for ARM builds), CPUID+XGETBV probe at
    startup, env override to downgrade, function-pointer tables per format, and
    a startup self-test that falls back one tier on mismatch.
13. Do not assume wider is faster: benchmark per format on Zen 5. pulsar measured
    AVX2 > AVX-512 for IQ2_XXS on a 9900X; codebook formats may keep an AVX2
    kernel even on AVX-512 machines. Decode is DRAM-bound; optimise bytes moved
    first, instruction count second.
14. AMX (Intel only) and signed x signed int8 dot instructions are not available
    on Zen 5; design the main path around vpdpbusd with unsigned weights and
    offset folding, so no emulation is needed.
