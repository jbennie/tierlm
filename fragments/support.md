# Support fragments: tokenizer, file readers, probing/planning, config, HTTP server, JSON

Scope: everything around the inference core. All pseudocode below is written in
our own words after reading the cited files; nothing is copied. Regex patterns
are described in prose (rule by rule), not reproduced. File-format facts
(GGUF, safetensors, GPT-2 byte map) are public specifications and are stated
precisely.

Topic codes: TOK = tokenizer, FMT = file readers, HW = hardware probing and
planning, CFG = configuration, SRV = HTTP server, JSON = JSON parser.

---

## F-TOK-01: Byte-level alphabet (GPT-2 byte to codepoint map)
Origin: [colibri] [ds4] [pulsar] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/tok.h; ds4 (MIT), ds4.c (byte-level BPE section); pulsar (MIT), crates/tokenizer/src/lib.rs -- read only, not copied
- Purpose: map every raw byte to a printable Unicode codepoint so that HF/GGUF vocab strings can be matched byte-exactly, and invert the map on decode.
- Inputs / outputs: none -> two tables: byte_to_cp[256] (u32) and cp_to_byte[0..511] (i16, -1 = not a byte char). Also cache the UTF-8 encoding of each byte_to_cp entry (1 or 2 bytes) so encoding a piece is a memcpy per byte.
- Pseudocode:
      printable(b) = b in 33..126 or b in 161..172 or b in 174..255
      next = 256
      for b in 0..255:
          if printable(b): cp = b
          else:            cp = next; next += 1
          byte_to_cp[b] = cp
          byte_utf8[b]  = utf8_encode(cp)          # 1 or 2 bytes
          cp_to_byte[cp] = b
- Notes: This is the public GPT-2 convention: 188 bytes map to themselves, the
  remaining 68 map to 256..323 in ascending byte order. A table of 512 entries
  is enough for the inverse. On decode, a codepoint not in the inverse table is
  NOT a byte char (it belongs to a control/added token string) and should be
  emitted as its UTF-8 bytes (pulsar does this) rather than dropped (colibri
  drops it, but routes added tokens around this path).
- Variants seen elsewhere: ds4 recomputes the codepoint per byte with a loop
  (fine because it is called per piece, but a table is cheaper). monolite's
  o200k vocab is exported as raw bytes, so it needs no byte map at all (see
  F-TOK-03).

## F-TOK-02: BPE merge loop, merges-list ranks
Origin: [colibri] [ds4] [pulsar] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/tok.h; ds4 (MIT), ds4.c (BPE emit piece) -- read only, not copied
- Purpose: turn one pre-tokenised piece into token ids using the HF/GGUF ordered merges list.
- Inputs / outputs: piece bytes -> appended ids. Needs: vocab hash (string -> id), merge hash (left SEP right -> rank, rank = index in merges list).
- Pseudocode:
      s = concat(byte_utf8[b] for b in piece)             # byte-level string
      if vocab has s: emit vocab[s]; return               # whole-piece shortcut
      syms = list of (offset,len) for each codepoint of s # initial symbols
      loop:
          best_rank = INF; best_i = none
          for i in 0 .. len(syms)-2:
              r = merge_rank(syms[i], syms[i+1])          # key = left + SEP + right
              if r found and r < best_rank: best_rank = r; best_i = i
          if best_i is none: break
          syms[best_i].len += syms[best_i+1].len          # adjacent in s, so just widen
          delete syms[best_i+1]
      for sym in syms:
          if vocab has sym: emit id
          else: emit id of each single byte-char inside sym (fallback)
- Notes: Because merged symbols are contiguous slices of s, a merge is a length
  update plus a delete; no string allocation is needed (colibri style). ds4
  allocates a new string per merge -- simpler but slower. Separator for the key:
  GGUF stores merges as "left right" (single space), HF JSON either as that
  string or as a 2-element array; store internally with a byte that can never
  appear in a byte-level token (NUL) to avoid ambiguity. The whole-piece
  shortcut matches HF "ignore_merges" behaviour for vocabularies that set it;
  for vocabularies that do not, it is still correct in practice because a
  vocabulary entry equal to the whole piece is what the merges would reach,
  but keep a flag to disable it for strict parity tests. O(n^2) per piece is
  acceptable because pieces are short (word-sized).
- Variants seen elsewhere: monolite keeps a per-position pair-rank array and
  only recomputes the two neighbours after a merge (F-TOK-03), O(n) per merge
  instead of a full rescan.

## F-TOK-03: tiktoken-style rank BPE (no merges list)
Origin: [colibri] [monolite]

Licence-of-origin: Apache-2.0 (colibri, monolite)
- Source: monolite (Apache-2.0), src/tokenizer.c; colibri (Apache-2.0), c/tok.h (rank mode when merges are absent) -- read only, not copied
- Purpose: encode with a vocabulary whose ids ARE the merge priority (tiktoken o200k/cl100k exports), so no merges list exists.
- Inputs / outputs: piece bytes -> ids. Needs only bytes -> rank hash.
- Pseudocode:
      if rank_of(piece) exists: emit it; return
      bounds = [0, 1, 2, ..., n]                 # n+1 boundaries, each byte a part
      pair_rank[i] = rank_of(bytes[bounds[i] : bounds[i+2]]) for each i
      while number of parts > 1:
          i = argmin(pair_rank)                  # ties -> leftmost
          if pair_rank[i] is NONE: break
          remove bounds[i+1]; remove pair_rank[i+1]
          recompute pair_rank[i-1] (if i>0) and pair_rank[i] from new bounds
      for each part: emit rank_of(part)          # must exist: all 256 bytes are in vocab
- Notes: This is exactly tiktoken's byte-pair merge: the pair to merge is the
  one whose CONCATENATION has the lowest id. Vocabulary format used by monolite:
  u32 count, then per id a u16 length and raw bytes (no byte map needed). Hash
  is open addressing with FNV-1a over raw bytes, table size = next power of two
  >= 2 * n_vocab, values stored as id+1 so zero means empty. Pieces longer than
  a fixed bound are rejected in monolite; we should instead fall back to a
  heap buffer. Special tokens are NOT encoded here; they are pushed by the chat
  renderer directly as ids.
- Variants seen elsewhere: colibri switches to this mode automatically when a
  tokenizer.json has an empty merges list, looking up the concatenation in the
  vocab hash instead of the merges hash.

## F-TOK-04: Hand-written pre-tokeniser (cl100k / GPT-4 family rules)
Origin: [colibri] [ds4] [pulsar] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/tok.h; ds4 (MIT), ds4.c (JoyAI / GLM / Qwen splitters) -- read only, not copied
- Purpose: split text into pieces exactly as the HF regex Split would, without a regex library.
- Inputs / outputs: UTF-8 text span (no special tokens inside) -> sequence of byte ranges, each fed to BPE.
- Pseudocode (rules tried IN ORDER at position i; first match wins, emit, continue):
      decode the span into arrays cp[] and byte_off[] once
      R1 contraction: apostrophe followed (case-insensitively) by one of
         s, t, m, d (2 codepoints) or re, ve, ll (3 codepoints)
      R2 word: optionally ONE leading codepoint that is not CR/LF, not a letter,
         not a number -- taken only if the next codepoint is a letter -- then a
         maximal run of letters
      R3 digits: 1 to 3 number codepoints
      R4 punctuation: optional single space, then a maximal run of codepoints
         that are neither whitespace, letter nor number, then any CR/LF run
      R5 whitespace ending in newline: within the maximal whitespace run starting
         at i, if it contains CR or LF, end just after the LAST CR/LF in the run
      R6 whitespace not followed by non-space: if the run reaches end of text
         take it all; otherwise take all but the last whitespace codepoint
         (that one becomes the optional prefix of the next word/punct piece)
      R7 single whitespace (R6 produced zero length): take one codepoint
      safety: if nothing matched, take one codepoint
- Notes: Unicode class tests (letter, number, whitespace, and for o200k
  upper-ish / lower-ish / mark) come from a generated range table with a bit
  flag per range, looked up by binary search (monolite: ~2.3k ranges). Generate
  the table offline from the Unicode database at the same version the reference
  tokenizer uses. Correctness check is token-for-token parity against the
  reference implementation on a corpus that includes code, CJK, numbers with
  leading spaces and CRLF. colibri reports that using the wrong family is
  silent and frequent: on a GPT-2-family model, cl100k rules matched only
  1560 of 1708 cases, every miss being a space-prefixed number or a non-space
  prefix glued to a word.
- Variants seen elsewhere: GPT-2 family (bare ByteLevel with regex): contractions
  are case-sensitive, letters/numbers/punct each take at most ONE optional
  leading SPACE (not any non-letter), number runs are unbounded, no newline tail
  on punctuation and no "whitespace ending in newline" rule. ds4's JoyAI family
  adds a CJK/kana run rule, a "one punct/symbol then ASCII letters" rule, and
  keeps trailing newlines with a punctuation run (the comment says splitting
  them broke long-context logits on code). Kimi family adds a Han-run rule that
  is tried first and masks Han out of the letter classes.

## F-TOK-05: o200k letter-run matcher with regex backtracking semantics
Origin: [colibri] [ds4] [pulsar] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: monolite (Apache-2.0), src/tokenizer.c; colibri (Apache-2.0), c/tok.h -- read only, not copied
- Purpose: reproduce the case-aware word rules of the o200k family, including the order in which a backtracking regex engine gives characters back.
- Inputs / outputs: cp[] with class flags, position i -> end index or none.
- Pseudocode:
      UPPERISH = Lu|Lt|Lm|Lo|M ; LOWERISH = Ll|Lm|Lo|M   (Lm, Lo, M are in both)
      rule A (zero-or-more upper, then one-or-more lower, optional contraction):
        for use_prefix in [true, false]:              # greedy optional prefix first
            if use_prefix and cp[i] is (not CR/LF, not letter, not number) and i+1<n:
                j0 = i+1
            elif use_prefix: continue
            else: j0 = i
            m = end of maximal UPPERISH run from j0
            for split from m down to j0:              # give back one at a time
                if split < n and cp[split] is LOWERISH:
                    k = end of LOWERISH run from split
                    return k + contraction_len(k)
      rule B (one-or-more upper, zero-or-more lower, optional contraction):
        same prefix loop; m = end of UPPERISH run from j0
        if m > j0: k = end of LOWERISH run from m; return k + contraction_len(k)
      return none
      then: numbers 1..3; punctuation run whose newline tail ALSO accepts '/';
      then whitespace rules as in F-TOK-04.
- Notes: Because Lm/Lo/M are in both classes, the "give back" loop matters:
  for a word of only Lo characters, the upper run swallows everything and must
  return the last codepoint so the lower run can match it. Contraction suffix is
  case-insensitive via ASCII OR 0x20.
- Variants seen elsewhere: monolite keeps upper-run positions in a fixed
  on-stack array (bounded run length); colibri walks indices of the pre-decoded
  codepoint array, which needs no bound. Prefer the latter.

## F-TOK-06: Special / added tokens as atomic units
Origin: [colibri] [ds4] [pulsar] [apus]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/tok.h; ds4 (MIT), ds4.c (rendered-chat tokenisation); pulsar (MIT), crates/tokenizer/src/lib.rs -- read only, not copied
- Purpose: never let BPE see control markers; split input on them first.
- Inputs / outputs: text -> ids, with a list of (string, id, is_control) sorted longest-first.
- Pseudocode:
      load added tokens from tokenizer.json "added_tokens" (content, id, special flag)
        or from GGUF: token_type array (control/user-defined) plus heuristic
        "looks like <|...|> or <...>" names
      sort by length descending
      encode(text, allow_specials):
          if not allow_specials: return pretok_and_bpe(text)
          i = 0
          while i < len:
              find the earliest position p >= i where any added token matches
                (at that position, the longest one wins)
              pretok_and_bpe(text[i:p])
              if no match: break
              emit that token id; i = p + its length
      decode(id): if id is an added token -> its literal text (or nothing if
          caller asked to hide control tokens); else byte-map inverse
- Notes: Keep two flags per id: "added" (render literally on decode) and
  "control" (special=true; never legitimate visible output, e.g. role markers).
  Tags like <think> or <tool_call> are added but not control, and must render.
  User-supplied message text must be encoded with allow_specials = false;
  otherwise a user can inject role markers. Only the template renderer's own
  scaffolding may produce special ids. Naive earliest-match scanning is
  O(len * n_specials); for many specials use a first-byte bucket or a trie.
  Load-time validation from colibri worth keeping: reject negative ids, missing
  id/content fields, and implausibly large max ids (they size id->string arrays).
- Variants seen elsewhere: ds4 hard-codes a per-family list of marker strings
  and ids. pulsar builds the stop set from GGUF eos/eot/eom ids plus a list of
  well-known end-of-turn spellings found in the vocab, minus the BOS id.

## F-TOK-07: Chat template rendering: hard-coded per family
Origin: [colibri] [ds4] [pulsar] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus)
- Source: monolite (Apache-2.0), src/harmony.c; ds4 (MIT), ds4_server.c (per-family prompt renderers) and ds4.c; pulsar (MIT), crates/tokenizer/src/lib.rs (marker-based renderers) -- read only, not copied
- Purpose: produce the exact token sequence a model was trained on without a template engine.
- Inputs / outputs: list of (role, content, optional channel/tool data), flags (add generation prompt, thinking on/off) -> ids.
- Pseudocode (token-level renderer, preferred):
      out = family.prologue()                         # e.g. BOS when the family wants it
      for msg in messages:
          out += [ROLE_OPEN_ID]                       # special id, pushed directly
          out += encode(role_name, allow_specials=false)   # if the format spells roles as text
          if msg has channel: out += [CHANNEL_ID] + encode(channel)
          out += [MESSAGE_ID] + encode(msg.content, allow_specials=false) + [END_ID]
      if add_generation_prompt:
          out += [ROLE_OPEN_ID] + encode("assistant") + family.think_prefix(mode)
      return out
  Output side (incremental parser fed one id at a time):
      state: in_channel_header, current_channel, done
      on CHANNEL id: start collecting channel name
      on MESSAGE id: close header; channel = analysis | final | unknown
      on END id: if channel was final -> done else reset channel
      on RETURN / EOS / CALL ids: done
      otherwise: decode bytes and route to the reasoning buffer or visible buffer
- Notes: Pushing marker ids directly (rather than rendering text and
  re-tokenising) removes a whole class of bugs (user text containing marker
  strings, BPE merging across a marker boundary). ds4 renders TEXT and then
  re-tokenises with special matching; that is simpler to keep in sync with a
  reference Jinja template but needs explicit escaping of closing sentinels
  inside tool payloads (ds4 does escape the exact closing tag). Thinking
  defaults: pulsar defaults reasoning OFF for families that can suppress it,
  because raw think tokens in a chat UI look broken.
- Variants seen elsewhere: pulsar detects the family from which marker tokens
  exist in the vocab (checked in a deliberate order so a newer family cannot
  fall into an older renderer).

## F-TOK-08: Chat template rendering: mini-Jinja path
Origin: [pulsar]

Licence-of-origin: MIT (pulsar)
- Source: pulsar (MIT), crates/tokenizer/src/chat_template.rs -- read only, not copied
- Purpose: support arbitrary models by executing the model's own Jinja chat template.
- Inputs / outputs: template string, messages, add_generation_prompt, bos/eos strings, tools JSON, extra kwargs -> prompt text (then tokenised with specials allowed).
- Pseudocode:
      template = first hit of: GGUF metadata "tokenizer.chat_template";
                 local cache file; tokenizer_config.json of the model id;
                 a curated catalog keyed by model id (network optional)
      strip non-standard block tags the engine does not know (generation markers)
      env = new template environment
      register python-compat string methods (format, strip, startswith, ...)
      register tojson filter that accepts and ignores ensure_ascii, honours indent,
          and HTML-escapes < > & ' inside the JSON
      register raise_exception(msg) -> render error
      context = { messages, add_generation_prompt, bos_token, eos_token, tools,
                  plus caller kwargs merged at top level }
      text = render(template, context)
      ids = encode(text, allow_specials = true)
- Notes: A minimal C Jinja subset that covers real HF templates needs: for/if/
  elif/else/set, loop.index/first/last, namespace(), string concat (~), slicing,
  comparison/in, filters (trim, tojson, length, default, items), string methods
  (startswith, endswith, strip, split, replace, format), whitespace control
  ({%- -%}), raise_exception. That is a large surface; recommended only as a
  fallback behind the hard-coded renderers (F-TOK-07). Note the security
  difference: the Jinja path renders user content as text, so specials inside
  user content WILL be matched on re-tokenisation unless contents are escaped
  or the renderer marks user spans as no-special.
- Variants seen elsewhere: ds4 and monolite avoid Jinja entirely and hard-code
  every supported family.

## F-TOK-09: Streaming detokenisation with partial UTF-8 and stop strings
Origin: [colibri] [ds4] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4), PolyForm-NC-1.0.0 (apus)
- Source: ds4 (MIT), ds4_server.c (stream-safe length helpers); monolite (Apache-2.0), src/tokenizer.c (decode one token) -- read only, not copied
- Purpose: emit text deltas per generated token without ever sending half a UTF-8 character or a prefix of a stop string that may still complete.
- Inputs / outputs: accumulated output bytes buf[0..len), already-sent offset sent, final flag, stop strings -> number of bytes safe to send now.
- Pseudocode:
      expected_len(lead) = 1 if <0x80; 2 if C2..DF; 3 if E0..EF; 4 if F0..F4; else 1
      utf8_safe_end(buf, start, limit, final):
          if final: return limit
          p = limit; k = 0
          while p > start and k < 4 and buf[p-1] is a continuation byte: p--, k++
          if p == limit:                                 # last byte is not a continuation
              return limit-1 if expected_len(buf[limit-1]) > 1 else limit
          if p == start and buf[p] is a continuation: return start   # orphan tail
          lead = p-1
          return lead if (limit - lead) < expected_len(buf[lead]) else limit
      stop_safe_end(len) = len - (max_stop_len - 1)   (clamped at 0), or len if no stops
      on each token:
          buf += decode_bytes(token)
          if any stop string occurs in buf[from scan_start:]: truncate at earliest, finish
          end = utf8_safe_end(buf, sent, stop_safe_end(len(buf)), final=false)
          if end > sent: send buf[sent:end]; sent = end
      on finish: send buf[sent:] (final=true)
- Notes: ds4 explains why this matters beyond cosmetics: clients replace an
  incomplete sequence with U+FFFD and send the corrupted text back next turn,
  which breaks KV prefix-cache matching. Search for stop strings starting a
  little before the previously scanned end (max_stop_len-1 back) so a stop
  spanning two tokens is found. Decode-one-token API should return the token's
  byte length even if the caller's buffer is smaller, so callers can detect
  truncation (monolite's contract).
- Variants seen elsewhere: monolite's CLI writes raw bytes to stdout per token
  (terminals reassemble UTF-8 themselves); fine for a TTY, wrong for JSON/SSE.

## F-FMT-01: safetensors header parse and tensor index
Origin: [colibri] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/st.h -- read only, not copied
- Purpose: index every tensor of one or more .safetensors shards without reading tensor data.
- Inputs / outputs: list of shard paths -> table of {name, fd, absolute offset, nbytes, dtype, shape[<=8], numel}; name -> index hash.
- Format facts (public spec): file = 8-byte little-endian u64 N, then N bytes of
  UTF-8 JSON (may be right-padded with spaces), then the data buffer. The JSON is
  one object: each key is a tensor name mapping to {"dtype": string, "shape":
  array of non-negative ints, "data_offsets": [begin, end]} where offsets are
  byte positions RELATIVE TO THE START OF THE DATA BUFFER (i.e. absolute =
  8 + N + begin), end exclusive. Optional key "__metadata__" maps to a
  string->string object. Data is row-major (C order), little-endian. dtype
  strings: F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL, F8_E4M3, F8_E5M2,
  (newer: F8_E8M0, U16/U32/U64 and sub-byte types in some writers).
- Pseudocode:
      for shard in sorted(shards):
          size = file size; N = read_u64_le(offset 0)
          require size >= 8 and N <= size - 8 and N <= HEADER_CAP
          hdr = read N bytes at offset 8; parse JSON (strict mode)
          require root is object
          data_start = 8 + N
          for (name, desc) in root, skipping "__metadata__":
              require dtype is string, data_offsets is array of 2 numbers,
                      shape is array of length <= MAX_RANK
              require offsets are finite, integral, >= 0, < 2^63, begin <= end,
                      data_start + end <= size
              numel = product(shape) with overflow check per multiply
              if dtype is a float type: require end-begin == numel * elem_size
              if name already indexed from another shard:
                  consult model.safetensors.index.json "weight_map" to pick the
                  authoritative shard; if the index cannot decide, refuse
              append entry {fd, data_start+begin, end-begin, dtype, shape}
      build open-addressing name hash (power of two >= 2 * count)
- Notes: Every number in the header is untrusted (mirrors, partial downloads).
  colibri caps the header at 512 MiB and checks overflow on shape products
  because a crafted shape can wrap numel to a small value that then passes the
  byte-span cross-check. Raw/quantised byte tensors (U8/I8) legitimately have
  numel != nbytes in some converters; only enforce the equality for float
  dtypes. Missing-tensor errors should say which shard is probably missing
  (colibri inspects the index and the last shard because the final norm and
  output head live at the end and are the first casualties of a truncated
  download).
- Variants seen elsewhere: none in scope; monolite uses its own format.

## F-FMT-02: Range reads: pread loop, O_DIRECT twin, mmap view
Origin: [colibri] [ds4] [pulsar] [monolite] [slipstream] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar, slipstream), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: colibri (Apache-2.0), c/st.h, c/compat.h, c/iobench.c -- read only, not copied
- Purpose: read an exact byte range of a tensor (or a slice of an expert) as fast as the device allows, with correct fallbacks.
- Inputs / outputs: (fd, offset, nbytes, dst, dst_capacity, flags {direct, drop_cache}) -> bytes in dst or fatal error.
- Pseudocode:
      read_full(fd, dst, n, off):
          while n > 0:
              chunk = min(n, 1 GiB)                  # single pread caps near 2 GiB on Linux
              r = pread(fd, dst, chunk, off); retry on EINTR; fail on 0 / error
              dst += r; off += r; n -= r
      read_range(fd, off, n, dst, cap, direct, drop):
          require 0 <= off, n <= cap, off + n <= file_size
          if direct and a direct-io twin fd exists:
              pad = off mod 4096
              window = dst - pad                     # caller reserved 4096 bytes of slack before dst
              if window is 4096-aligned:
                  bulk = round_down(pad + n, 4096), clipped to file end
                  if one aligned pread(twin, window, bulk, off - pad) returns exactly bulk:
                      read the unaligned tail (< 4096 B) with the buffered fd
                      return
          read_full(buffered fd, dst, n, off)        # always-correct fallback
          if drop: advise kernel the range is no longer needed
      map_view(tensor): page-aligned read-only mmap of [off, off+n); return pointer
          adjusted by (off - aligned_off) plus the handle for unmapping
- Notes: Open a direct-I/O twin for each shard once at index time (thread-safe
  lookups later). On macOS the equivalent is a cache-bypass fcntl on a normal
  fd (no alignment constraints); on Windows an unbuffered handle. Windows needs
  ReadFile with an explicit offset (CRT read is racy and text-mode can corrupt
  0x0A bytes) and 64-bit offsets everywhere. Direct I/O is drive-dependent:
  colibri measured +34% decode on one NVMe box and 4.25 -> 9.69 GB/s in its
  microbench on another, but neutral-to-negative on QLC/DRAM-less or
  virtualised disks; keep it a switch, default off, and let tune decide.
- Variants seen elsewhere: ds4 mmaps the whole GGUF once and uses pointer
  arithmetic; private mapping for CPU, shared mapping only when the GPU must
  wrap slices without copying (ds4 notes a macOS VM accounting panic when
  streaming a huge file through a shared map on the CPU path).

## F-FMT-03: GGUF header, metadata and tensor directory
Origin: [ds4] [pulsar]

Licence-of-origin: MIT (ds4, pulsar)
- Source: ds4 (MIT), ds4.c (GGUF parsing section); pulsar (MIT), crates/gguf/src/lib.rs -- read only, not copied
- Purpose: parse a GGUF file's header, key/value metadata and tensor infos, leaving tensor bytes in place.
- Inputs / outputs: mapped bytes (or the head of the file) -> {version, kv table (key, type, value offset), tensor table (name, dims, type, rel/abs offset, nbytes), alignment, data_start}.
- Format facts (public spec, all little-endian):
      header:  magic u32 = bytes 'G','G','U','F' (0x46554747 read as LE u32)
               version u32 (current = 3; v1 used u32 counts and is obsolete)
               tensor_count u64
               kv_count u64
      string:  len u64, then len bytes UTF-8, NO terminator
      kv:      key string, value_type u32, value
      value types: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool (1 byte),
               8 string, 9 array, 10 u64, 11 i64, 12 f64
      array:   elem_type u32, count u64, then count values of elem_type
               (arrays of strings are common: tokenizer.ggml.tokens / merges;
                arrays of arrays are legal)
      tensor info: name string, n_dims u32, dims u64[n_dims] (dims[0] is the
               innermost / fastest-varying, i.e. row length), ggml_type u32,
               offset u64 (relative to data_start, multiple of alignment)
      alignment = metadata "general.alignment" (u32) if present, else 32
      data_start = round_up(end of tensor infos, alignment)
      tensor nbytes = ceil(numel / block_elems) * block_bytes (per ggml_type)
- Pseudocode:
      cur = cursor(bytes, size)          # every read checks n <= size - pos
      require magic, version == 3 (or accept 2 with identical layout)
      require kv_count <= remaining bytes (each entry needs >= 1 byte) before allocating
      for each kv: key = read_string; type = read_u32; record value_pos = pos;
          if key == "general.alignment" and type == u32 and value != 0: alignment = value
          skip_value(type, depth=0)
      skip_value(t, depth): require depth <= 8
          scalar sizes {1,1,2,2,4,4,4,1,-,-,8,8,8}; string: skip u64 len + len
          array: read elem_type, count; if scalar elem: skip count*size (overflow-checked)
                 else repeat skip_value(elem_type, depth+1)
      same pre-allocation sanity for tensor_count
      for each tensor: name; n_dims in 1..MAX_DIMS; dims with overflow-checked product;
          type; rel_offset; nbytes from the block table (unknown type -> warn, nbytes 0)
      data_start = round_up(pos, alignment)
      for each tensor: abs = data_start + rel (overflow check);
          require abs + nbytes <= file size; track the max nbytes (scratch sizing)
- Notes: Store metadata as (key, type, value offset) and decode lazily; typed
  getters (get_u32, get_string, get_array) verify the stored type. Block table
  for the common ggml types (type id: elems per block / bytes per block):
  0 f32 1/4, 1 f16 1/2, 2 q4_0 32/18, 3 q4_1 32/20, 6 q5_0 32/22, 7 q5_1 32/24,
  8 q8_0 32/34, 9 q8_1 32/40, 10 q2_K 256/84, 11 q3_K 256/110, 12 q4_K 256/144,
  13 q5_K 256/176, 14 q6_K 256/210, 15 q8_K 256/292, 16 iq2_xxs 256/66,
  17 iq2_xs 256/74, 18 iq3_xxs 256/98, 19 iq1_s 256/50, 20 iq4_nl 32/18,
  21 iq3_s 256/110, 22 iq2_s 256/82, 23 iq4_xs 256/136, 24 i8 1/1, 25 i16 1/2,
  26 i32 1/4, 27 i64 1/8, 28 f64 1/8, 29 iq1_m 256/56, 30 bf16 1/2,
  39 mxfp4 32/17. (Verify iq1_s and iq4_nl against upstream ggml before use:
  the reference table we read lists different figures for those two rows than
  the upstream ggml definitions we know; treat upstream ggml as authoritative.)
  Split GGUFs ("-0000i-of-0000N" names) carry split metadata; pulsar resolves
  the sibling shard list from the first file's name.
- Variants seen elsewhere: pulsar parses from the first N bytes of an mmap and
  materialises metadata into a typed map up front (simpler API, more memory
  for 200k-entry token arrays).

## F-FMT-04: Tokenizer from GGUF metadata vs tokenizer.json
Origin: [colibri] [ds4] [pulsar]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, pulsar)
- Source: ds4 (MIT), ds4.c (vocab load); pulsar (MIT), crates/tokenizer/src/lib.rs; colibri (Apache-2.0), c/tok.h -- read only, not copied
- Purpose: build the same tokenizer object from either container.
- Inputs / outputs: GGUF kv table or tokenizer.json -> vocab hash, merges hash, specials, BOS/EOS/EOT ids, pre-tokeniser family, add-BOS flag.
- Pseudocode:
      if GGUF:
          tokens  = string array "tokenizer.ggml.tokens" (id = index)
          merges  = string array "tokenizer.ggml.merges" (rank = index; "left right")
          types   = optional i32 array "tokenizer.ggml.token_type" (normal/control/user-defined/byte...)
          bos/eos/eot/eom = u32 "tokenizer.ggml.<name>_token_id" when present
          add_bos = bool "tokenizer.ggml.add_bos_token" (else infer from tokenizer model kind)
          family  = string "tokenizer.ggml.pre"  -> choose splitter (F-TOK-04/05)
          template = string "tokenizer.chat_template" (for F-TOK-08)
      else tokenizer.json:
          vocab = model.vocab (string -> id), merges = model.merges (string or pair form)
          added_tokens[] with id, content, special flag
          family = inspect pre_tokenizer: bare ByteLevel with regex -> GPT-2 rules;
                   Split pattern containing case classes -> o200k rules;
                   containing a Han class -> Kimi rules; otherwise cl100k rules
          empty merges list -> rank mode (F-TOK-03)
- Notes: Detecting the family by inspecting the regex for marker substrings is
  what colibri does; more robust is to compare the whole pattern string against
  a small table of known patterns and refuse (with a clear message) on an
  unknown one, because a silent mismatch degrades quality rather than failing.
  Unknown "pre" names in GGUF should likewise be an error, not a default.
- Variants seen elsewhere: pulsar falls back to a default family for unknown
  pre names; ds4 selects by compiled-in model family.

## F-HW-01: Memory detection (total, available, cgroup, OS specifics)
Origin: [colibri] [ds4] [pulsar] [monolite] [slipstream]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar, slipstream)
- Source: slipstream (MIT), engine/src/peregrine_system_memory.c; colibri (Apache-2.0), c/compat.h and c/resource_plan.py; ds4 (MIT), ds4_linux_memory.h; monolite (Apache-2.0), src/monolite.c -- read only, not copied
- Purpose: one conservative {total, available, known} snapshot that every planner uses.
- Inputs / outputs: none -> {total_bytes, available_bytes, available_known}.
- Pseudocode:
      linux:
          total = phys_pages * page_size
          available = /proc/meminfo "MemAvailable" (kB means KiB: * 1024)
          for layout in [v2: memory.max / memory.current / memory.stat key inactive_file,
                         v1: memory/memory.limit_in_bytes / usage_in_bytes / total_inactive_file]:
              cap = read limit; skip if "max", 0, or > 2^63 (v1 "unlimited" sentinel)
              used = read usage (0 if unreadable)
              cap_avail = min(cap, max(0, cap - used) + inactive_file)
              total = min(total, cap); available = min(available, cap_avail)
              break
          (root of the cgroup tree overridable by an env var, for tests)
          optional: subtract CmaFree when sizing pinned GPU host buffers (ds4)
      macOS: total = sysctl hw.memsize; available = (free + inactive [+ speculative
             + purgeable]) pages * page size
      windows: GlobalMemoryStatusEx; available = min(avail_phys, avail_page_file)
             (commit limit, not just physical)
      unknown available -> available_known = false (planner must refuse, F-HW-05)
- Notes: Use MemAvailable, never MemFree: free excludes reclaimable page cache.
  memory.current inside a cgroup counts page cache, so add back inactive file
  pages or availability is badly understated. colibri found on Windows that
  budgeting from physical availability alone let the engine allocate until the
  commit charge was refused even though physical RAM was free; hence the
  min(phys, commit). Keep one implementation shared by the engine and the
  doctor/plan commands so their numbers agree (colibri had three engines with
  three different definitions before consolidating).
- Variants seen elsewhere: monolite uses MemAvailable only (no cgroup);
  colibri's Python planner uses MemAvailable plus vm_stat parsing on macOS.

## F-HW-02: CPU / SIMD / GPU inventory
Origin: [colibri] [pulsar] [monolite]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (pulsar)
- Source: monolite (Apache-2.0), src/monolite.c (doctor); colibri (Apache-2.0), c/resource_plan.py, c/telemetry.h; pulsar (MIT), crates/kernels/src/lib.rs -- read only, not copied
- Purpose: decide which kernels can run and report it.
- Inputs / outputs: none -> feature bitset, physical cores, sockets, GPU list {index, name, total, free, unified?}.
- Pseudocode:
      x86: use compiler cpu-feature builtins or cpuid leaves 1 and 7 (plus xgetbv
           to confirm OS saves AVX/AVX-512 state): avx2, fma, f16c, avx512f/bw/vl,
           avx512_vnni, avx512_bf16, avx_vnni, amx (needs an OS permission request
           on Linux before first use)
      arm64: getauxval hwcaps (asimddp, i8mm, sve, bf16); macOS: sysctl hw.optional.*
      physical cores: count unique (package, core) pairs from sysfs topology,
           not online logical CPUs; sockets = unique package ids
      GPUs: query the runtime if linked; otherwise parse vendor CLI output
           (CSV mode) as a best-effort report
      mark a GPU "placement-qualified" only when its free-memory figure is
           trustworthy on this platform; report others but do not plan with them
      dispatch: choose kernel tier once at startup from the bitset
- Notes: Print missing features with a marker and an explicit warning naming
  which kernel needs them (monolite). colibri warns loudly when physical-core
  detection falls back to 1, because a silent 1 halves throughput. Hidden trap
  from colibri: on unified-memory GPUs, VRAM and RAM budgets must be jointly
  constrained or the plan double-counts the same pages. Warn when a GPU is
  already more than ~25% occupied, because a plan made while another engine
  holds VRAM describes nothing.
- Variants seen elsewhere: pulsar measures each GPU's real DRAM bandwidth
  (F-HW-04) rather than trusting the model name.

## F-HW-03: Disk bandwidth probe that does not measure RAM
Origin: [colibri] [slipstream]

Licence-of-origin: Apache-2.0 (colibri), MIT (slipstream)
- Source: colibri (Apache-2.0), c/colibri.c (storage probe), c/iobench.c, docs/ENVIRONMENT.md -- read only, not copied
- Purpose: measure the model volume's real random-read throughput quickly at startup and cache the verdict.
- Inputs / outputs: model directory -> GB/s or "unknown" (+ reason).
- Pseudocode:
      cache_file = <model dir>/<hidden probe file>
      if cache parses under a strict grammar ("v2 <gbs> <device id>", plain digits,
         0 < gbs < 1000) and device id == stat(model dir).st_dev: return gbs
      shard = largest weight file
      if allocated_blocks*512 < file_size (sparse / still downloading): veto SPARSE
      if file offers < 64 MiB of 4 MiB windows: veto SMALL
      open with cache bypass, readahead off
      map file (address space only), get page residency vector (mincore)
      cold_windows = 4 MiB windows whose pages are all non-resident
      if total cold bytes < 64 MiB: veto RESIDENT
      buf = aligned 4 MiB
      until 0.35 s elapsed or 300 MiB read: pread a random cold window
      gbs = bytes / seconds; write cache file atomically; return gbs
      on veto: print one line naming the actual reason; write nothing; retry next start
  Offline microbench (doctor/tune): N threads, one fd per thread, random
      4 KiB-aligned offsets from a fixed seed, block size = typical expert size,
      buffered vs direct, report GB/s and ms per block.
- Notes: Measured in colibri: buffered reads of a warm file report 23-97 GB/s
  while the real cache-bypassing number was 14 GB/s; and cache bypass alone is
  not enough, because already-resident pages are still served from RAM (a warm
  shard measured 22.1 GB/s). Hence residency steering plus a veto. Block size
  must match the engine's real read size (fine-grained MoE experts can be under
  1 MiB). One fd per thread matters on Windows where a synchronous handle
  serialises reads. Device-id identity is best effort (ids can be recycled).
- Variants seen elsewhere: monolite skips probing and uses constants fitted
  from a measured sweep (F-HW-06).

## F-HW-04: Measured-bandwidth role assignment across devices
Origin: [colibri] [pulsar] [slipstream]

Licence-of-origin: Apache-2.0 (colibri), MIT (pulsar, slipstream)
- Source: pulsar (MIT), crates/kernels/src/lib.rs (VRAM bandwidth probe), crates/engine/src/lib.rs (device roles, FFN split), crates/engine/src/hybrid_miss.rs -- read only, not copied
- Purpose: give each device the job its measured bandwidth or capacity suits, and split work proportionally to measured speed.
- Inputs / outputs: device list -> per-device measured read GB/s; role map; split fractions.
- Pseudocode:
      probe(dev): allocate a 512 MiB buffer (larger than any L2), run one
          warm-up streaming-read kernel, then best of 3 timed reads; GB/s = size / time
      roles:
          primary (fastest link to host)  -> expert streaming / misses
          other devices                   -> capacity roles (attention stack,
                                             KV, resident layers), since residency
                                             is paid once at load
      two-way split of a layer's FFN:
          frac = bw_a / (bw_a + bw_b)  (0.5 if either probe failed)
          cut  = round_to_nearest_block(n_ff * frac), clamped to [n_ff/4, 3n_ff/4]
      three-way: shares proportional to the three bandwidths, same rounding
      host vs link: PCIe and host DRAM bandwidths come from env overrides with
          defaults; if host DRAM is faster than the link, compute host-cached
          experts on the CPU instead of copying them to the GPU
- Notes: pulsar's reasons: every sublayer is a barrier, so an even split on
  unequal cards runs at the slow card's pace (nsys showed the slower card at
  275 of 288 GB/s while the faster idled at 61%). Rounding to the nearest
  quant block rather than truncating mattered: one block of skew cost 47.8 ->
  46.2 tok/s on identical cards. The probe buffer must exceed L2 (Ada/
  Blackwell L2 up to 32 MB made a 64 MB probe report ~390 GB/s on every card).
  Device selection by free memory alone picked the slower card over a 69 MiB
  difference; sort by bandwidth first.
- Variants seen elsewhere: colibri splits multi-drive reads by a probed or
  user-given weight per drive.

## F-HW-05: Admission planner with typed refusal reasons
Origin: [colibri] [ds4] [slipstream]

Licence-of-origin: Apache-2.0 (colibri), MIT (ds4, slipstream)
- Source: slipstream (MIT), engine/src/peregrine_admission.c/.h, engine/src/peregrine_runtime.cpp, engine/src/peregrine_system_memory.c -- read only, not copied
- Purpose: decide before loading whether the model fits, how big the expert cache is, and refuse with an actionable message.
- Inputs / outputs: {total, available, known, model_bytes, expert_bytes, kv, overhead, staging, headroom, requested_cache (0 = auto), already_allocated} -> {status OK/WARN/REFUSE, mode RESIDENT/STREAMING, reason, ceiling, mandatory, cache, streamed, resident, free_after}.
- Pseudocode:
      default_headroom(total):
          macOS: min(3 GiB, total/4)
          other: clamp(total/8, 512 MiB, 3 GiB); if >= total: total/2
      plan:
          all additions overflow-checked; invalid inputs -> error + REFUSE
          if total <= headroom: REFUSE(HEADROOM_EXCEEDS_RAM)
          ceiling   = total - headroom
          mandatory = (model - experts) + kv + overhead + staging
          if mandatory > ceiling: REFUSE(MODEL_EXCEEDS_CEILING)
          room = ceiling - mandatory
          if requested == 0: cache = min(experts, room)
          else: cache = min(requested, experts); if cache > room: REFUSE(CACHE_TOO_LARGE)
          streamed = experts - cache; mode = STREAMING if streamed > 0 else RESIDENT
          if not available_known: REFUSE(AVAILABLE_UNKNOWN)          # fail closed
          future = resident - already_allocated
          need_now = future + max(headroom, 1 GiB)
          status = WARN(RESERVE_AT_RISK) if available < need_now else OK
      after plan: slots = cache / max_expert_bytes; require slots >= n_layers
          (each layer needs one slot for the expert it is computing)
      staging = n_layers * io_threads * max_expert_bytes (admit it, so raising
          I/O parallelism can never silently eat the reserve)
      live re-check after KV/compute allocations: available >= headroom
- Notes: Refusal messages state what was measured, what was wanted, and which
  knob to change, one message per reason (slipstream: a single generic message
  left users "holding a refusal with no idea which number to change"). Examples
  of the shape (our wording): "model needs X MiB resident before any cache but
  only Y MiB remains of Z after the R MiB reserve: too large for this host";
  "plan fits the machine but not the moment: X MiB free now, Y MiB planned,
  would break the R MiB reserve; close something or lower the cache";
  "cache of X MiB too small: L layers need one S KiB slot each, use at least N
  MiB or omit the flag". When suggesting a number, round UP at the printed
  precision so the suggestion itself is accepted. The macOS 3 GiB reserve was
  qualified on a 36 GiB machine; 1.5 GiB stalled at under 0.1 tok/s from GPU
  residency thrash with zero swap-outs.
- Variants seen elsewhere: colibri's engine refuses (exit code 2) when the
  projected peak exceeds MemAvailable unless an override env var is set.

## F-HW-06: Cache budget planner and throughput prediction (plan command)
Origin: [colibri] [monolite] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: monolite (Apache-2.0), src/monolite.c (plan); colibri (Apache-2.0), c/resource_plan.py -- read only, not copied
- Purpose: predict hit rate and tokens/s for candidate cache sizes, and derive tier budgets.
- Inputs / outputs: model geometry, expert stride, optional recorded router counts, budget -> table of (cache size, slots, % cached, predicted residency, disk GB/token, tok/s); tier plan {VRAM hot, RAM warm, disk cold}.
- Pseudocode:
      coverage curve cum[k] = mean over layers of the share of accesses served by
          the k+1 most used experts (from recorded per-(layer,expert) counts, else
          a built-in curve from a reference model read at the same normalised rank)
      for cache_mb in [512, 1024, 2048, 4096, 8192, 12288, 16384, budget]:
          slots = clamp(cache_mb / stride, 8, layers*experts)
          residency = lru_eff * interp(cum, slots / (layers*experts))   # lru_eff = 1 with own counts
          disk_gb = (1 - residency) * layers * top_k * stride
          ram_gb  = dense_read_per_token + residency * layers * top_k * stride
          t = ram_gb / RAM_BW + DISK_COST * disk_gb ; print 1/t
      dense_read_per_token = dense bytes - embedding table + one embedding row
          (output head is read in full every token)
      tier plan (colibri):
          runtime = fixed overhead + small multiple of max expert + kv * slots + workspace
          vram_budget = sum(max(0, free_vram - 2 GB)) over qualified GPUs, minus the
                        dense trunk if it is placed on the first GPU
          ram_budget  = explicit or 0.88 * available (unified memory: jointly clamped)
          cache = ram_budget - dense - runtime; hot = vram experts; warm = min(rest, cache)
          cold = rest; hit = (hot + warm) / experts
          bottleneck = disk if cold > 0; mixed if warm and GPU; compute if hit >= 0.99;
                       else memory bandwidth
- Notes: monolite fitted t_token = ram_bytes / 25.2 GB/s + 0.5703 s/GB * disk_bytes
  to three measured points (within 3.5%); the disk constant is 1.75 GB/s effective
  vs 1.99 GB/s raw, i.e. prefetch overlap bought ~12%, not 100%. LRU achieved
  0.91-0.96 of the optimal top-C hit rate (0.93 used). Extrapolating another
  model's curve under-predicted a bigger model by 19-32%; using its own recorded
  counts brought error within 16%. So: record counts during runs and feed them
  back. Defaults differ: monolite suggests 60% of MemAvailable (dense weights
  are mmapped on top of the cache), colibri 88% of available.
- Variants seen elsewhere: colibri's plan also emits "next actions" (measure
  storage, profile) instead of inventing numbers it has not measured.

## F-HW-07: doctor / plan / tune command structure
Origin: [colibri] [monolite]

Licence-of-origin: Apache-2.0 (colibri, monolite)
- Source: monolite (Apache-2.0), src/monolite.c; colibri (Apache-2.0), c/doctor.py, c/resource_plan.py, docs/SETTINGS.md -- read only, not copied
- Purpose: three read-only/measure commands that share the probing code.
- Inputs / outputs: CLI args -> human report (and --json).
- Pseudocode:
      doctor: CPU (cores, SIMD with missing marked), memory (total, free, cache,
          AVAILABLE flagged as the budget source), accelerators (detected and
          whether this build can use them), storage (free space, cached probe
          state and why it is missing), models found (layers, experts, dense/expert
          sizes, missing companion files), suggested cache budget
      plan MODEL [--budget] [--counts]: F-HW-06 table + tier plan + warnings
      tune MODEL: run short measured benchmarks of the few knobs that are
          hardware-dependent (direct I/O, I/O threads, prefetch depth, kernel
          tier); keep only quality-preserving settings; save a profile keyed by
          (model, machine) that later runs load unless disabled
- Notes: tune must only recommend knobs the selected engine actually reads
  (colibri found it was recommending settings for one engine that another
  ignored). Plans are advisory: applying a plan sets env defaults without
  overriding anything the user set explicitly (F-CFG-01).
- Variants seen elsewhere: ds4 has no planner; it validates the model and fails
  early on unknown layouts.

## F-CFG-01: Configuration precedence and derived defaults
Origin: [colibri] [ds4] [monolite]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4)
- Source: colibri (Apache-2.0), docs/ENVIRONMENT.md, docs/SETTINGS.md, c/resource_plan.py (applying a plan), c/coli; monolite (Apache-2.0), src/monolite.c; ds4 (MIT), ds4_server.c (option parsing) -- read only, not copied
- Purpose: one predictable resolution order for every knob.
- Inputs / outputs: argv, environment, saved tune profile, computed plan -> resolved config struct (+ a printed "effective config" header).
- Pseudocode:
      value(knob) = first defined of:
          explicit CLI flag
          environment variable (single documented prefix)
          saved tune profile for (model, machine)
          plan-derived default (from F-HW-06)
          platform default (e.g. different cache cap on unified-memory hosts)
          compiled default
      apply_plan(env): for each planned key: set only if not already set
      validate every numeric with full-string parse, finite check and range;
          unknown flags are an error, not ignored
      print the resolved config and the source of each non-default value
- Notes: Lessons recorded in colibri: (a) a single env var read by several
  binaries with different meanings is a trap; document which program reads
  which variable and warn on variables that the running binary ignores;
  (b) never use a generic name that collides with OS conventions (a TEMP
  variable for temperature collided with the Windows temp-directory variable);
  (c) do not set thread-affinity env vars from a launcher if the engine sets
  its own with no-overwrite semantics: the launcher's value wins and in one
  case pinned all threads to one core. monolite validates temperature, min-p
  and seed with strict parsing and rejects unknown options. ds4 defaults
  context to 32k and has separate "context size" and "default max tokens".
- Variants seen elsewhere: monolite is flag-only (plus one telemetry env var);
  colibri is env-first with a launcher translating flags to env.

## F-SRV-01: Minimal HTTP/1.1 request reader
Origin: [colibri] [ds4] [monolite] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4), no-licence (flashmoe)
- Source: ds4 (MIT), ds4_server.c (request reading); monolite (Apache-2.0), src/server.c -- read only, not copied
- Purpose: read exactly one request (headers + Content-Length body) safely.
- Inputs / outputs: connected socket -> {method, path (query stripped), body bytes} or failure.
- Pseudocode:
      buf = empty
      while header terminator not found and len(buf) < 64 KiB:
          recv up to 4 KiB (retry EINTR; 0 or error -> fail); append
          look for CRLFCRLF (also accept bare LFLF)
      parse request line: METHOD SP TARGET SP VERSION (bounded copies)
      cut TARGET at '?'
      scan header lines case-insensitively for Content-Length; parse strictly
          (digits only); reject negative or > MAX_BODY
      reject Transfer-Encoding: chunked with 411/501 (or implement it)
      if "Expect: 100-continue" present: send "HTTP/1.1 100 Continue\r\n\r\n"
      read until len(buf) >= header_end + content_length
      body = copy, NUL-terminated
  Responses: status line, Content-Length, Content-Type, optional CORS headers,
      "Connection: close"; write with a send-all loop; ignore SIGPIPE (or use
      no-signal send flags) so a client hang-up is not fatal
- Notes: Limits seen: ds4 64 KiB headers / 64 MiB body; monolite 4 MiB body.
  Set receive/send timeouts on client sockets. Install signal handlers without
  restart semantics so accept() returns EINTR and the server can stop
  (monolite: the default signal() set restart and only SIGKILL worked). Bind to
  127.0.0.1 by default; map "localhost" explicitly. Set TCP_NODELAY so SSE
  chunks are not delayed by Nagle. Routes: POST /v1/chat/completions,
  POST /v1/completions, GET /v1/models, GET /v1/models/{id}, GET /health,
  OPTIONS (204 for CORS preflight); 404 otherwise.
- Variants seen elsewhere: none use keep-alive; one request per connection is
  sufficient for OpenAI clients and removes pipelining edge cases.

## F-SRV-02: OpenAI chat request/response shape and SSE streaming
Origin: [colibri] [ds4] [pulsar] [monolite] [apus] [flashmoe]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4, pulsar), PolyForm-NC-1.0.0 (apus), no-licence (flashmoe)
- Source: ds4 (MIT), ds4_server.c (request parsing, SSE helpers, usage); monolite (Apache-2.0), src/server.c -- read only, not copied
- Purpose: speak enough of the OpenAI API for common clients and gateways.
- Inputs / outputs: request JSON -> JSON response or text/event-stream.
- Request fields to honour: model, messages[] {role: system|developer|user|
  assistant|tool, content: string OR array of parts {type:"text", text}},
  stream, stream_options.include_usage, max_tokens / max_completion_tokens,
  temperature, top_p, top_k, min_p, seed, stop (string or array), tools,
  tool_choice, user (useful as a session key). Unknown fields: skip.
- Pseudocode:
      validate: messages non-empty; numeric ranges (min_p in [0,1), seed integer
          in [0, 2^53-1]); clamp max_tokens to context; bounded copy of model name
      render prompt (F-TOK-07); if prompt_tokens >= ctx: 400 with
          code "context_length_exceeded", type invalid_request_error, param
          "messages", plus prompt token count and ctx
      non-stream: {id:"chatcmpl-...", object:"chat.completion", created, model,
          choices:[{index:0, message:{role:"assistant", content, [reasoning_content],
          [tool_calls]}, finish_reason:"stop"|"length"|"tool_calls"}],
          usage:{prompt_tokens, completion_tokens, total_tokens,
                 prompt_tokens_details:{cached_tokens}}}
      stream: headers 200, Content-Type text/event-stream, Cache-Control no-cache
          first event:  data: {..."object":"chat.completion.chunk", choices:[{index:0,
                         delta:{role:"assistant"}, finish_reason:null}]}\n\n
          per delta:    same with delta:{content:"..."} (text from F-TOK-09)
          final:        delta:{} and finish_reason set
          if include_usage: one chunk with choices:[] and usage
          terminator:   data: [DONE]\n\n
          errors mid-stream: "event: error" + data: {error:{message,type}}
      errors (non-stream): HTTP 4xx/5xx with {error:{message, type, [param], [code]}}
- Notes: Build each SSE chunk in a buffer sized from the payload, never a fixed
  buffer: a truncated chunk is malformed JSON and clients drop the stream
  (monolite). JSON-escape every string from the model or the request. Reasoning
  and tool-call text should be routed into reasoning_content / tool_calls
  deltas, not shown as content. cached_tokens must count only tokens actually
  served from the KV prefix cache (ds4 keeps newly prefilled tokens as a
  separate extension field so clients do not over-count).
- Variants seen elsewhere: ds4 also serves /v1/responses and an Anthropic-style
  /v1/messages from the same core; monolite adds a token-count endpoint and a
  status file for progress polling.

## F-SRV-03: Single-model request queueing, cancellation, prefix reuse
Origin: [ds4] [monolite] [apus]

Licence-of-origin: Apache-2.0 (monolite), MIT (ds4), PolyForm-NC-1.0.0 (apus)
- Source: monolite (Apache-2.0), src/server.c; ds4 (MIT), ds4_server.c (job queue, worker, disconnect polling) -- read only, not copied
- Purpose: serve one model with one KV cache safely under concurrent clients.
- Inputs / outputs: accepted connections -> serialized generation jobs.
- Pseudocode (thread-per-connection + one worker, ds4 style):
      accept loop: for each fd: set timeouts; spawn client thread
      client thread:
          read + parse request (F-SRV-01/02); cheap GETs answered directly
          job = {fd, request, done=false, mutex, cond}
          enqueue(job) under server mutex (refuse 503 if stopping); signal worker
          while not job.done:
              timed wait 100 ms on job.cond
              if peer disconnected (poll: hangup/error, or readable and recv == 0):
                  mark job cancelled (worker checks between tokens)
          close fd
      worker: loop { job = dequeue (blocking); generate(job) writing directly to
          job.fd (socket set non-blocking so a slow client fails instead of
          stalling everyone); mark done; signal }
      prefix reuse inside generate:
          if request session key equals the cached session key:
              reuse = common prefix length of new prompt ids vs cached ids
              if reuse == prompt length: reuse = prompt length - 1   # recompute last logits
          else reuse = 0 and adopt the new key
          prefill from position reuse; record ids as they enter the cache
  Simplest variant (monolite): single-threaded accept -> handle -> close; a
      second client waits in the listen backlog.
- Notes: monolite argues single-request is the honest behaviour for one KV
  cache and one expert cache: interleaving two conversations halves each
  cache. Its cost: health/status endpoints are unreachable mid-request, so it
  writes a status JSON file via write-temp-then-rename for progress polling. A
  bounded queue with 429/503 when full is better than an unbounded backlog.
  Always reserve at least one free context slot for generation (ds4 rejects
  prompt_len >= ctx). On prefill failure (I/O error paging experts) return 503
  with the specific error.
- Variants seen elsewhere: ds4 optionally runs several KV slots with a router
  that picks the slot with the longest reusable prefix and evicts stale ones.

## F-JSON-01: Minimal JSON parser design for C
Origin: [colibri] [ds4] [monolite] [apus]

Licence-of-origin: Apache-2.0 (colibri, monolite), MIT (ds4), PolyForm-NC-1.0.0 (apus)
- Source: colibri (Apache-2.0), c/json.h; monolite (Apache-2.0), src/json.h, src/json.c; ds4 (MIT), ds4_server.c (cursor helpers) -- read only, not copied
- Purpose: one small parser for safetensors headers, tokenizer.json, config.json and HTTP bodies.
- Inputs / outputs: (text, len) -> tree or NULL + error message + byte offset.
- Pseudocode:
      value = {type: null|bool|number|string|array|object; bool; double;
               string (ptr, len, NUL-terminated); array (items[], n);
               object (keys[], vals[], n) in document order}
      all nodes and strings allocated from an arena (linked 64 KiB blocks,
          16-byte aligned); one free call releases everything
      parse_value(depth):
          skip JSON whitespace only (space, tab, CR, LF)
          '{' / '[': depth+1, refuse beyond MAX_DEPTH; grow child arrays by doubling
              object key must be a string; require ':'; require ',' or close
          '"': first pass finds the closing quote honouring backslashes; allocate
              raw_len + 4 bytes (decoded never longer than source); decode escapes
              \" \\ \/ \b \f \n \r \t and \uXXXX (4 hex digits required);
              high surrogate followed by \u low surrogate -> combined codepoint;
              lone surrogate -> U+FFFD (or error in strict mode); raw control
              chars < 0x20 -> error; a key containing \u0000 -> error
          number: validate the grammar by hand (optional '-', '0' or [1-9][0-9]*,
              optional fraction with >=1 digit, optional exponent with >=1 digit)
              then convert with strtod over exactly that span (strtod alone accepts
              NaN/Infinity/hex)
          literals true/false/null by exact match
      top level: after value, only whitespace may remain, else error
      helpers: get(obj,key) linear scan; typed getters with fallback that return
          the fallback for both missing and wrong-typed fields; integer getter
          rejects non-integral, non-finite, out-of-range values
      writer: escape quotes, backslash and all control chars (\u00XX), with an
          snprintf-like return of the needed size
- Notes: colibri bugs fixed along the way, worth designing out: reading past
  the NUL when a key was not actually at a quote; a growable string buffer
  instead of a fixed 64 KiB one (long tokenizer strings were truncated); an
  arena that reallocated and invalidated earlier string pointers (use linked
  blocks, never realloc). Depth caps seen: 1024 (colibri), 256 for the server
  skip path (ds4). For 200k-entry vocab objects, linear get() is fine because
  the vocab is walked once into a hash; avoid repeated lookups in hot paths.
  Duplicate keys: define a policy (last wins) and make sure a re-parse into an
  already-filled field cannot double-free (ds4 nulls the output first).
- Variants seen elsewhere: ds4's server does not build a tree: it walks the
  request with a cursor, extracting known keys and recursively skipping the
  rest (cheaper, but each endpoint re-implements traversal).

---

## Recommended synthesis

- Tokenizer core: one byte-level BPE with two merge modes (merges-list ranks
  and tiktoken concatenation ranks), using contiguous-slice symbols and
  neighbour-only rank recomputation (F-TOK-02/03).
- Pre-tokenisers: hand-written matchers per family (GPT-2, cl100k, o200k,
  plus model-specific ones added on demand) driven by a generated Unicode
  class-range table; select the family by exact pattern/name match and refuse
  unknown ones; gate every family with a parity corpus (F-TOK-04/05, F-FMT-04).
- Specials: longest-first atomic matching only for renderer-produced text;
  user content always encoded with specials disabled; keep separate "added"
  and "control" flags (F-TOK-06).
- Chat templates: hard-coded token-level renderers for supported families as
  the default path; a Jinja subset only as an optional fallback (F-TOK-07/08).
- Streaming: hold back incomplete UTF-8 tails and the last (max_stop_len - 1)
  bytes; release on finish (F-TOK-09).
- File readers: a bounds-checked cursor for GGUF (lazy metadata, alignment from
  metadata with default 32, block-size table) and a hardened safetensors
  indexer (header cap, overflow-checked shapes, byte-span cross-check, index.json
  for duplicates); both produce the same tensor descriptor (F-FMT-01/03).
- I/O: one range-read primitive with chunked pread, optional aligned direct-I/O
  twin with buffered fallback, and an mmap view; direct I/O off by default and
  decided by tune (F-FMT-02).
- Memory snapshot: MemAvailable, min'd with cgroup v2/v1 limits (+inactive
  file pages), commit-aware on Windows, explicit "unknown" state; shared by
  engine and doctor/plan (F-HW-01).
- Admission: slipstream-style pure, overflow-checked planner with typed
  reasons, fail-closed on unknown availability, platform headroom default, and
  one actionable message per reason with rounded-up suggestions (F-HW-05).
- Planning: monolite-style calibrated throughput model fed by recorded
  per-expert counts, plus colibri-style hot/warm/cold tier split and bottleneck
  label; never invent unmeasured bandwidth (F-HW-06).
- Bandwidth: cache-steered, veto-capable disk probe cached per volume; GPU DRAM
  probe with an L2-defeating buffer; assign roles and split ratios by measured
  bandwidth with clamps (F-HW-03/04).
- Commands: doctor (inventory + why), plan (prediction table), tune (measure
  only hardware-dependent, quality-preserving knobs and save a profile) (F-HW-07).
- Config: flag > env (one prefix, no generic names) > tune profile > plan >
  platform default > compiled default; strict numeric parsing; print the
  effective config with sources (F-CFG-01).
- Server: thread-per-connection reader, one generation worker with a bounded
  queue, disconnect polling and cancellation, session-keyed prefix reuse,
  Connection: close, TCP_NODELAY, payload-sized SSE chunks, OpenAI error shapes
  including context_length_exceeded (F-SRV-01/02/03).
- JSON: arena-backed strict tree parser with depth cap, full escape and
  surrogate handling, hand-validated numbers, typed getters with fallbacks, and
  a matching escaper; use it everywhere instead of ad-hoc scanners (F-JSON-01).
