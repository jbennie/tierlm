# Tokenizer for Qwen3 (byte-level BPE + Qwen chat template)

Scope: encode/decode for Qwen2/Qwen3-family checkpoints (`tokenizer.json` with an
ordered merges list, ByteLevel decoder, no prefix space) and the Qwen3 chat
template without the tools branch. Implements the P-127 parity gate:
token-for-token against the reference tokenizer on a mixed corpus (code, CJK,
numbers, CRLF), chat rendering byte-exact against the reference template output.

Written after reading the cited implementations (read only, nothing copied) and
the public spec of the target checkpoint's `tokenizer.json` /
`tokenizer_config.json` (file-format facts are public and stated precisely).
Every rule below was additionally validated against the reference toolchain
(tokenizers 0.23.2, transformers apply_chat_template) with a throwaway probe
harness before porting to C: 100/100 encode, 100/100 decode, 20/20 template.

Topic code: TOK.

---

## F-TQ-01: Loading `tokenizer.json` (BPE, merges-list mode)
Origin: [colibri] [ds4] [pulsar] [apus] [monolite]

- Source: colibri (Apache-2.0), c/tok.h; ds4 (MIT), ds4.c; pulsar (MIT),
  crates/tokenizer/src/lib.rs; monolite (Apache-2.0), src/tokenizer.c --
  read only, not copied. Target-format facts: Qwen3-30B-A3B tokenizer.json
  (BPE model, 151,643 vocab entries, 151,387 merges, ByteLevel decoder with
  add_prefix_space=false, NFC normalizer, 26 added tokens).
- Purpose: build the in-memory tables one encode/decode session needs, and
  reject files that are not of the supported shape.
- Inputs / outputs: directory containing `tokenizer.json` -> handle.
- Pseudocode:
      parse with the existing JSON reader; require:
        model.type == "BPE", model.vocab object (string -> id),
        model.merges non-empty list, decoder.type == "ByteLevel",
        decoder.add_prefix_space == false,
        (normalizer null or "NFC" -- NFC is applied to the input, see notes)
      vocab  : open-addressed hash, key = vocab string bytes, value = id + 1
               (0 = empty slot), FNV-1a, table = next power of two >= 2 * n
      merges : same shape, key = left + 0x00 + right (NUL can never appear in
               a byte-level token), value = rank + 1
      added  : list of (content bytes, id, special), validated (no negative
               ids, no duplicate ids), kept in file order (ties in longest
               matching resolve to the earlier list entry = lower id); the
               26 entries of the target family span ids 151643..151668 and
               include bos (151643, special), the role markers im_start
               151644 / im_end 151645 (special), the think pair 151667/151668,
               the tool_response pair 151665/151666 and the tool_call pair
               151657/151658 (all non-special)
      build the GPT-2 byte map (F-TOK-01) once, globally
      note: added token ids may sit ABOVE the plain vocab size (151643 ==
      n_vocab); the decode table must therefore cover max(id) over vocab and
      added, not just n_vocab
- Notes: the vocab strings are already in byte-level form (e.g. the 0xE0 byte
  shows up as the character U+0100), so they hash directly against the
  byte-mapped pieces; no re-encoding step. Loading 151k strings + 151k merges
  is ~6-8 MB of keys: one contiguous pool, no per-entry allocation.

## F-TQ-02: Qwen-family pre-tokeniser (rule by rule, no regex engine)
Origin: [colibri] [ds4] [monolite] [apus]

- Source: colibri (Apache-2.0), c/tok.h; ds4 (MIT), ds4.c (Qwen splitter);
  monolite (Apache-2.0), src/tokenizer.c; apus (PolyForm-NC), c/tok.h --
  read only, not copied. Target-format facts: the target checkpoint's
  pre-tokenizer is an isolated split on the GPT-2/Qwen pattern (the seven
  alternatives below, stated from the target's tokenizer.json spec), followed
  by the ByteLevel mapper.
- Purpose: split an NFC text span into pieces exactly as the reference
  isolated-regex split would, without a regex library.
- Inputs / outputs: UTF-8 span (no special tokens inside) -> list of byte
  ranges. The reference matches leftmost, first listed rule wins; a piece is
  consumed and scanning continues at its end.
- Verified behaviour (probed against the reference pre-tokenizer piece for
  piece): rules tried IN ORDER at the current position (i over codepoints,
  n total, classes L = letter, N = number, W = whitespace; CR/LF are W):
      R1 contraction: the current codepoint is the ASCII apostrophe and the
         following 1..2 codepoints equal (case-insensitively, ASCII) one of
         s, t, re, ve, m, ll, d in that ALTERNATION ORDER (piece = 2 or 3
         codepoints). Eats the apostrophe plus that suffix only.
      R2 word: if current is not CR, not LF, not L and not N, and the next
         codepoint (if any) is L: take the current codepoint plus the maximal
         run of L starting at the next. Otherwise, if current is L: take the
         maximal run of L. (A digit before letters is taken as the one
         codepoint prefix; a space before letters is likewise a prefix, so a
         leading space glues onto the following word: " a" -> one piece.)
      R3 number: current is N: take exactly ONE codepoint.
         (Consequence: decimal digits are single pieces unless glued as a
         word prefix; "123" is three pieces.)
      R4 punct: if current is the ASCII space (0x20 only) and the next codepoint
         is not W, not L and not N: take that space. Then take the maximal run
         of codepoints that are not W, not L, not N; then the maximal run of
         CR/LF after it.
      R5 newline whitespace: let j advance over W codepoints that are not
         CR/LF, then over CR/LF; if any CR/LF was consumed: the piece runs
         from current to the end of that CR/LF run (leading non-CRLF spaces
         are swallowed in: "  \n" is one piece).
      R6 trailing whitespace: current is W: let the W run end at k.
         - if k == n: the piece is the whole run;
         - else if k - 1 > i: the piece is the run minus its LAST codepoint
           (the last one falls through to R7);
         - else (a single W before a non-W): no match, fall through to R7.
         (This is the greedy \s+ followed by the (?!\S) lookaround with
         backtracking; it makes every interior space before a word its own
         one-space piece: "  a" -> " ", " ", "a".)
      R7 whitespace: current is W: take the maximal W run.
      (Every position matches some rule for this rule set; an unmatched
      position can only occur for codepoints the reference regex also drops,
      which does not happen for this rule set -- the scanner never stalls.)
      Order subtleties: R2 is tried before R4, so a space followed by a letter
         is a word piece, not a punct piece. R5 is tried before R6/R7, so a
         space that precedes a newline is swallowed into the newline piece and
         any spaces AFTER the last newline are handled by R6/R7.
- Notes: Unicode class tables (L, N, W per codepoint) come from a committed
  generated range table (sorted [lo,hi] ranges, binary search), produced
  offline from the Unicode database of the version the reference tokenizer
  builds against; see tlm_uni.h. Verified whitespace set = the reference
  engine's \s: Unicode White_Space EXCEPT U+2007 FIGURE SPACE (and U+200B,
  which is not White_Space). Verified letter/number = the standard \p{L}
  (Lu Ll Lt Lm Lo) / \p{N} (Nd Nl No). CJK codepoints are L (Lo): an
  uninterrupted Han/kana run is one piece, as in the reference.

## F-TQ-03: BPE encode of one piece (ordered merges list)
Origin: [colibri] [ds4] [pulsar]

- Source: colibri (Apache-2.0), c/tok.h; ds4 (MIT), ds4.c; pulsar (MIT),
  crates/tokenizer/src/lib.rs -- read only, not copied.
- Purpose: turn one pre-tokenised piece (byte-level string) into token ids.
- Inputs / outputs: piece bytes -> ids appended to the output. Needs the
  vocab hash and the merge hash from F-TQ-01.
- Pseudocode (verified: the reference does NOT short-circuit on a whole-piece
  vocab hit; a well-formed vocab makes the two identical, so the shortcut is
  omitted):
      s = for each byte b of the piece: append the byte-map character of b
      syms = one entry per codepoint of s: (start offset, len 1)
      loop:
          best = none
          for i in 0 .. len(syms) - 2:
              r = merge_rank(syms[i].text, syms[i+1].text)   # by slice
              if r found and (best none or r < best.r): best = (i, r)
          if best none: break
          syms[best.i].len += syms[best.i+1].len
          drop syms[best.i+1]
      for each sym:
          id = vocab_id(sym.text)
          if id not found: ERROR (the reference fails on non-vocab output;
               for a consistent vocab -- every vocab entry reachable from its
               bytes by the merges list, which holds for this family -- this
               is unreachable)
          emit id
- Notes: symbols stay contiguous slices of s, so a merge is a length update +
  delete (no string copies). Ties keep the LEFTMOST best pair (the reference
  scans left to right and keeps strict improvements, so the first pair at the
  minimal rank wins). Merging one occurrence per iteration and rescan is
  equivalent to merging all occurrences of the best pair at once (all pairs
  touching a newly created symbol have strictly higher rank). O(len^3) worst
  case per piece with the naive rescan is fine for word-sized pieces at
  151k-vocab scale (a few thousand merge-hash probes per piece).

## F-TQ-04: Added tokens and top-level encode
Origin: [colibri] [ds4] [pulsar] [apus]

- Source: colibri (Apache-2.0), c/tok.h; ds4 (MIT), ds4.c; pulsar (MIT),
  crates/tokenizer/src/lib.rs -- read only, not copied.
- Purpose: match the reference's added-token behaviour, and give tierlm a
  stricter mode for raw user text (P-99).
- Inputs / outputs: NFC text, allow_added flag -> ids.
- Pseudocode (verified: the REFERENCE always matches added tokens, including
  in a bare encode; the flag below is tierlm's API split, not a reference
  behaviour):
      encode(text, allow_added):
          s = NFC-normalise(text)
          if not allow_added:
              for each piece of pretok(s): emit bpe(piece)
              return
          i = 0
          while i < len(s):
              best = none
              for each added token a:
                  if s[i:] starts with a.content and
                     (best none or a is longer than best): best = a
              if best:
                  emit best.id; i += best.length
                  continue
              # consume one codepoint of plain text, buffer consecutive runs
              pretok_and_bpe over the run up to the next added match (scan
              for the next added match first so runs stay whole)
      decode(ids, skip_special): for each id:
          if id is an added token: skip only when skip_special AND its
             special flag is set; else use its content
          else use the vocab string of id (unknown id: skip)
          content is appended to one shared byte buffer: every codepoint that
             is in the inverse byte map contributes its byte, a codepoint
             that is not (e.g. CJK) contributes its UTF-8 bytes
          one lossy UTF-8 decode of the whole buffer at the end
             (invalid sequences -> U+FFFD; no per-token boundaries)
- Notes: the reference matching is leftmost position, longest content, ties to
  the earlier added-list entry. tierlm's split (P-99): the bare encode API
  defaults to allow_added = false so a prompt cannot inject role markers;
  the chat renderer (F-TQ-05) must match the reference exactly and therefore
  encodes content segments with allow_added = true (a user message containing
  a marker string tokenises to the marker id in the reference too -- parity
  wins over sanitising there, and markers are still pushed as ids by the
  renderer itself, never re-encoded from rendered text).

## F-TQ-05: Qwen3 chat template, no-tools branch (hard-coded, token-level)
Origin: [ds4] [pulsar] [monolite] [colibri]

- Source: ds4 (MIT), ds4_server.c (per-family prompt renderers); pulsar
  (MIT), marker-based renderers; monolite (Apache-2.0), src/harmony.c;
  colibri (Apache-2.0), c/tok.h -- read only, not copied. Target-format
  facts: the target's tokenizer_config.json Jinja template (no-tools
  branch), as read 2026-10-13 and verified 20/20 against the reference
  apply_chat_template token-for-token.
- Purpose: produce exactly the id sequence the reference renders for a
  message list, without a template engine and without re-tokenising marker
  text.
- Inputs / outputs: list of messages (role in {system, user, assistant, tool},
  content: string or absent, optional reasoning_content string),
  add_generation_prompt, enable_thinking (tristate: unset, true, false)
  -> ids. A message whose content is absent (non-string in the reference)
  renders as "". A message carrying tool_calls is refused (TLM_EUNSUP) rather
  than silently diverging: the assistant tool_calls branch (and the tools
  system preamble) is deferred beyond the M1 text-turn gate.
- Pseudocode (M(x) = push the added-token id of marker x, E(x) = encode(x,
  allow_added=true), NL = E("\n"),
  markers: IM_S = im_start, IM_E = im_end, TH = think, THS = /think,
  TR = tool_response, TRS = /tool_response):
      # last_query_index pass: walk backwards; the last user message whose
      # content is a string and NOT (starts with TR and ends with TRS)
      lqi = n - 1
      for i = n-1 .. 0:
          if msg[i].role == user and msg[i].content is a string
             and not (content starts with TR and ends with TRS):
              lqi = i; break
      out = []
      if n and msg[0].role == system:
          out += M(IM_S) + E("system") + NL() + E(msg[0].content)
                + M(IM_E) + NL()
      for idx = 0 .. n-1:
          c = msg[idx].content or ""
          if role == user, or (role == system and idx != 0):
              out += M(IM_S) + E(role) + NL() + E(c) + M(IM_E) + NL()
          elif role == assistant:
              # reasoning split (verified):
              if reasoning_content is a string: r = it; keep c
              else if c contains THS:
                  r   = c before first THS, rstrip "\n",
                        after last TH within that, lstrip "\n"
                  c   = c after last THS, lstrip "\n"
              else: r = ""
              if idx > lqi and (idx == n-1 or r non-empty):
                  out += M(IM_S) + E("assistant") + NL()
                        + M(TH) + NL() + E(r with leading/trailing "\n"
                                       stripped) + NL()
                        + M(THS) + E("\n\n") + E(c lstrip "\n")
              else:
                  out += M(IM_S) + E("assistant") + NL() + E(c)
              out += M(IM_E) + NL()
          elif role == tool:
              if idx == 0 or msg[idx-1].role != tool:
                  out += M(IM_S) + E("user")
              out += NL() + M(TR) + NL() + E(c) + NL() + M(TRS)
              if idx == n-1 or msg[idx+1].role != tool:
                  out += M(IM_E) + NL()
      if add_generation_prompt:
          out += M(IM_S) + E("assistant") + NL()
          if enable_thinking is explicitly false:
              out += M(TH) + E("\n\n") + M(THS) + E("\n\n")
- Notes: markers are pushed as ids (never rendered-then-re-encoded). Verified
  subtleties: (1) the FIRST system message is handled in the prologue with the
  system role name, LATER system messages fall into the user branch with role
  "system"; (2) an assistant turn uses the thinking form only after the last
  real user query, and then only if it is the last message OR it has
  reasoning; (3) consecutive tool messages share one user wrapper: the
  opening marker is emitted only before the first of the run, the closing
  marker only after the last; (4) the empty-think scaffold appears only when
  enable_thinking is EXPLICITLY false (unset/true -> bare assistant marker);
  (5) user content containing marker strings is encoded with added matching
  ON, so it renders exactly like the reference (the marker becomes the marker
  id).
