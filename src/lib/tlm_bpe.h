/* tlm_bpe: byte-level BPE tokenizer loaded from tokenizer.json (merges-list
 * mode), Qwen2/3 family shape: ByteLevel decoder with add_prefix_space=false,
 * NFC input normaliser, pre-tokeniser per F-TQ-02, ordered-merges BPE with no
 * whole-piece shortcut (F-TQ-03), added-token matching (F-TQ-04).
 *
 * Pattern: F-TQ-01 (tokenizer_qwen3.md). Loading rules:
 *   - model.type must be "BPE" with a "vocab" object and a non-empty "merges"
 *     list; decoder.type must be "ByteLevel" with add_prefix_space=false;
 *     the normaliser must be null or "NFC" (both apply NFC to the input;
 *     anything else is refused); added_tokens entries need a string
 *     "content" and an integer "id"; duplicate added ids are an error.
 *
 * P-99: bare encode (allow_added=false) does not match added tokens, so a
 * prompt is the plain BPE of the text; chat rendering (tlm_chat) matches
 * them for exact parity with the reference renderers.
 *
 * P-103: decode is the ByteLevel chain: every token piece is turned into raw
 * bytes when all of its codepoints are GPT-2 byte chars, into raw UTF-8
 * otherwise; all pieces are concatenated and the whole buffer is UTF-8
 * decoded lossily (U+FFFD per invalid byte), once, at the end.
 */
#ifndef TLM_BPE_H
#define TLM_BPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tlm_bpe tlm_bpe;

/* Load <dir>/tokenizer.json. Returns NULL and fills err (when err != NULL)
 * on an unsupported shape or a parse error. */
tlm_bpe *tlm_bpe_load(const char *dir, char *err, size_t errcap);
void     tlm_bpe_free(tlm_bpe *b);

int tlm_bpe_vocab_size(const tlm_bpe *b);   /* number of BPE vocab entries */
int tlm_bpe_max_id(const tlm_bpe *b);       /* max id over vocab and added */

/* Encode UTF-8 `text` (len bytes; NUL may be inside). The text is NFC-
 * normalised, added tokens are matched leftmost-longest when allow_added
 * (ties: earlier entry in added_tokens), the remaining spans go through the
 * pre-tokeniser and BPE. Returns the number of ids written, or -1 when the
 * output would not fit in cap. */
int tlm_bpe_encode(const tlm_bpe *b, const char *text, size_t len, int allow_added,
                   int *ids, int cap);

/* Decode ids to UTF-8 (lossy per P-103). skip_special drops added tokens
 * whose "special" flag is set. Returns the number of bytes written, or -1
 * when the output would not fit in cap. */
int tlm_bpe_decode(const tlm_bpe *b, const int *ids, int n, int skip_special,
                   char *out, size_t cap);

/* Id of the added token whose content equals `content` (len bytes), or -1. */
int tlm_bpe_add_id(const tlm_bpe *b, const char *content, size_t len);

/* Incremental decoder for streaming output: tokens are pushed one by one,
 * bytes are emitted as soon as their UTF-8 sequence is complete; an
 * incomplete tail is held back and flushed (lossily) at the end.
 * dec_push returns the bytes written to out, or -1 when they would not fit.
 */
typedef struct tlm_bpe_dec tlm_bpe_dec;
tlm_bpe_dec *tlm_bpe_dec_new(const tlm_bpe *b, int skip_special);
void         tlm_bpe_dec_free(tlm_bpe_dec *d);
void         tlm_bpe_dec_reset(tlm_bpe_dec *d);
int          tlm_bpe_dec_push(tlm_bpe_dec *d, int id, char *out, size_t cap);
int          tlm_bpe_dec_flush(tlm_bpe_dec *d, char *out, size_t cap);

#ifdef __cplusplus
}
#endif
#endif
