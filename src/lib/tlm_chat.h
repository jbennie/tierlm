/* tlm_chat: Qwen3 chat template renderer, no-tools branch (F-TQ-05).
 *
 * Produces exactly the id sequence the reference apply_chat_template emits
 * for a message list, without a template engine: marker tokens are pushed
 * as added-token ids directly, content is encoded with added-token matching
 * on. Tool_calls messages and the tools system preamble are refused
 * (beyond the M1 text-turn gate).
 */
#ifndef TLM_CHAT_H
#define TLM_CHAT_H

#include <stddef.h>
#include "tlm_bpe.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *role;            /* "system" | "user" | "assistant" | "tool" */
    const char *content;         /* NULL = absent (renders as "") */
    size_t      content_len;     /* 0 = use strlen(content) */
    const char *reasoning;       /* optional separate reasoning content */
    size_t      reasoning_len;
} tlm_chat_msg;

typedef struct {
    int add_generation_prompt;   /* append the assistant generation marker */
    int enable_thinking;         /* tristate: 1 = true, 0 = false, -1 = unset */
} tlm_chat_opts;

/* Render the message list with the Qwen3 no-tools template. Returns the
 * number of ids written, or -1 when cap is too small, a marker token is
 * missing from the tokenizer, or the input is refused. */
int tlm_qwen3_chat_render(const tlm_bpe *b, const tlm_chat_msg *msgs, int nmsgs,
                          const tlm_chat_opts *opts, int *ids, int cap);

#ifdef __cplusplus
}
#endif
#endif
