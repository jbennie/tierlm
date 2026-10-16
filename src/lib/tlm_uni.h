/* tlm_uni: Unicode general-category class tests for the pre-tokeniser.
 *
 * The pre-tokeniser rules (F-TQ-02) need the reference regex engine's
 * character classes per codepoint: letter (\p{L}), number (\p{N}) and
 * whitespace (\s). The tables are committed generated data (tlm_uni.c).
 *
 * P-98: generated range tables, no regex library.
 */
#ifndef TLM_UNI_H
#define TLM_UNI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int tlm_uni_is_letter(uint32_t cp);
int tlm_uni_is_number(uint32_t cp);
int tlm_uni_is_space(uint32_t cp);

#ifdef __cplusplus
}
#endif
#endif
