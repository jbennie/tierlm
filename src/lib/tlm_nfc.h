/* NFC normalization (UAX #15), tokenizer-defined. The tables in tlm_nfc.c
 * are generated from the Unicode database (same precedent as tlm_uni.c).
 * The normalizer applies the same decomposition the reference tokenizer
 * engine applies (CDMA): Hangul syllables to jamo, 1-2 part canonical
 * decompositions, and nothing else; then composes adjacent pairs from
 * the composition table (primary-decomposition pairs, minus the 85
 * composition exclusions), with canonical reordering of combining
 * marks and re-composition of Hangul jamo runs. Validated token-for-token against the reference toolchain
 * (Python unicodedata / tokenizers) before the C tests run. */
#ifndef TLM_NFC_H
#define TLM_NFC_H

#include <stdint.h>

/* Write the NFC form of cps[0..n) to dst (capacity in codepoints; use
 * 3 * n, which always suffices). Returns the output length in
 * codepoints, or -1 if dst does not fit. dst may alias nothing; cps and
 * dst must not overlap. */
int tlm_nfc_normalize(const uint32_t *cps, int n, uint32_t *dst, int cap);

#endif /* TLM_NFC_H */
