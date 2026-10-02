/* tlm_st.h - safetensors reader (single file or sharded with
 * model.safetensors.index.json). Validates the header: offsets inside the
 * file, non-overlapping, sizes match dtype x shape.
 * Writer included for the synthetic model generator and tests.
 */
#ifndef TLM_ST_H
#define TLM_ST_H

#include <stdint.h>
#include <stddef.h>
#include "tlm_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char     name[160];
    tlm_dtype dtype;          /* F32, F16, BF16 supported; F8_E4M3 maps to raw bytes (dtype F32 invalid) */
    int      is_fp8;          /* safetensors "F8_E4M3" */
    int      ndim;
    int64_t  shape[4];
    int      file;            /* shard index */
    uint64_t offset;          /* absolute file offset of the data */
    uint64_t nbytes;
} tlm_st_tensor;

typedef struct tlm_st tlm_st;
/* path: a .safetensors file, or a directory containing model.safetensors or
 * model.safetensors.index.json. */
tlm_st *tlm_st_open(const char *path);
void    tlm_st_close(tlm_st *st);
int     tlm_st_count(const tlm_st *st);
const tlm_st_tensor *tlm_st_at(const tlm_st *st, int i);
const tlm_st_tensor *tlm_st_find(const tlm_st *st, const char *name);
/* Read a tensor converted to f32 (F32/F16/BF16 sources). dst holds numel floats. */
int     tlm_st_read_f32(tlm_st *st, const tlm_st_tensor *t, float *dst);
/* Read raw bytes. */
int     tlm_st_read_raw(tlm_st *st, const tlm_st_tensor *t, void *dst);
int64_t tlm_st_numel(const tlm_st_tensor *t);

/* Writer: collects tensors (F32 only) and writes one .safetensors file. */
typedef struct tlm_st_writer tlm_st_writer;
tlm_st_writer *tlm_st_writer_new(void);
int  tlm_st_writer_add(tlm_st_writer *w, const char *name, const float *data, int ndim, const int64_t *shape);
int  tlm_st_writer_save(tlm_st_writer *w, const char *path);
void tlm_st_writer_free(tlm_st_writer *w);

/* ---- notes (tlm_st.c) ------------------------------------------------------
 * Reader: header length must be >= 2, <= 256 MiB and fit the file; every
 * tensor needs dtype/shape/data_offsets; offsets are relative to the data
 * buffer, end-exclusive, inside the file; size must equal numel x element
 * size; non-empty tensors may not overlap; rank <= 4; names < 160 bytes.
 * Tensors of other dtypes (I64, F64, ...) are skipped with a warning.
 * F8_E4M3 tensors have is_fp8 = 1 and dtype TLM_DT_FP8E4M3_B128; their
 * tlm_st_read_f32 decodes raw e4m3 values WITHOUT block scales (apply the
 * companion scale tensor yourself). Sharded: the index's weight_map decides
 * which shard owns a name present in two shards; every listed name must exist.
 * Writer: tensors are written in insertion order, the header is padded with
 * spaces so the data buffer starts 8-byte aligned, the file is fsynced and
 * renamed into place (P-08). Names may not contain '"', '\\' or control
 * characters, nor be "__metadata__".
 */

#ifdef __cplusplus
}
#endif
#endif
