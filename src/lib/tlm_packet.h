/* tlm_packet.h - compiled model store: packets on drives + a manifest.
 *
 * Patterns: P-01 one expert = one contiguous record, P-03 validated container,
 * P-04 resumable conversion, P-05 verify bytes after write, P-08 tmp+rename.
 * Design: research/03-two-stage-design.md ("Packet format", "Placement").
 *
 * A store is one manifest file (<name>.tlmm) plus one packet file per drive
 * (<drive_dir>/<name>.tlmp). Every packet starts at a 4 KiB-aligned offset
 * with a 64-byte header followed by its payload, padded to 4 KiB.
 */
#ifndef TLM_PACKET_H
#define TLM_PACKET_H

#include <stdint.h>
#include <stddef.h>
#include "tlm_quant.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TLM_PACKET_MAGIC   0x504D4C54u   /* "TLMP" little-endian */
#define TLM_MANIFEST_MAGIC 0x4D4D4C54u   /* "TLMM" */
#define TLM_STORE_VERSION  2u
#define TLM_MAX_DRIVES     16
#define TLM_LAYER_NONE     0xFFFFu

typedef enum {
    TLM_PK_DENSE  = 1,   /* one layer's resident tensors (attention, norms, router, shared expert) */
    TLM_PK_EXPERT = 2,   /* one routed expert: gate|up|down */
    TLM_PK_BUNDLE = 3,   /* reserved: several small co-firing experts */
    TLM_PK_MTP    = 4,
    TLM_PK_EMBED  = 5,   /* token embeddings */
    TLM_PK_TABLE  = 6,
    TLM_PK_META   = 7,   /* final norm + lm_head and other global tensors */
    TLM_PK_TOKEN  = 8    /* tokenizer blob */
} tlm_packet_type;

#pragma pack(push, 1)
typedef struct {                 /* exactly 64 bytes on disk */
    uint32_t magic;              /* TLM_PACKET_MAGIC */
    uint16_t type;               /* tlm_packet_type */
    uint16_t version;            /* TLM_STORE_VERSION */
    uint16_t layer;              /* TLM_LAYER_NONE if global */
    uint16_t n_items;            /* 1, or members in a BUNDLE */
    uint32_t id;                 /* expert id / table id */
    uint32_t dtype;              /* main dtype of the payload (informational) */
    uint32_t layout;             /* kernel layout id; 0 = row-major plain */
    uint64_t payload_len;        /* bytes before padding */
    uint32_t crc32c;             /* of the payload */
    uint32_t item_off[3];        /* EXPERT: offsets of gate, up, down inside payload */
    uint8_t  reserved[16];
} tlm_packet_hdr;
#pragma pack(pop)

/* Tensor directory inside a DENSE/META/MTP/EMBED payload: the payload begins
 * with a u32 count, then that many tlm_tensor_ref, then 64-byte aligned data. */
#pragma pack(push, 1)
typedef struct {
    char     name[48];           /* e.g. "attn_q", "ffn_norm", "router" */
    uint32_t dtype;              /* tlm_dtype */
    uint32_t ndim;
    uint64_t shape[2];           /* [rows, cols]; 1-D tensors use rows=1 */
    uint64_t offset;             /* from start of payload */
    uint64_t nbytes;
} tlm_tensor_ref;                /* 88 bytes (packed) */
#pragma pack(pop)

/* Where a packet lives. */
typedef struct {
    uint16_t type, layer;
    uint32_t id;
    uint8_t  drive;              /* index into manifest drives */
    uint8_t  replica_drive;      /* 0xFF = none */
    uint16_t flags;              /* bit0 = pinned at load, bit1 = pruned */
    uint64_t offset;             /* of the header in the drive file */
    uint64_t replica_offset;
    uint64_t total_len;          /* header + payload + padding */
    uint32_t crc32c;
    float    heat;               /* calibration selection frequency (0..1) */
} tlm_packet_loc;

/* Model geometry stored in the manifest (generic MoE decoder). */
typedef struct {
    char     arch[32];           /* "tlm-moe" (synthetic/qwen3moe-like), ... */
    uint32_t n_layers, d_model, n_heads, n_kv_heads, head_dim;
    uint32_t vocab, n_experts, top_k, d_expert, n_shared, d_shared;
    uint32_t max_ctx;
    uint32_t router;             /* 0 = softmax top-k renorm, 1 = softmax top-k no renorm, 2 = sigmoid+bias renorm */
    float    rope_theta, norm_eps, routed_scale;
    uint32_t mtp_layers;         /* 0 if no MTP head */
    uint32_t rope_scale;         /* 0 = plain RoPE, 1 = YaRN (truncate), 2 = YaRN (no truncate) */
    float    rope_factor, rope_beta_fast, rope_beta_slow;   /* YaRN parameters */
    uint32_t rope_orig_len;      /* YaRN: original_max_position_embeddings */
} tlm_model_geom;

typedef struct {
    uint32_t       magic, version;
    tlm_model_geom geom;
    char           name[64];
    char           profile[16];        /* exact | balanced | aggressive */
    uint32_t       n_drives;
    char           drives[TLM_MAX_DRIVES][256];   /* drive directories at compile time */
    uint8_t        source_sha256[32];
    uint8_t        corpus_sha256[32];
    uint32_t       n_packets;
    tlm_packet_loc *packets;           /* n_packets entries, sorted by (type, layer, id) */
    uint32_t       n_pruned;           /* experts removed by prune pass */
    char          *report_json;        /* compiler report (NUL-terminated) or NULL */
    uint8_t        table_sha256[32];   /* over the packet table bytes */
} tlm_manifest;

/* ---- writer ------------------------------------------------------------- */
typedef struct tlm_store_writer tlm_store_writer;
tlm_store_writer *tlm_writer_open(const char *name, const char *const *drive_dirs, int n_drives);
/* Append a packet to `drive`; returns its location (offset filled) in *loc.
 * Pass replica_drive >= 0 to write a second copy. CRC computed here. */
int  tlm_writer_add(tlm_store_writer *w, tlm_packet_type type, uint16_t layer, uint32_t id,
                    uint32_t dtype, const void *payload, uint64_t len, const uint32_t item_off[3],
                    int drive, int replica_drive, tlm_packet_loc *loc);
/* Re-read every packet and verify its CRC (P-05), then close the drive files. */
int  tlm_writer_finish(tlm_store_writer *w, int verify);
void tlm_writer_free(tlm_store_writer *w);

/* ---- manifest ------------------------------------------------------------ */
int  tlm_manifest_save(const tlm_manifest *m, const char *path);    /* atomic */
int  tlm_manifest_load(tlm_manifest *m, const char *path);          /* validates SHA + bounds */
void tlm_manifest_free(tlm_manifest *m);
const tlm_packet_loc *tlm_manifest_find(const tlm_manifest *m, tlm_packet_type t,
                                        uint16_t layer, uint32_t id);
/* Path of the packet file for drive i: "<drives[i]>/<name>.tlmp". */
void tlm_manifest_drive_path(const tlm_manifest *m, int i, char *out, size_t cap);

/* ---- tensor directory helpers -------------------------------------------- */
/* Build a directory payload; returns malloc'd buffer and its length. */
typedef struct {
    const char *name; tlm_dtype dtype; uint64_t rows, cols; const void *data; uint64_t nbytes;
} tlm_tensor_in;
int  tlm_dir_build(const tlm_tensor_in *t, int n, void **payload, uint64_t *len);
/* Look up a tensor in a loaded directory payload. */
const tlm_tensor_ref *tlm_dir_find(const void *payload, uint64_t len, const char *name);
uint32_t tlm_dir_count(const void *payload);
const tlm_tensor_ref *tlm_dir_at(const void *payload, uint32_t i);

/* Read and verify one packet into `buf` (must hold loc->total_len bytes, 4 KiB aligned
 * for direct I/O). Returns pointer to payload inside buf via *payload. */
struct tlm_file_set;   /* opaque, see tlm_io.h */

/* ---- additions (tlm_packet.c) --------------------------------------------- */
/* Sort m->packets by (type, layer, id) in place so tlm_manifest_find works on
 * a manifest built in memory; TLM_EINVAL on a duplicate key. tlm_manifest_save
 * sorts its own copy, and tlm_manifest_load always returns a sorted table. */
int  tlm_manifest_sort(tlm_manifest *m);
/* Check a packet image read from disk (`buf` = loc->total_len bytes starting at
 * the 64-byte header): magic/version, type/layer/id equal to `loc`, payload
 * length consistent with total_len, payload CRC32C equal to header and `loc`.
 * On success *payload points inside buf. TLM_EFORMAT on any mismatch. */
int  tlm_packet_verify(const tlm_packet_loc *loc, const void *buf, uint64_t buf_len,
                       const void **payload, uint64_t *payload_len);
/* Synchronous read + verify through an tlm_io whose file i is manifest drive i
 * (open it with tlm_manifest_drive_path(m, i, ...) for i in [0, n_drives)).
 * Falls back to the replica copy when the primary read or check fails. */
struct tlm_io;
int  tlm_packet_read(struct tlm_io *io, const tlm_packet_loc *loc, void *buf,
                     const void **payload, uint64_t *payload_len);
/* Validate a whole tensor directory payload (bounds, alignment, names);
 * tlm_dir_find checks only the entry it returns. */
int  tlm_dir_validate(const void *payload, uint64_t len);
/* Note: tlm_dir_build's buffer is allocated with tlm_alloc_aligned (64-byte
 * aligned, zero padded); release it with free() or tlm_free_aligned(). */

#ifdef __cplusplus
}
#endif
#endif
