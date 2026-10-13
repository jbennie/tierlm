/* tlm_packet.c - packet writer, manifest load/save, tensor directories.
 *
 * Patterns: P-01 one expert = one contiguous 4 KiB-aligned record, P-03
 * validated container (every offset and size bounds-checked, CRC per packet,
 * SHA-256 over the packet table), P-05 verify bytes after writing, P-08
 * write-temp-then-rename for every file we publish.
 *
 * ---- Drive file (<dir>/<name>.tlmp) ------------------------------------
 * A plain sequence of packets. Each packet starts at a 4 KiB-aligned offset:
 * 64-byte tlm_packet_hdr, payload, zero padding to the next 4 KiB boundary.
 * The writer creates "<name>.tlmp.partial" and renames it only after every
 * packet has been fsynced and (optionally) re-read and CRC-checked, so a
 * crashed compile never leaves a file that looks complete.
 *
 * ---- Manifest (<name>.tlmm), all integers little-endian -------------------
 *   [fixed header, 512 bytes]
 *      0 u32 magic 'TLMM'        4 u32 version          8 u32 fixed size (512)
 *     12 u32 record size (48)   16 u32 n_drives        20 u32 n_packets
 *     24 u32 n_pruned           28 u32 report crc32c   32 u64 drives_off
 *     40 u64 table_off          48 u64 table_len       56 u64 report_off
 *     64 u64 report_len         72 u64 file_size       80 u8  table_sha256[32]
 *    112 u8  source_sha256[32] 144 u8  corpus_sha256[32]
 *    176 char name[64]         240 char profile[16]   256 char arch[32]
 *    288 17 x u32/f32 geometry (field order of tlm_model_geom after arch)
 *    356 zero                  508 u32 header crc32c over [0,508) + drives
 *   [drives: n_drives x 256 bytes, NUL-terminated paths]
 *   [packet table: n_packets x 48-byte records, sorted by (type, layer, id)]
 *      0 u16 type  2 u16 layer  4 u32 id  8 u8 drive  9 u8 replica_drive
 *     10 u16 flags 12 u32 crc32c 16 u64 offset 24 u64 replica_offset
 *     32 u64 total_len 40 f32 heat 44 u32 zero
 *   [report JSON: report_len bytes, no NUL]
 * The sections are contiguous and in this order; the loader requires the
 * stored offsets to equal the computed ones, so there is exactly one valid
 * encoding of a manifest. table_sha256 = SHA-256 of the packet table bytes.
 */
#include "tlm_packet.h"
#include "tlm_io.h"
#include "tlm_platform.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "tlm_packet.c writes packet headers in host order and assumes a little-endian host"
#endif

_Static_assert(sizeof(tlm_packet_hdr) == 64, "tlm_packet_hdr must be 64 bytes");
_Static_assert(sizeof(tlm_tensor_ref) == 88, "tlm_tensor_ref is 88 bytes on disk");

#define PKT_HDR_BYTES   64u
#define MH_FIXED        512u          /* fixed manifest header */
#define MH_CRC_AT       508u
#define MH_DRIVE_BYTES  256u
#define MREC_BYTES      48u
#define MAX_PACKETS     (1u << 24)    /* far above any real model (~30k experts) */
#define MAX_REPORT      (64ull << 20)
#define MAX_PAYLOAD     (1ull << 40)
#define KNOWN_FLAGS     0x3u          /* bit0 pinned, bit1 pruned */
#define VERIFY_CHUNK    (8u << 20)

/* ---- little-endian helpers ------------------------------------------------ */
static void put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static void put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static uint16_t get16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t get32(const uint8_t *p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
static uint64_t get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* ---- SHA-256 (FIPS 180-4), private to this file --------------------------- */
typedef struct {
    uint32_t h[8];
    uint64_t nbytes;
    uint8_t  blk[64];
    size_t   fill;
} sha256_ctx;

static const uint32_t SHA_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

#define ROR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_compress(uint32_t h[8], const uint8_t *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 |
               (uint32_t)p[4 * i + 2] << 8 | (uint32_t)p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR32(w[i - 15], 7) ^ ROR32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ROR32(w[i - 2], 17) ^ ROR32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = k + (ROR32(e, 6) ^ ROR32(e, 11) ^ ROR32(e, 25)) + ((e & f) ^ (~e & g)) +
                      SHA_K[i] + w[i];
        uint32_t t2 = (ROR32(a, 2) ^ ROR32(a, 13) ^ ROR32(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        k = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += k;
}

static void sha256_init(sha256_ctx *c) {
    static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    memcpy(c->h, iv, sizeof iv);
    c->nbytes = 0;
    c->fill = 0;
}

static void sha256_update(sha256_ctx *c, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    c->nbytes += len;
    if (c->fill) {
        size_t take = 64 - c->fill < len ? 64 - c->fill : len;
        memcpy(c->blk + c->fill, p, take);
        c->fill += take; p += take; len -= take;
        if (c->fill < 64) return;
        sha256_compress(c->h, c->blk);
        c->fill = 0;
    }
    for (; len >= 64; p += 64, len -= 64) sha256_compress(c->h, p);
    memcpy(c->blk, p, len);
    c->fill = len;
}

static void sha256_final(sha256_ctx *c, uint8_t out[32]) {
    uint64_t bits = c->nbytes * 8;
    uint8_t pad = 0x80;
    sha256_update(c, &pad, 1);
    pad = 0;
    while (c->fill != 56) sha256_update(c, &pad, 1);
    uint8_t len_be[8];
    for (int i = 0; i < 8; i++) len_be[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha256_update(c, len_be, 8);
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (uint8_t)(c->h[i] >> 24);
        out[4 * i + 1] = (uint8_t)(c->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(c->h[i] >> 8);
        out[4 * i + 3] = (uint8_t)c->h[i];
    }
}

static void sha256(const void *data, size_t len, uint8_t out[32]) {
    sha256_ctx c;
    sha256_init(&c);
    sha256_update(&c, data, len);
    sha256_final(&c, out);
}

/* ---- shared validation ---------------------------------------------------- */
static int str_ok(const char *s, size_t cap, int allow_empty) {
    return memchr(s, 0, cap) != NULL && (allow_empty || s[0] != 0);
}

/* The store name becomes a file-name component on every drive. */
static int name_ok(const char *name) {
    if (!name || !str_ok(name, 64, 0)) return 0;
    if (strchr(name, '/') || strcmp(name, ".") == 0 || strcmp(name, "..") == 0) return 0;
    return 1;
}

static int loc_cmp_key(const tlm_packet_loc *a, const tlm_packet_loc *b) {
    if (a->type != b->type) return a->type < b->type ? -1 : 1;
    if (a->layer != b->layer) return a->layer < b->layer ? -1 : 1;
    if (a->id != b->id) return a->id < b->id ? -1 : 1;
    return 0;
}
static int loc_qsort_cmp(const void *a, const void *b) {
    return loc_cmp_key((const tlm_packet_loc *)a, (const tlm_packet_loc *)b);
}

typedef struct { uint64_t off, len; uint32_t drive, idx; } extent;
static int extent_cmp(const void *a, const void *b) {
    const extent *x = (const extent *)a, *y = (const extent *)b;
    if (x->drive != y->drive) return x->drive < y->drive ? -1 : 1;
    if (x->off != y->off) return x->off < y->off ? -1 : 1;
    return 0;
}

/* Check one sorted packet table against n_drives (P-03): field ranges, unique
 * keys, alignment, no overflow, and no two packets sharing bytes on a drive. */
static int validate_packets(const tlm_packet_loc *p, uint32_t n, uint32_t n_drives) {
    if (n > MAX_PACKETS) return tlm_seterr(TLM_EFORMAT, "manifest: %u packets exceeds limit", n);
    for (uint32_t i = 0; i < n; i++) {
        const tlm_packet_loc *l = &p[i];
        if (l->type < TLM_PK_DENSE || l->type > TLM_PK_TOKEN)
            return tlm_seterr(TLM_EFORMAT, "manifest: packet %u has bad type %u", i, l->type);
        if (l->drive >= n_drives)
            return tlm_seterr(TLM_EFORMAT, "manifest: packet %u drive %u >= %u drives", i,
                              l->drive, n_drives);
        if (l->replica_drive != 0xFF &&
            (l->replica_drive >= n_drives || l->replica_drive == l->drive))
            return tlm_seterr(TLM_EFORMAT, "manifest: packet %u bad replica drive %u", i,
                              l->replica_drive);
        if (l->flags & ~KNOWN_FLAGS)
            return tlm_seterr(TLM_EFORMAT, "manifest: packet %u unknown flags 0x%x", i, l->flags);
        if (l->total_len < TLM_ALIGN_IO || l->total_len % TLM_ALIGN_IO ||
            l->total_len > MAX_PAYLOAD + TLM_ALIGN_IO)
            return tlm_seterr(TLM_EFORMAT, "manifest: packet %u bad length %llu", i,
                              (unsigned long long)l->total_len);
        if (l->offset % TLM_ALIGN_IO || l->offset > UINT64_MAX - l->total_len)
            return tlm_seterr(TLM_EFORMAT, "manifest: packet %u misaligned offset %llu", i,
                              (unsigned long long)l->offset);
        if (l->replica_drive != 0xFF &&
            (l->replica_offset % TLM_ALIGN_IO || l->replica_offset > UINT64_MAX - l->total_len))
            return tlm_seterr(TLM_EFORMAT, "manifest: packet %u misaligned replica offset", i);
        if (!isfinite(l->heat) || l->heat < 0.0f)
            return tlm_seterr(TLM_EFORMAT, "manifest: packet %u bad heat", i);
        if (i > 0 && loc_cmp_key(&p[i - 1], l) >= 0)
            return tlm_seterr(TLM_EFORMAT,
                              "manifest: packet table not strictly sorted at %u "
                              "(type %u layer %u id %u duplicated or out of order)",
                              i, l->type, l->layer, l->id);
    }
    /* Overlap check over primaries and replicas together. */
    if (n == 0) return TLM_OK;
    extent *ex = (extent *)malloc(sizeof(extent) * (size_t)n * 2);
    if (!ex) return tlm_seterr(TLM_ENOMEM, "manifest: out of memory");
    size_t m = 0;
    for (uint32_t i = 0; i < n; i++) {
        ex[m++] = (extent){p[i].offset, p[i].total_len, p[i].drive, i};
        if (p[i].replica_drive != 0xFF)
            ex[m++] = (extent){p[i].replica_offset, p[i].total_len, p[i].replica_drive, i};
    }
    qsort(ex, m, sizeof *ex, extent_cmp);
    int rc = TLM_OK;
    for (size_t i = 1; i < m; i++) {
        if (ex[i].drive == ex[i - 1].drive && ex[i - 1].off + ex[i - 1].len > ex[i].off) {
            rc = tlm_seterr(TLM_EFORMAT, "manifest: packets %u and %u overlap on drive %u",
                            ex[i - 1].idx, ex[i].idx, ex[i].drive);
            break;
        }
    }
    free(ex);
    return rc;
}

/* ---- writer --------------------------------------------------------------- */
typedef struct {
    uint16_t type, layer;
    uint32_t id, crc;
    uint8_t  drive;
    uint64_t off, total, payload_len;
} wrec;

struct tlm_store_writer {
    char     name[64];
    int      n_drives;
    char     partial[TLM_MAX_DRIVES][512];
    char     final_path[TLM_MAX_DRIVES][512];
    char     dirs[TLM_MAX_DRIVES][256];
    int      fd[TLM_MAX_DRIVES];
    uint64_t end[TLM_MAX_DRIVES];
    wrec    *recs;
    size_t   n_recs, cap_recs;
    int      finished, failed;
};

static void fsync_dir(const char *dir) {
    int fd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        (void)fsync(fd);
        close(fd);
    }
}

tlm_store_writer *tlm_writer_open(const char *name, const char *const *drive_dirs, int n_drives) {
    if (!name_ok(name)) {
        tlm_seterr(TLM_EINVAL, "writer: bad store name");
        return NULL;
    }
    if (!drive_dirs || n_drives < 1 || n_drives > TLM_MAX_DRIVES) {
        tlm_seterr(TLM_EINVAL, "writer: need 1..%d drive directories", TLM_MAX_DRIVES);
        return NULL;
    }
    tlm_store_writer *w = (tlm_store_writer *)calloc(1, sizeof *w);
    if (!w) {
        tlm_seterr(TLM_ENOMEM, "writer: out of memory");
        return NULL;
    }
    snprintf(w->name, sizeof w->name, "%s", name);
    w->n_drives = n_drives;
    for (int i = 0; i < TLM_MAX_DRIVES; i++) w->fd[i] = -1;
    dev_t devs[TLM_MAX_DRIVES];
    ino_t inos[TLM_MAX_DRIVES];
    for (int i = 0; i < n_drives; i++) {
        const char *d = drive_dirs[i];
        if (!d || !d[0] || strlen(d) >= sizeof w->dirs[i]) {
            tlm_seterr(TLM_EINVAL, "writer: drive directory %d missing or longer than 255", i);
            goto fail;
        }
        snprintf(w->dirs[i], sizeof w->dirs[i], "%s", d);
        if (tlm_mkdirs(d, 0755) != TLM_OK) goto fail;
        struct stat st;
        if (stat(d, &st) != 0) {
            tlm_seterr(TLM_EIO, "writer: stat %s: %s", d, strerror(errno));
            goto fail;
        }
        /* Two entries naming the same directory would clobber one file. */
        for (int j = 0; j < i; j++)
            if (devs[j] == st.st_dev && inos[j] == st.st_ino) {
                tlm_seterr(TLM_EINVAL, "writer: drives %d and %d are the same directory", j, i);
                goto fail;
            }
        devs[i] = st.st_dev;
        inos[i] = st.st_ino;
        snprintf(w->final_path[i], sizeof w->final_path[i], "%s/%s.tlmp", d, name);
        snprintf(w->partial[i], sizeof w->partial[i], "%s/%s.tlmp.partial", d, name);
        w->fd[i] = open(w->partial[i], O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (w->fd[i] < 0) {
            tlm_seterr(TLM_EIO, "writer: create %s: %s", w->partial[i], strerror(errno));
            goto fail;
        }
    }
    return w;
fail:
    tlm_writer_free(w);
    return NULL;
}

static int write_packet(tlm_store_writer *w, int d, const tlm_packet_hdr *h, const void *payload,
                        uint64_t len, uint64_t total, uint64_t *off_out) {
    static const uint8_t zeros[TLM_ALIGN_IO];
    uint64_t off = w->end[d];
    int rc = tlm_pwrite_full(w->fd[d], h, PKT_HDR_BYTES, off);
    if (rc == TLM_OK && len) rc = tlm_pwrite_full(w->fd[d], payload, len, off + PKT_HDR_BYTES);
    uint64_t pad = total - PKT_HDR_BYTES - len;   /* < 4096 by construction */
    if (rc == TLM_OK && pad) rc = tlm_pwrite_full(w->fd[d], zeros, pad, off + PKT_HDR_BYTES + len);
    if (rc != TLM_OK) {
        w->failed = 1;   /* file contents now unknown: finish must refuse */
        return tlm_seterr(TLM_EIO, "writer: %s: %s", w->partial[d], tlm_errmsg());
    }
    w->end[d] = off + total;
    *off_out = off;
    return TLM_OK;
}

static int push_rec(tlm_store_writer *w, const wrec *r) {
    if (w->n_recs == w->cap_recs) {
        size_t cap = w->cap_recs ? w->cap_recs * 2 : 256;
        wrec *n = (wrec *)realloc(w->recs, cap * sizeof *n);
        if (!n) return tlm_seterr(TLM_ENOMEM, "writer: out of memory");
        w->recs = n;
        w->cap_recs = cap;
    }
    w->recs[w->n_recs++] = *r;
    return TLM_OK;
}

int tlm_writer_add(tlm_store_writer *w, tlm_packet_type type, uint16_t layer, uint32_t id,
                   uint32_t dtype, const void *payload, uint64_t len, const uint32_t item_off[3],
                   int drive, int replica_drive, tlm_packet_loc *loc) {
    if (!w || w->finished) return tlm_seterr(TLM_EINVAL, "writer: not open");
    if (w->failed) return tlm_seterr(TLM_EIO, "writer: an earlier write failed");
    if ((int)type < TLM_PK_DENSE || (int)type > TLM_PK_TOKEN)
        return tlm_seterr(TLM_EINVAL, "writer: bad packet type %d", (int)type);
    if (drive < 0 || drive >= w->n_drives)
        return tlm_seterr(TLM_EINVAL, "writer: drive %d out of range", drive);
    if (replica_drive >= w->n_drives || replica_drive == drive)
        return tlm_seterr(TLM_EINVAL, "writer: replica drive %d invalid", replica_drive);
    if (!payload && len) return tlm_seterr(TLM_EINVAL, "writer: NULL payload");
    if (len > MAX_PAYLOAD) return tlm_seterr(TLM_ERANGE, "writer: payload too large");

    tlm_packet_hdr h;
    memset(&h, 0, sizeof h);
    h.magic = TLM_PACKET_MAGIC;
    h.type = (uint16_t)type;
    h.version = TLM_STORE_VERSION;
    h.layer = layer;
    h.n_items = 1;
    h.id = id;
    h.dtype = dtype;
    h.layout = 0;
    h.payload_len = len;
    h.crc32c = len ? tlm_crc32c(0, payload, (size_t)len) : 0;
    if (item_off) memcpy(h.item_off, item_off, sizeof h.item_off);
    uint64_t total = TLM_ALIGN_UP(PKT_HDR_BYTES + len, TLM_ALIGN_IO);

    uint64_t off = 0, roff = 0;
    int rc = write_packet(w, drive, &h, payload, len, total, &off);
    if (rc == TLM_OK && replica_drive >= 0)
        rc = write_packet(w, replica_drive, &h, payload, len, total, &roff);
    if (rc != TLM_OK) return rc;

    wrec r = {h.type, layer, id, h.crc32c, (uint8_t)drive, off, total, len};
    rc = push_rec(w, &r);
    if (rc == TLM_OK && replica_drive >= 0) {
        r.drive = (uint8_t)replica_drive;
        r.off = roff;
        rc = push_rec(w, &r);
    }
    if (rc != TLM_OK) return rc;

    if (loc) {
        memset(loc, 0, sizeof *loc);
        loc->type = h.type;
        loc->layer = layer;
        loc->id = id;
        loc->drive = (uint8_t)drive;
        loc->replica_drive = replica_drive >= 0 ? (uint8_t)replica_drive : 0xFF;
        loc->flags = 0;
        loc->offset = off;
        loc->replica_offset = replica_drive >= 0 ? roff : 0;
        loc->total_len = total;
        loc->crc32c = h.crc32c;
        loc->heat = 0.0f;
    }
    return TLM_OK;
}

/* Re-read one packet in chunks (through O_DIRECT when available, so the bytes
 * come from the device rather than the page cache) and check header, payload
 * CRC and zero padding (P-05). */
static int verify_written(const tlm_file *f, uint8_t *buf, const wrec *r) {
    uint64_t pos = 0, left = r->payload_len;
    uint32_t crc = 0;
    while (pos < r->total) {
        uint64_t n = r->total - pos < VERIFY_CHUNK ? r->total - pos : VERIFY_CHUNK;
        int rc = tlm_pread_full(f, buf, n, r->off + pos, 1);
        if (rc != TLM_OK) return rc;
        const uint8_t *p = buf;
        uint64_t avail = n;
        if (pos == 0) {
            tlm_packet_hdr h;
            memcpy(&h, buf, sizeof h);
            if (h.magic != TLM_PACKET_MAGIC || h.version != TLM_STORE_VERSION ||
                h.type != r->type || h.layer != r->layer || h.id != r->id ||
                h.payload_len != r->payload_len || h.crc32c != r->crc)
                return tlm_seterr(TLM_EFORMAT, "verify %s: header mismatch at offset %llu",
                                  f->path, (unsigned long long)r->off);
            p += PKT_HDR_BYTES;
            avail -= PKT_HDR_BYTES;
        }
        uint64_t take = avail < left ? avail : left;
        crc = tlm_crc32c(crc, p, (size_t)take);
        left -= take;
        for (uint64_t i = take; i < avail; i++)
            if (p[i])
                return tlm_seterr(TLM_EFORMAT, "verify %s: nonzero padding in packet at %llu",
                                  f->path, (unsigned long long)r->off);
        pos += n;
    }
    if (crc != r->crc)
        return tlm_seterr(TLM_EFORMAT,
                          "verify %s: CRC mismatch for packet type %u layer %u id %u at %llu",
                          f->path, r->type, r->layer, r->id, (unsigned long long)r->off);
    return TLM_OK;
}

int tlm_writer_finish(tlm_store_writer *w, int verify) {
    if (!w || w->finished) return tlm_seterr(TLM_EINVAL, "writer: not open");
    if (w->failed) return tlm_seterr(TLM_EIO, "writer: an earlier write failed");
    for (int d = 0; d < w->n_drives; d++)
        if (fsync(w->fd[d]) != 0)
            return tlm_seterr(TLM_EIO, "writer: fsync %s: %s", w->partial[d], strerror(errno));
    if (verify) {
        uint8_t *buf = (uint8_t *)tlm_alloc_aligned(VERIFY_CHUNK, TLM_ALIGN_IO);
        if (!buf) return tlm_seterr(TLM_ENOMEM, "writer: out of memory");
        int rc = TLM_OK;
        for (int d = 0; d < w->n_drives && rc == TLM_OK; d++) {
            tlm_file f;
            rc = tlm_file_open(&f, w->partial[d], TLM_F_DIRECT);
            for (size_t i = 0; i < w->n_recs && rc == TLM_OK; i++)
                if (w->recs[i].drive == d) rc = verify_written(&f, buf, &w->recs[i]);
            tlm_file_close(&f);
        }
        tlm_free_aligned(buf);
        if (rc != TLM_OK) return rc;
    }
    for (int d = 0; d < w->n_drives; d++) {
        close(w->fd[d]);
        w->fd[d] = -1;
        if (rename(w->partial[d], w->final_path[d]) != 0)
            return tlm_seterr(TLM_EIO, "writer: rename %s: %s", w->partial[d], strerror(errno));
        fsync_dir(w->dirs[d]);
    }
    w->finished = 1;
    return TLM_OK;
}

void tlm_writer_free(tlm_store_writer *w) {
    if (!w) return;
    for (int d = 0; d < w->n_drives; d++) {
        if (w->fd[d] >= 0) close(w->fd[d]);
        /* An unfinished store is never left behind looking usable. */
        if (!w->finished && w->partial[d][0]) unlink(w->partial[d]);
    }
    free(w->recs);
    free(w);
}

/* ---- manifest ------------------------------------------------------------- */
static void put_geom(uint8_t *p, const tlm_model_geom *g) {
    memcpy(p, g->arch, 32);
    p[31] = 0;
    const uint32_t v[22] = {g->n_layers, g->d_model,   g->n_heads,  g->n_kv_heads,
                            g->head_dim, g->vocab,     g->n_experts, g->top_k,
                            g->d_expert, g->n_shared,  g->d_shared, g->max_ctx,
                            g->router,   f2u(g->rope_theta), f2u(g->norm_eps),
                            f2u(g->routed_scale), g->mtp_layers,
                            g->rope_scale, f2u(g->rope_factor),
                            f2u(g->rope_beta_fast), f2u(g->rope_beta_slow),
                            g->rope_orig_len};
    for (int i = 0; i < 22; i++) put32(p + 32 + 4 * i, v[i]);
}

static void get_geom(const uint8_t *p, tlm_model_geom *g) {
    memset(g, 0, sizeof *g);
    memcpy(g->arch, p, 32);
    uint32_t v[22];
    for (int i = 0; i < 22; i++) v[i] = get32(p + 32 + 4 * i);
    g->n_layers = v[0]; g->d_model = v[1]; g->n_heads = v[2]; g->n_kv_heads = v[3];
    g->head_dim = v[4]; g->vocab = v[5]; g->n_experts = v[6]; g->top_k = v[7];
    g->d_expert = v[8]; g->n_shared = v[9]; g->d_shared = v[10]; g->max_ctx = v[11];
    g->router = v[12]; g->rope_theta = u2f(v[13]); g->norm_eps = u2f(v[14]);
    g->routed_scale = u2f(v[15]); g->mtp_layers = v[16];
    g->rope_scale = v[17]; g->rope_factor = u2f(v[18]);
    g->rope_beta_fast = u2f(v[19]); g->rope_beta_slow = u2f(v[20]);
    g->rope_orig_len = v[21];
}

static void put_rec(uint8_t *p, const tlm_packet_loc *l) {
    put16(p, l->type);
    put16(p + 2, l->layer);
    put32(p + 4, l->id);
    p[8] = l->drive;
    p[9] = l->replica_drive;
    put16(p + 10, l->flags);
    put32(p + 12, l->crc32c);
    put64(p + 16, l->offset);
    put64(p + 24, l->replica_drive == 0xFF ? 0 : l->replica_offset);
    put64(p + 32, l->total_len);
    put32(p + 40, f2u(l->heat));
    put32(p + 44, 0);
}

static void get_rec(const uint8_t *p, tlm_packet_loc *l) {
    memset(l, 0, sizeof *l);
    l->type = get16(p);
    l->layer = get16(p + 2);
    l->id = get32(p + 4);
    l->drive = p[8];
    l->replica_drive = p[9];
    l->flags = get16(p + 10);
    l->crc32c = get32(p + 12);
    l->offset = get64(p + 16);
    l->replica_offset = get64(p + 24);
    l->total_len = get64(p + 32);
    l->heat = u2f(get32(p + 40));
}

static uint32_t header_crc(const uint8_t *buf, uint32_t n_drives) {
    uint32_t c = tlm_crc32c(0, buf, MH_CRC_AT);
    return tlm_crc32c(c, buf + MH_FIXED, (size_t)n_drives * MH_DRIVE_BYTES);
}

int tlm_manifest_sort(tlm_manifest *m) {
    if (!m) return tlm_seterr(TLM_EINVAL, "manifest: NULL");
    if (m->n_packets && !m->packets) return tlm_seterr(TLM_EINVAL, "manifest: no packet table");
    if (m->n_packets) qsort(m->packets, m->n_packets, sizeof *m->packets, loc_qsort_cmp);
    for (uint32_t i = 1; i < m->n_packets; i++)
        if (loc_cmp_key(&m->packets[i - 1], &m->packets[i]) == 0)
            return tlm_seterr(TLM_EINVAL, "manifest: duplicate packet type %u layer %u id %u",
                              m->packets[i].type, m->packets[i].layer, m->packets[i].id);
    return TLM_OK;
}

int tlm_manifest_save(const tlm_manifest *m, const char *path) {
    if (!m || !path) return tlm_seterr(TLM_EINVAL, "manifest save: NULL argument");
    if (!name_ok(m->name)) return tlm_seterr(TLM_EINVAL, "manifest save: bad name");
    if (!str_ok(m->profile, sizeof m->profile, 1) || !str_ok(m->geom.arch, sizeof m->geom.arch, 1))
        return tlm_seterr(TLM_EINVAL, "manifest save: profile/arch not NUL-terminated");
    if (m->n_drives < 1 || m->n_drives > TLM_MAX_DRIVES)
        return tlm_seterr(TLM_EINVAL, "manifest save: n_drives %u out of 1..%d", m->n_drives,
                          TLM_MAX_DRIVES);
    for (uint32_t i = 0; i < m->n_drives; i++)
        if (!str_ok(m->drives[i], MH_DRIVE_BYTES, 0))
            return tlm_seterr(TLM_EINVAL, "manifest save: drive %u path empty or too long", i);
    if (m->n_packets > MAX_PACKETS || (m->n_packets && !m->packets))
        return tlm_seterr(TLM_EINVAL, "manifest save: bad packet table");
    if (m->n_pruned > m->n_packets)
        return tlm_seterr(TLM_EINVAL, "manifest save: n_pruned > n_packets");
    size_t rlen = m->report_json ? strlen(m->report_json) : 0;
    if (rlen > MAX_REPORT) return tlm_seterr(TLM_EINVAL, "manifest save: report too large");

    /* Serialise a sorted copy so the caller's order does not matter. */
    tlm_packet_loc *sorted = NULL;
    if (m->n_packets) {
        sorted = (tlm_packet_loc *)malloc(sizeof *sorted * m->n_packets);
        if (!sorted) return tlm_seterr(TLM_ENOMEM, "manifest save: out of memory");
        memcpy(sorted, m->packets, sizeof *sorted * m->n_packets);
        for (uint32_t i = 0; i < m->n_packets; i++)
            if (sorted[i].replica_drive == 0xFF) sorted[i].replica_offset = 0;
        qsort(sorted, m->n_packets, sizeof *sorted, loc_qsort_cmp);
    }
    int rc = validate_packets(sorted, m->n_packets, m->n_drives);
    if (rc != TLM_OK) {
        free(sorted);
        return rc == TLM_EFORMAT ? TLM_EINVAL : rc;   /* caller's data, not a file problem */
    }

    uint64_t drives_off = MH_FIXED;
    uint64_t table_off = drives_off + (uint64_t)m->n_drives * MH_DRIVE_BYTES;
    uint64_t table_len = (uint64_t)m->n_packets * MREC_BYTES;
    uint64_t report_off = table_off + table_len;
    uint64_t file_size = report_off + rlen;
    uint8_t *buf = (uint8_t *)calloc(1, (size_t)file_size);
    if (!buf) {
        free(sorted);
        return tlm_seterr(TLM_ENOMEM, "manifest save: out of memory");
    }
    for (uint32_t i = 0; i < m->n_packets; i++) put_rec(buf + table_off + (size_t)i * MREC_BYTES, &sorted[i]);
    free(sorted);
    if (rlen) memcpy(buf + report_off, m->report_json, rlen);
    for (uint32_t i = 0; i < m->n_drives; i++)
        memcpy(buf + drives_off + (size_t)i * MH_DRIVE_BYTES, m->drives[i], MH_DRIVE_BYTES);

    put32(buf + 0, TLM_MANIFEST_MAGIC);
    put32(buf + 4, TLM_STORE_VERSION);
    put32(buf + 8, MH_FIXED);
    put32(buf + 12, MREC_BYTES);
    put32(buf + 16, m->n_drives);
    put32(buf + 20, m->n_packets);
    put32(buf + 24, m->n_pruned);
    put32(buf + 28, rlen ? tlm_crc32c(0, m->report_json, rlen) : 0);
    put64(buf + 32, drives_off);
    put64(buf + 40, table_off);
    put64(buf + 48, table_len);
    put64(buf + 56, report_off);
    put64(buf + 64, rlen);
    put64(buf + 72, file_size);
    sha256(buf + table_off, (size_t)table_len, buf + 80);
    memcpy(buf + 112, m->source_sha256, 32);
    memcpy(buf + 144, m->corpus_sha256, 32);
    memcpy(buf + 176, m->name, 64);
    memcpy(buf + 240, m->profile, 16);
    put_geom(buf + 256, &m->geom);
    put32(buf + MH_CRC_AT, header_crc(buf, m->n_drives));

    rc = tlm_write_file_atomic(path, buf, (size_t)file_size);
    free(buf);
    return rc;
}

int tlm_manifest_load(tlm_manifest *m, const char *path) {
    if (!m || !path) return tlm_seterr(TLM_EINVAL, "manifest load: NULL argument");
    memset(m, 0, sizeof *m);
    char *raw = NULL;
    size_t size = 0;
    int rc = tlm_read_file(path, &raw, &size);
    if (rc != TLM_OK) return rc;
    const uint8_t *b = (const uint8_t *)raw;
#define BAD(...) do { rc = tlm_seterr(TLM_EFORMAT, __VA_ARGS__); goto out; } while (0)
    if (size < MH_FIXED) BAD("manifest %s: truncated (%zu bytes)", path, size);
    if (get32(b) != TLM_MANIFEST_MAGIC) BAD("manifest %s: bad magic", path);
    if (get32(b + 4) != TLM_STORE_VERSION)
        BAD("manifest %s: version %u, this build reads %u", path, get32(b + 4), TLM_STORE_VERSION);
    if (get32(b + 8) != MH_FIXED || get32(b + 12) != MREC_BYTES)
        BAD("manifest %s: unexpected header geometry", path);
    uint32_t n_drives = get32(b + 16), n_packets = get32(b + 20), n_pruned = get32(b + 24);
    if (n_drives < 1 || n_drives > TLM_MAX_DRIVES) BAD("manifest %s: n_drives %u", path, n_drives);
    if (n_packets > MAX_PACKETS) BAD("manifest %s: n_packets %u too large", path, n_packets);
    if (n_pruned > n_packets) BAD("manifest %s: n_pruned > n_packets", path);
    uint64_t drives_off = get64(b + 32), table_off = get64(b + 40), table_len = get64(b + 48);
    uint64_t report_off = get64(b + 56), report_len = get64(b + 64), file_size = get64(b + 72);
    /* Exactly one valid layout: every section where the writer puts it. */
    if (drives_off != MH_FIXED || table_off != drives_off + (uint64_t)n_drives * MH_DRIVE_BYTES ||
        table_len != (uint64_t)n_packets * MREC_BYTES || report_off != table_off + table_len ||
        report_len > MAX_REPORT || file_size != report_off + report_len || file_size != size)
        BAD("manifest %s: section bounds inconsistent with file size %zu", path, size);
    if (get32(b + MH_CRC_AT) != header_crc(b, n_drives))
        BAD("manifest %s: header checksum mismatch", path);
    uint8_t digest[32];
    sha256(b + table_off, (size_t)table_len, digest);
    if (memcmp(digest, b + 80, 32) != 0) BAD("manifest %s: packet table SHA-256 mismatch", path);
    uint32_t rcrc = report_len ? tlm_crc32c(0, b + report_off, (size_t)report_len) : 0;
    if (rcrc != get32(b + 28)) BAD("manifest %s: report checksum mismatch", path);
    if (report_len && memchr(b + report_off, 0, (size_t)report_len))
        BAD("manifest %s: NUL inside report", path);

    m->magic = TLM_MANIFEST_MAGIC;
    m->version = TLM_STORE_VERSION;
    memcpy(m->table_sha256, b + 80, 32);
    memcpy(m->source_sha256, b + 112, 32);
    memcpy(m->corpus_sha256, b + 144, 32);
    memcpy(m->name, b + 176, 64);
    memcpy(m->profile, b + 240, 16);
    get_geom(b + 256, &m->geom);
    if (!name_ok(m->name)) BAD("manifest %s: bad store name", path);
    if (!str_ok(m->profile, sizeof m->profile, 1) || !str_ok(m->geom.arch, sizeof m->geom.arch, 1))
        BAD("manifest %s: unterminated string", path);
    m->n_drives = n_drives;
    for (uint32_t i = 0; i < n_drives; i++) {
        memcpy(m->drives[i], b + drives_off + (size_t)i * MH_DRIVE_BYTES, MH_DRIVE_BYTES);
        if (!str_ok(m->drives[i], MH_DRIVE_BYTES, 0)) BAD("manifest %s: bad drive %u path", path, i);
    }
    m->n_pruned = n_pruned;
    m->n_packets = n_packets;
    if (n_packets) {
        m->packets = (tlm_packet_loc *)malloc(sizeof *m->packets * n_packets);
        if (!m->packets) {
            rc = tlm_seterr(TLM_ENOMEM, "manifest load: out of memory");
            goto out;
        }
        for (uint32_t i = 0; i < n_packets; i++)
            get_rec(b + table_off + (size_t)i * MREC_BYTES, &m->packets[i]);
    }
    rc = validate_packets(m->packets, n_packets, n_drives);
    if (rc != TLM_OK) goto out;
    if (report_len) {
        m->report_json = (char *)malloc((size_t)report_len + 1);
        if (!m->report_json) {
            rc = tlm_seterr(TLM_ENOMEM, "manifest load: out of memory");
            goto out;
        }
        memcpy(m->report_json, b + report_off, (size_t)report_len);
        m->report_json[report_len] = 0;
    }
#undef BAD
out:
    free(raw);
    if (rc != TLM_OK) tlm_manifest_free(m);
    return rc;
}

void tlm_manifest_free(tlm_manifest *m) {
    if (!m) return;
    free(m->packets);
    free(m->report_json);
    memset(m, 0, sizeof *m);
}

const tlm_packet_loc *tlm_manifest_find(const tlm_manifest *m, tlm_packet_type t, uint16_t layer,
                                        uint32_t id) {
    if (!m || !m->packets) return NULL;
    tlm_packet_loc key;
    memset(&key, 0, sizeof key);
    key.type = (uint16_t)t;
    key.layer = layer;
    key.id = id;
    uint32_t lo = 0, hi = m->n_packets;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        int c = loc_cmp_key(&m->packets[mid], &key);
        if (c == 0) return &m->packets[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}

void tlm_manifest_drive_path(const tlm_manifest *m, int i, char *out, size_t cap) {
    if (!out || cap == 0) return;
    out[0] = 0;
    if (!m || i < 0 || (uint32_t)i >= m->n_drives) return;
    snprintf(out, cap, "%s/%s.tlmp", m->drives[i], m->name);
}

/* ---- packet read/verify --------------------------------------------------- */
int tlm_packet_verify(const tlm_packet_loc *loc, const void *buf, uint64_t buf_len,
                      const void **payload, uint64_t *payload_len) {
    if (!loc || !buf) return tlm_seterr(TLM_EINVAL, "packet verify: NULL argument");
    if (buf_len < PKT_HDR_BYTES || buf_len < loc->total_len)
        return tlm_seterr(TLM_EINVAL, "packet verify: buffer smaller than packet");
    tlm_packet_hdr h;
    memcpy(&h, buf, sizeof h);
    if (h.magic != TLM_PACKET_MAGIC || h.version != TLM_STORE_VERSION)
        return tlm_seterr(TLM_EFORMAT, "packet type %u layer %u id %u: bad magic/version",
                          loc->type, loc->layer, loc->id);
    if (h.type != loc->type || h.layer != loc->layer || h.id != loc->id)
        return tlm_seterr(TLM_EFORMAT,
                          "packet header (type %u layer %u id %u) does not match manifest "
                          "(type %u layer %u id %u)",
                          h.type, h.layer, h.id, loc->type, loc->layer, loc->id);
    if (h.payload_len > loc->total_len - PKT_HDR_BYTES ||
        TLM_ALIGN_UP(PKT_HDR_BYTES + h.payload_len, TLM_ALIGN_IO) != loc->total_len)
        return tlm_seterr(TLM_EFORMAT, "packet type %u layer %u id %u: length mismatch",
                          loc->type, loc->layer, loc->id);
    const uint8_t *p = (const uint8_t *)buf + PKT_HDR_BYTES;
    uint32_t crc = h.payload_len ? tlm_crc32c(0, p, (size_t)h.payload_len) : 0;
    if (crc != h.crc32c || crc != loc->crc32c)
        return tlm_seterr(TLM_EFORMAT, "packet type %u layer %u id %u: CRC mismatch",
                          loc->type, loc->layer, loc->id);
    if (payload) *payload = p;
    if (payload_len) *payload_len = h.payload_len;
    return TLM_OK;
}

int tlm_packet_read(struct tlm_io *io, const tlm_packet_loc *loc, void *buf, const void **payload,
                    uint64_t *payload_len) {
    if (!io || !loc || !buf) return tlm_seterr(TLM_EINVAL, "packet read: NULL argument");
    int rc = tlm_io_read_sync(io, loc->drive, loc->offset, buf, loc->total_len);
    if (rc == TLM_OK) rc = tlm_packet_verify(loc, buf, loc->total_len, payload, payload_len);
    if (rc != TLM_OK && loc->replica_drive != 0xFF) {
        /* A bad copy must never be served; the replica is checked the same way. */
        tlm_log(TLM_LOG_WARN, "%s; trying replica on drive %u", tlm_errmsg(), loc->replica_drive);
        rc = tlm_io_read_sync(io, loc->replica_drive, loc->replica_offset, buf, loc->total_len);
        if (rc == TLM_OK) rc = tlm_packet_verify(loc, buf, loc->total_len, payload, payload_len);
    }
    return rc;
}

/* ---- tensor directories --------------------------------------------------- */
#define DIR_HDR_BYTES 8u

int tlm_dir_build(const tlm_tensor_in *t, int n, void **payload, uint64_t *len) {
    if (!payload || !len || n < 0 || (n > 0 && !t))
        return tlm_seterr(TLM_EINVAL, "dir build: bad arguments");
    *payload = NULL;
    *len = 0;
    if (n > 1 << 20) return tlm_seterr(TLM_ERANGE, "dir build: too many tensors");
    uint64_t cur = TLM_ALIGN_UP(DIR_HDR_BYTES + (uint64_t)n * sizeof(tlm_tensor_ref), TLM_ALIGN_SIMD);
    for (int i = 0; i < n; i++) {
        size_t nl = t[i].name ? strlen(t[i].name) : 0;
        if (nl == 0 || nl >= sizeof(((tlm_tensor_ref *)0)->name))
            return tlm_seterr(TLM_EINVAL, "dir build: tensor %d name empty or longer than 47", i);
        if (!t[i].data && t[i].nbytes)
            return tlm_seterr(TLM_EINVAL, "dir build: tensor %s has no data", t[i].name);
        for (int j = 0; j < i; j++)
            if (strcmp(t[i].name, t[j].name) == 0)
                return tlm_seterr(TLM_EINVAL, "dir build: duplicate tensor %s", t[i].name);
        if (t[i].nbytes > MAX_PAYLOAD || cur > MAX_PAYLOAD)
            return tlm_seterr(TLM_ERANGE, "dir build: tensors too large");
        cur = TLM_ALIGN_UP(cur + t[i].nbytes, TLM_ALIGN_SIMD);
    }
    uint8_t *buf = (uint8_t *)tlm_alloc_aligned((size_t)cur, TLM_ALIGN_SIMD);   /* zeroed */
    if (!buf) return tlm_seterr(TLM_ENOMEM, "dir build: out of memory");
    put32(buf, (uint32_t)n);
    uint64_t off = TLM_ALIGN_UP(DIR_HDR_BYTES + (uint64_t)n * sizeof(tlm_tensor_ref), TLM_ALIGN_SIMD);
    for (int i = 0; i < n; i++) {
        tlm_tensor_ref r;
        memset(&r, 0, sizeof r);
        snprintf(r.name, sizeof r.name, "%s", t[i].name);
        r.dtype = (uint32_t)t[i].dtype;
        r.ndim = t[i].rows == 1 ? 1 : 2;
        r.shape[0] = t[i].rows;
        r.shape[1] = t[i].cols;
        r.offset = off;
        r.nbytes = t[i].nbytes;
        memcpy(buf + DIR_HDR_BYTES + (size_t)i * sizeof r, &r, sizeof r);
        if (t[i].nbytes) memcpy(buf + off, t[i].data, (size_t)t[i].nbytes);
        off = TLM_ALIGN_UP(off + t[i].nbytes, TLM_ALIGN_SIMD);
    }
    *payload = buf;
    *len = cur;
    return TLM_OK;
}

static int dir_ref_ok(const tlm_tensor_ref *r, uint64_t len) {
    return memchr(r->name, 0, sizeof r->name) != NULL && r->name[0] != 0 &&
           (r->ndim == 1 || r->ndim == 2) && r->offset % TLM_ALIGN_SIMD == 0 &&
           r->offset <= len && r->nbytes <= len - r->offset;
}

int tlm_dir_validate(const void *payload, uint64_t len) {
    if (!payload || len < DIR_HDR_BYTES) return tlm_seterr(TLM_EFORMAT, "tensor directory truncated");
    uint32_t n = get32((const uint8_t *)payload);
    if ((uint64_t)n * sizeof(tlm_tensor_ref) > len - DIR_HDR_BYTES)
        return tlm_seterr(TLM_EFORMAT, "tensor directory: %u entries exceed payload", n);
    uint64_t data_start = DIR_HDR_BYTES + (uint64_t)n * sizeof(tlm_tensor_ref);
    for (uint32_t i = 0; i < n; i++) {
        const tlm_tensor_ref *r = tlm_dir_at(payload, i);
        if (!dir_ref_ok(r, len) || r->offset < data_start)
            return tlm_seterr(TLM_EFORMAT, "tensor directory: entry %u out of bounds", i);
    }
    return TLM_OK;
}

uint32_t tlm_dir_count(const void *payload) {
    return payload ? get32((const uint8_t *)payload) : 0;
}

const tlm_tensor_ref *tlm_dir_at(const void *payload, uint32_t i) {
    if (!payload || i >= tlm_dir_count(payload)) return NULL;
    return (const tlm_tensor_ref *)((const uint8_t *)payload + DIR_HDR_BYTES +
                                    (size_t)i * sizeof(tlm_tensor_ref));
}

const tlm_tensor_ref *tlm_dir_find(const void *payload, uint64_t len, const char *name) {
    if (!payload || !name || len < DIR_HDR_BYTES) return NULL;
    uint32_t n = tlm_dir_count(payload);
    if ((uint64_t)n * sizeof(tlm_tensor_ref) > len - DIR_HDR_BYTES) return NULL;
    for (uint32_t i = 0; i < n; i++) {
        const tlm_tensor_ref *r = tlm_dir_at(payload, i);
        if (strncmp(r->name, name, sizeof r->name) == 0 && memchr(r->name, 0, sizeof r->name))
            return dir_ref_ok(r, len) ? r : NULL;
    }
    return NULL;
}
