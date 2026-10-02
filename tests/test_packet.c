/* test_packet.c - packet writer, manifest round trip and validation, packet
 * verification, tensor directories. */
#include "tlm_io.h"
#include "tlm_packet.h"
#include "tlm_platform.h"
#include "tlm_test.h"

#include <fcntl.h>
#include <ftw.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

static char g_root[128];
static char g_dirs[3][200];
static char g_mpath[320];

typedef struct {
    tlm_packet_loc loc;
    uint8_t *data;
    uint64_t len;
} pk;
static pk g_pk[64];
static int g_npk;

static int rm_cb(const char *p, const struct stat *sb, int flag, struct FTW *ftw) {
    (void)sb; (void)flag; (void)ftw;
    return remove(p);
}
static void rm_rf(const char *p) { nftw(p, rm_cb, 16, FTW_DEPTH | FTW_PHYS); }

static int file_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

static void fill(uint8_t *p, uint64_t n, uint64_t seed) {
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
    for (uint64_t i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        p[i] = (uint8_t)(x >> 56);
    }
}

static uint64_t le64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}
static void wle32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static int write_bytes(const char *path, const void *b, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t w = fwrite(b, 1, n, f);
    fclose(f);
    return w == n ? 0 : -1;
}

/* SHA-256 of a byte range via coreutils, as an independent reference.
 * Returns 0 and fills out[32], or -1 when sha256sum is unavailable. */
static int ext_sha256(const uint8_t *b, size_t n, uint8_t out[32]) {
    char tmp[400], cmd[512], hex[80] = {0};
    snprintf(tmp, sizeof tmp, "%s/sha_input.bin", g_root);
    if (write_bytes(tmp, b, n) != 0) return -1;
    snprintf(cmd, sizeof cmd, "sha256sum '%s' 2>/dev/null", tmp);
    FILE *p = popen(cmd, "r");
    if (!p) return -1;
    int ok = fscanf(p, "%64s", hex) == 1;
    pclose(p);
    unlink(tmp);
    if (!ok || strlen(hex) != 64) return -1;
    for (int i = 0; i < 32; i++) {
        unsigned v;
        if (sscanf(hex + 2 * i, "%2x", &v) != 1) return -1;
        out[i] = (uint8_t)v;
    }
    return 0;
}

/* After editing a manifest image, make its table SHA-256 and header CRC valid
 * again so that only the semantic check under test can reject it. */
static int fix_hashes(uint8_t *b, size_t n) {
    uint64_t toff = le64(b + 40), tlen = le64(b + 48);
    uint32_t n_drives = (uint32_t)b[16] | (uint32_t)b[17] << 8;
    if (toff + tlen > n) return -1;
    if (ext_sha256(b + toff, (size_t)tlen, b + 80) != 0) return -1;
    uint32_t c = tlm_crc32c(0, b, 508);
    c = tlm_crc32c(c, b + 512, (size_t)n_drives * 256);
    wle32(b + 508, c);
    return 0;
}

static int load_image(const uint8_t *b, size_t n, tlm_manifest *m) {
    char p[400];
    snprintf(p, sizeof p, "%s/tampered.tlmm", g_root);
    if (write_bytes(p, b, n) != 0) return TLM_EIO;
    return tlm_manifest_load(m, p);
}

/* ---- tests ---------------------------------------------------------------- */
static void test_sha_empty_table(void) {
    tlm_manifest m;
    memset(&m, 0, sizeof m);
    snprintf(m.name, sizeof m.name, "empty");
    snprintf(m.profile, sizeof m.profile, "exact");
    m.n_drives = 1;
    snprintf(m.drives[0], sizeof m.drives[0], "%s", g_dirs[0]);
    char path[400];
    snprintf(path, sizeof path, "%s/empty.tlmm", g_root);
    CHECK_EQ_INT(tlm_manifest_save(&m, path), 0);
    tlm_manifest l;
    CHECK_EQ_INT(tlm_manifest_load(&l, path), 0);
    /* SHA-256("") - FIPS 180-4 test vector */
    static const uint8_t empty[32] = {0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
                                      0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
                                      0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
                                      0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55};
    CHECK(memcmp(l.table_sha256, empty, 32) == 0);
    CHECK_EQ_INT(l.n_packets, 0);
    CHECK(l.packets == NULL);
    CHECK(l.report_json == NULL);
    CHECK(tlm_manifest_find(&l, TLM_PK_EXPERT, 0, 0) == NULL);
    tlm_manifest_free(&l);
}

static void add_pk(tlm_store_writer *w, tlm_packet_type t, uint16_t layer, uint32_t id, uint8_t *data,
                   uint64_t len, int drive, int rep) {
    uint32_t item[3] = {0, (uint32_t)(len / 3), (uint32_t)(2 * len / 3)};
    pk *p = &g_pk[g_npk++];
    p->data = data;
    p->len = len;
    CHECK_EQ_INT(tlm_writer_add(w, t, layer, id, TLM_DT_Q8_0, data, len, item, drive, rep, &p->loc), 0);
    CHECK_EQ_INT(p->loc.offset % 4096, 0);
    CHECK_EQ_INT(p->loc.total_len % 4096, 0);
    CHECK(p->loc.total_len >= len + 64);
    CHECK(p->loc.total_len < len + 64 + 4096);
    CHECK_EQ_INT(p->loc.replica_drive, rep >= 0 ? rep : 0xFF);
}

static void test_write_roundtrip(void) {
    const char *dirs[3] = {g_dirs[0], g_dirs[1], g_dirs[2]};
    tlm_store_writer *w = tlm_writer_open("model", dirs, 3);
    CHECK(w != NULL);
    if (!w) return;
    /* DENSE: a tensor directory per layer; layer 1 also replicated. */
    for (int l = 0; l < 3; l++) {
        float q[32];
        for (int i = 0; i < 32; i++) q[i] = (float)(l * 100 + i);
        uint8_t norm[16];
        fill(norm, sizeof norm, (uint64_t)l);
        tlm_tensor_in t[2] = {{"attn_q", TLM_DT_F32, 4, 8, q, sizeof q},
                              {"ffn_norm", TLM_DT_F32, 1, 4, norm, sizeof norm}};
        void *payload = NULL;
        uint64_t plen = 0;
        CHECK_EQ_INT(tlm_dir_build(t, 2, &payload, &plen), 0);
        add_pk(w, TLM_PK_DENSE, (uint16_t)l, 0, (uint8_t *)payload, plen, l % 3,
               l == 1 ? 2 : -1);
    }
    /* EXPERT packets with sizes around the 4 KiB padding edges. */
    static const uint64_t sizes[6] = {1, 4032, 4033, 10000, 70000, 123457};
    for (int l = 0; l < 3; l++)
        for (int e = 0; e < 6; e++) {
            uint64_t n = sizes[(e + l) % 6];
            uint8_t *d = (uint8_t *)malloc(n);
            fill(d, n, (uint64_t)(l * 1009 + e + 7));
            int drive = (l + e) % 3;
            add_pk(w, TLM_PK_EXPERT, (uint16_t)l, (uint32_t)e, d, n, drive,
                   e % 2 == 0 ? (drive + 1) % 3 : -1);
        }
    uint8_t *meta = (uint8_t *)malloc(5000);
    fill(meta, 5000, 99);
    add_pk(w, TLM_PK_META, TLM_LAYER_NONE, 0, meta, 5000, 0, 1);
    /* A zero-length payload is legal (header + padding only). */
    add_pk(w, TLM_PK_TABLE, TLM_LAYER_NONE, 3, NULL, 0, 2, -1);

    char partial[400], final_[400];
    snprintf(partial, sizeof partial, "%s/model.tlmp.partial", g_dirs[0]);
    snprintf(final_, sizeof final_, "%s/model.tlmp", g_dirs[0]);
    CHECK(file_exists(partial));
    CHECK(!file_exists(final_));
    CHECK_EQ_INT(tlm_writer_finish(w, 1), 0);
    CHECK(!file_exists(partial));
    CHECK(file_exists(final_));
    /* Adding after finish is refused. */
    uint8_t one = 1;
    CHECK(tlm_writer_add(w, TLM_PK_EXPERT, 9, 9, 0, &one, 1, NULL, 0, -1, NULL) != 0);
    tlm_writer_free(w);

    tlm_manifest m;
    memset(&m, 0, sizeof m);
    snprintf(m.geom.arch, sizeof m.geom.arch, "tlm-moe");
    m.geom.n_layers = 3; m.geom.d_model = 64; m.geom.n_heads = 4; m.geom.n_kv_heads = 2;
    m.geom.head_dim = 16; m.geom.vocab = 300; m.geom.n_experts = 6; m.geom.top_k = 2;
    m.geom.d_expert = 32; m.geom.n_shared = 1; m.geom.d_shared = 32; m.geom.max_ctx = 128;
    m.geom.router = 2; m.geom.rope_theta = 10000.0f; m.geom.norm_eps = 1e-6f;
    m.geom.routed_scale = 2.5f; m.geom.mtp_layers = 1;
    snprintf(m.name, sizeof m.name, "model");
    snprintf(m.profile, sizeof m.profile, "balanced");
    m.n_drives = 3;
    for (int i = 0; i < 3; i++) snprintf(m.drives[i], sizeof m.drives[i], "%.255s", g_dirs[i]);
    for (int i = 0; i < 32; i++) {
        m.source_sha256[i] = (uint8_t)i;
        m.corpus_sha256[i] = (uint8_t)(255 - i);
    }
    m.n_pruned = 1;
    m.report_json = (char *)"{\"eval\": {\"top1\": 0.991}}";
    m.n_packets = (uint32_t)g_npk;
    m.packets = (tlm_packet_loc *)malloc(sizeof *m.packets * (size_t)g_npk);
    for (int i = 0; i < g_npk; i++) {   /* reversed: save must sort */
        g_pk[i].loc.heat = (float)i / 64.0f;
        if (i == 5) g_pk[i].loc.flags = 1;
        if (i == 6) g_pk[i].loc.flags = 2;
        m.packets[g_npk - 1 - i] = g_pk[i].loc;
    }
    snprintf(g_mpath, sizeof g_mpath, "%s/model.tlmm", g_root);
    CHECK_EQ_INT(tlm_manifest_save(&m, g_mpath), 0);
    free(m.packets);

    tlm_manifest l;
    CHECK_EQ_INT(tlm_manifest_load(&l, g_mpath), 0);
    CHECK_EQ_INT(l.magic, TLM_MANIFEST_MAGIC);
    CHECK_EQ_INT(l.version, TLM_STORE_VERSION);
    CHECK(strcmp(l.name, "model") == 0);
    CHECK(strcmp(l.profile, "balanced") == 0);
    CHECK(strcmp(l.geom.arch, "tlm-moe") == 0);
    CHECK_EQ_INT(l.geom.n_layers, 3);
    CHECK_EQ_INT(l.geom.vocab, 300);
    CHECK_EQ_INT(l.geom.router, 2);
    CHECK_EQ_INT(l.geom.mtp_layers, 1);
    CHECK(l.geom.rope_theta == 10000.0f && l.geom.norm_eps == 1e-6f && l.geom.routed_scale == 2.5f);
    CHECK_EQ_INT(l.n_drives, 3);
    for (int i = 0; i < 3; i++) CHECK(strcmp(l.drives[i], g_dirs[i]) == 0);
    CHECK(memcmp(l.source_sha256, m.source_sha256, 32) == 0);
    CHECK(memcmp(l.corpus_sha256, m.corpus_sha256, 32) == 0);
    CHECK_EQ_INT(l.n_pruned, 1);
    CHECK(l.report_json && strcmp(l.report_json, m.report_json) == 0);
    CHECK_EQ_INT(l.n_packets, g_npk);
    for (uint32_t i = 1; i < l.n_packets; i++) {
        const tlm_packet_loc *a = &l.packets[i - 1], *b = &l.packets[i];
        CHECK(a->type < b->type || (a->type == b->type && (a->layer < b->layer ||
              (a->layer == b->layer && a->id < b->id))));
    }
    for (int i = 0; i < g_npk; i++) {
        const tlm_packet_loc *e = &g_pk[i].loc;
        const tlm_packet_loc *f = tlm_manifest_find(&l, (tlm_packet_type)e->type, e->layer, e->id);
        CHECK(f != NULL);
        if (!f) continue;
        CHECK(f->drive == e->drive && f->replica_drive == e->replica_drive);
        CHECK(f->offset == e->offset && f->total_len == e->total_len && f->crc32c == e->crc32c);
        CHECK(f->replica_drive == 0xFF || f->replica_offset == e->replica_offset);
        CHECK(f->heat == e->heat && f->flags == e->flags);
    }
    CHECK(tlm_manifest_find(&l, TLM_PK_EXPERT, 0, 6) == NULL);
    CHECK(tlm_manifest_find(&l, TLM_PK_EXPERT, 3, 0) == NULL);
    CHECK(tlm_manifest_find(&l, TLM_PK_MTP, 0, 0) == NULL);
    char dp[512];
    tlm_manifest_drive_path(&l, 2, dp, sizeof dp);
    char want[512];
    snprintf(want, sizeof want, "%s/model.tlmp", g_dirs[2]);
    CHECK(strcmp(dp, want) == 0);
    tlm_manifest_drive_path(&l, 3, dp, sizeof dp);
    CHECK(dp[0] == 0);

    /* Read every packet (and every replica) back through tlm_io, direct on. */
    const char *paths[3];
    char pbuf[3][512];
    for (int i = 0; i < 3; i++) {
        tlm_manifest_drive_path(&l, i, pbuf[i], sizeof pbuf[i]);
        paths[i] = pbuf[i];
    }
    tlm_io_opts o = {TLM_IO_PREAD, 1, 2, 8};
    tlm_io *io = tlm_io_open(paths, 3, &o);
    CHECK(io != NULL);
    for (int i = 0; io && i < g_npk; i++) {
        const tlm_packet_loc *f = tlm_manifest_find(&l, (tlm_packet_type)g_pk[i].loc.type,
                                                    g_pk[i].loc.layer, g_pk[i].loc.id);
        void *buf = tlm_alloc_aligned((size_t)f->total_len, 4096);
        const void *payload = NULL;
        uint64_t plen = 0;
        CHECK_EQ_INT(tlm_packet_read(io, f, buf, &payload, &plen), 0);
        CHECK_EQ_INT(plen, g_pk[i].len);
        CHECK(plen == 0 || memcmp(payload, g_pk[i].data, (size_t)plen) == 0);
        tlm_packet_hdr h;
        memcpy(&h, buf, sizeof h);
        CHECK_EQ_INT(h.item_off[1], (uint32_t)(g_pk[i].len / 3));
        CHECK_EQ_INT(h.n_items, 1);
        if (f->replica_drive != 0xFF) {
            memset(buf, 0, (size_t)f->total_len);
            CHECK_EQ_INT(tlm_io_read_sync(io, f->replica_drive, f->replica_offset, buf, f->total_len), 0);
            CHECK_EQ_INT(tlm_packet_verify(f, buf, f->total_len, &payload, &plen), 0);
            CHECK(plen == 0 || memcmp(payload, g_pk[i].data, (size_t)plen) == 0);
        }
        if (f->type == TLM_PK_DENSE) {   /* the directory survived the trip */
            CHECK_EQ_INT(tlm_dir_validate(payload, plen), 0);
            const tlm_tensor_ref *r = tlm_dir_find(payload, plen, "attn_q");
            CHECK(r != NULL && r->nbytes == 32 * sizeof(float));
            if (r) {
                float v;
                memcpy(&v, (const uint8_t *)payload + r->offset + 4 * 3, 4);
                CHECK_EQ_INT((int)v, f->layer * 100 + 3);
            }
        }
        tlm_free_aligned(buf);
    }
    tlm_io_close(io);
    tlm_manifest_free(&l);
}

static void test_sha_matches_coreutils(void) {
    char *raw = NULL;
    size_t n = 0;
    CHECK_EQ_INT(tlm_read_file(g_mpath, &raw, &n), 0);
    if (!raw) return;
    const uint8_t *b = (const uint8_t *)raw;
    uint64_t toff = le64(b + 40), tlen = le64(b + 48);
    CHECK(toff + tlen <= n);
    CHECK(tlen >= 64 * 2);   /* several SHA blocks, so padding across blocks is exercised */
    uint8_t ref[32];
    if (ext_sha256(b + toff, (size_t)tlen, ref) != 0) {
        printf("  (sha256sum unavailable: skipped external SHA check)\n");
    } else {
        tlm_manifest m;
        CHECK_EQ_INT(tlm_manifest_load(&m, g_mpath), 0);
        CHECK(memcmp(m.table_sha256, ref, 32) == 0);
        tlm_manifest_free(&m);
    }
    free(raw);
}

static void test_tamper_detection(void) {
    char *raw = NULL;
    size_t n = 0;
    CHECK_EQ_INT(tlm_read_file(g_mpath, &raw, &n), 0);
    if (!raw) return;
    uint8_t *img = (uint8_t *)malloc(n);
    uint64_t toff = le64((uint8_t *)raw + 40), roff = le64((uint8_t *)raw + 56);
    tlm_manifest m;

    memcpy(img, raw, n); img[toff + 40] ^= 0x01;   /* heat bit of record 0 */
    CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);
    CHECK(strstr(tlm_errmsg(), "SHA-256") != NULL);
    CHECK(m.packets == NULL && m.n_packets == 0);   /* nothing leaks on failure */

    memcpy(img, raw, n); img[176] ^= 0x20;          /* store name */
    CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);
    CHECK(strstr(tlm_errmsg(), "header checksum") != NULL);

    memcpy(img, raw, n); img[512 + 3] ^= 0x01;      /* drive path */
    CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);

    memcpy(img, raw, n); img[roff + 2] ^= 0x01;     /* report */
    CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);
    CHECK(strstr(tlm_errmsg(), "report") != NULL);

    CHECK_EQ_INT(load_image((const uint8_t *)raw, n - 1, &m), TLM_EFORMAT);   /* truncated */
    CHECK_EQ_INT(load_image((const uint8_t *)raw, 100, &m), TLM_EFORMAT);

    memcpy(img, raw, n); img[0] ^= 0xFF;            /* magic */
    CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);

    /* Semantic bounds checks behind valid hashes (needs sha256sum). */
    uint8_t probe[32];
    if (ext_sha256((const uint8_t *)"abc", 3, probe) != 0) {
        printf("  (sha256sum unavailable: skipped bounds-behind-valid-hash checks)\n");
    } else {
        static const uint8_t abc[32] = {0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
                                        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
                                        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
                                        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
        CHECK(memcmp(probe, abc, 32) == 0);   /* the reference itself is sane */

        memcpy(img, raw, n); CHECK_EQ_INT(fix_hashes(img, n), 0);
        CHECK_EQ_INT(load_image(img, n, &m), 0);   /* control: fix_hashes is faithful */
        tlm_manifest_free(&m);

        memcpy(img, raw, n); img[toff + 8] = 7;     /* drive index >= n_drives */
        CHECK_EQ_INT(fix_hashes(img, n), 0);
        CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);
        CHECK(strstr(tlm_errmsg(), "drive") != NULL);

        memcpy(img, raw, n); img[toff + 16] |= 0x08; /* offset no longer 4 KiB aligned */
        CHECK_EQ_INT(fix_hashes(img, n), 0);
        CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);
        CHECK(strstr(tlm_errmsg(), "misaligned") != NULL);

        memcpy(img, raw, n); img[toff + 32] = 0x10;  /* total_len 4096+16: not a multiple */
        CHECK_EQ_INT(fix_hashes(img, n), 0);
        CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);

        memcpy(img, raw, n);                          /* record 1 duplicates record 0's key */
        memcpy(img + toff + 48, img + toff, 8);
        CHECK_EQ_INT(fix_hashes(img, n), 0);
        CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);
        CHECK(strstr(tlm_errmsg(), "sorted") != NULL);

        memcpy(img, raw, n); img[toff + 10] = 0x80;  /* unknown flag bit */
        CHECK_EQ_INT(fix_hashes(img, n), 0);
        CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);

        memcpy(img, raw, n); img[20] = 0xFF; img[21] = 0xFF; img[22] = 0xFF; img[23] = 0x7F;
        CHECK_EQ_INT(fix_hashes(img, n), 0);         /* n_packets absurd vs table_len */
        CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);

        memcpy(img, raw, n); img[4] = 2;              /* future version */
        CHECK_EQ_INT(fix_hashes(img, n), 0);
        CHECK_EQ_INT(load_image(img, n, &m), TLM_EFORMAT);
        CHECK(strstr(tlm_errmsg(), "version") != NULL);
    }
    free(img);
    free(raw);
}

static void flip_byte(const char *path, uint64_t off) {
    int fd = open(path, O_RDWR);
    CHECK(fd >= 0);
    if (fd < 0) return;
    uint8_t b = 0;
    CHECK(pread(fd, &b, 1, (off_t)off) == 1);
    b ^= 0x5A;
    CHECK(pwrite(fd, &b, 1, (off_t)off) == 1);
    close(fd);
}

static void test_crc_corruption_on_read(void) {
    tlm_manifest m;
    CHECK_EQ_INT(tlm_manifest_load(&m, g_mpath), 0);
    char pbuf[3][512];
    const char *paths[3];
    for (int i = 0; i < 3; i++) {
        tlm_manifest_drive_path(&m, i, pbuf[i], sizeof pbuf[i]);
        paths[i] = pbuf[i];
    }
    tlm_io_opts o = {TLM_IO_PREAD, 1, 1, 8};
    tlm_io *io = tlm_io_open(paths, 3, &o);
    CHECK(io != NULL);
    if (!io) { tlm_manifest_free(&m); return; }
    tlm_log_level(TLM_LOG_ERROR);   /* the replica fallback logs a warning by design */

    /* Expert (layer 0, id 1): no replica. Corrupt one payload byte. */
    const tlm_packet_loc *a = tlm_manifest_find(&m, TLM_PK_EXPERT, 0, 1);
    CHECK(a && a->replica_drive == 0xFF);
    void *buf = tlm_alloc_aligned(256 * 1024, 4096);
    const void *payload;
    uint64_t plen;
    if (a) {
        flip_byte(paths[a->drive], a->offset + 64 + 100);
        CHECK_EQ_INT(tlm_packet_read(io, a, buf, &payload, &plen), TLM_EFORMAT);
        CHECK(strstr(tlm_errmsg(), "CRC") != NULL);
        flip_byte(paths[a->drive], a->offset + 64 + 100);   /* restore */
        CHECK_EQ_INT(tlm_packet_read(io, a, buf, &payload, &plen), 0);
    }
    /* Expert (layer 1, id 2): has a replica; a corrupt primary is never
     * served, the replica is. */
    const tlm_packet_loc *b = tlm_manifest_find(&m, TLM_PK_EXPERT, 1, 2);
    CHECK(b && b->replica_drive != 0xFF);
    if (b) {
        flip_byte(paths[b->drive], b->offset + 64 + 7);
        CHECK_EQ_INT(tlm_packet_read(io, b, buf, &payload, &plen), 0);
        int idx = -1;
        for (int i = 0; i < g_npk; i++)
            if (g_pk[i].loc.type == TLM_PK_EXPERT && g_pk[i].loc.layer == 1 && g_pk[i].loc.id == 2) idx = i;
        CHECK(idx >= 0 && plen == g_pk[idx].len && memcmp(payload, g_pk[idx].data, (size_t)plen) == 0);
        /* Corrupt the replica too: now it must fail. */
        flip_byte(paths[b->replica_drive], b->replica_offset + 64 + 7);
        CHECK_EQ_INT(tlm_packet_read(io, b, buf, &payload, &plen), TLM_EFORMAT);
        flip_byte(paths[b->drive], b->offset + 64 + 7);
        flip_byte(paths[b->replica_drive], b->replica_offset + 64 + 7);
    }
    /* Header that names another packet is rejected even with a valid CRC. */
    const tlm_packet_loc *c = tlm_manifest_find(&m, TLM_PK_EXPERT, 2, 1);
    CHECK(c != NULL);
    if (c) {
        tlm_packet_loc wrong = *c;
        wrong.id = 4;
        CHECK_EQ_INT(tlm_packet_read(io, &wrong, buf, &payload, &plen), TLM_EFORMAT);
        CHECK(strstr(tlm_errmsg(), "does not match") != NULL);
        /* A short buffer is an argument error, not a format error. */
        CHECK_EQ_INT(tlm_packet_verify(c, buf, 32, &payload, &plen), TLM_EINVAL);
    }
    tlm_log_level(TLM_LOG_WARN);
    tlm_free_aligned(buf);
    tlm_io_close(io);
    tlm_manifest_free(&m);
}

static void test_writer_verify_and_abort(void) {
    char d[2][400];
    snprintf(d[0], sizeof d[0], "%s/v0", g_root);
    snprintf(d[1], sizeof d[1], "%s/v1", g_root);
    const char *dirs[2] = {d[0], d[1]};
    uint8_t data[9000];
    fill(data, sizeof data, 5);

    /* Corruption between write and finish is caught by the verify pass. */
    tlm_store_writer *w = tlm_writer_open("vstore", dirs, 2);
    CHECK(w != NULL);
    if (!w) return;
    tlm_packet_loc loc;
    CHECK_EQ_INT(tlm_writer_add(w, TLM_PK_EXPERT, 0, 0, 0, data, sizeof data, NULL, 0, 1, &loc), 0);
    CHECK_EQ_INT(tlm_writer_add(w, TLM_PK_EXPERT, 0, 1, 0, data, 100, NULL, 1, -1, NULL), 0);
    char part[500], fin[500];
    snprintf(part, sizeof part, "%s/vstore.tlmp.partial", d[1]);
    snprintf(fin, sizeof fin, "%s/vstore.tlmp", d[1]);
    flip_byte(part, loc.replica_offset + 64 + 4000);
    CHECK_EQ_INT(tlm_writer_finish(w, 1), TLM_EFORMAT);
    tlm_writer_free(w);   /* unfinished: partial files removed, nothing published */
    CHECK(!file_exists(part));
    CHECK(!file_exists(fin));

    /* Nonzero padding is also caught. */
    w = tlm_writer_open("vstore", dirs, 2);
    CHECK_EQ_INT(tlm_writer_add(w, TLM_PK_EXPERT, 0, 0, 0, data, 100, NULL, 1, -1, &loc), 0);
    flip_byte(part, loc.offset + 64 + 100 + 10);
    CHECK_EQ_INT(tlm_writer_finish(w, 1), TLM_EFORMAT);
    CHECK(strstr(tlm_errmsg(), "padding") != NULL);
    tlm_writer_free(w);

    /* Abort without finish. */
    w = tlm_writer_open("vstore", dirs, 2);
    CHECK_EQ_INT(tlm_writer_add(w, TLM_PK_DENSE, 0, 0, 0, data, 10, NULL, 0, -1, NULL), 0);
    tlm_writer_free(w);
    CHECK(!file_exists(part) && !file_exists(fin));

    /* Argument checks. */
    const char *same[2] = {d[0], d[0]};
    CHECK(tlm_writer_open("x", same, 2) == NULL);
    CHECK(tlm_writer_open("a/b", dirs, 2) == NULL);
    CHECK(tlm_writer_open("x", dirs, 0) == NULL);
    w = tlm_writer_open("args", dirs, 2);
    CHECK(w != NULL);
    CHECK_EQ_INT(tlm_writer_add(w, TLM_PK_EXPERT, 0, 0, 0, data, 10, NULL, 2, -1, NULL), TLM_EINVAL);
    CHECK_EQ_INT(tlm_writer_add(w, TLM_PK_EXPERT, 0, 0, 0, data, 10, NULL, 0, 0, NULL), TLM_EINVAL);
    CHECK_EQ_INT(tlm_writer_add(w, (tlm_packet_type)42, 0, 0, 0, data, 10, NULL, 0, -1, NULL), TLM_EINVAL);
    CHECK_EQ_INT(tlm_writer_add(w, TLM_PK_EXPERT, 0, 0, 0, NULL, 10, NULL, 0, -1, NULL), TLM_EINVAL);
    CHECK_EQ_INT(tlm_writer_finish(w, 1), 0);   /* an empty store is valid */
    tlm_writer_free(w);
}

static void test_manifest_save_rejects(void) {
    tlm_manifest m;
    memset(&m, 0, sizeof m);
    snprintf(m.name, sizeof m.name, "bad");
    m.n_drives = 2;
    snprintf(m.drives[0], 256, "/a");
    snprintf(m.drives[1], 256, "/b");
    tlm_packet_loc p[2];
    memset(p, 0, sizeof p);
    for (int i = 0; i < 2; i++) {
        p[i].type = TLM_PK_EXPERT; p[i].layer = 0; p[i].id = (uint32_t)i;
        p[i].drive = 0; p[i].replica_drive = 0xFF;
        p[i].offset = (uint64_t)i * 8192; p[i].total_len = 8192;
    }
    m.packets = p;
    m.n_packets = 2;
    char path[400];
    snprintf(path, sizeof path, "%s/bad.tlmm", g_root);
    CHECK_EQ_INT(tlm_manifest_save(&m, path), 0);   /* control */

    p[1].id = 0;                                    /* duplicate key */
    CHECK_EQ_INT(tlm_manifest_save(&m, path), TLM_EINVAL);
    CHECK_EQ_INT(tlm_manifest_sort(&m), TLM_EINVAL);
    p[1].id = 1;
    p[1].drive = 2;                                 /* drive out of range */
    CHECK_EQ_INT(tlm_manifest_save(&m, path), TLM_EINVAL);
    p[1].drive = 0;
    p[1].offset = 4096;                             /* overlaps packet 0 */
    CHECK_EQ_INT(tlm_manifest_save(&m, path), TLM_EINVAL);
    p[1].offset = 8192 + 512;                       /* misaligned */
    CHECK_EQ_INT(tlm_manifest_save(&m, path), TLM_EINVAL);
    p[1].offset = 8192;
    p[1].replica_drive = 0;                         /* replica on the same drive */
    CHECK_EQ_INT(tlm_manifest_save(&m, path), TLM_EINVAL);
    p[1].replica_drive = 1;
    p[1].replica_offset = 0;
    CHECK_EQ_INT(tlm_manifest_save(&m, path), 0);
    p[1].heat = NAN;
    CHECK_EQ_INT(tlm_manifest_save(&m, path), TLM_EINVAL);
    p[1].heat = 0.5f;
    m.n_drives = 0;
    CHECK_EQ_INT(tlm_manifest_save(&m, path), TLM_EINVAL);
    m.n_drives = 2;
    snprintf(m.name, sizeof m.name, "..");
    CHECK_EQ_INT(tlm_manifest_save(&m, path), TLM_EINVAL);
    snprintf(m.name, sizeof m.name, "bad");

    /* In-memory sort makes find work without a save/load. */
    tlm_packet_loc q[3];
    memcpy(q, p, sizeof p);
    q[2] = p[0];
    q[2].type = TLM_PK_DENSE; q[2].offset = 65536;
    tlm_packet_loc t = q[0]; q[0] = q[2]; q[2] = t;
    m.packets = q;
    m.n_packets = 3;
    CHECK_EQ_INT(tlm_manifest_sort(&m), 0);
    CHECK(tlm_manifest_find(&m, TLM_PK_DENSE, 0, 0) == &q[0]);
    CHECK(tlm_manifest_find(&m, TLM_PK_EXPERT, 0, 1) == &q[2]);
}

static void test_dir(void) {
    float a[8 * 4];
    for (int i = 0; i < 32; i++) a[i] = (float)i * 0.5f;
    uint8_t r[68];
    fill(r, sizeof r, 3);
    float n1[5] = {1, 2, 3, 4, 5};
    tlm_tensor_in t[4] = {
        {"attn_q", TLM_DT_F32, 8, 4, a, sizeof a},
        {"router", TLM_DT_Q8_0, 2, 32, r, sizeof r},
        {"attn_norm", TLM_DT_F32, 1, 5, n1, sizeof n1},
        {"empty", TLM_DT_F32, 0, 0, NULL, 0},
    };
    void *p = NULL;
    uint64_t len = 0;
    CHECK_EQ_INT(tlm_dir_build(t, 4, &p, &len), 0);
    if (!p) return;
    CHECK_EQ_INT((uintptr_t)p % 64, 0);
    CHECK_EQ_INT(len % 64, 0);
    CHECK_EQ_INT(tlm_dir_count(p), 4);
    CHECK_EQ_INT(tlm_dir_validate(p, len), 0);
    for (uint32_t i = 0; i < 4; i++) {
        const tlm_tensor_ref *ref = tlm_dir_at(p, i);
        CHECK(ref != NULL);
        CHECK_EQ_INT(ref->offset % 64, 0);
        CHECK(ref->offset >= 8 + 4 * sizeof(tlm_tensor_ref));
        CHECK(strcmp(ref->name, t[i].name) == 0);
        CHECK_EQ_INT(ref->nbytes, t[i].nbytes);
        CHECK(t[i].nbytes == 0 || memcmp((uint8_t *)p + ref->offset, t[i].data, (size_t)t[i].nbytes) == 0);
        CHECK(tlm_dir_find(p, len, t[i].name) == ref);
    }
    CHECK(tlm_dir_at(p, 4) == NULL);
    const tlm_tensor_ref *q = tlm_dir_find(p, len, "router");
    CHECK(q && q->dtype == TLM_DT_Q8_0 && q->ndim == 2 && q->shape[0] == 2 && q->shape[1] == 32);
    q = tlm_dir_find(p, len, "attn_norm");
    CHECK(q && q->ndim == 1 && q->shape[1] == 5);
    CHECK(tlm_dir_find(p, len, "missing") == NULL);
    CHECK(tlm_dir_find(p, len, "attn") == NULL);
    CHECK(tlm_dir_find(p, 16, "attn_q") == NULL);          /* table exceeds length */
    CHECK_EQ_INT(tlm_dir_validate(p, 16), TLM_EFORMAT);
    q = tlm_dir_find(p, len, "router");
    CHECK(tlm_dir_find(p, q->offset + 10, "router") == NULL); /* data exceeds length */
    free(p);

    tlm_tensor_in dup[2] = {{"x", TLM_DT_F32, 1, 1, a, 4}, {"x", TLM_DT_F32, 1, 1, a, 4}};
    CHECK_EQ_INT(tlm_dir_build(dup, 2, &p, &len), TLM_EINVAL);
    char long_name[49];
    memset(long_name, 'n', 48);
    long_name[48] = 0;
    tlm_tensor_in ln = {long_name, TLM_DT_F32, 1, 1, a, 4};
    CHECK_EQ_INT(tlm_dir_build(&ln, 1, &p, &len), TLM_EINVAL);
    long_name[47] = 0;   /* 47 characters fit */
    CHECK_EQ_INT(tlm_dir_build(&ln, 1, &p, &len), 0);
    CHECK(tlm_dir_find(p, len, long_name) != NULL);
    free(p);
    CHECK_EQ_INT(tlm_dir_build(NULL, 0, &p, &len), 0);   /* empty directory */
    CHECK_EQ_INT(tlm_dir_count(p), 0);
    CHECK_EQ_INT(tlm_dir_validate(p, len), 0);
    free(p);
}

int main(void) {
    snprintf(g_root, sizeof g_root, "%s/tlm-test-packet-%ld", tlm_test_tmpbase(), (long)getpid());
    rm_rf(g_root);
    for (int i = 0; i < 3; i++) snprintf(g_dirs[i], sizeof g_dirs[i], "%s/drive%d", g_root, i);
    CHECK_EQ_INT(tlm_mkdirs(g_root, 0700), 0);
    RUN(test_sha_empty_table);
    RUN(test_write_roundtrip);
    RUN(test_sha_matches_coreutils);
    RUN(test_tamper_detection);
    RUN(test_crc_corruption_on_read);
    RUN(test_writer_verify_and_abort);
    RUN(test_manifest_save_rejects);
    RUN(test_dir);
    for (int i = 0; i < g_npk; i++) free(g_pk[i].data);
    rm_rf(g_root);
    TEST_MAIN_END();
}
