/* test_st.c - safetensors reader/writer: round trip, dtype conversion,
 * sharded index, and rejection of malformed headers (P-104, P-03). */
#include "tlm_platform.h"
#include "tlm_quant.h"
#include "tlm_st.h"
#include "tlm_test.h"

#include <ftw.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

static char g_root[160];

static int rm_cb(const char *p, const struct stat *sb, int flag, struct FTW *ftw) {
    (void)sb; (void)flag; (void)ftw;
    return remove(p);
}
static void rm_rf(const char *p) { nftw(p, rm_cb, 16, FTW_DEPTH | FTW_PHYS); }

static void path_of(char *out, size_t cap, const char *name) { snprintf(out, cap, "%s/%s", g_root, name); }

/* Write a safetensors file from a header string and a data blob. */
static int write_st(const char *path, const char *hdr, const void *data, size_t dlen) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint64_t n = strlen(hdr);
    uint8_t lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(n >> (8 * i));
    int ok = fwrite(lb, 1, 8, f) == 8 && fwrite(hdr, 1, n, f) == n &&
             (dlen == 0 || fwrite(data, 1, dlen, f) == dlen);
    fclose(f);
    return ok ? 0 : -1;
}

static int write_text(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t n = strlen(text);
    int ok = fwrite(text, 1, n, f) == n;
    fclose(f);
    return ok ? 0 : -1;
}

static void test_roundtrip(void) {
    char p[256];
    path_of(p, sizeof p, "rt.safetensors");
    tlm_st_writer *w = tlm_st_writer_new();
    CHECK(w != NULL);
    float a[6] = {1, -2, 3.5f, 0, 1e-7f, -1e30f}, b[24], s1 = 42.0f;
    for (int i = 0; i < 24; i++) b[i] = tlm_test_randf(-5, 5);
    int64_t sa[2] = {2, 3}, sb[3] = {2, 3, 4}, s0[1] = {0};
    CHECK_EQ_INT(tlm_st_writer_add(w, "model.a.weight", a, 2, sa), 0);
    CHECK_EQ_INT(tlm_st_writer_add(w, "b", b, 3, sb), 0);
    CHECK_EQ_INT(tlm_st_writer_add(w, "scalar", &s1, 0, NULL), 0);
    CHECK_EQ_INT(tlm_st_writer_add(w, "empty", NULL, 1, s0), 0);
    /* rejected names / duplicates */
    CHECK(tlm_st_writer_add(w, "b", b, 3, sb) != 0);
    CHECK(tlm_st_writer_add(w, "bad\"name", b, 1, sa) != 0);
    CHECK(tlm_st_writer_add(w, "__metadata__", b, 1, sa) != 0);
    CHECK(tlm_st_writer_add(w, "", b, 1, sa) != 0);
    char longname[200];
    memset(longname, 'x', sizeof longname - 1);
    longname[sizeof longname - 1] = 0;
    CHECK(tlm_st_writer_add(w, longname, b, 1, sa) != 0);
    CHECK_EQ_INT(tlm_st_writer_save(w, p), 0);
    tlm_st_writer_free(w);

    /* data buffer is 8-byte aligned */
    FILE *f = fopen(p, "rb");
    uint8_t lb[8] = {0};
    CHECK(f && fread(lb, 1, 8, f) == 8);
    if (f) fclose(f);
    uint64_t hl = 0;
    for (int i = 7; i >= 0; i--) hl = (hl << 8) | lb[i];
    CHECK((8 + hl) % 8 == 0);

    tlm_st *st = tlm_st_open(p);
    CHECK(st != NULL);
    if (!st) { fprintf(stderr, "  %s\n", tlm_errmsg()); return; }
    CHECK_EQ_INT(tlm_st_count(st), 4);
    const tlm_st_tensor *t = tlm_st_find(st, "model.a.weight");
    CHECK(t != NULL);
    if (t) {
        CHECK_EQ_INT(t->ndim, 2);
        CHECK_EQ_INT(t->shape[0], 2);
        CHECK_EQ_INT(t->shape[1], 3);
        CHECK_EQ_INT(t->dtype, TLM_DT_F32);
        CHECK_EQ_INT(tlm_st_numel(t), 6);
        float r[6];
        CHECK_EQ_INT(tlm_st_read_f32(st, t, r), 0);
        CHECK(memcmp(r, a, sizeof a) == 0);
    }
    t = tlm_st_find(st, "b");
    CHECK(t != NULL);
    if (t) {
        float r[24];
        CHECK_EQ_INT(t->ndim, 3);
        CHECK_EQ_INT(tlm_st_read_f32(st, t, r), 0);
        CHECK(memcmp(r, b, sizeof b) == 0);
        uint8_t raw[96];
        CHECK_EQ_INT(tlm_st_read_raw(st, t, raw), 0);
        CHECK(memcmp(raw, b, sizeof b) == 0);
    }
    t = tlm_st_find(st, "scalar");
    CHECK(t && t->ndim == 0 && tlm_st_numel(t) == 1);
    if (t) {
        float r = 0;
        CHECK_EQ_INT(tlm_st_read_f32(st, t, &r), 0);
        CHECK(r == 42.0f);
    }
    t = tlm_st_find(st, "empty");
    CHECK(t && t->nbytes == 0 && tlm_st_numel(t) == 0);
    CHECK(tlm_st_find(st, "nope") == NULL);
    CHECK(tlm_st_at(st, 4) == NULL);
    CHECK(tlm_st_at(st, -1) == NULL);
    for (int i = 0; i < tlm_st_count(st); i++) CHECK(tlm_st_find(st, tlm_st_at(st, i)->name) == tlm_st_at(st, i));
    tlm_st_close(st);

    /* a directory holding model.safetensors opens too */
    char d[256], p2[300];
    path_of(d, sizeof d, "rtdir");
    tlm_mkdirs(d, 0755);
    snprintf(p2, sizeof p2, "%s/model.safetensors", d);
    CHECK_EQ_INT(rename(p, p2), 0);
    st = tlm_st_open(d);
    CHECK(st != NULL);
    if (st) CHECK_EQ_INT(tlm_st_count(st), 4);
    tlm_st_close(st);
}

/* Many tensors with long names: the writer's header grows as needed. */
static void test_writer_many(void) {
    char p[256];
    path_of(p, sizeof p, "many.safetensors");
    tlm_st_writer *w = tlm_st_writer_new();
    float v[4] = {1, 2, 3, 4};
    int64_t sh[4] = {1, 1, 2, 2};
    char name[160];
    for (int i = 0; i < 300; i++) {
        memset(name, 'a' + i % 26, 150);
        snprintf(name + 150, 10, "%05u", (unsigned)i % 100000u);
        CHECK_EQ_INT(tlm_st_writer_add(w, name, v, 4, sh), 0);
    }
    CHECK_EQ_INT(tlm_st_writer_save(w, p), 0);
    tlm_st_writer_free(w);
    tlm_st *st = tlm_st_open(p);
    CHECK(st != NULL);
    if (st) {
        CHECK_EQ_INT(tlm_st_count(st), 300);
        const tlm_st_tensor *t = tlm_st_at(st, 299);
        float r[4];
        CHECK(t && tlm_st_read_f32(st, t, r) == 0 && memcmp(r, v, sizeof v) == 0);
    }
    tlm_st_close(st);
}

/* F16, BF16 and F8_E4M3 sources convert to f32. */
static void test_dtypes(void) {
    char p[256];
    path_of(p, sizeof p, "dt.safetensors");
    const float vals[4] = {1.0f, -2.5f, 0.15625f, 448.0f};
    uint8_t data[4 * 2 + 4 * 2 + 4];
    for (int i = 0; i < 4; i++) {
        uint16_t h = tlm_f32_to_f16(vals[i]), bf = tlm_f32_to_bf16(vals[i]);
        data[2 * i] = (uint8_t)h;
        data[2 * i + 1] = (uint8_t)(h >> 8);
        data[8 + 2 * i] = (uint8_t)bf;
        data[8 + 2 * i + 1] = (uint8_t)(bf >> 8);
        data[16 + i] = tlm_f32_to_e4m3(vals[i]);
    }
    const char *hdr =
        "{\"h\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,8]},"
        "\"bf\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[8,16]},"
        "\"f8\":{\"dtype\":\"F8_E4M3\",\"shape\":[4],\"data_offsets\":[16,20]},"
        "\"__metadata__\":{\"format\":\"pt\"}}";
    CHECK_EQ_INT(write_st(p, hdr, data, sizeof data), 0);
    tlm_st *st = tlm_st_open(p);
    CHECK(st != NULL);
    if (!st) { fprintf(stderr, "  %s\n", tlm_errmsg()); return; }
    float r[4];
    const tlm_st_tensor *t = tlm_st_find(st, "h");
    CHECK(t && t->dtype == TLM_DT_F16 && !t->is_fp8);
    CHECK(t && tlm_st_read_f32(st, t, r) == 0);
    for (int i = 0; i < 4; i++) CHECK(r[i] == vals[i]);
    t = tlm_st_find(st, "bf");
    CHECK(t && t->dtype == TLM_DT_BF16);
    CHECK(t && tlm_st_read_f32(st, t, r) == 0);
    for (int i = 0; i < 4; i++) CHECK_NEAR(r[i], vals[i], fabsf(vals[i]) * 0.008f);
    t = tlm_st_find(st, "f8");
    CHECK(t && t->is_fp8 && t->nbytes == 4);
    CHECK(t && tlm_st_read_f32(st, t, r) == 0);
    for (int i = 0; i < 4; i++) CHECK_NEAR(r[i], vals[i], fabsf(vals[i]) * 0.07f);
    CHECK(tlm_st_find(st, "__metadata__") == NULL);
    tlm_st_close(st);

    /* an unsupported dtype is skipped with a warning, not fatal */
    const char *hdr2 = "{\"i\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[0,8]},"
                       "\"f\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}";
    uint8_t d2[12] = {0};
    CHECK_EQ_INT(write_st(p, hdr2, d2, sizeof d2), 0);
    tlm_log_level(TLM_LOG_ERROR);
    st = tlm_st_open(p);
    tlm_log_level(TLM_LOG_WARN);
    CHECK(st != NULL);
    if (st) {
        CHECK_EQ_INT(tlm_st_count(st), 1);
        CHECK(tlm_st_find(st, "i") == NULL);
    }
    tlm_st_close(st);
}

static int open_fails(const char *hdr, size_t dlen, int raw_len_override, uint64_t len_value) {
    char p[256];
    path_of(p, sizeof p, "bad.safetensors");
    static uint8_t zeros[256];
    if (raw_len_override) {
        FILE *f = fopen(p, "wb");
        if (!f) return 0;
        uint8_t lb[8];
        for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(len_value >> (8 * i));
        fwrite(lb, 1, 8, f);
        fwrite(hdr, 1, strlen(hdr), f);
        fclose(f);
    } else if (write_st(p, hdr, zeros, dlen) != 0) {
        return 0;
    }
    tlm_st *st = tlm_st_open(p);
    int failed = st == NULL;
    tlm_st_close(st);
    return failed;
}

static void test_malformed(void) {
    tlm_log_level(TLM_LOG_ERROR);
    /* sanity: a valid one opens */
    CHECK(!open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", 8, 0, 0));
    /* offsets past the end of the data */
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", 4, 0, 0));
    /* begin > end */
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[8,4]}}", 8, 0, 0));
    /* size does not match dtype x shape */
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,8]}}", 16, 0, 0));
    CHECK(open_fails("{\"a\":{\"dtype\":\"BF16\",\"shape\":[2,2],\"data_offsets\":[0,4]}}", 16, 0, 0));
    /* overlapping tensors */
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
                     "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}", 16, 0, 0));
    /* negative / fractional / huge dims */
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[-2],\"data_offsets\":[0,8]}}", 8, 0, 0));
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[2.5],\"data_offsets\":[0,8]}}", 8, 0, 0));
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[4611686018427387904,4611686018427387904],"
                     "\"data_offsets\":[0,8]}}", 8, 0, 0));
    /* rank above 4 */
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[1,1,1,1,2],\"data_offsets\":[0,8]}}", 8, 0, 0));
    /* missing fields, wrong types */
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[2]}}", 8, 0, 0));
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0]}}", 8, 0, 0));
    CHECK(open_fails("{\"a\":{\"dtype\":7,\"shape\":[2],\"data_offsets\":[0,8]}}", 8, 0, 0));
    CHECK(open_fails("{\"a\":[1,2]}", 8, 0, 0));
    CHECK(open_fails("[1,2,3]", 8, 0, 0));
    /* JSON syntax errors */
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}", 8, 0, 0));
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8],}}", 8, 0, 0));
    /* header length larger than the file / absurd */
    CHECK(open_fails("{}", 0, 1, 1000));
    CHECK(open_fails("{}", 0, 1, ~0ull));
    CHECK(open_fails("{}", 0, 1, 1));
    /* duplicate tensor names inside one shard */
    CHECK(open_fails("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
                     "\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", 8, 0, 0));
    /* file shorter than 8 bytes */
    char p[256];
    path_of(p, sizeof p, "tiny.safetensors");
    CHECK_EQ_INT(write_text(p, "abc"), 0);
    CHECK(tlm_st_open(p) == NULL);
    /* missing file / directory without a model */
    path_of(p, sizeof p, "does-not-exist.safetensors");
    CHECK(tlm_st_open(p) == NULL);
    char d[256];
    path_of(d, sizeof d, "emptydir");
    tlm_mkdirs(d, 0755);
    CHECK(tlm_st_open(d) == NULL);
    tlm_log_level(TLM_LOG_WARN);
}

static void write_shard(const char *dir, const char *name, const char *tname, float v) {
    char p[400];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    tlm_st_writer *w = tlm_st_writer_new();
    int64_t sh[1] = {2};
    float d[2] = {v, v + 1};
    tlm_st_writer_add(w, tname, d, 1, sh);
    tlm_st_writer_save(w, p);
    tlm_st_writer_free(w);
}

static void test_sharded(void) {
    char d[256], ip[400];
    path_of(d, sizeof d, "sharded");
    tlm_mkdirs(d, 0755);
    write_shard(d, "model-00001-of-00002.safetensors", "x", 1.0f);
    write_shard(d, "model-00002-of-00002.safetensors", "y", 5.0f);
    snprintf(ip, sizeof ip, "%s/model.safetensors.index.json", d);
    CHECK_EQ_INT(write_text(ip, "{\"metadata\":{\"total_size\":16},\"weight_map\":{"
                                "\"x\":\"model-00001-of-00002.safetensors\","
                                "\"y\":\"model-00002-of-00002.safetensors\"}}"), 0);
    tlm_st *st = tlm_st_open(d);
    CHECK(st != NULL);
    if (st) {
        CHECK_EQ_INT(tlm_st_count(st), 2);
        const tlm_st_tensor *x = tlm_st_find(st, "x"), *y = tlm_st_find(st, "y");
        CHECK(x && y && x->file != y->file);
        float r[2];
        CHECK(y && tlm_st_read_f32(st, y, r) == 0 && r[0] == 5.0f && r[1] == 6.0f);
        CHECK(x && tlm_st_read_f32(st, x, r) == 0 && r[0] == 1.0f);
    } else {
        fprintf(stderr, "  %s\n", tlm_errmsg());
    }
    tlm_st_close(st);
    /* the index json path itself opens too */
    st = tlm_st_open(ip);
    CHECK(st != NULL);
    tlm_st_close(st);

    tlm_log_level(TLM_LOG_ERROR);
    /* index names a tensor no shard has */
    CHECK_EQ_INT(write_text(ip, "{\"weight_map\":{\"x\":\"model-00001-of-00002.safetensors\","
                                "\"z\":\"model-00002-of-00002.safetensors\"}}"), 0);
    CHECK(tlm_st_open(d) == NULL);
    /* index names a missing shard */
    CHECK_EQ_INT(write_text(ip, "{\"weight_map\":{\"x\":\"model-00001-of-00002.safetensors\","
                                "\"y\":\"model-00009-of-00009.safetensors\"}}"), 0);
    CHECK(tlm_st_open(d) == NULL);
    /* path traversal in a shard name */
    CHECK_EQ_INT(write_text(ip, "{\"weight_map\":{\"x\":\"../model-00001-of-00002.safetensors\"}}"), 0);
    CHECK(tlm_st_open(d) == NULL);
    /* no weight_map */
    CHECK_EQ_INT(write_text(ip, "{\"metadata\":{}}"), 0);
    CHECK(tlm_st_open(d) == NULL);
    /* same tensor in two shards: the index decides */
    write_shard(d, "model-00002-of-00002.safetensors", "x", 9.0f);
    CHECK_EQ_INT(write_text(ip, "{\"weight_map\":{\"x\":\"model-00002-of-00002.safetensors\"}}"), 0);
    st = tlm_st_open(d);
    tlm_log_level(TLM_LOG_WARN);
    CHECK(st != NULL);
    if (st) {
        const tlm_st_tensor *x = tlm_st_find(st, "x");
        float r[2];
        CHECK(x && tlm_st_read_f32(st, x, r) == 0 && r[0] == 9.0f);
    }
    tlm_st_close(st);
}

int main(void) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || !tmp[0]) tmp = "/tmp";
    snprintf(g_root, sizeof g_root, "%s/tlm-test-st-%ld", tmp, (long)getpid());
    rm_rf(g_root);
    CHECK_EQ_INT(tlm_mkdirs(g_root, 0700), 0);
    RUN(test_roundtrip);
    RUN(test_writer_many);
    RUN(test_dtypes);
    RUN(test_malformed);
    RUN(test_sharded);
    rm_rf(g_root);
    TEST_MAIN_END();
}
