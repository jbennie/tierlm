/* test_cfg.c - key = value config: syntax, precedence (incl. environment),
 * includes, conf.d order, sizes, enums, prefix keys, strict errors with
 * line numbers, origins, sorted iteration and writer round trip. */
#include "tlm_cfg.h"
#include "tlm_platform.h"
#include "tlm_test.h"

#include <ftw.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

static char g_tmp[512];

static int rm_cb(const char *p, const struct stat *st, int flag, struct FTW *f) {
    (void)st; (void)flag; (void)f;
    return remove(p);
}
static void rm_rf(const char *p) { nftw(p, rm_cb, 16, FTW_DEPTH | FTW_PHYS); }

static void put_file(const char *rel, const char *text) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", g_tmp, rel);
    char dir[1024];
    snprintf(dir, sizeof dir, "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) { *slash = 0; tlm_mkdirs(dir, 0755); }
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(2); }
    fputs(text, f);
    fclose(f);
}

static int has_diag(const tlm_cfg *c, const char *needle) {
    for (int i = 0; i < tlm_cfg_ndiag(c); i++)
        if (strstr(tlm_cfg_diag(c, i, NULL), needle)) return 1;
    return 0;
}

static tlm_cfg *quiet_cfg(void) {
    tlm_cfg *c = tlm_cfg_new();
    tlm_cfg_quiet(c, 1);
    return c;
}

static void test_defaults(void) {
    tlm_cfg *c = quiet_cfg();
    CHECK(c != NULL);
    CHECK(strcmp(tlm_cfg_str(c, "log.level"), "warn") == 0);
    CHECK(tlm_cfg_is_auto(c, "cpu.threads"));
    CHECK_EQ_INT(tlm_cfg_int(c, "cpu.threads", -1), -1);          /* auto -> caller default */
    CHECK_EQ_INT(tlm_cfg_int(c, "io.queue_depth", 0), 32);
    CHECK_EQ_INT(tlm_cfg_int(c, "io.threads_per_drive", 0), 4);
    CHECK_NEAR(tlm_cfg_frac(c, "memory.pin_frac", 0), 0.30, 1e-12);
    CHECK_NEAR(tlm_cfg_frac(c, "kv.vram_frac", 0), 0.05, 1e-12);
    CHECK_EQ_INT(tlm_cfg_bool(c, "cpu.pin", 0), 1);
    CHECK_EQ_INT(tlm_cfg_bool(c, "io.direct", 1), 0);
    CHECK(strcmp(tlm_cfg_str(c, "io.backend"), "pread") == 0);
    CHECK(strcmp(tlm_cfg_str(c, "spec.mode"), "off") == 0);
    CHECK(strcmp(tlm_cfg_str(c, "cpu.thread_groups"), "ccd") == 0);
    CHECK(strcmp(tlm_cfg_str(c, "decide.backend"), "rules") == 0);
    CHECK(strcmp(tlm_cfg_str(c, "content.default"), "standard") == 0);
    CHECK(tlm_cfg_str(c, "decide.encoder_model") == NULL);
    const char *items[8];
    CHECK_EQ_INT(tlm_cfg_list(c, "io.drives", items, 8), 0);
    CHECK_EQ_INT(tlm_cfg_list(c, "kv.device", items, 8), 1);
    CHECK(strcmp(items[0], "cpu") == 0);
    const char *where = NULL;
    CHECK_EQ_INT(tlm_cfg_origin_of(c, "io.backend", &where), TLM_ORIGIN_DEFAULT);
    CHECK(strcmp(where, "default") == 0);
    CHECK(tlm_cfg_schema_find("content.code.python") != NULL);
    CHECK(tlm_cfg_schema_find("content.default")->def != NULL);         /* exact beats prefix */
    CHECK(tlm_cfg_schema_find("hardware.drive.0.read_gbs")->type == TLM_CFG_STR);
    CHECK(tlm_cfg_schema_find("content.") == NULL);
    CHECK(tlm_cfg_schema_find("bogus.key") == NULL);
    tlm_cfg_free(c);
}

static void test_syntax(void) {
    tlm_cfg *c = quiet_cfg();
    const char *text =
        "# comment line\n"
        "\n"
        "   cpu.threads=12   # trailing comment\n"
        "paths.models = \"/srv/a#b\"      # '#' inside quotes kept\n"
        "paths.log = \"  lead and trail  \"\n"
        "paths.kv_store = \"q\\\"uote\\\\back\"\r\n"
        "io.direct = ON\n"
        "cpu.pin\t=\toff\n"
        "kv.dtype = F8_E4M3\n"
        "io.drives = /d0 ,/d1,  /d2\n"
        "experts.exclude =\n";
    int rc = tlm_cfg_load_string(c, text, "syn.conf", 1);
    if (rc) fprintf(stderr, "    %s\n", tlm_errmsg());
    CHECK_EQ_INT(rc, 0);
    CHECK_EQ_INT(tlm_cfg_int(c, "cpu.threads", 0), 12);
    CHECK(strcmp(tlm_cfg_str(c, "paths.models"), "/srv/a#b") == 0);
    CHECK(strcmp(tlm_cfg_str(c, "paths.log"), "  lead and trail  ") == 0);
    CHECK(strcmp(tlm_cfg_str(c, "paths.kv_store"), "q\"uote\\back") == 0);
    CHECK_EQ_INT(tlm_cfg_bool(c, "io.direct", 0), 1);
    CHECK(strcmp(tlm_cfg_str(c, "io.direct"), "on") == 0);
    CHECK_EQ_INT(tlm_cfg_bool(c, "cpu.pin", 1), 0);
    CHECK(strcmp(tlm_cfg_str(c, "kv.dtype"), "f8_e4m3") == 0);              /* canonical spelling */
    CHECK(strcmp(tlm_cfg_str(c, "io.drives"), "/d0, /d1, /d2") == 0);       /* normalised list */
    const char *items[2];
    CHECK_EQ_INT(tlm_cfg_list(c, "io.drives", items, 2), 2);                /* truncated to max */
    CHECK(strcmp(items[1], "/d1") == 0);
    const char *all[8];
    CHECK_EQ_INT(tlm_cfg_list(c, "io.drives", all, 8), 3);
    CHECK(strcmp(all[2], "/d2") == 0);
    CHECK(strcmp(tlm_cfg_str(c, "experts.exclude"), "") == 0);
    const char *where = NULL;
    CHECK_EQ_INT(tlm_cfg_origin_of(c, "cpu.threads", &where), TLM_ORIGIN_FILE);
    CHECK(strcmp(where, "syn.conf:3") == 0);
    tlm_cfg_free(c);

    /* syntax errors are fatal even when not strict, and name file:line */
    static const struct { const char *text; const char *needle; } bad[] = {
        {"cpu.threads 12\n", "bad.conf:1: expected 'key = value'"},
        {"\n\nCpu.Threads = 4\n", "bad.conf:3: invalid key"},
        {"paths.log = \"open\n", "bad.conf:1: unterminated"},
        {"paths.log = \"a\" b\n", "bad.conf:1: unterminated or malformed"},
        {" = 4\n", "bad.conf:1: missing key"},
        {"cpu..threads = 4\n", "invalid key"},
        {"include\n", "expected 'key = value'"},
    };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        tlm_cfg *b = quiet_cfg();
        rc = tlm_cfg_load_string(b, bad[i].text, "bad.conf", 0);
        CHECK_EQ_INT(rc, TLM_EFORMAT);
        if (!strstr(tlm_errmsg(), bad[i].needle)) fprintf(stderr, "    got '%s' want '%s'\n", tlm_errmsg(), bad[i].needle);
        CHECK(strstr(tlm_errmsg(), bad[i].needle) != NULL);
        tlm_cfg_free(b);
    }
}

static void test_values_and_ranges(void) {
    /* non-strict: bad values are reported and ignored, the load succeeds */
    tlm_cfg *c = quiet_cfg();
    const char *text =
        "kv.vram_frac = 0.2\n"           /* > 0.10 */
        "io.queue_depth = 0\n"           /* < 1 */
        "io.backend = spdk\n"            /* not an enum value */
        "cpu.threads = many\n"           /* not an integer */
        "memory.ram_reserve = 12X\n"     /* bad size */
        "no.such.key = 1\n"              /* unknown */
        "prune.max_ratio = 0.5\n";       /* boundary ok */
    CHECK_EQ_INT(tlm_cfg_load_string(c, text, "v.conf", 0), 0);
    CHECK_NEAR(tlm_cfg_frac(c, "kv.vram_frac", 0), 0.05, 1e-12);
    CHECK_EQ_INT(tlm_cfg_int(c, "io.queue_depth", 0), 32);
    CHECK(strcmp(tlm_cfg_str(c, "io.backend"), "pread") == 0);
    CHECK(tlm_cfg_is_auto(c, "cpu.threads"));
    CHECK_NEAR(tlm_cfg_frac(c, "prune.max_ratio", 0), 0.5, 1e-12);
    CHECK(has_diag(c, "v.conf:1: kv.vram_frac: 0.2 is out of range: expected a number in [0, 0.1]"));
    CHECK(has_diag(c, "v.conf:2: io.queue_depth: 0 is out of range"));
    CHECK(has_diag(c, "v.conf:3: io.backend: 'spdk' is not allowed: expected one of pread|io_uring"));
    CHECK(has_diag(c, "v.conf:4: cpu.threads: 'many' is not an integer in [1, 4096] or auto"));
    CHECK(has_diag(c, "v.conf:5"));
    CHECK(has_diag(c, "v.conf:6: unknown key 'no.such.key' (ignored)"));
    int is_err = -1;
    int last = tlm_cfg_ndiag(c) - 1;
    CHECK(strstr(tlm_cfg_diag(c, last, &is_err), "unknown key") != NULL);
    CHECK_EQ_INT(is_err, 0);                                     /* unknown key = warning */
    tlm_cfg_free(c);

    /* strict: the same file fails, first error reported with file:line */
    c = quiet_cfg();
    int rc = tlm_cfg_load_string(c, text, "v.conf", 1);
    CHECK_EQ_INT(rc, TLM_ERANGE);
    CHECK(strstr(tlm_errmsg(), "v.conf:1: kv.vram_frac") != NULL);
    tlm_cfg_free(c);

    c = quiet_cfg();
    rc = tlm_cfg_load_string(c, "log.level = info\n\nmystery = 3\n", "s.conf", 1);
    CHECK_EQ_INT(rc, TLM_EINVAL);
    CHECK(strstr(tlm_errmsg(), "s.conf:3: unknown key 'mystery'") != NULL);
    CHECK(strcmp(tlm_cfg_str(c, "log.level"), "info") == 0);       /* valid lines still applied */
    tlm_cfg_free(c);

    /* enums, keywords, bools */
    char norm[128], why[256];
    CHECK_EQ_INT(tlm_cfg_validate("compile.profile", "Balanced", norm, sizeof norm, why, sizeof why), 0);
    CHECK(strcmp(norm, "balanced") == 0);
    CHECK_EQ_INT(tlm_cfg_validate("spec.depth", "AUTO", norm, sizeof norm, NULL, 0), 0);
    CHECK(strcmp(norm, "auto") == 0);
    CHECK_EQ_INT(tlm_cfg_validate("spec.depth", "5", norm, sizeof norm, why, sizeof why), TLM_ERANGE);
    CHECK(strstr(why, "[0, 4]") != NULL);
    CHECK_EQ_INT(tlm_cfg_validate("io.prefetch", "auto", NULL, 0, NULL, 0), TLM_ERANGE);  /* not listed */
    CHECK_EQ_INT(tlm_cfg_validate("io.direct", "maybe", NULL, 0, why, sizeof why), TLM_EINVAL);
    CHECK_EQ_INT(tlm_cfg_validate("nope.key", "1", NULL, 0, why, sizeof why), TLM_ENOTFOUND);
    CHECK_EQ_INT(tlm_cfg_validate("Bad.Key", "1", NULL, 0, why, sizeof why), TLM_EINVAL);
    CHECK_EQ_INT(tlm_cfg_validate("memory.pin_frac", "30%", NULL, 0, NULL, 0), 0);
    CHECK_EQ_INT(tlm_cfg_validate("memory.pin_frac", "nan", NULL, 0, NULL, 0), TLM_EINVAL);
    CHECK_EQ_INT(tlm_cfg_validate("memory.pin_frac", "1.5", NULL, 0, NULL, 0), TLM_ERANGE);
    CHECK_EQ_INT(tlm_cfg_validate("prune.max_ratio", "0.6", NULL, 0, NULL, 0), TLM_ERANGE);
    CHECK_EQ_INT(tlm_cfg_validate("prune.min_keep_per_layer", "0.4", NULL, 0, NULL, 0), TLM_ERANGE);
    CHECK_EQ_INT(tlm_cfg_validate("server.listen", "0.0.0.0:8080", NULL, 0, NULL, 0), 0);
    CHECK_EQ_INT(tlm_cfg_validate("server.listen", "[::1]:8000", NULL, 0, NULL, 0), 0);
    CHECK_EQ_INT(tlm_cfg_validate("server.listen", "localhost", NULL, 0, NULL, 0), TLM_EINVAL);
    CHECK_EQ_INT(tlm_cfg_validate("server.listen", "h:70000", NULL, 0, NULL, 0), TLM_EINVAL);
    /* list item grammars */
    CHECK_EQ_INT(tlm_cfg_validate("experts.protect", "12:178, 17:44", norm, sizeof norm, NULL, 0), 0);
    CHECK(strcmp(norm, "12:178, 17:44") == 0);
    CHECK_EQ_INT(tlm_cfg_validate("experts.protect", "12-178", NULL, 0, why, sizeof why), TLM_EINVAL);
    CHECK(strstr(why, "layer:expert") != NULL);
    CHECK_EQ_INT(tlm_cfg_validate("experts.exclude", "1:2,,3:4", NULL, 0, why, sizeof why), TLM_EINVAL);
    CHECK(strstr(why, "empty item") != NULL);
    CHECK_EQ_INT(tlm_cfg_validate("kv.device", "hip:0, hip:1", NULL, 0, NULL, 0), 0);
    CHECK_EQ_INT(tlm_cfg_validate("kv.device", "cpu, hip:0", NULL, 0, NULL, 0), TLM_EINVAL);
    CHECK_EQ_INT(tlm_cfg_validate("kv.device", "gpu0", NULL, 0, NULL, 0), TLM_EINVAL);
    CHECK_EQ_INT(tlm_cfg_validate("kv.device", " ", NULL, 0, why, sizeof why), TLM_EINVAL);   /* empty: no device */
    CHECK(strstr(why, "hip:0") != NULL);
    CHECK_EQ_INT(tlm_cfg_validate("io.drives", "", norm, sizeof norm, NULL, 0), 0);          /* empty list ok */
    CHECK(strcmp(norm, "") == 0);
}

static void test_sizes(void) {
    uint64_t b = 0;
    CHECK_EQ_INT(tlm_cfg_parse_size("4096", &b), 0); CHECK(b == 4096);
    CHECK_EQ_INT(tlm_cfg_parse_size("12G", &b), 0); CHECK(b == 12ull << 30);
    CHECK_EQ_INT(tlm_cfg_parse_size("12g", &b), 0); CHECK(b == 12ull << 30);
    CHECK_EQ_INT(tlm_cfg_parse_size("512M", &b), 0); CHECK(b == 512ull << 20);
    CHECK_EQ_INT(tlm_cfg_parse_size("512MiB", &b), 0); CHECK(b == 512ull << 20);
    CHECK_EQ_INT(tlm_cfg_parse_size("64KB", &b), 0); CHECK(b == 64ull << 10);
    CHECK_EQ_INT(tlm_cfg_parse_size("2T", &b), 0); CHECK(b == 2ull << 40);
    CHECK_EQ_INT(tlm_cfg_parse_size("1.5G", &b), 0); CHECK(b == 3ull << 29);
    CHECK_EQ_INT(tlm_cfg_parse_size("100B", &b), 0); CHECK(b == 100);
    CHECK_EQ_INT(tlm_cfg_parse_size("0", &b), 0); CHECK(b == 0);
    CHECK(tlm_cfg_parse_size("12X", &b) != 0);
    CHECK(tlm_cfg_parse_size("-1G", &b) != 0);
    CHECK(tlm_cfg_parse_size("1e3", &b) != 0);
    CHECK(tlm_cfg_parse_size("", &b) != 0);
    CHECK(tlm_cfg_parse_size("12 G", &b) != 0);
    CHECK(tlm_cfg_parse_size("1.G", &b) != 0);
    CHECK(tlm_cfg_parse_size("99999999999999999999", &b) == TLM_ERANGE);
    CHECK(tlm_cfg_parse_size("20000000T", &b) == TLM_ERANGE);
    int64_t i = 0;
    CHECK_EQ_INT(tlm_cfg_parse_int("4K", &i), 0); CHECK_EQ_INT(i, 4096);
    CHECK_EQ_INT(tlm_cfg_parse_int("-7", &i), 0); CHECK_EQ_INT(i, -7);
    CHECK(tlm_cfg_parse_int("1.5", &i) != 0);
    CHECK(tlm_cfg_parse_int("12abc", &i) != 0);
    CHECK(tlm_cfg_parse_int("9223372036854775808", &i) == TLM_ERANGE);
    double f = 0;
    CHECK_EQ_INT(tlm_cfg_parse_frac(".5", &f), 0); CHECK_NEAR(f, 0.5, 0);
    CHECK_EQ_INT(tlm_cfg_parse_frac("5%", &f), 0); CHECK_NEAR(f, 0.05, 1e-15);
    CHECK(tlm_cfg_parse_frac("inf", &f) != 0);
    CHECK(tlm_cfg_parse_frac("0x1p3", &f) != 0);
    char s[32];
    tlm_cfg_fmt_size(12ull << 30, s, sizeof s); CHECK(strcmp(s, "12G") == 0);
    tlm_cfg_fmt_size(1536ull << 20, s, sizeof s); CHECK(strcmp(s, "1536M") == 0);
    tlm_cfg_fmt_size(1000, s, sizeof s); CHECK(strcmp(s, "1000") == 0);
    tlm_cfg_fmt_size(0, s, sizeof s); CHECK(strcmp(s, "0") == 0);

    tlm_cfg *c = quiet_cfg();
    CHECK_EQ_INT(tlm_cfg_load_string(c, "memory.ram_reserve = 12G\nmemory.expert_cache = 100G\n", "z", 1), 0);
    CHECK(tlm_cfg_size(c, "memory.ram_reserve", 0) == 12ull << 30);
    CHECK(tlm_cfg_size(c, "memory.expert_cache", 0) == 100ull << 30);
    CHECK(tlm_cfg_int(c, "memory.ram_reserve", 0) == (int64_t)(12ull << 30));
    CHECK_EQ_INT(tlm_cfg_set(c, "memory.expert_cache", "auto"), 0);
    CHECK(tlm_cfg_size(c, "memory.expert_cache", 7) == 7);
    CHECK(tlm_cfg_is_auto(c, "memory.expert_cache"));
    tlm_cfg_free(c);
}

static void test_prefix_keys(void) {
    tlm_cfg *c = quiet_cfg();
    const char *text =
        "content.code = deep\n"
        "content.code.python = Deep\n"
        "content.lang.ja = exclude\n"
        "content.code.c_cpp = shallow\n"
        "hardware.cpu.cores = 24\n"
        "hardware.drive.0.path = /srv/tlm/nvme0\n"
        "hardware.mem.total = 128G\n";
    CHECK_EQ_INT(tlm_cfg_load_string(c, text, "p.conf", 1), 0);
    CHECK(strcmp(tlm_cfg_str(c, "content.code.python"), "deep") == 0);
    CHECK(strcmp(tlm_cfg_str(c, "content.lang.ja"), "exclude") == 0);
    CHECK_EQ_INT(tlm_cfg_int(c, "hardware.cpu.cores", 0), 24);
    CHECK(tlm_cfg_size(c, "hardware.mem.total", 0) == 128ull << 30);
    CHECK(tlm_cfg_str(c, "content.law") == NULL);
    int rc = tlm_cfg_load_string(c, "content.law = maybe\n", "p2.conf", 1);
    CHECK_EQ_INT(rc, TLM_ERANGE);
    CHECK(strstr(tlm_errmsg(), "p2.conf:1: content.law") != NULL);
    CHECK(strstr(tlm_errmsg(), "exclude|shallow|standard|deep") != NULL);
    CHECK_EQ_INT(tlm_cfg_set(c, "content.medicine", "exclude"), 0);
    CHECK_EQ_INT(tlm_cfg_set(c, "content", "deep"), TLM_ENOTFOUND);
    CHECK_EQ_INT(tlm_cfg_set(c, "made.up", "1"), TLM_ENOTFOUND);
    CHECK(strstr(tlm_errmsg(), "unknown key 'made.up'") != NULL);

    /* sorted iteration over schema keys plus extra prefix keys */
    int it = 0, n = 0, sorted = 1, saw_python = 0, saw_hw = 0;
    const char *k, *v, *prev = NULL;
    while (tlm_cfg_next(c, &it, &k, &v)) {
        if (prev && strcmp(prev, k) >= 0) sorted = 0;
        if (strcmp(k, "content.code.python") == 0) saw_python = 1;
        if (strcmp(k, "hardware.drive.0.path") == 0 && strcmp(v, "/srv/tlm/nvme0") == 0) saw_hw = 1;
        prev = k;
        n++;
    }
    CHECK(sorted);
    CHECK(saw_python && saw_hw);
    CHECK(n > 40);
    tlm_cfg_free(c);
}

static void test_precedence_env(void) {
    tlm_cfg *c = quiet_cfg();
    CHECK_EQ_INT(tlm_cfg_load_string(c, "cpu.threads = 8\nio.queue_depth = 16\ncontent.code.c_cpp = deep\n", "f.conf", 1), 0);
    setenv("TIERLM_CPU_THREADS", "12", 1);
    setenv("TIERLM_CPU_THREAD_GROUPS", "none", 1);          /* underscore inside a key segment */
    setenv("TIERLM_CONTENT_CODE_C_CPP", "exclude", 1);      /* existing key with '_' */
    setenv("TIERLM_CONTENT_LAW", "exclude", 1);             /* new prefix key */
    setenv("TIERLM_KV_VRAM_FRAC", "0.5", 1);                /* invalid: out of range */
    int rc = tlm_cfg_apply_env(c);
    CHECK_EQ_INT(rc, TLM_ERANGE);
    CHECK(strstr(tlm_errmsg(), "env TIERLM_KV_VRAM_FRAC") != NULL);
    CHECK_EQ_INT(tlm_cfg_int(c, "cpu.threads", 0), 12);
    CHECK(strcmp(tlm_cfg_str(c, "cpu.thread_groups"), "none") == 0);
    CHECK(strcmp(tlm_cfg_str(c, "content.code.c_cpp"), "exclude") == 0);
    CHECK(strcmp(tlm_cfg_str(c, "content.law"), "exclude") == 0);
    CHECK_NEAR(tlm_cfg_frac(c, "kv.vram_frac", 0), 0.05, 1e-12);
    CHECK_EQ_INT(tlm_cfg_int(c, "io.queue_depth", 0), 16);              /* file value untouched */
    const char *where = NULL;
    CHECK_EQ_INT(tlm_cfg_origin_of(c, "cpu.threads", &where), TLM_ORIGIN_ENV);
    CHECK(strcmp(where, "env TIERLM_CPU_THREADS") == 0);
    CHECK_EQ_INT(tlm_cfg_origin_of(c, "io.queue_depth", &where), TLM_ORIGIN_FILE);
    CHECK(strcmp(where, "f.conf:2") == 0);
    /* a later file load cannot override env; an explicit set (flag) can */
    CHECK_EQ_INT(tlm_cfg_load_string(c, "cpu.threads = 4\nio.queue_depth = 64\n", "g.conf", 1), 0);
    CHECK_EQ_INT(tlm_cfg_int(c, "cpu.threads", 0), 12);
    CHECK_EQ_INT(tlm_cfg_int(c, "io.queue_depth", 0), 64);              /* file over file: later wins */
    CHECK_EQ_INT(tlm_cfg_set_origin(c, "cpu.threads", "20", TLM_ORIGIN_SET, "--threads"), 0);
    CHECK_EQ_INT(tlm_cfg_int(c, "cpu.threads", 0), 20);
    CHECK_EQ_INT(tlm_cfg_origin_of(c, "cpu.threads", &where), TLM_ORIGIN_SET);
    CHECK(strcmp(where, "--threads") == 0);
    CHECK_EQ_INT(tlm_cfg_apply_env(c), TLM_ERANGE);
    CHECK_EQ_INT(tlm_cfg_int(c, "cpu.threads", 0), 20);                 /* env below set */
    CHECK_EQ_INT(tlm_cfg_set(c, "cpu.threads", "0"), TLM_ERANGE);       /* validated */
    CHECK_EQ_INT(tlm_cfg_int(c, "cpu.threads", 0), 20);
    CHECK_EQ_INT(tlm_cfg_unset(c, "cpu.threads"), 0);
    CHECK(tlm_cfg_str(c, "cpu.threads") == NULL);
    unsetenv("TIERLM_CPU_THREADS");
    unsetenv("TIERLM_CPU_THREAD_GROUPS");
    unsetenv("TIERLM_CONTENT_CODE_C_CPP");
    unsetenv("TIERLM_CONTENT_LAW");
    unsetenv("TIERLM_KV_VRAM_FRAC");
    tlm_cfg_free(c);
}

static void test_includes(void) {
    char path[1024];
    put_file("inc/main.conf",
             "cpu.threads = 2\n"
             "include parts/*.conf\n"
             "include \"/nonexistent-dir-tlm/*.conf\"\n"     /* glob without matches is fine */
             "io.queue_depth = 8\n");
    put_file("inc/parts/20-b.conf", "cpu.threads = 20\nkv.page_tokens = 128\n");
    put_file("inc/parts/10-a.conf", "cpu.threads = 10\nio.queue_depth = 99\nkv.page_tokens = 32\n");
    put_file("inc/parts/README", "not a conf file = ignored by the glob\n");
    tlm_cfg *c = quiet_cfg();
    snprintf(path, sizeof path, "%s/inc/main.conf", g_tmp);
    int rc = tlm_cfg_load_file(c, path, 1);
    if (rc) fprintf(stderr, "    %s\n", tlm_errmsg());
    CHECK_EQ_INT(rc, 0);
    CHECK_EQ_INT(tlm_cfg_int(c, "cpu.threads", 0), 20);           /* lexical: 10-a then 20-b */
    CHECK_EQ_INT(tlm_cfg_int(c, "kv.page_tokens", 0), 128);
    CHECK_EQ_INT(tlm_cfg_int(c, "io.queue_depth", 0), 8);         /* line after include wins */
    const char *where = NULL;
    tlm_cfg_origin_of(c, "cpu.threads", &where);
    CHECK(strstr(where, "inc/parts/20-b.conf:1") != NULL);
    tlm_cfg_free(c);

    /* missing literal include: warning, strict error */
    put_file("inc2/main.conf", "include missing.conf\n");
    snprintf(path, sizeof path, "%s/inc2/main.conf", g_tmp);
    c = quiet_cfg();
    CHECK_EQ_INT(tlm_cfg_load_file(c, path, 0), 0);
    CHECK(has_diag(c, "no such file"));
    CHECK(tlm_cfg_load_file(c, path, 1) != 0);
    tlm_cfg_free(c);

    /* depth: 4 nested includes are allowed, 5 are not */
    put_file("deep/d0.conf", "include d1.conf\n");
    put_file("deep/d1.conf", "include d2.conf\n");
    put_file("deep/d2.conf", "include d3.conf\n");
    put_file("deep/d3.conf", "include d4.conf\n");
    put_file("deep/d4.conf", "cpu.threads = 4\n");
    snprintf(path, sizeof path, "%s/deep/d0.conf", g_tmp);
    c = quiet_cfg();
    CHECK_EQ_INT(tlm_cfg_load_file(c, path, 1), 0);
    CHECK_EQ_INT(tlm_cfg_int(c, "cpu.threads", 0), 4);
    tlm_cfg_free(c);
    put_file("deep/d4.conf", "include d5.conf\n");
    put_file("deep/d5.conf", "cpu.threads = 5\n");
    c = quiet_cfg();
    rc = tlm_cfg_load_file(c, path, 0);
    CHECK_EQ_INT(rc, TLM_EFORMAT);
    CHECK(strstr(tlm_errmsg(), "d4.conf:1: include nesting deeper than 4") != NULL);
    tlm_cfg_free(c);
    /* a cycle stops at the depth limit instead of looping */
    put_file("cyc/a.conf", "include b.conf\n");
    put_file("cyc/b.conf", "include a.conf\n");
    snprintf(path, sizeof path, "%s/cyc/a.conf", g_tmp);
    c = quiet_cfg();
    CHECK_EQ_INT(tlm_cfg_load_file(c, path, 0), TLM_EFORMAT);
    tlm_cfg_free(c);
    /* missing top-level file */
    c = quiet_cfg();
    snprintf(path, sizeof path, "%s/none.conf", g_tmp);
    CHECK_EQ_INT(tlm_cfg_load_file(c, path, 0), TLM_ENOTFOUND);
    tlm_cfg_free(c);
}

static void test_load_etc(void) {
    char etc[1024];
    snprintf(etc, sizeof etc, "%s/etc", g_tmp);
    put_file("etc/hardware.conf", "hardware.cpu.cores = 24\nhardware.io_uring = yes\n");
    put_file("etc/tierlm.conf",
             "include conf.d/*.conf\n"
             "cpu.threads = 8\n"            /* conf.d must still win */
             "io.queue_depth = 16\n"
             "log.level = info\n");
    put_file("etc/conf.d/50-local.conf", "cpu.threads = 6\nweird.key = 1\n");
    put_file("etc/conf.d/90-last.conf", "io.queue_depth = 48\n");
    put_file("etc/conf.d/README", "this is not loaded\n");
    setenv("TIERLM_LOG_LEVEL", "debug", 1);
    tlm_cfg *c = quiet_cfg();
    int rc = tlm_cfg_load_etc(c, etc, 0);
    if (rc) fprintf(stderr, "    %s\n", tlm_errmsg());
    CHECK_EQ_INT(rc, 0);
    CHECK_EQ_INT(tlm_cfg_int(c, "cpu.threads", 0), 6);
    CHECK_EQ_INT(tlm_cfg_int(c, "io.queue_depth", 0), 48);
    CHECK_EQ_INT(tlm_cfg_int(c, "hardware.cpu.cores", 0), 24);
    CHECK(strcmp(tlm_cfg_str(c, "log.level"), "debug") == 0);
    const char *where = NULL;
    CHECK_EQ_INT(tlm_cfg_origin_of(c, "cpu.threads", &where), TLM_ORIGIN_FILE);
    CHECK(strstr(where, "conf.d/50-local.conf:1") != NULL);
    CHECK_EQ_INT(tlm_cfg_origin_of(c, "log.level", &where), TLM_ORIGIN_ENV);
    int unknown = 0;
    for (int i = 0; i < tlm_cfg_ndiag(c); i++)
        if (strstr(tlm_cfg_diag(c, i, NULL), "weird.key")) unknown++;
    CHECK_EQ_INT(unknown, 1);                                   /* reported once, not twice */
    tlm_cfg_free(c);
    /* strict: unknown key in conf.d is fatal and names the file and line */
    c = quiet_cfg();
    rc = tlm_cfg_load_etc(c, etc, 1);
    CHECK_EQ_INT(rc, TLM_EINVAL);
    CHECK(strstr(tlm_errmsg(), "50-local.conf:2: unknown key 'weird.key'") != NULL);
    tlm_cfg_free(c);
    unsetenv("TIERLM_LOG_LEVEL");
    /* missing etc dir: defaults only */
    c = quiet_cfg();
    snprintf(etc, sizeof etc, "%s/no-etc", g_tmp);
    CHECK_EQ_INT(tlm_cfg_load_etc(c, etc, 1), 0);
    CHECK(tlm_cfg_is_auto(c, "cpu.threads"));
    tlm_cfg_free(c);
}

static void test_check(void) {
    tlm_cfg *c = quiet_cfg();
    CHECK_EQ_INT(tlm_cfg_check(c), 0);
    CHECK_EQ_INT(tlm_cfg_set(c, "decide.ask_threshold", "0.95"), 0);
    CHECK_EQ_INT(tlm_cfg_check(c), TLM_EINVAL);
    CHECK(strstr(tlm_errmsg(), "decide.ask_threshold") != NULL);
    tlm_cfg_free(c);
}

static void test_writer_roundtrip(void) {
    tlm_cfg_line lines[] = {
        {NULL, NULL, "-- CPU --"},
        {"cpu.threads", "24", "1 per physical core"},
        {"cpu.thread_groups", "ccd", NULL},
        {NULL, NULL, NULL},
        {"paths.models", "/srv/a#b", "hash needs quotes"},
        {"paths.log", " lead", NULL},
        {"paths.kv_store", "q\"uote\\x", NULL},
        {"io.drives", "/srv/tlm/nvme0, /srv/tlm/nvme1", NULL},
        {"experts.exclude", "", NULL},
        {"content.code.python", "deep", NULL},
        {"include", "conf.d/*.conf", NULL},
    };
    int n = (int)(sizeof lines / sizeof lines[0]);
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    CHECK_EQ_INT(tlm_cfg_write(f, "test header\nsecond line", lines, n), 0);
    fclose(f);
    CHECK(strncmp(buf, "# test header\n# second line\n\n", 29) == 0);
    CHECK(strstr(buf, "cpu.threads            = 24            # 1 per physical core\n") != NULL);
    CHECK(strstr(buf, "paths.models           = \"/srv/a#b\"    # hash needs quotes\n") != NULL);
    CHECK(strstr(buf, "include conf.d/*.conf\n") != NULL);
    CHECK(strstr(buf, "# -- CPU --\n") != NULL);
    CHECK(strstr(buf, "\nexperts.exclude        =\n") != NULL);             /* empty: no trailing space */
    tlm_cfg *c = quiet_cfg();
    char name[1100];
    snprintf(name, sizeof name, "%s/rt.conf", g_tmp);               /* include resolved next to it */
    int rc = tlm_cfg_load_string(c, buf, name, 1);
    if (rc) fprintf(stderr, "    %s\n%s", tlm_errmsg(), buf);
    CHECK_EQ_INT(rc, 0);
    for (int i = 0; i < n; i++) {
        if (!lines[i].key || strcmp(lines[i].key, "include") == 0) continue;
        const char *v = tlm_cfg_str(c, lines[i].key);
        if (!v || strcmp(v, lines[i].value) != 0) fprintf(stderr, "    %s: '%s' vs '%s'\n", lines[i].key, v, lines[i].value);
        CHECK(v && strcmp(v, lines[i].value) == 0);
    }
    tlm_cfg_free(c);
    free(buf);
    /* writer refuses what it cannot represent */
    tlm_cfg_line bad[] = {{"paths.log", "a\nb", NULL}};
    f = open_memstream(&buf, &len);
    CHECK_EQ_INT(tlm_cfg_write(f, NULL, bad, 1), TLM_EINVAL);
    fclose(f);
    free(buf);
}

static int tax_cb(void *ctx, const char *key, const char *value, int line) {
    int *n = (int *)ctx;
    if (line == 2) CHECK(strcmp(key, "code.python") == 0 && strcmp(value, "Python # not a comment") == 0);
    (*n)++;
    return 0;
}

static void test_parse_text(void) {
    int n = 0;
    CHECK_EQ_INT(tlm_cfg_parse_text("code = Programming\ncode.python = \"Python # not a comment\" # c\n\n# x\n",
                                    "tax", tax_cb, &n), 0);
    CHECK_EQ_INT(n, 2);
    CHECK_EQ_INT(tlm_cfg_parse_text("ok = 1\nbroken line\n", "tax", tax_cb, &n), TLM_EFORMAT);
    CHECK(strstr(tlm_errmsg(), "tax:2:") != NULL);
}

int main(void) {
    printf("test_cfg\n");
    const char *td = getenv("TMPDIR");
    snprintf(g_tmp, sizeof g_tmp, "%s/tlm-test-cfg-XXXXXX", td && *td ? td : "/tmp");
    if (!mkdtemp(g_tmp)) { perror("mkdtemp"); return 2; }
    RUN(test_defaults);
    RUN(test_syntax);
    RUN(test_values_and_ranges);
    RUN(test_sizes);
    RUN(test_prefix_keys);
    RUN(test_precedence_env);
    RUN(test_includes);
    RUN(test_load_etc);
    RUN(test_check);
    RUN(test_writer_roundtrip);
    RUN(test_parse_text);
    rm_rf(g_tmp);
    TEST_MAIN_END();
}
