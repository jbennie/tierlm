/* test_json.c - strict JSON reader/writer: valid and invalid corpus, unicode,
 * depth limit, numbers, getters, writer round trip. */
#include "tlm_json.h"
#include "tlm_test.h"

#include <stdint.h>

static tlm_json *P(const char *s) {
    char err[256];
    return tlm_json_parse(s, strlen(s), err, sizeof err);
}

static int parses(const char *s) {
    tlm_json *j = P(s);
    int ok = j != NULL;
    tlm_json_free(j);
    return ok;
}

static void test_valid_corpus(void) {
    static const char *ok[] = {
        "null", "true", "false", "0", "-0", "1", "-1", "0.5", "-0.5e-3", "1E10", "1e+2",
        "\"\"", "\"abc\"", "[]", "{}", " [ ] ", "\t{\n}\r\n", "[1,2,3]", "[[[]]]",
        "{\"a\":1}", "{\"a\":{\"b\":[true,false,null]}}", "{\"\":0}",
        "\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"", "\"\\u0041\"", "[1.0e0, -2.5E-1, 3e2]",
        "\"caf\xc3\xa9\"", "\"\xe2\x82\xac\"", "\"\xf0\x9f\x98\x80\"",
        "\xef\xbb\xbf{}",                              /* leading BOM */
        "{\"a\":1,\"a\":2}",                           /* duplicate keys allowed */
        "\"\\u0000\"",                                 /* NUL in a value is fine */
        NULL};
    for (int i = 0; ok[i]; i++) {
        if (!parses(ok[i])) fprintf(stderr, "    should parse: %s\n", ok[i]);
        CHECK(parses(ok[i]));
    }
}

static void test_invalid_corpus(void) {
    static const char *bad[] = {
        "", " ", "nul", "tru", "falsee", "True", "NULL", "[", "]", "{", "}", "[1,]", "[,1]", "[1 2]",
        "{\"a\"}", "{\"a\":}", "{\"a\":1,}", "{a:1}", "{'a':1}", "{\"a\" 1}", "{1:2}",
        "01", "-", "+1", "1.", ".5", "1e", "1e+", "0x10", "NaN", "Infinity", "-Infinity", "1e400",
        "\"abc", "\"\\x\"", "\"\\u12\"", "\"\\u12G4\"", "\"\\ud800\"", "\"\\udc00\"",
        "\"\\ud800\\u0041\"", "\"a\nb\"", "\"tab\there\"", "\"\x01\"",
        "\"\xc3\"", "\"\xc0\x80\"", "\"\xed\xa0\x80\"", "\"\xf5\x80\x80\x80\"", "\"\xff\"",
        "1 2", "[] []", "{}x", "{\"a\\u0000b\":1}", "// c\n1", "[1] // c",
        NULL};
    for (int i = 0; bad[i]; i++) {
        char err[256] = "";
        tlm_json *j = tlm_json_parse(bad[i], strlen(bad[i]), err, sizeof err);
        if (j) fprintf(stderr, "    should fail: %s\n", bad[i]);
        CHECK(j == NULL);
        CHECK(strstr(err, "offset") != NULL);
        tlm_json_free(j);
    }
    /* embedded NUL after a valid value (explicit length) */
    char err[256];
    CHECK(tlm_json_parse("[1]\0", 4, err, sizeof err) == NULL);
    CHECK(strstr(err, "offset 3") != NULL);
    /* error offset points at the problem */
    CHECK(tlm_json_parse("{\"a\":1,}", 8, err, sizeof err) == NULL);
    CHECK(strstr(err, "offset 7") != NULL);
    CHECK(tlm_json_parse("[1,\n  x]", 8, err, sizeof err) == NULL);
    CHECK(strstr(err, "line 2, column 3") != NULL);
    /* NULL input and no error buffer must not crash */
    CHECK(tlm_json_parse(NULL, 0, err, sizeof err) == NULL);
    CHECK(tlm_json_parse("[", 1, NULL, 0) == NULL);
}

static void test_unicode(void) {
    tlm_json *j = P("[\"\\u00e9\", \"\\ud83d\\ude00\", \"\\u20AC\", \"a\\u0000b\", \"\\u007f\"]");
    CHECK(j != NULL);
    if (!j) return;
    const tlm_jval *r = tlm_json_root(j);
    CHECK_EQ_INT(tlm_jlen(r), 5);
    const tlm_jval *s0 = tlm_jat(r, 0), *s1 = tlm_jat(r, 1), *s2 = tlm_jat(r, 2), *s3 = tlm_jat(r, 3);
    CHECK(strcmp(tlm_jstr(s0, ""), "\xc3\xa9") == 0);
    CHECK(strcmp(tlm_jstr(s1, ""), "\xf0\x9f\x98\x80") == 0);
    CHECK_EQ_INT(s1->u.str.len, 4);
    CHECK(strcmp(tlm_jstr(s2, ""), "\xe2\x82\xac") == 0);
    CHECK_EQ_INT(s3->u.str.len, 3);
    CHECK(memcmp(s3->u.str.s, "a\0b", 3) == 0);
    tlm_json_free(j);

    /* highest code point via surrogates and raw */
    j = P("\"\\udbff\\udfff\xf4\x8f\xbf\xbf\"");
    CHECK(j != NULL);
    if (j) {
        CHECK(memcmp(tlm_jstr(tlm_json_root(j), ""), "\xf4\x8f\xbf\xbf\xf4\x8f\xbf\xbf", 8) == 0);
        tlm_json_free(j);
    }
}

static void test_depth(void) {
    char buf[300];
    int n = TLM_JSON_MAX_DEPTH;
    for (int i = 0; i < n; i++) buf[i] = '[';
    for (int i = 0; i < n; i++) buf[n + i] = ']';
    buf[2 * n] = 0;
    CHECK(parses(buf));
    n = TLM_JSON_MAX_DEPTH + 1;
    for (int i = 0; i < n; i++) buf[i] = '[';
    for (int i = 0; i < n; i++) buf[n + i] = ']';
    buf[2 * n] = 0;
    char err[256];
    CHECK(tlm_json_parse(buf, strlen(buf), err, sizeof err) == NULL);
    CHECK(strstr(err, "nesting") != NULL);
    /* objects count too */
    char ob[700];
    size_t o = 0;
    for (int i = 0; i < TLM_JSON_MAX_DEPTH + 1; i++) o += (size_t)sprintf(ob + o, "{\"k\":");
    o += (size_t)sprintf(ob + o, "1");
    for (int i = 0; i < TLM_JSON_MAX_DEPTH + 1; i++) ob[o++] = '}';
    ob[o] = 0;
    CHECK(!parses(ob));
}

static void test_numbers(void) {
    tlm_json *j = P("[0, -0, 42, -17, 9223372036854775807, -9223372036854775808, 9223372036854775808,"
                    " 1.5, 1e3, 2.0, 1e-320, 123456789012345678901234567890, 0.1]");
    CHECK(j != NULL);
    if (!j) return;
    const tlm_jval *r = tlm_json_root(j);
    CHECK_EQ_INT(tlm_jint(tlm_jat(r, 0), 99), 0);
    CHECK(tlm_jat(r, 1)->u.num.is_int);
    CHECK(signbit(tlm_jnum(tlm_jat(r, 1), 1)));
    CHECK_EQ_INT(tlm_jint(tlm_jat(r, 2), 0), 42);
    CHECK_EQ_INT(tlm_jint(tlm_jat(r, 3), 0), -17);
    CHECK(tlm_jint(tlm_jat(r, 4), 0) == INT64_MAX);
    CHECK(tlm_jint(tlm_jat(r, 5), 0) == INT64_MIN);
    CHECK(!tlm_jat(r, 6)->u.num.is_int);                     /* 2^63 does not fit */
    CHECK_EQ_INT(tlm_jint(tlm_jat(r, 6), -1), -1);
    CHECK_NEAR(tlm_jnum(tlm_jat(r, 6), 0), 9223372036854775808.0, 1);
    CHECK(!tlm_jat(r, 7)->u.num.is_int);
    CHECK_NEAR(tlm_jnum(tlm_jat(r, 7), 0), 1.5, 0);
    CHECK_EQ_INT(tlm_jint(tlm_jat(r, 7), -1), -1);
    CHECK_EQ_INT(tlm_jint(tlm_jat(r, 8), 0), 1000);
    CHECK_EQ_INT(tlm_jint(tlm_jat(r, 9), 0), 2);
    CHECK(tlm_jnum(tlm_jat(r, 10), 1) > 0 && tlm_jnum(tlm_jat(r, 10), 1) < 1e-300);   /* denormal ok */
    CHECK(!tlm_jat(r, 11)->u.num.is_int);
    CHECK_NEAR(tlm_jnum(tlm_jat(r, 11), 0), 1.2345678901234568e29, 1e15);
    CHECK(tlm_jnum(tlm_jat(r, 12), 0) == 0.1);
    tlm_json_free(j);
}

static void test_getters(void) {
    tlm_json *j = P("{\"s\":\"x\",\"n\":3,\"b\":true,\"a\":[1,2],\"o\":{\"k\":null},\"d\":1,\"d\":2}");
    CHECK(j != NULL);
    if (!j) return;
    const tlm_jval *r = tlm_json_root(j);
    CHECK(strcmp(tlm_jtype_name(r->type), "object") == 0);
    CHECK(strcmp(tlm_jstr(tlm_jobj_get(r, "s"), "?"), "x") == 0);
    CHECK(strcmp(tlm_jstr(tlm_jobj_get(r, "n"), "?"), "?") == 0);       /* wrong type -> default */
    CHECK_EQ_INT(tlm_jint(tlm_jobj_get(r, "n"), 0), 3);
    CHECK_EQ_INT(tlm_jint(tlm_jobj_get(r, "s"), -5), -5);
    CHECK_EQ_INT(tlm_jbool(tlm_jobj_get(r, "b"), 0), 1);
    CHECK_EQ_INT(tlm_jbool(tlm_jobj_get(r, "n"), 7), 7);
    CHECK_EQ_INT(tlm_jlen(tlm_jobj_get(r, "a")), 2);
    CHECK_EQ_INT(tlm_jint(tlm_jat(tlm_jobj_get(r, "a"), 1), 0), 2);
    CHECK(tlm_jat(tlm_jobj_get(r, "a"), 2) == NULL);
    CHECK(tlm_jobj_get(tlm_jobj_get(r, "o"), "k")->type == TLM_J_NULL);
    CHECK(tlm_jobj_get(r, "missing") == NULL);
    CHECK(tlm_jobj_get(NULL, "x") == NULL);
    CHECK_EQ_INT(tlm_jint(tlm_jobj_get(r, "d"), 0), 2);                 /* last duplicate wins */
    CHECK_EQ_INT(tlm_jlen(r), 7);
    CHECK(strcmp(r->u.obj.keys[0], "s") == 0);                          /* insertion order */
    CHECK_NEAR(tlm_jnum(NULL, 1.25), 1.25, 0);
    tlm_json_free(j);
}

static void test_big_document(void) {
    /* many items: scratch stacks and arena blocks must grow without moving data */
    size_t cap = 2u << 20, o = 0;
    char *b = (char *)malloc(cap);
    o += (size_t)sprintf(b + o, "{\"items\":[");
    for (int i = 0; i < 50000; i++) o += (size_t)sprintf(b + o, "%s{\"id\":%d,\"name\":\"item-%d\"}", i ? "," : "", i, i);
    o += (size_t)sprintf(b + o, "],\"tail\":\"");
    for (int i = 0; i < 200000; i++) b[o++] = 'z';                       /* string > one arena block */
    o += (size_t)sprintf(b + o, "\"}");
    char err[256];
    tlm_json *j = tlm_json_parse(b, o, err, sizeof err);
    CHECK(j != NULL);
    if (j) {
        const tlm_jval *items = tlm_jobj_get(tlm_json_root(j), "items");
        CHECK_EQ_INT(tlm_jlen(items), 50000);
        CHECK_EQ_INT(tlm_jint(tlm_jobj_get(tlm_jat(items, 49999), "id"), 0), 49999);
        CHECK(strcmp(tlm_jstr(tlm_jobj_get(tlm_jat(items, 123), "name"), ""), "item-123") == 0);
        CHECK_EQ_INT(tlm_jobj_get(tlm_json_root(j), "tail")->u.str.len, 200000);
        tlm_json_free(j);
    }
    free(b);
}

static void test_writer_roundtrip(void) {
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    tlm_jw w;
    tlm_jw_init(&w, f);
    tlm_jw_begin_obj(&w);
    tlm_jw_str(&w, "s", "quote\" back\\ nl\n tab\t ctl\x01 utf8 \xc3\xa9");
    tlm_jw_num(&w, "pi", 3.141592653589793);
    tlm_jw_num(&w, "tenth", 0.1);
    tlm_jw_num(&w, "nan", NAN);
    tlm_jw_int(&w, "big", INT64_MIN);
    tlm_jw_bool(&w, "t", 1);
    tlm_jw_null(&w, "z");
    tlm_jw_begin_arr(&w, "arr");
    tlm_jw_int(&w, NULL, 1);
    tlm_jw_begin_obj(&w);
    tlm_jw_str(&w, "k", "v");
    tlm_jw_end_obj(&w);
    tlm_jw_begin_arr(&w, NULL);
    tlm_jw_end_arr(&w);
    tlm_jw_str(&w, NULL, NULL);
    tlm_jw_end_arr(&w);
    tlm_jw_key_obj(&w, "nested");
    tlm_jw_num(&w, "x", -2.5e-8);
    tlm_jw_end_obj(&w);
    tlm_jw_end_obj(&w);
    fclose(f);
    char err[256];
    tlm_json *j = tlm_json_parse(buf, len, err, sizeof err);
    if (!j) fprintf(stderr, "    %s\n    %s\n", err, buf);
    CHECK(j != NULL);
    if (j) {
        const tlm_jval *r = tlm_json_root(j);
        CHECK(strcmp(tlm_jstr(tlm_jobj_get(r, "s"), ""), "quote\" back\\ nl\n tab\t ctl\x01 utf8 \xc3\xa9") == 0);
        CHECK(tlm_jnum(tlm_jobj_get(r, "pi"), 0) == 3.141592653589793);
        CHECK(tlm_jnum(tlm_jobj_get(r, "tenth"), 0) == 0.1);
        CHECK(tlm_jobj_get(r, "nan")->type == TLM_J_NULL);
        CHECK(tlm_jint(tlm_jobj_get(r, "big"), 0) == INT64_MIN);
        CHECK_EQ_INT(tlm_jbool(tlm_jobj_get(r, "t"), 0), 1);
        const tlm_jval *a = tlm_jobj_get(r, "arr");
        CHECK_EQ_INT(tlm_jlen(a), 4);
        CHECK(strcmp(tlm_jstr(tlm_jobj_get(tlm_jat(a, 1), "k"), ""), "v") == 0);
        CHECK(tlm_jat(a, 2)->type == TLM_J_ARR);
        CHECK(tlm_jat(a, 3)->type == TLM_J_NULL);
        CHECK(tlm_jnum(tlm_jobj_get(tlm_jobj_get(r, "nested"), "x"), 0) == -2.5e-8);
        tlm_json_free(j);
    }
    free(buf);

    char small[8];
    size_t need = tlm_json_escape(small, sizeof small, "a\"b\n", 4);
    CHECK_EQ_INT(need, 6);
    CHECK(strcmp(small, "a\\\"b\\n") == 0);
    need = tlm_json_escape(small, 4, "abcdef", 6);                     /* truncates, reports need */
    CHECK_EQ_INT(need, 6);
    CHECK(strcmp(small, "abc") == 0);
}

int main(void) {
    printf("test_json\n");
    RUN(test_valid_corpus);
    RUN(test_invalid_corpus);
    RUN(test_unicode);
    RUN(test_depth);
    RUN(test_numbers);
    RUN(test_getters);
    RUN(test_big_document);
    RUN(test_writer_roundtrip);
    TEST_MAIN_END();
}
