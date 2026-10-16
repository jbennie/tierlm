/* test_bpe.c - byte-level BPE tokenizer: byte-exact encode/decode against a
 * frozen reference-oracle fixture (tokenizers 0.23.2 shape: ByteLevel decoder,
 * GPT-2 byte chars, Qwen pre-tokeniser, ordered merges). The fixture is a
 * minimal BPE vocab (all 256 GPT-2 byte chars + aa/YY/YYY/(\n+space) merges)
 * plus two added tokens; every expected id/text below was captured from the
 * reference and matches the C implementation byte-for-byte.
 *
 * Regression coverage (each case fails on the corresponding past bug):
 *   - ordered-merge leftmost tie-break: "aaYYY" -> [aa, YYY], not [aa, Y, YY]
 *     (the rank-0 "aa" merge shuffles the heap, exposing a wrong overlapping
 *      pair pick among the equal-rank YY pairs of the YYY run)
 *   - pre-tokeniser whitespace/newline run: "a\n \nb" -> [a, \n+space, \n, b]
 *     (a lone-newline stop instead of the maximal \s*[\r\n]+ run mis-splits it)
 *   - ByteLevel decode, byte-exact per-codepoint output: ASCII + high-byte
 *     (raw-byte branch) and a CJK added token (raw-UTF-8 fallback branch)
 *   - added-token match (allow_added) and skip_special decode.
 *
 * The fixture JSON is hex-encoded below (self-contained; written to a temp dir
 * at test time and loaded via tlm_bpe_load). */
#include "tlm_bpe.h"
#include "tlm_platform.h"
#include "tlm_test.h"

#include <unistd.h>

static const char FIXTURE_HEX[] =
    "7b2276657273696f6e223a22312e30222c227472756e636174696f6e223a6e75"
    "6c6c2c2270616464696e67223a6e756c6c2c2261646465645f746f6b656e7322"
    "3a5b7b226964223a3236302c22636f6e74656e74223a223c656f733e222c2273"
    "696e676c655f776f7264223a66616c73652c226c7374726970223a66616c7365"
    "2c22727374726970223a66616c73652c226e6f726d616c697a6564223a66616c"
    "73652c227370656369616c223a747275657d2c7b226964223a3236312c22636f"
    "6e74656e74223a22e4b8ad222c2273696e676c655f776f7264223a66616c7365"
    "2c226c7374726970223a66616c73652c22727374726970223a66616c73652c22"
    "6e6f726d616c697a6564223a66616c73652c227370656369616c223a66616c73"
    "657d5d2c226e6f726d616c697a6572223a7b2274797065223a224e4643227d2c"
    "227072655f746f6b656e697a6572223a7b2274797065223a2253657175656e63"
    "65222c22707265746f6b656e697a657273223a5b7b2274797065223a2253706c"
    "6974222c227061747465726e223a7b225265676578223a22283f693a27737c27"
    "747c2772657c2776657c276d7c276c6c7c2764297c5b5e5c5c725c5c6e5c5c70"
    "7b4c7d5c5c707b4e7d5d3f5c5c707b4c7d2b7c5c5c707b4e7d7c203f5b5e5c5c"
    "735c5c707b4c7d5c5c707b4e7d5d2b5b5c5c725c5c6e5d2a7c5c5c732a5b5c5c"
    "725c5c6e5d2b7c5c5c732b283f215c5c53297c5c5c732b227d2c226265686176"
    "696f72223a2249736f6c61746564222c22696e76657274223a66616c73657d2c"
    "7b2274797065223a22427974654c6576656c222c226164645f7072656669785f"
    "7370616365223a66616c73652c227472696d5f6f666673657473223a66616c73"
    "652c227573655f7265676578223a66616c73657d5d7d2c22706f73745f70726f"
    "636573736f72223a6e756c6c2c226465636f646572223a7b2274797065223a22"
    "427974654c6576656c222c226164645f7072656669785f7370616365223a6661"
    "6c73652c227472696d5f6f666673657473223a66616c73652c227573655f7265"
    "676578223a66616c73657d2c226d6f64656c223a7b2274797065223a22425045"
    "222c2264726f706f7574223a6e756c6c2c22756e6b5f746f6b656e223a6e756c"
    "6c2c22667573655f756e6b223a66616c73652c22766f636162223a7b22c48022"
    "3a302c22c481223a312c22c482223a322c22c483223a332c22c484223a342c22"
    "c485223a352c22c486223a362c22c487223a372c22c488223a382c22c489223a"
    "392c22c48a223a31302c22c48b223a31312c22c48c223a31322c22c48d223a31"
    "332c22c48e223a31342c22c48f223a31352c22c490223a31362c22c491223a31"
    "372c22c492223a31382c22c493223a31392c22c494223a32302c22c495223a32"
    "312c22c496223a32322c22c497223a32332c22c498223a32342c22c499223a32"
    "352c22c49a223a32362c22c49b223a32372c22c49c223a32382c22c49d223a32"
    "392c22c49e223a33302c22c49f223a33312c22c4a0223a33322c2221223a3333"
    "2c225c22223a33342c2223223a33352c2224223a33362c2225223a33372c2226"
    "223a33382c2227223a33392c2228223a34302c2229223a34312c222a223a3432"
    "2c222b223a34332c222c223a34342c222d223a34352c222e223a34362c222f22"
    "3a34372c2230223a34382c2231223a34392c2232223a35302c2233223a35312c"
    "2234223a35322c2235223a35332c2236223a35342c2237223a35352c2238223a"
    "35362c2239223a35372c223a223a35382c223b223a35392c223c223a36302c22"
    "3d223a36312c223e223a36322c223f223a36332c2240223a36342c2241223a36"
    "352c2242223a36362c2243223a36372c2244223a36382c2245223a36392c2246"
    "223a37302c2247223a37312c2248223a37322c2249223a37332c224a223a3734"
    "2c224b223a37352c224c223a37362c224d223a37372c224e223a37382c224f22"
    "3a37392c2250223a38302c2251223a38312c2252223a38322c2253223a38332c"
    "2254223a38342c2255223a38352c2256223a38362c2257223a38372c2258223a"
    "38382c2259223a38392c225a223a39302c225b223a39312c225c5c223a39322c"
    "225d223a39332c225e223a39342c225f223a39352c2260223a39362c2261223a"
    "39372c2262223a39382c2263223a39392c2264223a3130302c2265223a313031"
    "2c2266223a3130322c2267223a3130332c2268223a3130342c2269223a313035"
    "2c226a223a3130362c226b223a3130372c226c223a3130382c226d223a313039"
    "2c226e223a3131302c226f223a3131312c2270223a3131322c2271223a313133"
    "2c2272223a3131342c2273223a3131352c2274223a3131362c2275223a313137"
    "2c2276223a3131382c2277223a3131392c2278223a3132302c2279223a313231"
    "2c227a223a3132322c227b223a3132332c227c223a3132342c227d223a313235"
    "2c227e223a3132362c22c4a1223a3132372c22c4a2223a3132382c22c4a3223a"
    "3132392c22c4a4223a3133302c22c4a5223a3133312c22c4a6223a3133322c22"
    "c4a7223a3133332c22c4a8223a3133342c22c4a9223a3133352c22c4aa223a31"
    "33362c22c4ab223a3133372c22c4ac223a3133382c22c4ad223a3133392c22c4"
    "ae223a3134302c22c4af223a3134312c22c4b0223a3134322c22c4b1223a3134"
    "332c22c4b2223a3134342c22c4b3223a3134352c22c4b4223a3134362c22c4b5"
    "223a3134372c22c4b6223a3134382c22c4b7223a3134392c22c4b8223a313530"
    "2c22c4b9223a3135312c22c4ba223a3135322c22c4bb223a3135332c22c4bc22"
    "3a3135342c22c4bd223a3135352c22c4be223a3135362c22c4bf223a3135372c"
    "22c580223a3135382c22c581223a3135392c22c582223a3136302c22c2a1223a"
    "3136312c22c2a2223a3136322c22c2a3223a3136332c22c2a4223a3136342c22"
    "c2a5223a3136352c22c2a6223a3136362c22c2a7223a3136372c22c2a8223a31"
    "36382c22c2a9223a3136392c22c2aa223a3137302c22c2ab223a3137312c22c2"
    "ac223a3137322c22c583223a3137332c22c2ae223a3137342c22c2af223a3137"
    "352c22c2b0223a3137362c22c2b1223a3137372c22c2b2223a3137382c22c2b3"
    "223a3137392c22c2b4223a3138302c22c2b5223a3138312c22c2b6223a313832"
    "2c22c2b7223a3138332c22c2b8223a3138342c22c2b9223a3138352c22c2ba22"
    "3a3138362c22c2bb223a3138372c22c2bc223a3138382c22c2bd223a3138392c"
    "22c2be223a3139302c22c2bf223a3139312c22c380223a3139322c22c381223a"
    "3139332c22c382223a3139342c22c383223a3139352c22c384223a3139362c22"
    "c385223a3139372c22c386223a3139382c22c387223a3139392c22c388223a32"
    "30302c22c389223a3230312c22c38a223a3230322c22c38b223a3230332c22c3"
    "8c223a3230342c22c38d223a3230352c22c38e223a3230362c22c38f223a3230"
    "372c22c390223a3230382c22c391223a3230392c22c392223a3231302c22c393"
    "223a3231312c22c394223a3231322c22c395223a3231332c22c396223a323134"
    "2c22c397223a3231352c22c398223a3231362c22c399223a3231372c22c39a22"
    "3a3231382c22c39b223a3231392c22c39c223a3232302c22c39d223a3232312c"
    "22c39e223a3232322c22c39f223a3232332c22c3a0223a3232342c22c3a1223a"
    "3232352c22c3a2223a3232362c22c3a3223a3232372c22c3a4223a3232382c22"
    "c3a5223a3232392c22c3a6223a3233302c22c3a7223a3233312c22c3a8223a32"
    "33322c22c3a9223a3233332c22c3aa223a3233342c22c3ab223a3233352c22c3"
    "ac223a3233362c22c3ad223a3233372c22c3ae223a3233382c22c3af223a3233"
    "392c22c3b0223a3234302c22c3b1223a3234312c22c3b2223a3234322c22c3b3"
    "223a3234332c22c3b4223a3234342c22c3b5223a3234352c22c3b6223a323436"
    "2c22c3b7223a3234372c22c3b8223a3234382c22c3b9223a3234392c22c3ba22"
    "3a3235302c22c3bb223a3235312c22c3bc223a3235322c22c3bd223a3235332c"
    "22c3be223a3235342c22c3bf223a3235352c226161223a3235362c225959223a"
    "3235372c22595959223a3235382c22c48ac4a0223a3235397d2c226d65726765"
    "73223a5b5b2261222c2261225d2c5b2259222c2259225d2c5b225959222c2259"
    "225d2c5b22c48a222c22c4a0225d5d7d7d";


static tlm_bpe *g_bpe;
static char     g_dir[512];

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* decode the hex fixture to <dir>/tokenizer.json; 0 on success */
static int write_fixture(const char *hex, const char *dir, size_t *nbytes) {
    char path[600];
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t i = 0;
    for (; hex[i] && hex[i + 1]; i += 2) {
        int hi = hexval(hex[i]), lo = hexval(hex[i + 1]);
        if (hi < 0 || lo < 0) { fclose(f); return -1; }
        if (fputc((hi << 4) | lo, f) == EOF) { fclose(f); return -1; }
    }
    *nbytes = i / 2;
    return fclose(f) == 0 ? 0 : -1;
}

static void test_load(void) {
    CHECK(g_bpe != NULL);
    CHECK_EQ_INT(tlm_bpe_vocab_size(g_bpe), 260);
    CHECK_EQ_INT(tlm_bpe_max_id(g_bpe), 261);
}

/* encode(allow_added=1): leftmost-longest added-token match, then BPE */
static const char *enc1_text[] = {
    "aaYYY", "YYY", "YY", "YYYY", "Y", "caf\xc3\xa9", "\xe4\xb8\xad",
    "\xf0\x9f\x98\x80", "a\n \nb", "x\n\ny", "a\r\nb\rc", "hi", "",
    "z\n  \nw", "\xe5\x89\x8d\xe4\xb8\xad", "a \n b", "aa"
};
static const int enc1_ids[] = {
    256, 258,                    /* aaYYY  -> [aa, YYY]   (tie-break)   */
    258,                         /* YYY    -> [YYY]                        */
    257,                         /* YY     -> [YY]                         */
    257, 257,                    /* YYYY   -> [YY, YY]                     */
    89,                          /* Y      -> [Y]                          */
    99, 97, 102, 195, 169,       /* cafe   -> [c a f, 2-byte Latin-1]      */
    261,                         /* 中      -> [added token]               */
    240, 159, 152, 128,          /* U+1F600 -> 4-byte emoji bytes          */
    97, 259, 10, 98,             /* a\n \nb  -> [a, \n+space, \n, b]       */
    120, 10, 10, 121,            /* x\n\ny   -> [x, \n, \n, y]             */
    97, 13, 10, 98, 13, 99,      /* a\r\nb\rc-> [a, \r, \n, b, \r, c]      */
    104, 105,                    /* hi     -> [h, i]                       */
    /* (empty)                                                            */
    122, 259, 32, 10, 119,       /* z\n  \nw -> [z, \n+space, space, \n, w]*/
    229, 137, 141, 261,          /* 前中     -> [前 3 bytes, 中 added]     */
    97, 32, 10, 32, 98,          /* a \n b   -> [a, sp, \n, sp, b]         */
    256                          /* aa     -> [aa]                         */
};
static const int enc1_len[] = {2, 1, 1, 2, 1, 5, 1, 4, 4, 4, 6, 2, 0, 5, 4, 5, 1};

static void test_encode_allow(void) {
    size_t off = 0;
    for (size_t t = 0; t < sizeof enc1_text / sizeof enc1_text[0]; t++) {
        const char *s = enc1_text[t];
        int n = (int)strlen(s);
        int cap = n * 4 + 8, r;
        int *ids = (int *)malloc(sizeof(int) * (size_t)cap);
        r = tlm_bpe_encode(g_bpe, s, (size_t)n, 1, ids, cap);
        CHECK_EQ_INT(r, enc1_len[t]);
        if (r > 0)
            for (int i = 0; i < r; i++)
                CHECK_EQ_INT(ids[i], enc1_ids[off + (size_t)i]);
        off += (size_t)enc1_len[t];
        free(ids);
    }
}

/* encode(allow_added=0): bare BPE, no added-token match. "中" is the byte
 * level of E4 B8 AD, not the added token 261. */
static const char *enc0_text[] = { "Y", "hi", "caf\xc3\xa9", "\xe4\xb8\xad" };
static const int enc0_ids[] = { 89, 104, 105, 99, 97, 102, 195, 169, 228, 184, 173 };
static const int enc0_len[] = { 1, 2, 5, 3 };

static void test_encode_bare(void) {
    size_t off = 0;
    for (size_t t = 0; t < sizeof enc0_text / sizeof enc0_text[0]; t++) {
        const char *s = enc0_text[t];
        int n = (int)strlen(s);
        int cap = n * 4 + 8, r;
        int *ids = (int *)malloc(sizeof(int) * (size_t)cap);
        r = tlm_bpe_encode(g_bpe, s, (size_t)n, 0, ids, cap);
        CHECK_EQ_INT(r, enc0_len[t]);
        if (r > 0)
            for (int i = 0; i < r; i++)
                CHECK_EQ_INT(ids[i], enc0_ids[off + (size_t)i]);
        off += (size_t)enc0_len[t];
        free(ids);
    }
}

/* decode(ids, skip_special): ByteLevel chain, lossy UTF-8 at the end */
static const int dec_ids[] = {
    256, 258,               /* aaYYY                  */
    258,                    /* YYY                     */
    99, 97, 102, 195, 169,  /* cafe  (2-byte Latin-1)  */
    228, 184, 173,          /* 中       (3-byte CJK)   */
    240, 159, 152, 128,     /* U+1F600 (4-byte emoji)  */
    97, 259, 10, 98,        /* a\n \nb  (\n+space fused) */
    261, 89,                /* 中Y     (3-byte via added) */
    260, 89, 260,           /* skip=1 -> Y             */
    260, 89, 260,           /* skip=0 -> <eos>Y<eos>    */
    257, 258,               /* YY YYY -> YYYYY          */
    /* (empty)                                */
};
static const int dec_len[]  = {2, 1, 5, 3, 4, 4, 2, 3, 3, 2, 0};
static const int dec_skip[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1};
static const char *dec_exp[] = {
    "aaYYY", "YYY", "caf\xc3\xa9", "\xe4\xb8\xad", "\xf0\x9f\x98\x80",
    "a\n \nb", "\xe4\xb8\xadY", "Y", "<eos>Y<eos>", "YYYYY", ""
};

static void test_decode(void) {
    size_t off = 0;
    for (size_t t = 0; t < sizeof dec_len / sizeof dec_len[0]; t++) {
        char out[512];
        int r = tlm_bpe_decode(g_bpe, dec_ids + off, dec_len[t], dec_skip[t],
                               out, sizeof out);
        if (r >= 0) out[r] = 0;
        CHECK_EQ_INT(r, (int)strlen(dec_exp[t]));
        if (r >= 0) CHECK(strcmp(out, dec_exp[t]) == 0);
        off += (size_t)dec_len[t];
    }
}

int main(void) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || !tmp[0]) tmp = "/tmp";
    snprintf(g_dir, sizeof g_dir, "%s/tlm-test-bpe-%ld", tmp, (long)getpid());
    CHECK_EQ_INT(tlm_mkdirs(g_dir, 0700), 0);
    size_t nbytes = 0;
    CHECK_EQ_INT(write_fixture(FIXTURE_HEX, g_dir, &nbytes), 0);
    char err[256] = "";
    g_bpe = tlm_bpe_load(g_dir, err, sizeof err);
    RUN(test_load);
    RUN(test_encode_allow);
    RUN(test_encode_bare);
    RUN(test_decode);
    if (g_bpe) tlm_bpe_free(g_bpe);
    (void)remove(g_dir);
    (void)rmdir(g_dir);
    TEST_MAIN_END();
}
