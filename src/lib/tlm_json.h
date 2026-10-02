/* tlm_json.h - small strict JSON reader/writer.
 * Arena-allocated DOM; depth limit 64; numbers parsed as double (and int64
 * when integral). Used for safetensors headers, config.json, manifests'
 * reports and the cluster/server protocols.
 */
#ifndef TLM_JSON_H
#define TLM_JSON_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { TLM_J_NULL, TLM_J_BOOL, TLM_J_NUM, TLM_J_STR, TLM_J_ARR, TLM_J_OBJ } tlm_jtype;

typedef struct tlm_jval tlm_jval;
struct tlm_jval {
    tlm_jtype type;
    union {
        int      b;
        struct { double d; int64_t i; int is_int; } num;
        struct { char *s; size_t len; } str;
        struct { tlm_jval **items; size_t n; } arr;
        struct { char **keys; tlm_jval **vals; size_t n; } obj;   /* insertion order */
    } u;
};

typedef struct tlm_json tlm_json;          /* owns the arena */
tlm_json *tlm_json_parse(const char *text, size_t len, char *err, size_t errcap);
void      tlm_json_free(tlm_json *j);
const tlm_jval *tlm_json_root(const tlm_json *j);

const tlm_jval *tlm_jobj_get(const tlm_jval *obj, const char *key);
const char     *tlm_jstr(const tlm_jval *v, const char *dflt);
double          tlm_jnum(const tlm_jval *v, double dflt);
int64_t         tlm_jint(const tlm_jval *v, int64_t dflt);
int             tlm_jbool(const tlm_jval *v, int dflt);
size_t          tlm_jlen(const tlm_jval *v);              /* array/object length */
const tlm_jval *tlm_jat(const tlm_jval *arr, size_t i);

/* Minimal streaming writer. */
typedef struct { FILE *f; int depth; int need_comma[64]; } tlm_jw;
void tlm_jw_begin_obj(tlm_jw *w);
void tlm_jw_end_obj(tlm_jw *w);
void tlm_jw_begin_arr(tlm_jw *w, const char *key);   /* key NULL inside arrays */
void tlm_jw_end_arr(tlm_jw *w);
void tlm_jw_key_obj(tlm_jw *w, const char *key);     /* "key": { */
void tlm_jw_str(tlm_jw *w, const char *key, const char *v);
void tlm_jw_num(tlm_jw *w, const char *key, double v);
void tlm_jw_int(tlm_jw *w, const char *key, int64_t v);
void tlm_jw_bool(tlm_jw *w, const char *key, int v);

/* ---- additions (implemented in tlm_json.c) --------------------------
 * Parser policy: RFC 8259 strict. Input must be valid UTF-8 (a leading
 * UTF-8 BOM is skipped); lone surrogates, raw control characters, leading
 * zeros, NaN/Infinity, trailing commas and trailing garbage are errors; a
 * number that overflows a double is an error. Duplicate object keys are
 * kept in document order and tlm_jobj_get returns the LAST one. Object keys
 * containing U+0000 are rejected. Errors read "json: <what> at offset N
 * (line L, column C)". Numbers: is_int is set when the literal has no
 * fraction/exponent and fits int64 (exact), or when its double value is
 * integral and within +-2^63. tlm_jat also indexes object values. */
#define TLM_JSON_MAX_DEPTH 64
tlm_json   *tlm_json_parse_file(const char *path, char *err, size_t errcap);
const char *tlm_jtype_name(tlm_jtype t);
/* Escape `len` bytes of `s` as a JSON string body (no quotes) into dst;
 * returns the length needed (excluding NUL), like snprintf. */
size_t      tlm_json_escape(char *dst, size_t cap, const char *s, size_t len);
void        tlm_json_write_str(FILE *f, const char *s, size_t len);   /* "quoted" */
void tlm_jw_init(tlm_jw *w, FILE *f);
void tlm_jw_null(tlm_jw *w, const char *key);

#ifdef __cplusplus
}
#endif
#endif
