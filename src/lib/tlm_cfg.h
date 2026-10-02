/* tlm_cfg.h - conventional unix key = value configuration.
 *
 * Files live in /etc/tierlm/ by default (override with TIERLM_ETC or --etc).
 * Precedence, later wins: built-in defaults < tierlm.conf < conf.d/<name>.conf
 * (lexical) < TIERLM_<KEY> environment (dots -> underscores, upper case)
 * < explicit tlm_cfg_set() calls (command-line flags).
 *
 * Syntax:  key = value      # comment
 *          include <glob>   (max nesting depth 4)
 * Keys: lowercase dotted names [a-z0-9_.]. Values: integers with K/M/G/T
 * suffix (powers of 1024), fractions, yes/no/on/off/true/false, comma lists,
 * "auto", strings (double-quote when containing '#' or edge spaces).
 */
#ifndef TLM_CFG_H
#define TLM_CFG_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TLM_ETC_DEFAULT "/etc/tierlm"

typedef enum { TLM_CFG_STR, TLM_CFG_INT, TLM_CFG_SIZE, TLM_CFG_FRAC,
               TLM_CFG_BOOL, TLM_CFG_LIST, TLM_CFG_ENUM } tlm_cfg_type;

/* Schema entry: every key the engine understands. Unknown keys warn. */
typedef struct {
    const char  *key;
    tlm_cfg_type type;
    const char  *def;          /* default as text; NULL = unset */
    double       min, max;     /* numeric range (ignored when min > max) */
    const char  *enum_values;  /* "a|b|c" for TLM_CFG_ENUM; "auto" always allowed if listed */
    const char  *help;         /* one line */
} tlm_cfg_key;

/* The built-in schema (tlm_cfg_schema.c). Terminated by key == NULL. */
extern const tlm_cfg_key tlm_cfg_schema[];
const tlm_cfg_key *tlm_cfg_schema_find(const char *key);

typedef struct tlm_cfg tlm_cfg;

/* origin of an effective value */
typedef enum { TLM_ORIGIN_DEFAULT, TLM_ORIGIN_FILE, TLM_ORIGIN_ENV, TLM_ORIGIN_SET } tlm_cfg_origin;

tlm_cfg *tlm_cfg_new(void);                    /* loaded with schema defaults */
void     tlm_cfg_free(tlm_cfg *c);
/* Load "<etc>/tierlm.conf" (missing file is fine) then its includes and
 * "<etc>/conf.d/<name>.conf", then the environment. strict != 0 makes unknown
 * keys and range errors fatal. */
int      tlm_cfg_load_etc(tlm_cfg *c, const char *etc_dir, int strict);
int      tlm_cfg_load_file(tlm_cfg *c, const char *path, int strict);
int      tlm_cfg_load_string(tlm_cfg *c, const char *text, const char *name, int strict);
int      tlm_cfg_apply_env(tlm_cfg *c);
int      tlm_cfg_set(tlm_cfg *c, const char *key, const char *value);   /* validated */

const char *tlm_cfg_str(const tlm_cfg *c, const char *key);  /* NULL if unset */
int      tlm_cfg_is_auto(const tlm_cfg *c, const char *key);
int64_t  tlm_cfg_int(const tlm_cfg *c, const char *key, int64_t dflt);
uint64_t tlm_cfg_size(const tlm_cfg *c, const char *key, uint64_t dflt);
double   tlm_cfg_frac(const tlm_cfg *c, const char *key, double dflt);
int      tlm_cfg_bool(const tlm_cfg *c, const char *key, int dflt);
/* Split a list value; returns count, fills up to max pointers into an
 * internal copy valid until the next call on this key. */
int      tlm_cfg_list(const tlm_cfg *c, const char *key, const char **out, int max);
tlm_cfg_origin tlm_cfg_origin_of(const tlm_cfg *c, const char *key, const char **where);

/* Iterate effective keys in sorted order (schema keys and extra keys such
 * as content.* that match a schema prefix rule). Returns 0 at end. */
int      tlm_cfg_next(const tlm_cfg *c, int *iter, const char **key, const char **value);

/* Write `key = value` lines (with optional per-key comments) to `out`. */
typedef struct { const char *key, *value, *comment; } tlm_cfg_line;
int      tlm_cfg_write(FILE *out, const char *header, const tlm_cfg_line *lines, int n);

/* ---- additions (tlm_cfg.c, tlm_cfg_schema.c) ---------------------------
 *
 * Schema conventions:
 *   - a key ending in ".*" is a prefix rule: "content.*" validates any
 *     well-formed key below "content." (longest prefix wins; exact keys win
 *     over prefix rules). The built-in rules are content.* (enum
 *     exclude|shallow|standard|deep) and hardware.* (string, probe output).
 *   - for non-ENUM types, `enum_values` lists keywords accepted verbatim
 *     besides the typed value (e.g. "auto" for cpu.threads); for ENUM it is
 *     the complete set. Enum and keyword matching ignores case and stores the
 *     schema spelling.
 *
 * Loading rules:
 *   - tlm_cfg_load_etc also loads "<etc>/hardware.conf" (probe facts,
 *     hardware.* keys) before tierlm.conf when it exists. Files included from
 *     tierlm.conf that also live in conf.d are re-applied after tierlm.conf
 *     so the documented precedence holds wherever the include line sits.
 *   - Precedence is enforced by origin rank, not call order: a value never
 *     replaces one from a higher origin (DEFAULT < FILE < ENV < SET); within
 *     one origin the later value wins.
 *   - Syntax errors (no '=', bad key, unterminated quote, include nesting > 4)
 *     are always errors (TLM_EFORMAT). Unknown keys warn; invalid or
 *     out-of-range values are reported as errors and ignored. With strict,
 *     both make the load fail (TLM_EINVAL / TLM_ERANGE). Loading continues
 *     past problems so every diagnostic is reported; tlm_errmsg() holds the
 *     first fatal one as "file:line: text".
 *   - Environment: every exact schema key and every key already present maps
 *     to TIERLM_<KEY> (dots -> underscores, upper case). Other
 *     TIERLM_CONTENT_* / TIERLM_HARDWARE_* variables map '_' to '.', so a
 *     content key containing '_' must already exist in a file to be
 *     overridden from the environment.
 *   - tlm_cfg_list returns the number of items stored in `out` (<= max).
 */

/* Validate `value` for `key`. On success returns 0 and writes the canonical
 * form to `norm` (may be NULL). Errors: TLM_ENOTFOUND unknown key,
 * TLM_EINVAL malformed key or value, TLM_ERANGE outside the range / not an
 * allowed value; `why` (may be NULL) gets a one-line reason that names what
 * is allowed. */
int  tlm_cfg_validate(const char *key, const char *value, char *norm, size_t normcap,
                      char *why, size_t whycap);
int  tlm_cfg_key_valid(const char *key);                /* 1 if [a-z][a-z0-9_]*(.[a-z0-9_]+)* */
void tlm_cfg_describe(const tlm_cfg_key *k, char *out, size_t cap);   /* "integer in [1, 4096] or auto" */

/* tlm_cfg_set with an explicit origin and label (e.g. "--threads"). */
int  tlm_cfg_set_origin(tlm_cfg *c, const char *key, const char *value,
                        tlm_cfg_origin origin, const char *where);
int  tlm_cfg_unset(tlm_cfg *c, const char *key);        /* removes the value (any origin) */
/* Cross-key checks, e.g. decide.ask_threshold <= decide.auto_threshold and
 * kv.device not mixing "cpu" with GPUs. 0 or TLM_EINVAL (+ diagnostic). */
int  tlm_cfg_check(tlm_cfg *c);

/* Diagnostics recorded while loading ("file:line: text"). */
void        tlm_cfg_quiet(tlm_cfg *c, int quiet);       /* 1 = record only, do not tlm_log */
int         tlm_cfg_ndiag(const tlm_cfg *c);
const char *tlm_cfg_diag(const tlm_cfg *c, int i, int *is_error);
void        tlm_cfg_clear_diag(tlm_cfg *c);
const char *tlm_cfg_origin_name(tlm_cfg_origin o);    /* "default", "file", "env", "set" */

/* Raw syntax pass with no schema (used for content-taxonomy.conf): calls fn
 * for every `key = value` line (values unquoted); an `include` line is
 * passed as key "include". Returns 0, the first non-zero fn result, or
 * TLM_EFORMAT on a syntax error (message "name:line: ..."). */
typedef int (*tlm_cfg_line_fn)(void *ctx, const char *key, const char *value, int line);
int  tlm_cfg_parse_text(const char *text, const char *name, tlm_cfg_line_fn fn, void *ctx);

/* Strict full-string parsers shared by the getters and validation. */
int  tlm_cfg_parse_int(const char *s, int64_t *out);    /* "-3", "4K" (powers of 1024) */
int  tlm_cfg_parse_size(const char *s, uint64_t *out);  /* "4096", "12G", "1.5G", "512MiB" */
int  tlm_cfg_parse_frac(const char *s, double *out);    /* "0.3", ".5", "30%" */
int  tlm_cfg_parse_bool(const char *s, int *out);       /* yes/no/on/off/true/false/1/0 */
/* Bytes with the largest exact K/M/G/T suffix ("12G", "1536M", "1000"). */
void tlm_cfg_fmt_size(uint64_t bytes, char *out, size_t cap);

#ifdef __cplusplus
}
#endif
#endif
