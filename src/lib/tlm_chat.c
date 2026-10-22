/* tlm_chat.c - Qwen3 chat template, no-tools branch (F-TQ-05).
 *
 * Token-level port of the target's Jinja template: markers are pushed as
 * added-token ids, content is encoded with tlm_bpe_encode(allow_added=1).
 * The reasoning split, last-real-query pass, tool-run folding and the
 * empty-think scaffold are the verified subtleties of the fragment.
 */
#include "tlm_chat.h"
#include "tlm_bpe.h"

#include <stdlib.h>
#include <string.h>

#define MK_IM_S  "<|im_start|>"
#define MK_IM_E  "<|im_end|>"
#define MK_TH    "<think>"
#define MK_THS   "\x3c/think\x3e"         /* no pipes: Qwen3 think-close marker */
#define MK_TR    "\x3ctool_response\x3e"
#define MK_TRS   "\x3c/tool_response\x3e"

typedef struct {
    int *v;
    int  n, cap;
    int  err;
} outb;

static int ob_push(outb *o, int id) {
    if (o->err) return -1;
    if (o->n == o->cap) { o->err = 1; return -1; }
    o->v[o->n++] = id;
    return 0;
}


/* Encode text with added matching; append the ids. len 0 = strlen. */
static int ob_enc(outb *o, const tlm_bpe *b, const char *s, size_t len) {
    if (o->err) return -1;
    if (!s) s = "";
    if (len == 0) len = strlen(s);
    int cap = (int)(len * 2 + 16);   /* ids per byte upper bound is 1; headroom */
    int *tmp = (int *)malloc(sizeof(int) * (size_t)cap);
    if (!tmp) { o->err = 1; return -1; }
    int n = 0;
    for (;;) {
        n = tlm_bpe_encode(b, s, len, 1, tmp, cap);
        if (n >= 0) break;
        if (cap > (1 << 28)) { free(tmp); o->err = 1; return -1; }
        cap *= 2;
        int *nt = (int *)realloc(tmp, sizeof(int) * (size_t)cap);
        if (!nt) { free(tmp); o->err = 1; return -1; }
        tmp = nt;
    }
    for (int i = 0; i < n; i++)
        if (ob_push(o, tmp[i])) { free(tmp); return -1; }
    free(tmp);
    return 0;
}


static int msg_is_real_query(const tlm_chat_msg *m) {
    if (!m->content || strcmp(m->role, "user") != 0) return 0;
    size_t l = m->content_len ? m->content_len : strlen(m->content);
    size_t trl = strlen(MK_TR), trsl = strlen(MK_TRS);
    return !(l >= trl + trsl && memcmp(m->content, MK_TR, trl) == 0 &&
             memcmp(m->content + l - trsl, MK_TRS, trsl) == 0);
}

int tlm_qwen3_chat_render(const tlm_bpe *b, const tlm_chat_msg *msgs, int nmsgs,
                          const tlm_chat_opts *opts, int *ids, int cap) {
    int id_im_s = tlm_bpe_add_id(b, MK_IM_S, strlen(MK_IM_S));
    int id_im_e = tlm_bpe_add_id(b, MK_IM_E, strlen(MK_IM_E));
    int id_th   = tlm_bpe_add_id(b, MK_TH, strlen(MK_TH));
    int id_ths  = tlm_bpe_add_id(b, MK_THS, strlen(MK_THS));
    int id_tr   = tlm_bpe_add_id(b, MK_TR, strlen(MK_TR));
    int id_trs  = tlm_bpe_add_id(b, MK_TRS, strlen(MK_TRS));
    if (id_im_s < 0 || id_im_e < 0 || id_th < 0 || id_ths < 0 || id_tr < 0 || id_trs < 0)
        return -1;

    outb o = {ids, 0, cap, 0};

    /* last real query index */
    int lqi = nmsgs - 1;
    for (int i = nmsgs - 1; i >= 0; i--)
        if (msg_is_real_query(&msgs[i])) { lqi = i; break; }

    if (nmsgs > 0 && strcmp(msgs[0].role, "system") == 0) {
        if (ob_push(&o, id_im_s) || ob_enc(&o, b, "system", 0) || ob_enc(&o, b, "\n", 1) ||
            ob_enc(&o, b, msgs[0].content, msgs[0].content_len) ||
            ob_push(&o, id_im_e) || ob_enc(&o, b, "\n", 1))
            return -1;
    }

    for (int idx = 0; idx < nmsgs && !o.err; idx++) {
        const tlm_chat_msg *m = &msgs[idx];
        if (idx == 0 && strcmp(m->role, "system") == 0)
            continue;   /* rendered in the prologue */
        if (strcmp(m->role, "user") == 0 || strcmp(m->role, "system") == 0) {
            if (ob_push(&o, id_im_s) || ob_enc(&o, b, m->role, 0) || ob_enc(&o, b, "\n", 1) ||
                ob_enc(&o, b, m->content, m->content_len) ||
                ob_push(&o, id_im_e) || ob_enc(&o, b, "\n", 1))
                return -1;
        } else if (strcmp(m->role, "assistant") == 0) {
            /* reasoning split */
            const char *r = NULL;
            size_t rlen = 0;
            const char *c = m->content ? m->content : "";
            size_t clen = m->content_len ? m->content_len : strlen(c);
            const char *cend = c + clen;
            if (m->reasoning) {
                r = m->reasoning;
                rlen = m->reasoning_len ? m->reasoning_len : strlen(r);
            } else if (strchr(c, '<') != NULL) {
                const char *ths1 = strstr(c, MK_THS);
                if (ths1 && ths1 < cend) {
                    const char *ths = ths1;
                    for (const char *p = strstr(ths1 + 1, MK_THS); p && p < cend;
                         p = strstr(p + 1, MK_THS))
                        ths = p;
                    /* r = c before first THS, rstrip "\n"; within that prefix,
                     * after the last TH, lstrip "\n"; c = after last THS,
                     * lstrip "\n" */
                    size_t pre = (size_t)(ths1 - c);
                    while (pre > 0 && c[pre - 1] == '\n') pre--;
                    const char *th_last = NULL;
                    for (const char *p = strstr(c, MK_TH); p && p < ths1; p = strstr(p + 1, MK_TH))
                        th_last = p;
                    if (th_last) {
                        r = th_last + strlen(MK_TH);
                        while (r < c + pre && *r == '\n') r++;
                        rlen = (size_t)(c + pre - r);
                    } else {
                        r = c;
                        rlen = pre;
                    }
                    c = ths + strlen(MK_THS);
                    while (c < cend && *c == '\n') c++;
                    clen = (size_t)(cend - c);
                }
            }
            if (idx > lqi && (idx == nmsgs - 1 || rlen > 0)) {
                /* thinking form */
                const char *rr = r ? r : "";
                size_t rl = rlen;
                while (rl > 0 && rr[0] == '\n') { rr++; rl--; }
                while (rl > 0 && rr[rl - 1] == '\n') rl--;
                if (ob_push(&o, id_im_s) || ob_enc(&o, b, "assistant", 0) || ob_enc(&o, b, "\n", 1) ||
                    ob_push(&o, id_th) || ob_enc(&o, b, "\n", 1) ||
                    ob_enc(&o, b, rr, rl) || ob_enc(&o, b, "\n", 1) ||
                    ob_push(&o, id_ths) || ob_enc(&o, b, "\n\n", 2) ||
                    ob_enc(&o, b, c, clen))
                    return -1;
            } else {
                if (ob_push(&o, id_im_s) || ob_enc(&o, b, "assistant", 0) || ob_enc(&o, b, "\n", 1) ||
                    ob_enc(&o, b, c, clen))
                    return -1;
            }
            if (ob_push(&o, id_im_e) || ob_enc(&o, b, "\n", 1))
                return -1;
        } else if (strcmp(m->role, "tool") == 0) {
            int first = idx == 0 || strcmp(msgs[idx - 1].role, "tool") != 0;
            int last = idx == nmsgs - 1 || strcmp(msgs[idx + 1].role, "tool") != 0;
            if (first)
                if (ob_push(&o, id_im_s) || ob_enc(&o, b, "user", 0)) return -1;
            if (ob_enc(&o, b, "\n", 1) || ob_push(&o, id_tr) || ob_enc(&o, b, "\n", 1) ||
                ob_enc(&o, b, m->content, m->content_len) || ob_enc(&o, b, "\n", 1) ||
                ob_push(&o, id_trs))
                return -1;
            if (last && (ob_push(&o, id_im_e) || ob_enc(&o, b, "\n", 1)))
                return -1;
        } else {
            return -1;   /* unknown role */
        }
    }
    if (o.err) return -1;

    if (opts && opts->add_generation_prompt) {
        if (ob_push(&o, id_im_s) || ob_enc(&o, b, "assistant", 0) || ob_enc(&o, b, "\n", 1))
            return -1;
        if (opts->enable_thinking == 0)
            if (ob_push(&o, id_th) || ob_enc(&o, b, "\n\n", 2) ||
                ob_push(&o, id_ths) || ob_enc(&o, b, "\n\n", 2))
                return -1;
    }
    return o.n;
}
