/* tlm_store.c - routed-expert RAM cache over the packet store.
 *
 * Patterns:
 *   P-10 bounded cache: one arena of equal 4 KiB-aligned slots, allocated once.
 *   P-12 slot lifecycle: FREE -> LOADING -> READY; a slot is published READY
 *        only after the whole packet was read and its header and CRC checked
 *        against the manifest; a failed load returns the slot to FREE (first
 *        victim) and every holder gets the error from tlm_store_wait.
 *   P-15 frequency-aware eviction: every selection (hit or miss) counts,
 *        counts halve every DECAY_PERIOD ticks, recency only breaks ties.
 *   P-18 one global cache across layers (C-07 default).
 *   P-19 warm pins: compiler-chosen pins first, then manifest heat, then the
 *        warm sidecar, capped at pin_frac of the slots (always leaving >= 1
 *        streaming slot).
 *   P-34 speculation never hurts demand: hints only take FREE slots or READY
 *        slots whose expert has frequency 0, are dropped while demand reads are
 *        backed up, publish at the lowest priority, and are bounded in flight
 *        so that top_k unpinned slots always remain for demand misses (a
 *        LOADING slot is never a victim, so this reserve is what keeps a hint
 *        from turning a demand miss into TLM_EBUSY).
 *   P-07/C-09 warm sidecar: binary (layer, id, count), atomically written,
 *        merge rule = stored count decayed by 0.99 plus this run's selections.
 *
 * Concurrency: one mutex guards all store state. I/O completions arrive on
 * tlm_io threads; CRC verification runs there without the lock (the slot is
 * LOADING, so nothing else touches its buffer). Lock order: store -> io.
 *
 * Failure semantics of tlm_store_acquire: every out[i] with slot >= 0 holds
 * a reference and must be released, whatever the return code; entries with
 * slot == -1 were not acquired (unknown/pruned expert: TLM_ENOTFOUND; every
 * slot referenced or pinned: TLM_EBUSY). The model must size the budget for
 * at least top_k unpinned slots (more for prefetch to be useful).
 */
#include "tlm_store.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define DECAY_PERIOD      16     /* P-15: halve selection counts every 16 tokens */
#define PF_DEMAND_BACKLOG 4      /* P-34: drop hints while more demand reads are in flight */
#define PF_MAX_INFLIGHT   16     /* P-34: bound on speculative reads in flight */
#define WARM_MAGIC        0x574D4C54u   /* "TLMW" */
#define WARM_VERSION      1u
#define WARM_HDR          24u
#define WARM_REC          12u
#define PKF_PINNED        0x1u
#define PKF_PRUNED        0x2u

enum { SLOT_FREE = 0, SLOT_LOADING = 1, SLOT_READY = 2 };
enum { LOAD_DEMAND = 0, LOAD_PREFETCH = 1, LOAD_PIN = 2 };

typedef struct {
    tlm_packet_loc loc;
    uint32_t freq;    /* decayed selection count (eviction priority) */
    uint32_t seed;    /* count loaded from the warm sidecar */
    uint64_t sel;     /* selections this run, undecayed (sidecar) */
    int      slot;    /* resident or loading slot, -1 if none */
} st_expert;

typedef struct st_slot {
    struct tlm_store *s;
    int      idx;
    int      state;
    int      ex;          /* expert index, -1 = none */
    int      refcount;
    int      pinned;
    int      prefetched;  /* filled by a hint and not yet used by a demand acquire */
    int      demand;      /* in-flight load counts as demand (P-34 backlog) */
    int      attempt;     /* 0 = either copy, 1 = primary only, 2 = replica only */
    int      err;         /* failed load (state FREE, refcount > 0) */
    uint64_t last_use;
    uint8_t *buf;
    uint64_t payload_off, payload_len;
} st_slot;

struct tlm_store {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    tlm_io         *io;
    tlm_store_opts  o;
    tlm_model_geom  geom;
    int             n_ex;
    st_expert      *ex;
    uint32_t       *hmap;     /* expert index + 1; 0 = empty */
    uint32_t        hmask;
    int             n_slots;
    uint64_t        slot_size;
    uint8_t        *arena;
    size_t          arena_bytes;
    st_slot        *slots;
    uint64_t        clock, ticks;
    int             loading, demand_loading, pf_loading;
    int             n_pinned;
    int             reserve;  /* unpinned slots kept for demand: max(1, top_k) (P-34) */
    tlm_store_stats st;       /* counters only; slots/pinned/in_use computed on demand */
};

/* ---- lookup -------------------------------------------------------------- */
static uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

static int find_ex(const tlm_store *s, int layer, int id) {
    if (!s->hmap || layer < 0 || layer > 0xFFFF || id < 0) return -1;
    uint64_t key = ((uint64_t)layer << 32) | (uint32_t)id;
    for (uint32_t h = (uint32_t)mix64(key) & s->hmask;; h = (h + 1) & s->hmask) {
        uint32_t v = s->hmap[h];
        if (!v) return -1;
        const tlm_packet_loc *l = &s->ex[v - 1].loc;
        if (l->layer == (uint32_t)layer && l->id == (uint32_t)id) return (int)(v - 1);
    }
}

/* ---- slot lifecycle (P-12), all called with s->mu held --------------------- */
static void on_read_done(void *user, int status);

static int submit_slot(tlm_store *s, int v) {
    st_slot *sl = &s->slots[v];
    const tlm_packet_loc *l = &s->ex[sl->ex].loc;
    int file = l->drive, alt = l->replica_drive != 0xFF ? l->replica_drive : -1;
    uint64_t off = l->offset, alt_off = l->replica_offset;
    if (sl->attempt == 1) {
        alt = -1;                       /* retry after a bad copy: primary only ... */
    } else if (sl->attempt == 2) {
        file = l->replica_drive;        /* ... then the replica only */
        off = l->replica_offset;
        alt = -1;
    }
    return tlm_io_submit(s->io, file, off, alt, alt_off, sl->buf, l->total_len, on_read_done, sl);
}

static void fail_slot(tlm_store *s, int v, int err) {
    st_slot *sl = &s->slots[v];
    if (sl->ex >= 0 && s->ex[sl->ex].slot == v) s->ex[sl->ex].slot = -1;
    if (sl->state == SLOT_LOADING) {
        s->loading--;
        if (sl->demand) s->demand_loading--;
        else s->pf_loading--;
    }
    sl->state = SLOT_FREE;   /* first victim once unreferenced (P-12) */
    if (sl->pinned) s->n_pinned--;
    sl->pinned = sl->prefetched = sl->demand = 0;
    sl->last_use = 0;
    if (sl->refcount > 0) {
        sl->err = err;       /* holders see it in tlm_store_wait */
    } else {
        sl->err = 0;
        sl->ex = -1;
    }
    pthread_cond_broadcast(&s->cv);
}

/* Put expert e into slot v (FREE, or an unreferenced unpinned READY slot,
 * which is evicted in place) and start reading it (P-12 reserve step). */
static int start_load(tlm_store *s, int v, int e, int kind) {
    st_slot *sl = &s->slots[v];
    if (sl->state == SLOT_READY && sl->ex >= 0) {
        s->ex[sl->ex].slot = -1;
        s->st.evictions++;
    }
    s->loading++;
    sl->state = SLOT_LOADING;
    sl->ex = e;
    s->ex[e].slot = v;
    sl->refcount = kind == LOAD_PREFETCH ? 0 : 1;
    sl->pinned = kind == LOAD_PIN;
    if (sl->pinned) s->n_pinned++;
    sl->prefetched = kind == LOAD_PREFETCH;
    sl->demand = kind != LOAD_PREFETCH;
    if (sl->demand) s->demand_loading++;
    else s->pf_loading++;
    sl->attempt = 0;
    sl->err = 0;
    /* Hints publish with the lowest priority so only a real router hit
     * promotes them (F-pf-22). */
    sl->last_use = kind == LOAD_PREFETCH ? 0 : s->clock;
    int rc = submit_slot(s, v);
    if (rc != TLM_OK) fail_slot(s, v, rc);
    return rc;
}

enum { PICK_DEMAND = 0, PICK_PREFETCH = 1, PICK_PIN = 2 };

/* Victim choice (P-15/P-18/P-34): never a referenced, pinned or LOADING
 * slot. Any FREE slot first (a failed load is FREE: first victim, P-12);
 * else the READY slot with the lowest (freq, last_use) across all layers;
 * hints may only take a READY slot whose expert has freq 0. -1 if none.
 * Linear scan: fine for thousands of slots (F-cache-08 has sampled variants). */
static int pick_victim(const tlm_store *s, int mode) {
    int best = -1;
    uint32_t bf = 0;
    uint64_t bl = 0;
    for (int i = 0; i < s->n_slots; i++) {
        const st_slot *sl = &s->slots[i];
        if (sl->refcount > 0 || sl->pinned) continue;
        if (sl->state == SLOT_FREE) return i;
        if (sl->state != SLOT_READY) continue;
        uint32_t f = s->ex[sl->ex].freq;
        if (mode == PICK_PREFETCH && f > 0) continue;
        if (best < 0 || f < bf || (f == bf && sl->last_use < bl)) {
            best = i;
            bf = f;
            bl = sl->last_use;
        }
    }
    return best;
}

static void take_ref(tlm_store *s, int v, uint64_t stamp, tlm_expert_ref *r) {
    st_slot *sl = &s->slots[v];
    sl->refcount++;
    sl->last_use = stamp;
    if (sl->prefetched) {
        sl->prefetched = 0;
        s->st.prefetch_used++;
    }
    if (sl->state == SLOT_LOADING && !sl->demand) {   /* a hint became demand */
        sl->demand = 1;
        s->pf_loading--;
        s->demand_loading++;
    }
    r->slot = v;
    if (sl->state == SLOT_READY) {
        r->payload = sl->buf + sl->payload_off;
        r->payload_len = sl->payload_len;
    }
}

static void drop_ref_locked(tlm_store *s, int v) {
    st_slot *sl = &s->slots[v];
    if (sl->refcount <= 0) {
        tlm_log(TLM_LOG_WARN, "store: release of unreferenced slot %d", v);
        return;
    }
    if (--sl->refcount == 0 && sl->state == SLOT_FREE) {
        sl->err = 0;
        sl->ex = -1;
    }
}

/* I/O completion, on an I/O thread (P-12 publish step). */
static void on_read_done(void *user, int status) {
    st_slot *sl = (st_slot *)user;
    tlm_store *s = sl->s;
    pthread_mutex_lock(&s->mu);
    int e = sl->ex;   /* fixed while LOADING: a LOADING slot is never a victim */
    pthread_mutex_unlock(&s->mu);

    const void *payload = NULL;
    uint64_t plen = 0;
    int rc = status;
    /* Verify without the lock: the slot is LOADING, nobody else touches buf;
     * ex[].loc is immutable after open. A slot is READY only after this. */
    if (rc == TLM_OK)
        rc = tlm_packet_verify(&s->ex[e].loc, sl->buf, s->ex[e].loc.total_len, &payload, &plen);
    char why[256] = "";
    if (rc != TLM_OK) snprintf(why, sizeof why, "%s", tlm_errmsg());   /* thread-local message */

    pthread_mutex_lock(&s->mu);
    int v = sl->idx;
    if (rc == TLM_OK) {
        sl->state = SLOT_READY;
        sl->payload_off = (uint64_t)((const uint8_t *)payload - sl->buf);
        sl->payload_len = plen;
        s->loading--;
        if (sl->demand) s->demand_loading--;
        else s->pf_loading--;
        s->st.bytes_read += s->ex[e].loc.total_len;
        pthread_cond_broadcast(&s->cv);
    } else if (s->ex[e].loc.replica_drive != 0xFF && sl->attempt < 2) {
        /* Never serve a bad copy; try each copy explicitly before giving up. */
        tlm_log(TLM_LOG_WARN, "store: layer %u expert %u: %s; re-reading the %s copy",
                s->ex[e].loc.layer, s->ex[e].loc.id, why, sl->attempt == 0 ? "primary" : "replica");
        sl->attempt++;
        int src = submit_slot(s, v);
        if (src != TLM_OK) fail_slot(s, v, src);
    } else {
        if (sl->prefetched)
            tlm_log(TLM_LOG_INFO, "store: prefetch of layer %u expert %u failed: %s",
                    s->ex[e].loc.layer, s->ex[e].loc.id, why);
        else
            tlm_log(TLM_LOG_WARN, "store: layer %u expert %u failed: %s", s->ex[e].loc.layer,
                    s->ex[e].loc.id, why);
        fail_slot(s, v, rc);
    }
    pthread_mutex_unlock(&s->mu);
}

/* ---- warm sidecar (P-07) -------------------------------------------------- */
static void put32le(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}
static uint32_t get32le(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

/* Placement data only: a bad or foreign sidecar costs speed, never results,
 * so it is ignored with a warning rather than failing the open. */
static void load_warm(tlm_store *s, const char *path) {
    char *raw = NULL;
    size_t n = 0;
    if (tlm_read_file(path, &raw, &n) != TLM_OK) {
        tlm_log(TLM_LOG_INFO, "store: no warm sidecar at %s", path);
        return;
    }
    const uint8_t *b = (const uint8_t *)raw;
    const char *bad = NULL;
    uint32_t count = 0;
    if (n < WARM_HDR || get32le(b) != WARM_MAGIC) bad = "not a warm sidecar";
    else if (get32le(b + 4) != WARM_VERSION) bad = "unsupported version";
    else if (get32le(b + 8) != s->geom.n_layers || get32le(b + 12) != s->geom.n_experts)
        bad = "written for a different model geometry";
    else if ((count = get32le(b + 16)) != (n - WARM_HDR) / WARM_REC || (n - WARM_HDR) % WARM_REC)
        bad = "truncated";
    else if (tlm_crc32c(0, b + WARM_HDR, n - WARM_HDR) != get32le(b + 20))
        bad = "checksum mismatch";
    if (bad) {
        tlm_log(TLM_LOG_WARN, "store: ignoring warm sidecar %s: %s", path, bad);
        free(raw);
        return;
    }
    int used = 0;
    for (uint32_t i = 0; i < count; i++) {
        const uint8_t *r = b + WARM_HDR + (size_t)i * WARM_REC;
        int e = find_ex(s, (int)(r[0] | r[1] << 8), (int)get32le(r + 4));
        if (e >= 0) {
            s->ex[e].seed = get32le(r + 8);
            used++;
        }
    }
    tlm_log(TLM_LOG_INFO, "store: warm sidecar %s: %d of %u records apply", path, used, count);
    free(raw);
}

int tlm_store_save_warm(const tlm_store *cs, const char *path) {
    if (!cs || !path) return tlm_seterr(TLM_EINVAL, "store save warm: NULL argument");
    tlm_store *s = (tlm_store *)cs;   /* lock only; no logical mutation */
    pthread_mutex_lock(&s->mu);
    uint32_t count = 0;
    uint32_t *val = (uint32_t *)calloc(s->n_ex ? (size_t)s->n_ex : 1, sizeof *val);
    if (!val) {
        pthread_mutex_unlock(&s->mu);
        return tlm_seterr(TLM_ENOMEM, "store save warm: out of memory");
    }
    for (int i = 0; i < s->n_ex; i++) {
        /* C-09: decay the stored history (~0.99 per save, round half up) and add
         * this run's selections, so one short run neither erases nor freezes it. */
        uint64_t v = ((uint64_t)s->ex[i].seed * 99 + 50) / 100 + s->ex[i].sel;
        val[i] = v > UINT32_MAX ? UINT32_MAX : (uint32_t)v;
        if (val[i]) count++;
    }
    size_t len = WARM_HDR + (size_t)count * WARM_REC;
    uint8_t *buf = (uint8_t *)calloc(1, len);
    if (!buf) {
        pthread_mutex_unlock(&s->mu);
        free(val);
        return tlm_seterr(TLM_ENOMEM, "store save warm: out of memory");
    }
    uint8_t *r = buf + WARM_HDR;
    for (int i = 0; i < s->n_ex; i++) {
        if (!val[i]) continue;
        r[0] = (uint8_t)s->ex[i].loc.layer;
        r[1] = (uint8_t)(s->ex[i].loc.layer >> 8);
        put32le(r + 4, s->ex[i].loc.id);
        put32le(r + 8, val[i]);
        r += WARM_REC;
    }
    put32le(buf, WARM_MAGIC);
    put32le(buf + 4, WARM_VERSION);
    put32le(buf + 8, s->geom.n_layers);
    put32le(buf + 12, s->geom.n_experts);
    put32le(buf + 16, count);
    put32le(buf + 20, tlm_crc32c(0, buf + WARM_HDR, len - WARM_HDR));
    pthread_mutex_unlock(&s->mu);
    free(val);
    int rc = tlm_write_file_atomic(path, buf, len);   /* P-08 */
    free(buf);
    return rc;
}

/* ---- open / close ---------------------------------------------------------- */
tlm_store *tlm_store_open(const tlm_manifest *m, tlm_io *io, const tlm_store_opts *o) {
    if (!m || (m->n_packets && !m->packets)) {
        tlm_seterr(TLM_EINVAL, "store: no manifest");
        return NULL;
    }
    tlm_store_opts def;
    memset(&def, 0, sizeof def);
    if (!o) o = &def;
    tlm_store *s = (tlm_store *)calloc(1, sizeof *s);
    if (!s) {
        tlm_seterr(TLM_ENOMEM, "store: out of memory");
        return NULL;
    }
    s->io = io;
    s->o = *o;
    s->o.warm_path = NULL;   /* not retained: the caller's string may not outlive us */
    s->geom = m->geom;
    s->reserve = m->geom.top_k > 0 ? (int)(m->geom.top_k > 1024 ? 1024 : m->geom.top_k) : 1;
    (void)tlm_crc32c(0, "", 0);   /* build the CRC table before I/O threads verify */

    for (uint32_t i = 0; i < m->n_packets; i++)
        if (m->packets[i].type == TLM_PK_EXPERT && !(m->packets[i].flags & PKF_PRUNED)) s->n_ex++;
    if (s->n_ex && !io) {
        tlm_seterr(TLM_EINVAL, "store: an I/O backend is required");
        goto fail;
    }
    if (s->n_ex) {
        s->ex = (st_expert *)calloc((size_t)s->n_ex, sizeof *s->ex);
        uint32_t cap = 16;
        while (cap < (uint32_t)s->n_ex * 2) cap <<= 1;
        s->hmap = (uint32_t *)calloc(cap, sizeof *s->hmap);
        s->hmask = cap - 1;
        if (!s->ex || !s->hmap) {
            tlm_seterr(TLM_ENOMEM, "store: out of memory");
            goto fail;
        }
    }
    int n = 0;
    for (uint32_t i = 0; i < m->n_packets; i++) {
        const tlm_packet_loc *l = &m->packets[i];
        if (l->type != TLM_PK_EXPERT || (l->flags & PKF_PRUNED)) continue;
        if (l->total_len < TLM_ALIGN_IO || l->total_len % TLM_ALIGN_IO) {
            tlm_seterr(TLM_EFORMAT, "store: expert packet with bad length");
            goto fail;
        }
        if (find_ex(s, l->layer, (int)l->id) >= 0) {
            tlm_seterr(TLM_EFORMAT, "store: duplicate expert layer %u id %u", l->layer, l->id);
            goto fail;
        }
        s->ex[n].loc = *l;
        s->ex[n].slot = -1;
        uint64_t key = ((uint64_t)l->layer << 32) | l->id;
        uint32_t h = (uint32_t)mix64(key) & s->hmask;
        while (s->hmap[h]) h = (h + 1) & s->hmask;
        s->hmap[h] = (uint32_t)n + 1;
        if (l->total_len > s->slot_size) s->slot_size = l->total_len;
        n++;
    }

    if (s->n_ex) {
        uint64_t ns = o->budget_bytes / s->slot_size;   /* whole slots only */
        if (ns < 1) ns = 1;
        if (ns > (uint64_t)s->n_ex) ns = (uint64_t)s->n_ex;
        if (ns > (uint64_t)(SIZE_MAX / s->slot_size) || ns > 0x7FFFFFFF) {
            tlm_seterr(TLM_ERANGE, "store: budget too large");
            goto fail;
        }
        s->n_slots = (int)ns;
        s->arena_bytes = (size_t)ns * (size_t)s->slot_size;
        s->arena = (uint8_t *)tlm_map_anon(s->arena_bytes, o->huge, o->lock);
        s->slots = (st_slot *)calloc((size_t)s->n_slots, sizeof *s->slots);
        if (!s->arena || !s->slots) {
            tlm_seterr(TLM_ENOMEM, "store: cannot allocate %zu bytes of expert slots",
                       s->arena_bytes);
            goto fail;
        }
        for (int i = 0; i < s->n_slots; i++) {
            s->slots[i].s = s;
            s->slots[i].idx = i;
            s->slots[i].ex = -1;
            s->slots[i].buf = s->arena + (size_t)i * s->slot_size;
        }
    }
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv, NULL);
    if (o->warm_path && o->warm_path[0]) load_warm(s, o->warm_path);
    tlm_log(TLM_LOG_INFO, "store: %d experts, %d slots x %llu bytes (%.1f MiB)", s->n_ex,
            s->n_slots, (unsigned long long)s->slot_size, (double)s->arena_bytes / 1048576.0);
    return s;
fail:
    if (s->arena) tlm_unmap_anon(s->arena, s->arena_bytes);
    free(s->slots);
    free(s->hmap);
    free(s->ex);
    free(s);
    return NULL;
}

void tlm_store_close(tlm_store *s) {
    if (!s) return;
    pthread_mutex_lock(&s->mu);
    while (s->loading > 0) pthread_cond_wait(&s->cv, &s->mu);   /* no read may land in freed memory */
    int held = 0;
    for (int i = 0; i < s->n_slots; i++) held += s->slots[i].refcount > 0;
    pthread_mutex_unlock(&s->mu);
    if (held) tlm_log(TLM_LOG_WARN, "store: closing with %d referenced slot(s)", held);
    pthread_mutex_destroy(&s->mu);
    pthread_cond_destroy(&s->cv);
    if (s->arena) tlm_unmap_anon(s->arena, s->arena_bytes);
    free(s->slots);
    free(s->hmap);
    free(s->ex);
    free(s);
}

int tlm_store_n_slots(const tlm_store *s) { return s ? s->n_slots : 0; }
uint64_t tlm_store_slot_bytes(const tlm_store *s) { return s ? s->slot_size : 0; }

/* ---- warm pins (P-19) ------------------------------------------------------ */
typedef struct {
    int      e, compiler_pin;
    float    heat;
    uint32_t seed;
    uint16_t layer;
    uint32_t id;
} pin_cand;

static int pin_cmp(const void *a, const void *b) {
    const pin_cand *x = (const pin_cand *)a, *y = (const pin_cand *)b;
    if (x->compiler_pin != y->compiler_pin) return x->compiler_pin > y->compiler_pin ? -1 : 1;
    if (x->heat != y->heat) return x->heat > y->heat ? -1 : 1;
    if (x->seed != y->seed) return x->seed > y->seed ? -1 : 1;
    if (x->layer != y->layer) return x->layer < y->layer ? -1 : 1;
    return x->id < y->id ? -1 : x->id > y->id;
}

int tlm_store_warm(tlm_store *s) {
    if (!s) return tlm_seterr(TLM_EINVAL, "store warm: NULL");
    if (s->n_slots == 0) return TLM_OK;
    double frac = s->o.pin_frac;
    if (!(frac > 0.0)) return TLM_OK;
    if (frac > 0.9) frac = 0.9;
    int target = (int)floor(frac * s->n_slots);
    if (target > s->n_slots - 1) target = s->n_slots - 1;   /* keep a streaming slot */
    if (target <= 0) return TLM_OK;

    pin_cand *c = (pin_cand *)malloc(sizeof *c * (size_t)s->n_ex);
    int *mine = (int *)malloc(sizeof *mine * (size_t)target);
    if (!c || !mine) {
        free(c);
        free(mine);
        return tlm_seterr(TLM_ENOMEM, "store warm: out of memory");
    }
    pthread_mutex_lock(&s->mu);
    int nc = 0;
    for (int i = 0; i < s->n_ex; i++) {
        const st_expert *x = &s->ex[i];
        int cp = (x->loc.flags & PKF_PINNED) != 0;
        if (!cp && !(x->loc.heat > 0.0f) && x->seed == 0) continue;   /* never selected: no pin */
        c[nc++] = (pin_cand){i, cp, x->loc.heat, x->seed, x->loc.layer, x->loc.id};
    }
    qsort(c, (size_t)nc, sizeof *c, pin_cmp);
    int pinned = s->n_pinned, n_mine = 0, rc = TLM_OK;
    for (int i = 0; i < nc && pinned < target; i++) {
        int e = c[i].e, v = s->ex[e].slot;
        if (v >= 0) {   /* already resident or loading: pin in place */
            if (!s->slots[v].pinned) {
                s->slots[v].pinned = 1;
                s->n_pinned++;
                pinned++;
            }
            continue;
        }
        v = pick_victim(s, PICK_PIN);
        if (v < 0) break;
        start_load(s, v, e, LOAD_PIN);   /* holds a reference until we have waited */
        mine[n_mine++] = v;
        pinned++;
    }
    for (int i = 0; i < n_mine; i++) {
        st_slot *sl = &s->slots[mine[i]];
        while (sl->state == SLOT_LOADING) pthread_cond_wait(&s->cv, &s->mu);
        if (sl->state != SLOT_READY && rc == TLM_OK) {
            rc = sl->err ? sl->err : TLM_EIO;
            tlm_seterr(rc, "store warm: pinned expert failed to load");
        }
        drop_ref_locked(s, mine[i]);
    }
    pinned = s->n_pinned;
    pthread_mutex_unlock(&s->mu);
    tlm_log(TLM_LOG_INFO, "store: pinned %d of %d slots", pinned, s->n_slots);
    free(c);
    free(mine);
    return rc;
}

/* ---- demand path ----------------------------------------------------------- */
int tlm_store_acquire(tlm_store *s, int layer, const int *ids, int k, tlm_expert_ref *out) {
    if (!s || k < 0 || (k > 0 && (!ids || !out)))
        return tlm_seterr(TLM_EINVAL, "store acquire: bad arguments");
    int rc = TLM_OK;
    pthread_mutex_lock(&s->mu);
    uint64_t stamp = ++s->clock;
    for (int i = 0; i < k; i++) {
        out[i].slot = -1;
        out[i].payload = NULL;
        out[i].payload_len = 0;
        int e = find_ex(s, layer, ids[i]);
        if (e < 0) {
            if (rc == TLM_OK)
                rc = tlm_seterr(TLM_ENOTFOUND, "store: layer %d expert %d has no packet (absent or pruned)",
                                layer, ids[i]);
            continue;
        }
        if (s->ex[e].freq < UINT32_MAX) s->ex[e].freq++;   /* every selection counts (P-15) */
        s->ex[e].sel++;
    }
    /* Pass 1: reference every hit before any miss chooses a victim, so a miss
     * can never evict an expert this same step needs (F-cache-08). */
    for (int i = 0; i < k; i++) {
        int e = find_ex(s, layer, ids[i]);
        if (e < 0 || s->ex[e].slot < 0) continue;
        take_ref(s, s->ex[e].slot, stamp, &out[i]);
        s->st.hits++;
    }
    /* Pass 2: misses. */
    for (int i = 0; i < k; i++) {
        if (out[i].slot >= 0) continue;
        int e = find_ex(s, layer, ids[i]);
        if (e < 0) continue;
        if (s->ex[e].slot >= 0) {   /* repeated id: loaded earlier in this pass */
            take_ref(s, s->ex[e].slot, stamp, &out[i]);
            s->st.hits++;
            continue;
        }
        int v = pick_victim(s, PICK_DEMAND);
        if (v < 0) {
            if (rc == TLM_OK)
                rc = tlm_seterr(TLM_EBUSY,
                                "store: all %d slots are referenced or pinned; the expert budget "
                                "must hold at least top_k unpinned slots",
                                s->n_slots);
            continue;
        }
        s->st.misses++;
        int lrc = start_load(s, v, e, LOAD_DEMAND);
        out[i].slot = v;   /* referenced even if the submit failed: wait reports it */
        if (lrc != TLM_OK && rc == TLM_OK) rc = lrc;
    }
    pthread_mutex_unlock(&s->mu);
    return rc;
}

int tlm_store_wait(tlm_store *s, tlm_expert_ref *ref) {
    if (!s || !ref) return tlm_seterr(TLM_EINVAL, "store wait: NULL argument");
    if (ref->slot < 0 || ref->slot >= s->n_slots)
        return tlm_seterr(TLM_EINVAL, "store wait: expert was not acquired");
    pthread_mutex_lock(&s->mu);
    st_slot *sl = &s->slots[ref->slot];
    uint64_t t0 = 0;
    while (sl->state == SLOT_LOADING) {
        if (!t0) t0 = tlm_now_ns();
        pthread_cond_wait(&s->cv, &s->mu);
    }
    if (t0) s->st.wait_ns += tlm_now_ns() - t0;
    int rc = TLM_OK;
    if (sl->state == SLOT_READY) {
        ref->payload = sl->buf + sl->payload_off;
        ref->payload_len = sl->payload_len;
    } else {
        rc = sl->err ? sl->err : TLM_EIO;
        if (sl->ex >= 0)
            tlm_seterr(rc, "store: layer %u expert %u could not be loaded (%s)",
                       s->ex[sl->ex].loc.layer, s->ex[sl->ex].loc.id,
                       rc == TLM_EFORMAT ? "header or CRC mismatch" : "read error");
        else
            tlm_seterr(rc, "store: expert could not be loaded");
        ref->payload = NULL;
        ref->payload_len = 0;
    }
    pthread_mutex_unlock(&s->mu);
    return rc;
}

int tlm_store_is_ready(tlm_store *s, const tlm_expert_ref *ref) {
    if (!s || !ref || ref->slot < 0 || ref->slot >= s->n_slots) return 0;
    pthread_mutex_lock(&s->mu);
    int ready = s->slots[ref->slot].state == SLOT_READY;
    pthread_mutex_unlock(&s->mu);
    return ready;
}

void tlm_store_release(tlm_store *s, tlm_expert_ref *ref) {
    if (!s || !ref) return;
    if (ref->slot >= 0 && ref->slot < s->n_slots) {
        pthread_mutex_lock(&s->mu);
        drop_ref_locked(s, ref->slot);
        pthread_mutex_unlock(&s->mu);
    }
    ref->slot = -1;   /* makes a second release harmless */
    ref->payload = NULL;
    ref->payload_len = 0;
}

/* ---- speculation (P-33 hints, P-34 rules) ---------------------------------- */
void tlm_store_prefetch(tlm_store *s, int layer, const int *ids, int k) {
    if (!s || !s->o.prefetch || !ids || k <= 0) return;
    pthread_mutex_lock(&s->mu);
    /* In-flight hints are not victims, so they must leave `reserve` unpinned
     * slots for the next demand acquire (P-34); with no spare slot every hint
     * is dropped. */
    int cap = s->n_slots - s->n_pinned - s->reserve;
    if (cap > PF_MAX_INFLIGHT) cap = PF_MAX_INFLIGHT;
    for (int i = 0; i < k; i++) {
        int e = find_ex(s, layer, ids[i]);
        if (e < 0 || s->ex[e].slot >= 0) continue;   /* unknown, resident or already loading */
        if (s->demand_loading > PF_DEMAND_BACKLOG || s->pf_loading >= cap) {
            s->st.prefetch_dropped++;
            continue;
        }
        int v = pick_victim(s, PICK_PREFETCH);
        if (v < 0) {
            s->st.prefetch_dropped++;
            continue;
        }
        if (start_load(s, v, e, LOAD_PREFETCH) == TLM_OK) s->st.prefetch_issued++;
        else s->st.prefetch_dropped++;   /* non-fatal (F-pf-22) */
    }
    pthread_mutex_unlock(&s->mu);
}

void tlm_store_tick(tlm_store *s) {
    if (!s) return;
    pthread_mutex_lock(&s->mu);
    if (++s->ticks % DECAY_PERIOD == 0)
        for (int i = 0; i < s->n_ex; i++) s->ex[i].freq >>= 1;
    pthread_mutex_unlock(&s->mu);
}

void tlm_store_get_stats(const tlm_store *cs, tlm_store_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    if (!cs) return;
    tlm_store *s = (tlm_store *)cs;   /* lock only */
    pthread_mutex_lock(&s->mu);
    *out = s->st;
    out->slots = s->n_slots;
    out->pinned = s->n_pinned;
    for (int i = 0; i < s->n_slots; i++) out->in_use += s->slots[i].refcount > 0;
    pthread_mutex_unlock(&s->mu);
}
