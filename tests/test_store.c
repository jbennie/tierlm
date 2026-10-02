/* test_store.c - expert slot cache over a real packet store on three drive
 * directories: bytes identical across budgets and prefetch (P-10, P-34),
 * frequency eviction with decay (P-15, P-18), pins (P-19), TLM_EBUSY when
 * every slot is held, CRC/header failures never published (P-12), warm
 * sidecar (P-07), concurrent use. */
#include "tlm_io.h"
#include "tlm_packet.h"
#include "tlm_platform.h"
#include "tlm_store.h"
#include "tlm_test.h"

#include <fcntl.h>
#include <ftw.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <sys/stat.h>
#include <unistd.h>

#define NL     4
#define NE     8
#define TOPK   2
#define PRU_L  2      /* (2, 7) is flagged pruned in the manifest */
#define PRU_E  7
#define CPIN_L 3      /* (3, 5) carries the compiler pin flag */
#define CPIN_E 5
#define MAXSTEPS 512

static char g_root[128];
static char g_dirs[3][200];
static char g_mpath[300];
static char g_paths[3][512];
static const char *g_pp[3];
static tlm_manifest g_m;
static uint8_t *g_data[NL][NE];
static uint64_t g_len[NL][NE];
static uint64_t g_slot_bytes;   /* largest non-pruned expert packet */
static int g_n_live;            /* non-pruned experts */
/* TLM_TEST_IO_BACKEND=uring runs the suite over io_uring (make URING=1). */
static tlm_io_backend g_backend = TLM_IO_PREAD;

/* ---- fixture ------------------------------------------------------------- */
static int rm_cb(const char *p, const struct stat *sb, int flag, struct FTW *ftw) {
    (void)sb; (void)flag; (void)ftw;
    return remove(p);
}
static void rm_rf(const char *p) { nftw(p, rm_cb, 16, FTW_DEPTH | FTW_PHYS); }

static void fill(uint8_t *p, uint64_t n, uint64_t seed) {
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull;
    for (uint64_t i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        p[i] = (uint8_t)(x >> 56);
    }
}

static int pruned(int l, int e) { return l == PRU_L && e == PRU_E; }

static float heat_of(int l, int e) {
    if (l == 0 && e == 3) return 0.9f;
    if (l == 1 && e == 6) return 0.8f;
    if (l == 2 && e == 1) return 0.7f;
    if (l == 3 && e == 0) return 0.6f;
    if (l == CPIN_L && e == CPIN_E) return 0.05f;
    return 0.0f;
}

static const tlm_packet_loc *loc_of(int l, int e) {
    return tlm_manifest_find(&g_m, TLM_PK_EXPERT, (uint16_t)l, (uint32_t)e);
}

static int build_store(void) {
    const char *dirs[3] = {g_dirs[0], g_dirs[1], g_dirs[2]};
    tlm_store_writer *w = tlm_writer_open("st", dirs, 3);
    if (!w) return -1;
    tlm_packet_loc locs[NL * NE + NL];
    int n = 0;
    for (int l = 0; l < NL; l++) {   /* non-expert packets the store must ignore */
        uint8_t dense[300];
        fill(dense, sizeof dense, 1000u + (unsigned)l);
        if (tlm_writer_add(w, TLM_PK_DENSE, (uint16_t)l, 0, TLM_DT_F32, dense, sizeof dense, NULL,
                           l % 3, -1, &locs[n++]) != 0)
            return -1;
    }
    for (int l = 0; l < NL; l++)
        for (int e = 0; e < NE; e++) {
            uint64_t len = 1000 + (uint64_t)((l * NE + e) * 2711 % 15000);
            g_len[l][e] = len;
            g_data[l][e] = (uint8_t *)malloc(len);
            fill(g_data[l][e], len, (uint64_t)(l * 131 + e));
            int drive = (l * NE + e) % 3, rep = e % 3 == 0 ? (drive + 1) % 3 : -1;
            uint32_t item[3] = {0, (uint32_t)(len / 3), (uint32_t)(2 * len / 3)};
            tlm_packet_loc *lc = &locs[n++];
            if (tlm_writer_add(w, TLM_PK_EXPERT, (uint16_t)l, (uint32_t)e, TLM_DT_Q8_0, g_data[l][e],
                               len, item, drive, rep, lc) != 0)
                return -1;
            lc->heat = heat_of(l, e);
            if (pruned(l, e)) lc->flags = 2;
            if (l == CPIN_L && e == CPIN_E) lc->flags = 1;
        }
    if (tlm_writer_finish(w, 1) != 0) return -1;
    tlm_writer_free(w);

    tlm_manifest m;
    memset(&m, 0, sizeof m);
    snprintf(m.geom.arch, sizeof m.geom.arch, "tlm-moe");
    m.geom.n_layers = NL;
    m.geom.n_experts = NE;
    m.geom.top_k = TOPK;
    m.geom.d_model = 64;
    snprintf(m.name, sizeof m.name, "st");
    snprintf(m.profile, sizeof m.profile, "exact");
    m.n_drives = 3;
    for (int i = 0; i < 3; i++) snprintf(m.drives[i], sizeof m.drives[i], "%.255s", g_dirs[i]);
    m.n_pruned = 1;
    m.n_packets = (uint32_t)n;
    m.packets = locs;
    snprintf(g_mpath, sizeof g_mpath, "%s/st.tlmm", g_root);
    if (tlm_manifest_save(&m, g_mpath) != 0) return -1;
    if (tlm_manifest_load(&g_m, g_mpath) != 0) return -1;
    for (int i = 0; i < 3; i++) {
        tlm_manifest_drive_path(&g_m, i, g_paths[i], sizeof g_paths[i]);
        g_pp[i] = g_paths[i];
    }
    g_slot_bytes = 0;
    g_n_live = 0;
    for (int l = 0; l < NL; l++)
        for (int e = 0; e < NE; e++) {
            if (pruned(l, e)) continue;
            g_n_live++;
            if (loc_of(l, e)->total_len > g_slot_bytes) g_slot_bytes = loc_of(l, e)->total_len;
        }
    return 0;
}

static tlm_io *open_io(int direct) {
    tlm_io_opts o = {g_backend, direct, 2, 16};
    return tlm_io_open(g_pp, 3, &o);
}

static tlm_store *open_store_m(const tlm_manifest *m, tlm_io *io, int slots, int prefetch,
                               double pin_frac, const char *warm) {
    tlm_store_opts o;
    memset(&o, 0, sizeof o);
    o.budget_bytes = (uint64_t)slots * g_slot_bytes + g_slot_bytes / 2;   /* rounded down */
    o.pin_frac = pin_frac;
    o.prefetch = prefetch;
    o.warm_path = warm;
    tlm_store *s = tlm_store_open(m, io, &o);
    CHECK(s != NULL);
    if (s) {
        CHECK_EQ_INT(tlm_store_slot_bytes(s), g_slot_bytes);
        CHECK_EQ_INT(tlm_store_n_slots(s), slots < g_n_live ? slots : g_n_live);
    }
    return s;
}
static tlm_store *open_store(tlm_io *io, int slots, int prefetch, double pin_frac) {
    return open_store_m(&g_m, io, slots, prefetch, pin_frac, NULL);
}

static int payload_ok(const tlm_expert_ref *r, int l, int e) {
    return r->payload && r->payload_len == g_len[l][e] &&
           memcmp(r->payload, g_data[l][e], (size_t)g_len[l][e]) == 0;
}

static void flip_byte(const char *path, uint64_t off) {
    int fd = open(path, O_RDWR);
    CHECK(fd >= 0);
    if (fd < 0) return;
    uint8_t b = 0;
    CHECK(pread(fd, &b, 1, (off_t)off) == 1);
    b ^= 0xA5;
    CHECK(pwrite(fd, &b, 1, (off_t)off) == 1);
    close(fd);
}

/* Acquire one expert, wait, check bytes, release. Returns 1 for a hit
 * (counted as such by the store), 0 for a miss, -1 on any error. */
static int touch(tlm_store *s, int l, int e) {
    tlm_store_stats a, b;
    tlm_store_get_stats(s, &a);
    tlm_expert_ref r;
    int rc = tlm_store_acquire(s, l, &e, 1, &r);
    if (rc != 0 || r.slot < 0) {
        tlm_store_release(s, &r);
        return -1;
    }
    rc = tlm_store_wait(s, &r);
    int ok = rc == 0 && payload_ok(&r, l, e);
    tlm_store_release(s, &r);
    tlm_store_get_stats(s, &b);
    if (!ok) return -1;
    if (b.hits == a.hits + 1 && b.misses == a.misses) return 1;
    if (b.misses == a.misses + 1 && b.hits == a.hits) return 0;
    return -1;
}

/* A pinned (or otherwise resident) expert: acquire is a hit and the slot is
 * READY before any wait. */
static int resident_now(tlm_store *s, int l, int e) {
    tlm_store_stats a, b;
    tlm_store_get_stats(s, &a);
    tlm_expert_ref r;
    int rc = tlm_store_acquire(s, l, &e, 1, &r);
    int ready = rc == 0 && r.slot >= 0 && tlm_store_is_ready(s, &r) && payload_ok(&r, l, e);
    tlm_store_release(s, &r);
    tlm_store_get_stats(s, &b);
    return ready && b.misses == a.misses && b.hits == a.hits + 1;
}

/* ---- deterministic workloads --------------------------------------------- */
typedef struct { int layer, ids[TOPK]; } step_t;

static uint64_t xs(uint64_t *x) {
    *x ^= *x << 13; *x ^= *x >> 7; *x ^= *x << 17;
    return *x;
}

/* Skewed: 70% of picks from experts {0,1,2} of the layer, so there are both
 * hits and misses at small budgets. */
static void make_seq(step_t *seq, int n, uint64_t seed) {
    uint64_t x = seed | 1;
    for (int i = 0; i < n; i++) {
        seq[i].layer = i % NL;
        for (int j = 0; j < TOPK; j++) {
            for (;;) {
                uint64_t r = xs(&x);
                int id = (int)(r % 10 < 7 ? (r >> 8) % 3 : (r >> 8) % NE);
                if (pruned(seq[i].layer, id) || (j > 0 && id == seq[i].ids[0])) continue;
                seq[i].ids[j] = id;
                break;
            }
        }
    }
}

static uint32_t ref_digest(const step_t *seq, int n) {
    uint32_t d = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < TOPK; j++)
            d = tlm_crc32c(d, g_data[seq[i].layer][seq[i].ids[j]],
                           (size_t)g_len[seq[i].layer][seq[i].ids[j]]);
    return d;
}

/* Run a workload like the model does (acquire, optional lookahead hints for
 * the next step, wait in rank order, release, tick). Returns the digest of
 * every payload seen, in order; *bad counts byte or status mismatches. */
static uint32_t run_seq(tlm_store *s, const step_t *seq, int n, int prefetch, int *bad) {
    uint32_t d = 0;
    *bad = 0;
    for (int i = 0; i < n; i++) {
        tlm_expert_ref r[TOPK];
        int rc = tlm_store_acquire(s, seq[i].layer, seq[i].ids, TOPK, r);
        if (rc != 0) (*bad)++;
        if (prefetch && i + 1 < n) {
            /* one right guess, one sometimes-right guess, one usually wrong */
            int hint[3] = {seq[i + 1].ids[0], i % 2 ? seq[i + 1].ids[1] : (i * 5 + 1) % NE,
                           (i * 3) % NE};
            tlm_store_prefetch(s, seq[i + 1].layer, hint, 3);
        }
        for (int j = 0; j < TOPK; j++) {
            if (r[j].slot < 0 || tlm_store_wait(s, &r[j]) != 0 ||
                !payload_ok(&r[j], seq[i].layer, seq[i].ids[j])) {
                (*bad)++;
                continue;
            }
            d = tlm_crc32c(d, r[j].payload, (size_t)r[j].payload_len);
        }
        for (int j = 0; j < TOPK; j++) tlm_store_release(s, &r[j]);
        tlm_store_tick(s);
    }
    return d;
}

/* ---- tests ----------------------------------------------------------------- */
static void test_open_args(void) {
    tlm_io *io = open_io(0);
    CHECK(io != NULL);
    CHECK(tlm_store_open(NULL, io, NULL) == NULL);
    CHECK(tlm_store_open(&g_m, NULL, NULL) == NULL);   /* experts but no I/O */
    /* Default options: a zero budget still yields one slot. */
    tlm_store *s = tlm_store_open(&g_m, io, NULL);
    CHECK(s != NULL);
    CHECK_EQ_INT(tlm_store_n_slots(s), 1);
    CHECK_EQ_INT(tlm_store_slot_bytes(s), g_slot_bytes);
    CHECK_EQ_INT(tlm_store_warm(s), 0);   /* pin_frac 0: nothing to do */
    tlm_store_close(s);

    /* A manifest without experts needs no I/O and has no slots. */
    tlm_manifest empty = g_m;
    empty.n_packets = 0;
    empty.packets = NULL;
    s = tlm_store_open(&empty, NULL, NULL);
    CHECK(s != NULL);
    CHECK_EQ_INT(tlm_store_n_slots(s), 0);
    int id = 0;
    tlm_expert_ref r;
    CHECK_EQ_INT(tlm_store_acquire(s, 0, &id, 1, &r), TLM_ENOTFOUND);
    CHECK_EQ_INT(r.slot, -1);
    tlm_store_close(s);
    tlm_io_close(io);
}

static void test_bytes_across_budgets(void) {
    static step_t seq[MAXSTEPS];
    const int n = 400;
    make_seq(seq, n, 0x5EED1234u);
    uint32_t want = ref_digest(seq, n);
    int used[NL][NE];
    memset(used, 0, sizeof used);
    uint64_t distinct_bytes = 0;
    int distinct = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < TOPK; j++)
            if (!used[seq[i].layer][seq[i].ids[j]]++) {
                distinct++;
                distinct_bytes += loc_of(seq[i].layer, seq[i].ids[j])->total_len;
            }

    static const int budgets[4] = {2, 3, 5, 64};
    for (int pf = 0; pf <= 1; pf++)
        for (int b = 0; b < 4; b++) {
            tlm_io *io = open_io((b + pf) % 2);   /* direct on and off */
            CHECK(io != NULL);
            if (!io) continue;
            tlm_store *s = open_store(io, budgets[b], pf, 0.0);
            if (!s) {
                tlm_io_close(io);
                continue;
            }
            int bad = 0;
            uint32_t got = run_seq(s, seq, n, pf, &bad);
            CHECK_EQ_INT(bad, 0);
            CHECK(got == want);   /* same bytes whatever the budget or prefetch */
            tlm_store_stats st;
            tlm_store_get_stats(s, &st);
            CHECK_EQ_INT(st.hits + st.misses, n * TOPK);
            CHECK_EQ_INT(st.in_use, 0);
            CHECK_EQ_INT(st.pinned, 0);
            CHECK_EQ_INT(st.slots, budgets[b] < g_n_live ? budgets[b] : g_n_live);
            CHECK((int)(st.misses + st.prefetch_used) >= distinct);   /* each loaded at least once */
            CHECK(st.bytes_read >= distinct_bytes);
            CHECK(st.prefetch_used <= st.prefetch_issued);
            if (budgets[b] <= 3) CHECK(st.evictions > 0);
            if (budgets[b] >= g_n_live && !pf) {   /* everything fits: one miss per expert */
                CHECK_EQ_INT(st.misses, distinct);
                CHECK_EQ_INT(st.evictions, 0);
                CHECK_EQ_INT(st.bytes_read, distinct_bytes);
            }
            if (!pf) CHECK(st.prefetch_issued == 0 && st.prefetch_dropped == 0 && st.prefetch_used == 0);
            /* P-34 reserve: with budget == top_k every hint is dropped; with
             * spare slots some are issued and some are used. */
            if (pf && budgets[b] == TOPK) CHECK_EQ_INT(st.prefetch_issued, 0);
            if (pf && budgets[b] == TOPK) CHECK(st.prefetch_dropped > 0);
            if (pf && budgets[b] > TOPK) CHECK(st.prefetch_issued > 0 && st.prefetch_used > 0);
            tlm_store_close(s);
            tlm_io_close(io);
        }
}

static void test_lfu_decay_and_hints(void) {
    tlm_io *io = open_io(1);
    tlm_store *s = open_store(io, 3, 1, 0.0);
    if (!s) {
        tlm_io_close(io);
        return;
    }
    /* Layer 0, one expert per acquire. Expected outcomes follow lowest
     * (decayed frequency, last use) eviction over unreferenced READY slots. */
    static const int script[][2] = {   /* {expert, 1 = hit / 0 = miss} */
        {0, 0}, {0, 1}, {0, 1}, {0, 1},   /* f0 = 4 */
        {1, 0}, {2, 0},                   /* slots {0,1,2} full */
        {4, 0},                           /* evicts 1 (f1, older than 2) */
        {0, 1}, {2, 1},
        {1, 0},                           /* evicts 4 (f1) */
        {4, 0},                           /* evicts 2 (f2, older than 1's f2) */
        {0, 1}, {1, 1},
        {2, 0},                           /* evicts 4 (f2 < 1's f3 < 0's f6) */
    };
    int nsc = (int)(sizeof script / sizeof script[0]);
    for (int i = 0; i < nsc; i++) {
        int got = touch(s, 0, script[i][0]);
        if (got != script[i][1])
            fprintf(stderr, "  step %d expert %d: got %d want %d\n", i, script[i][0], got, script[i][1]);
        CHECK_EQ_INT(got, script[i][1]);
    }
    tlm_store_stats st;
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.evictions, 4);

    /* Every resident expert has freq > 0: a hint may not evict any. */
    int five = 5, one = 1;
    tlm_store_prefetch(s, 0, &five, 1);
    tlm_store_prefetch(s, 0, &one, 1);   /* resident: ignored, not counted */
    tlm_store_stats st2;
    tlm_store_get_stats(s, &st2);
    CHECK_EQ_INT(st2.prefetch_issued, 0);
    CHECK_EQ_INT(st2.prefetch_dropped, 1);

    /* Three halvings (every 16 ticks, P-15) take every count to 0 ... */
    for (int t = 0; t < 48; t++) tlm_store_tick(s);
    tlm_store_prefetch(s, 0, &five, 1);   /* ... so now the hint may evict */
    tlm_store_get_stats(s, &st2);
    CHECK_EQ_INT(st2.prefetch_issued, 1);
    CHECK_EQ_INT(st2.evictions, 5);       /* expert 0 (lowest last use) despite its old f6 */
    CHECK_EQ_INT(touch(s, 0, 5), 1);      /* the hint is a hit and counts as used */
    tlm_store_get_stats(s, &st2);
    CHECK_EQ_INT(st2.prefetch_used, 1);
    CHECK_EQ_INT(touch(s, 0, 0), 0);      /* 0 was really evicted; evicts 1 (f0, oldest) */
    CHECK_EQ_INT(touch(s, 0, 2), 1);
    CHECK_EQ_INT(touch(s, 0, 1), 0);
    tlm_store_get_stats(s, &st2);
    CHECK_EQ_INT(st2.in_use, 0);
    CHECK_EQ_INT(st2.hits + st2.misses, nsc + 4);
    tlm_store_close(s);

    /* prefetch = 0 in the options: hints are ignored entirely. */
    s = open_store(io, 3, 0, 0.0);
    tlm_store_prefetch(s, 0, &five, 1);
    tlm_store_get_stats(s, &st2);
    CHECK(st2.prefetch_issued == 0 && st2.prefetch_dropped == 0 && st2.bytes_read == 0);
    tlm_store_close(s);
    tlm_io_close(io);
}

/* P-34 reserve: in-flight hints are not victims, so they may never occupy
 * the top_k slots the next demand acquire needs. A completion needs the
 * store lock to publish, so during one prefetch call the in-flight count
 * cannot drop and these counts do not depend on I/O timing. */
static void test_prefetch_reserve(void) {
    tlm_io *io = open_io(1);
    tlm_store_stats st;
    int hints[4] = {0, 1, 2, 3}, demand[2] = {4, 5};
    tlm_expert_ref r[2];

    tlm_store *s = open_store(io, 3, 1, 0.0);   /* 3 - top_k 2 = 1 hint in flight */
    tlm_store_prefetch(s, 1, hints, 4);
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.prefetch_issued, 1);
    CHECK_EQ_INT(st.prefetch_dropped, 3);
    CHECK_EQ_INT(tlm_store_acquire(s, 1, demand, 2, r), 0);   /* never EBUSY because of a hint */
    CHECK(r[0].slot >= 0 && r[1].slot >= 0);
    for (int j = 0; j < 2; j++) {
        CHECK_EQ_INT(tlm_store_wait(s, &r[j]), 0);
        CHECK(payload_ok(&r[j], 1, demand[j]));
        tlm_store_release(s, &r[j]);
    }
    CHECK_EQ_INT(touch(s, 1, 0), 1);   /* the issued hint was hints[0] and is used */
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.prefetch_used, 1);
    tlm_store_close(s);

    s = open_store(io, 2, 1, 0.0);     /* budget == top_k: every hint dropped */
    tlm_store_prefetch(s, 1, hints, 4);
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.prefetch_issued, 0);
    CHECK_EQ_INT(st.prefetch_dropped, 4);
    CHECK_EQ_INT(st.bytes_read, 0);
    tlm_store_close(s);

    s = open_store(io, 5, 1, 0.2);     /* 1 pin: 5 - 1 - 2 = 2 hints in flight */
    CHECK_EQ_INT(tlm_store_warm(s), 0);
    tlm_store_prefetch(s, 1, hints, 4);
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.pinned, 1);
    CHECK_EQ_INT(st.prefetch_issued, 2);
    CHECK_EQ_INT(st.prefetch_dropped, 2);
    CHECK_EQ_INT(tlm_store_acquire(s, 1, demand, 2, r), 0);
    for (int j = 0; j < 2; j++) {
        CHECK_EQ_INT(tlm_store_wait(s, &r[j]), 0);
        tlm_store_release(s, &r[j]);
    }
    tlm_store_close(s);
    tlm_io_close(io);
}

static void test_pins(void) {
    tlm_io *io = open_io(1);
    /* 6 slots x 0.5 = 3 pins: the compiler-pinned expert first, then the two
     * hottest by manifest heat (P-19). */
    tlm_store *s = open_store(io, 6, 1, 0.5);
    if (!s) {
        tlm_io_close(io);
        return;
    }
    CHECK_EQ_INT(tlm_store_warm(s), 0);
    tlm_store_stats st;
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.pinned, 3);
    CHECK_EQ_INT(st.in_use, 0);
    CHECK_EQ_INT(st.misses, 0);
    CHECK_EQ_INT(st.bytes_read, loc_of(CPIN_L, CPIN_E)->total_len + loc_of(0, 3)->total_len +
                                    loc_of(1, 6)->total_len);
    CHECK_EQ_INT(tlm_store_warm(s), 0);   /* idempotent */
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.pinned, 3);

    /* Heavy churn through the 3 streaming slots never touches the pins. */
    static step_t seq[MAXSTEPS];
    make_seq(seq, 300, 0xABCDEFu);
    for (int i = 0; i < 300; i += 3) {   /* spread wider than the skewed default */
        int id = (seq[i].ids[0] + 3 + i / 3) % NE;
        while (id == seq[i].ids[0] || pruned(seq[i].layer, id)) id = (id + 1) % NE;
        seq[i].ids[1] = id;
    }
    int bad = 0;
    uint32_t got = run_seq(s, seq, 300, 1, &bad);
    CHECK_EQ_INT(bad, 0);
    CHECK(got == ref_digest(seq, 300));
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.pinned, 3);
    CHECK(st.evictions > 0);
    CHECK(resident_now(s, CPIN_L, CPIN_E));
    CHECK(resident_now(s, 0, 3));
    CHECK(resident_now(s, 1, 6));
    tlm_store_close(s);

    /* pin_frac is capped so a streaming slot always survives. */
    s = open_store(io, 2, 0, 0.9);
    CHECK_EQ_INT(tlm_store_warm(s), 0);
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.pinned, 1);
    CHECK(resident_now(s, CPIN_L, CPIN_E));
    CHECK_EQ_INT(touch(s, 0, 0), 0);
    CHECK_EQ_INT(touch(s, 0, 1), 0);   /* only one streaming slot: evicts 0, not the pin */
    CHECK(resident_now(s, CPIN_L, CPIN_E));
    tlm_store_close(s);
    tlm_io_close(io);
}

static void test_ebusy_and_notfound(void) {
    tlm_io *io = open_io(0);
    tlm_store *s = open_store(io, 2, 1, 0.0);
    if (!s) {
        tlm_io_close(io);
        return;
    }
    int ids01[2] = {0, 1}, zero = 0;
    tlm_expert_ref r[2], x;
    CHECK_EQ_INT(tlm_store_acquire(s, 0, ids01, 2, r), 0);
    CHECK_EQ_INT(tlm_store_wait(s, &r[0]), 0);
    CHECK_EQ_INT(tlm_store_wait(s, &r[1]), 0);
    /* Both slots referenced: a third expert cannot be placed. */
    CHECK_EQ_INT(tlm_store_acquire(s, 1, &zero, 1, &x), TLM_EBUSY);
    CHECK_EQ_INT(x.slot, -1);
    CHECK_EQ_INT(tlm_store_wait(s, &x), TLM_EINVAL);
    CHECK_EQ_INT(tlm_store_is_ready(s, &x), 0);
    tlm_store_release(s, &x);           /* slot -1: no-op */
    tlm_store_stats st;
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.in_use, 2);
    CHECK(payload_ok(&r[0], 0, 0) && payload_ok(&r[1], 0, 1));   /* still intact */
    /* The same expert again is a hit on a referenced slot, not EBUSY. */
    CHECK_EQ_INT(tlm_store_acquire(s, 0, &zero, 1, &x), 0);
    CHECK_EQ_INT(x.slot, r[0].slot);
    tlm_store_release(s, &x);
    tlm_store_release(s, &r[0]);
    tlm_store_release(s, &r[0]);        /* second release of the same ref: harmless */
    CHECK_EQ_INT(r[0].slot, -1);
    CHECK_EQ_INT(tlm_store_acquire(s, 1, &zero, 1, &x), 0);
    CHECK_EQ_INT(tlm_store_wait(s, &x), 0);
    CHECK(payload_ok(&x, 1, 0));
    CHECK(x.slot != r[1].slot);          /* evicted the unreferenced slot */
    tlm_store_release(s, &x);
    tlm_store_release(s, &r[1]);

    /* Pruned and unknown experts are not found; the rest of the call works. */
    int pr = PRU_E;
    CHECK_EQ_INT(tlm_store_acquire(s, PRU_L, &pr, 1, &x), TLM_ENOTFOUND);
    CHECK_EQ_INT(x.slot, -1);
    int mixed[2] = {2, 99};
    CHECK_EQ_INT(tlm_store_acquire(s, 3, mixed, 2, r), TLM_ENOTFOUND);
    CHECK(r[0].slot >= 0 && r[1].slot == -1);
    CHECK_EQ_INT(tlm_store_wait(s, &r[0]), 0);
    CHECK(payload_ok(&r[0], 3, 2));
    tlm_store_release(s, &r[0]);
    tlm_store_release(s, &r[1]);
    CHECK_EQ_INT(tlm_store_acquire(s, NL, &zero, 1, &x), TLM_ENOTFOUND);
    CHECK_EQ_INT(tlm_store_acquire(s, -1, &zero, 1, &x), TLM_ENOTFOUND);
    CHECK_EQ_INT(tlm_store_acquire(s, 0, NULL, 1, &x), TLM_EINVAL);
    CHECK_EQ_INT(tlm_store_acquire(s, 0, ids01, 0, NULL), 0);   /* k = 0 */
    tlm_store_prefetch(s, PRU_L, &pr, 1);   /* unknown hints are ignored */
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.in_use, 0);
    CHECK_EQ_INT(st.prefetch_issued, 0);
    tlm_store_close(s);

    /* With a pin, 3 slots leave 2 streaming slots: the third reference fails. */
    s = open_store(io, 3, 0, 0.5);
    CHECK_EQ_INT(tlm_store_warm(s), 0);
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.pinned, 1);
    CHECK_EQ_INT(tlm_store_acquire(s, 0, ids01, 2, r), 0);
    CHECK_EQ_INT(tlm_store_acquire(s, 1, &zero, 1, &x), TLM_EBUSY);
    tlm_store_release(s, &x);
    tlm_store_release(s, &r[0]);
    tlm_store_release(s, &r[1]);
    tlm_store_close(s);   /* closes with reads possibly in flight: must wait */
    tlm_io_close(io);
}

static void test_crc_failures(void) {
    tlm_log_level(TLM_LOG_ERROR);   /* failed loads and replica retries log by design */
    tlm_io *io = open_io(1);
    tlm_store *s = open_store(io, 2, 0, 0.0);
    if (!s) {
        tlm_io_close(io);
        tlm_log_level(TLM_LOG_WARN);
        return;
    }
    tlm_expert_ref r[2];
    int one = 1, dup[2] = {1, 1};

    /* (0,1) has no replica: a flipped payload byte is never published. */
    const tlm_packet_loc *a = loc_of(0, 1);
    CHECK(a && a->replica_drive == 0xFF);
    flip_byte(g_paths[a->drive], a->offset + 64 + 10);
    CHECK_EQ_INT(tlm_store_acquire(s, 0, dup, 2, r), 0);
    CHECK_EQ_INT(r[0].slot, r[1].slot);   /* both holders share the slot ... */
    CHECK_EQ_INT(tlm_store_wait(s, &r[0]), TLM_EFORMAT);   /* ... and both see the error */
    CHECK_EQ_INT(tlm_store_wait(s, &r[1]), TLM_EFORMAT);
    CHECK(r[0].payload == NULL && r[0].payload_len == 0);
    CHECK_EQ_INT(tlm_store_is_ready(s, &r[0]), 0);
    tlm_store_release(s, &r[0]);
    tlm_store_release(s, &r[1]);
    /* Not cached: the next acquire reads (and fails) again. */
    tlm_store_stats st0, st1;
    tlm_store_get_stats(s, &st0);
    CHECK_EQ_INT(tlm_store_acquire(s, 0, &one, 1, r), 0);
    CHECK_EQ_INT(tlm_store_wait(s, &r[0]), TLM_EFORMAT);
    tlm_store_release(s, &r[0]);
    tlm_store_get_stats(s, &st1);
    CHECK_EQ_INT(st1.misses, st0.misses + 1);
    CHECK_EQ_INT(st1.bytes_read, st0.bytes_read);   /* failed loads are not counted as read */
    /* The failed slot went back to FREE: both slots still serve other experts. */
    int ids23[2] = {2, 4};
    CHECK_EQ_INT(tlm_store_acquire(s, 0, ids23, 2, r), 0);
    CHECK_EQ_INT(tlm_store_wait(s, &r[0]), 0);
    CHECK_EQ_INT(tlm_store_wait(s, &r[1]), 0);
    CHECK(payload_ok(&r[0], 0, 2) && payload_ok(&r[1], 0, 4));
    tlm_store_release(s, &r[0]);
    tlm_store_release(s, &r[1]);
    flip_byte(g_paths[a->drive], a->offset + 64 + 10);   /* repair */
    CHECK_EQ_INT(touch(s, 0, 1), 0);

    /* A header naming another packet (id field) is rejected too. */
    const tlm_packet_loc *b = loc_of(1, 2);
    CHECK(b && b->replica_drive == 0xFF);
    flip_byte(g_paths[b->drive], b->offset + 12);
    CHECK_EQ_INT(touch(s, 1, 2), -1);
    int two = 2;
    CHECK_EQ_INT(tlm_store_acquire(s, 1, &two, 1, r), 0);
    CHECK_EQ_INT(tlm_store_wait(s, &r[0]), TLM_EFORMAT);
    tlm_store_release(s, &r[0]);
    flip_byte(g_paths[b->drive], b->offset + 12);
    CHECK_EQ_INT(touch(s, 1, 2), 0);

    /* (0,3) has a replica: a bad primary is served from the replica; with
     * both copies bad the load fails. */
    const tlm_packet_loc *c = loc_of(0, 3);
    CHECK(c && c->replica_drive != 0xFF);
    flip_byte(g_paths[c->drive], c->offset + 64 + 500);
    CHECK_EQ_INT(touch(s, 0, 3), 0);   /* correct bytes, checked inside touch */
    CHECK(touch(s, 0, 0) >= 0);        /* push (0,3) out: the hotter (0,1) stays, */
    CHECK(touch(s, 0, 2) >= 0);        /* so (0,3) is the victim (checked below)  */
    flip_byte(g_paths[c->replica_drive], c->replica_offset + 64 + 500);
    int three = 3;
    CHECK_EQ_INT(tlm_store_acquire(s, 0, &three, 1, r), 0);
    CHECK_EQ_INT(tlm_store_wait(s, &r[0]), TLM_EFORMAT);
    tlm_store_release(s, &r[0]);
    flip_byte(g_paths[c->drive], c->offset + 64 + 500);
    flip_byte(g_paths[c->replica_drive], c->replica_offset + 64 + 500);
    CHECK_EQ_INT(touch(s, 0, 3), 0);

    /* A failing hint is non-fatal and leaves nothing behind. */
    tlm_store_close(s);
    s = open_store(io, 3, 1, 0.0);
    flip_byte(g_paths[a->drive], a->offset + 64 + 10);
    tlm_store_prefetch(s, 0, &one, 1);
    tlm_store_stats st;
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.prefetch_issued, 1);
    CHECK_EQ_INT(tlm_store_acquire(s, 0, &one, 1, r), 0);   /* hit on the in-flight hint, or a new miss */
    CHECK_EQ_INT(tlm_store_wait(s, &r[0]), TLM_EFORMAT);
    tlm_store_release(s, &r[0]);
    flip_byte(g_paths[a->drive], a->offset + 64 + 10);
    CHECK_EQ_INT(touch(s, 0, 1), 0);
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.in_use, 0);
    tlm_store_close(s);
    tlm_io_close(io);
    tlm_log_level(TLM_LOG_WARN);
}

static void test_warm_sidecar(void) {
    char wpath[400], bad[400];
    snprintf(wpath, sizeof wpath, "%s/st.warm", g_root);
    snprintf(bad, sizeof bad, "%s/bad.warm", g_root);
    tlm_io *io = open_io(0);
    tlm_store *s = open_store(io, 4, 0, 0.0);
    if (!s) {
        tlm_io_close(io);
        return;
    }
    for (int i = 0; i < 10; i++) CHECK(touch(s, 1, 2) >= 0);
    for (int i = 0; i < 7; i++) CHECK(touch(s, 2, 3) >= 0);
    for (int i = 0; i < 4; i++) CHECK(touch(s, 0, 5) >= 0);
    CHECK(touch(s, 3, 4) >= 0);
    CHECK(touch(s, 0, 6) >= 0);
    CHECK_EQ_INT(tlm_store_save_warm(s, wpath), 0);
    CHECK_EQ_INT(tlm_store_save_warm(s, "/nonexistent-tlm-test/no-such-dir/x/y.warm") != 0, 1);
    tlm_store_close(s);

    /* Same packets, no heat and no compiler pins: only the sidecar ranks. */
    tlm_manifest cold = g_m;
    cold.packets = (tlm_packet_loc *)malloc(sizeof *cold.packets * g_m.n_packets);
    memcpy(cold.packets, g_m.packets, sizeof *cold.packets * g_m.n_packets);
    for (uint32_t i = 0; i < cold.n_packets; i++) {
        cold.packets[i].heat = 0.0f;
        cold.packets[i].flags &= (uint16_t)~1u;
    }
    s = open_store_m(&cold, io, 6, 0, 0.5, wpath);
    CHECK_EQ_INT(tlm_store_warm(s), 0);
    tlm_store_stats st;
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.pinned, 3);
    CHECK(resident_now(s, 1, 2));
    CHECK(resident_now(s, 2, 3));
    CHECK(resident_now(s, 0, 5));
    /* Re-save after this run: decayed history + new selections; a third
     * store built from it ranks the same way. */
    CHECK_EQ_INT(tlm_store_save_warm(s, wpath), 0);
    tlm_store_close(s);
    s = open_store_m(&cold, io, 6, 0, 0.5, wpath);
    CHECK_EQ_INT(tlm_store_warm(s), 0);
    CHECK(resident_now(s, 1, 2) && resident_now(s, 2, 3) && resident_now(s, 0, 5));
    tlm_store_close(s);

    /* Garbage, truncated or foreign sidecars are ignored (speed-only data). */
    tlm_log_level(TLM_LOG_ERROR);
    FILE *f = fopen(bad, "wb");
    CHECK(f != NULL);
    if (f) {
        fputs("not a sidecar at all, but long enough to have a header", f);
        fclose(f);
    }
    s = open_store_m(&cold, io, 6, 0, 0.5, bad);
    CHECK_EQ_INT(tlm_store_warm(s), 0);
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.pinned, 0);
    tlm_store_close(s);
    char *raw = NULL;
    size_t n = 0;
    CHECK_EQ_INT(tlm_read_file(wpath, &raw, &n), 0);
    if (raw && n > 30) {
        CHECK_EQ_INT(tlm_write_file_atomic(bad, raw, n - 5), 0);   /* truncated */
        s = open_store_m(&cold, io, 6, 0, 0.5, bad);
        CHECK_EQ_INT(tlm_store_warm(s), 0);
        tlm_store_get_stats(s, &st);
        CHECK_EQ_INT(st.pinned, 0);
        tlm_store_close(s);
        raw[n - 1] ^= 0x40;                                       /* checksum */
        CHECK_EQ_INT(tlm_write_file_atomic(bad, raw, n), 0);
        s = open_store_m(&cold, io, 6, 0, 0.5, bad);
        CHECK_EQ_INT(tlm_store_warm(s), 0);
        tlm_store_get_stats(s, &st);
        CHECK_EQ_INT(st.pinned, 0);
        tlm_store_close(s);
    }
    free(raw);
    tlm_manifest other = cold;
    other.geom.n_experts = NE + 1;                                  /* different model */
    s = open_store_m(&other, io, 6, 0, 0.5, wpath);
    CHECK_EQ_INT(tlm_store_warm(s), 0);
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.pinned, 0);
    tlm_store_close(s);
    s = open_store_m(&cold, io, 6, 0, 0.5, "/nonexistent-tlm-test/no-such.warm");   /* missing */
    CHECK(s != NULL);
    tlm_store_close(s);
    tlm_log_level(TLM_LOG_WARN);
    free(cold.packets);
    tlm_io_close(io);
}

/* ---- concurrent use -------------------------------------------------------- */
typedef struct {
    tlm_store  *s;
    int         tid;
    atomic_int *bad;
    atomic_int *busy;
} worker_arg;

static void *worker(void *p) {
    worker_arg *a = (worker_arg *)p;
    step_t seq[300];
    make_seq(seq, 300, 0x1000u + (unsigned)a->tid * 7919u);
    for (int i = 0; i < 300; i++) {
        tlm_expert_ref r[TOPK];
        int layer = (seq[i].layer + a->tid) % NL;
        if (pruned(layer, seq[i].ids[0]) || pruned(layer, seq[i].ids[1])) layer = 0;
        int rc;
        while ((rc = tlm_store_acquire(a->s, layer, seq[i].ids, TOPK, r)) == TLM_EBUSY) {
            for (int j = 0; j < TOPK; j++) tlm_store_release(a->s, &r[j]);   /* back off */
            atomic_fetch_add(a->busy, 1);
            sched_yield();
        }
        if (rc != 0) atomic_fetch_add(a->bad, 1);
        int hint[2] = {seq[(i + 1) % 300].ids[0], (i + a->tid) % NE};
        tlm_store_prefetch(a->s, (layer + 1) % NL, hint, 2);
        for (int j = 0; j < TOPK; j++) {
            if (r[j].slot < 0 || tlm_store_wait(a->s, &r[j]) != 0 ||
                !payload_ok(&r[j], layer, seq[i].ids[j]))
                atomic_fetch_add(a->bad, 1);
        }
        for (int j = 0; j < TOPK; j++) tlm_store_release(a->s, &r[j]);
        if (a->tid == 0) tlm_store_tick(a->s);
    }
    return NULL;
}

static void test_concurrent(void) {
    tlm_io *io = open_io(1);
    enum { T = 4 };
    tlm_store *s = open_store(io, T * TOPK + 2, 1, 0.1);   /* one pin, 9 streaming slots */
    if (!s) {
        tlm_io_close(io);
        return;
    }
    CHECK_EQ_INT(tlm_store_warm(s), 0);
    atomic_int bad, busy;
    atomic_init(&bad, 0);
    atomic_init(&busy, 0);
    pthread_t th[T];
    worker_arg args[T];
    for (int t = 0; t < T; t++) {
        args[t] = (worker_arg){s, t, &bad, &busy};
        CHECK_EQ_INT(pthread_create(&th[t], NULL, worker, &args[t]), 0);
    }
    for (int t = 0; t < T; t++) pthread_join(th[t], NULL);
    CHECK_EQ_INT(atomic_load(&bad), 0);
    tlm_store_stats st;
    tlm_store_get_stats(s, &st);
    CHECK_EQ_INT(st.in_use, 0);
    CHECK_EQ_INT(st.pinned, 1);
    CHECK(st.hits + st.misses >= (uint64_t)(T * 300 * TOPK));
    CHECK(st.prefetch_used <= st.prefetch_issued);
    tlm_store_close(s);
    tlm_io_close(io);
}

int main(void) {
    snprintf(g_root, sizeof g_root, "%s/tlm-test-store-%ld", tlm_test_tmpbase(), (long)getpid());
    const char *be = getenv("TLM_TEST_IO_BACKEND");
    if (be && strcmp(be, "uring") == 0) g_backend = TLM_IO_URING;
    rm_rf(g_root);
    CHECK_EQ_INT(tlm_mkdirs(g_root, 0700), 0);
    for (int i = 0; i < 3; i++) snprintf(g_dirs[i], sizeof g_dirs[i], "%s/drive%d", g_root, i);
    if (build_store() != 0) {
        fprintf(stderr, "  FAIL fixture: %s\n", tlm_errmsg());
        rm_rf(g_root);
        return 1;
    }
    RUN(test_open_args);
    RUN(test_bytes_across_budgets);
    RUN(test_lfu_decay_and_hints);
    RUN(test_prefetch_reserve);
    RUN(test_pins);
    RUN(test_ebusy_and_notfound);
    RUN(test_crc_failures);
    RUN(test_warm_sidecar);
    RUN(test_concurrent);
    tlm_manifest_free(&g_m);
    for (int l = 0; l < NL; l++)
        for (int e = 0; e < NE; e++) free(g_data[l][e]);
    rm_rf(g_root);
    TEST_MAIN_END();
}
