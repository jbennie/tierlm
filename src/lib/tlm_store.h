/* tlm_store.h - routed-expert RAM cache over the packet store.
 *
 * Patterns: P-10 bounded slot pool allocated once, P-12 slot lifecycle
 * (reserve -> loading -> ready -> released; publish only after a full CRC-
 * verified read; never evict a slot in use), P-15 frequency-aware eviction
 * with periodic decay (recency as tie-break), P-18 one global cache across
 * layers, P-19 warm pins, P-34 speculative loads never hurt demand, P-33
 * router-lookahead prefetch (the caller supplies predicted ids).
 *
 * The bytes a caller sees are exactly the packet payload from disk: cache
 * size, prefetch and I/O backend can change speed, never results.
 */
#ifndef TLM_STORE_H
#define TLM_STORE_H

#include <stdint.h>
#include "tlm_packet.h"
#include "tlm_io.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tlm_store tlm_store;

typedef struct {
    uint64_t budget_bytes;   /* total slot memory; rounded down to whole slots */
    double   pin_frac;       /* share of slots for pinned hot experts (0..0.9) */
    int      prefetch;       /* accept tlm_store_prefetch hints */
    int      huge, lock;     /* hugepages / mlock for the slot arena */
    const char *warm_path;   /* optional heat sidecar to load/save */
} tlm_store_opts;

typedef struct {
    int      slot;           /* >= 0 */
    const void *payload;     /* valid once ready; points inside the slot */
    uint64_t payload_len;
} tlm_expert_ref;

/* Store over the manifest's EXPERT packets. Slot size = largest expert packet. */
tlm_store *tlm_store_open(const tlm_manifest *m, tlm_io *io, const tlm_store_opts *o);
void       tlm_store_close(tlm_store *s);
int        tlm_store_n_slots(const tlm_store *s);

/* Pin the hottest experts (by manifest heat, then warm sidecar) up to
 * pin_frac of slots; loads them synchronously. */
int        tlm_store_warm(tlm_store *s);

/* Begin acquiring k experts of one layer. Hits are referenced immediately;
 * misses are submitted to I/O. Never blocks. Returns 0 or TLM_E*. */
int        tlm_store_acquire(tlm_store *s, int layer, const int *ids, int k, tlm_expert_ref *out);
/* Block until out[i] is ready (CRC verified). Returns 0, or TLM_EFORMAT/TLM_EIO. */
int        tlm_store_wait(tlm_store *s, tlm_expert_ref *ref);
int        tlm_store_is_ready(tlm_store *s, const tlm_expert_ref *ref);
/* Drop the caller's reference; slot becomes evictable when unreferenced. */
void       tlm_store_release(tlm_store *s, tlm_expert_ref *ref);
/* Advisory: load these if a free/evictable slot exists and no demand load
 * is waiting; never evicts pinned or in-use slots; may drop the request. */
void       tlm_store_prefetch(tlm_store *s, int layer, const int *ids, int k);
/* Call once per generated token: advances the decay clock. */
void       tlm_store_tick(tlm_store *s);

typedef struct {
    uint64_t hits, misses, prefetch_issued, prefetch_used, prefetch_dropped;
    uint64_t bytes_read, evictions, wait_ns;
    int      slots, pinned, in_use;
} tlm_store_stats;
void       tlm_store_get_stats(const tlm_store *s, tlm_store_stats *out);
int        tlm_store_save_warm(const tlm_store *s, const char *path);

/* ---- additions (tlm_store.c) ----------------------------------------------
 * Contract details:
 *  - File i of `io` must be manifest drive i (see tlm_packet_read).
 *  - tlm_store_acquire: every out[i] with slot >= 0 holds a reference and must
 *    be released whatever the return code; out[i].slot == -1 means not
 *    acquired: TLM_ENOTFOUND (no packet / pruned) or TLM_EBUSY (every slot is
 *    referenced or pinned). Size budget_bytes for at least top_k unpinned
 *    slots, i.e. >= (top_k + pinned) * tlm_store_slot_bytes(); prefetch needs
 *    spare slots beyond that to help.
 *  - A LOADING slot is never a victim. tlm_store_prefetch therefore keeps at
 *    most (slots - pinned - max(1, geom.top_k)) hints in flight (and at most
 *    16), takes only FREE slots or READY slots whose expert has decayed
 *    frequency 0, and drops hints while more than 4 demand reads are in
 *    flight; with budget == (top_k + pinned) slots every hint is dropped.
 *  - tlm_store_wait returns TLM_EFORMAT (header/CRC mismatch on every copy)
 *    or TLM_EIO; the slot is never READY in that case.
 *  - tlm_store_release on a ref with slot == -1 is a no-op and clears the ref.
 *  - Close the store before the io (store_close waits for in-flight reads).
 *  - The warm sidecar is never written implicitly; call tlm_store_save_warm.
 *    Its count = stored count * 0.99 + selections in this run (C-09). */
/* Bytes per slot: the largest EXPERT packet (header + payload + padding). */
uint64_t   tlm_store_slot_bytes(const tlm_store *s);

#ifdef __cplusplus
}
#endif
#endif
