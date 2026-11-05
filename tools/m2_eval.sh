#!/usr/bin/env bash
# tools/m2_eval.sh - M2 measurement harness: expert-cache bandwidth sweep + A/B.
#
# Runs `tierlm bench` across a set of expert-cache sizes (and an I/O A/B at a
# fixed size) on a compiled store, then emits a machine-readable summary for the
# M2 gate:
#
#     gate = per-drive expert-read bandwidth / that drive's raw fio sequential
#            read bandwidth                      (need >= 0.85)
#
# Expert-read bandwidth comes from the runtime's own counters: each drive reports
# the bytes it read and the wall time it spent reading (busy_ns), so per-drive
# GiB/s = bytes / busy_ns. That is the bandwidth the drive actually delivered to
# the runtime - the quantity to compare against fio (which measures the same
# drive's raw sequential-read bandwidth). GiB/s here = 2^30 bytes/s.
#
#     tools/m2_eval.sh STORE DRIVES OUT
#
#       STORE    compiled store manifest (m.tlmm)
#       DRIVES   comma-separated drive dirs, in the store's manifest order (the
#                gate indexes the fio baseline to the bench's drive index)
#       OUT      output directory (created if missing)
#
# Env (all optional):
#     BUILD        build dir          (default build)
#     CACHE_LIST   cache sizes to sweep (default "1G 4G 8G 16G 32G 64G")
#     AB_CACHE     cache size for the A/B runs (default 16G)
#     PLEN         prompt length      (default 512)
#     GEN          tokens generated   (default 128)
#     FIO_RUNTIME  fio seconds per drive (default 30; 0 = skip fio)
#     NO_PROBE=1   skip the tierlm-setup probe
#
# Outputs (under OUT):
#     probe.txt         tierlm-setup probe + recommendations (if the tool is built)
#     fio.txt           raw fio output per drive (if fio is installed)
#     report.txt        tlmc report STORE
#     bench/<name>.json one --stats-json line per sweep point and A/B run
#     summary.json      parsed per-run fields + derived bandwidth + gate
#     table.md          the sweep as a markdown table
set -euo pipefail

BUILD=${BUILD:-build}
B=$BUILD/bin
STORE=${1:?usage: m2_eval.sh STORE DRIVES OUT}
DRIVES=${2:?usage: m2_eval.sh STORE DRIVES OUT}
OUT=${3:?usage: m2_eval.sh STORE DRIVES OUT}
CACHE_LIST=${CACHE_LIST:-"1G 4G 8G 16G 32G 64G"}
AB_CACHE=${AB_CACHE:-16G}
PLEN=${PLEN:-512}
GEN=${GEN:-128}
FIO_RUNTIME=${FIO_RUNTIME:-30}

command -v jq >/dev/null 2>&1 || { echo "m2_eval: jq is required" >&2; exit 1; }
[ -x "$B/tierlm" ] && [ -x "$B/tlmc" ] || { echo "m2_eval: build the binaries first (make BUILD=$BUILD bins)" >&2; exit 1; }
[ -f "$STORE" ] || { echo "m2_eval: no such store: $STORE" >&2; exit 1; }
mkdir -p "$OUT/bench"

say() { printf 'm2_eval: %s\n' "$*"; }

# bench NAME [-- extra bench args]: run one bench, save its --stats-json line.
bench() {
    local name=$1; shift
    say "bench $name"
    "$B/tierlm" bench "$STORE" --prompt-len "$PLEN" -n "$GEN" --stats-json "$@" \
        > "$OUT/bench/$name.json"
}

# ---- tierlm-setup probe (hardware context) ----------------------------------
if [ -x "$B/tierlm-setup" ] && [ "${NO_PROBE:-0}" != 1 ]; then
    say "tierlm-setup probe"
    "$B/tierlm-setup" </dev/null > "$OUT/probe.txt" 2>&1 || say "probe failed (see probe.txt)"
fi

# ---- store report ------------------------------------------------------------
say "tlmc report"
"$B/tlmc" report "$STORE" > "$OUT/report.txt" 2>&1

# ---- fio baseline (per-drive sequential read at the expert-read size) --------
FIO_GIBS_JSON='{}'
if [ "$FIO_RUNTIME" != 0 ] && command -v fio >/dev/null 2>&1; then
    say "fio baseline ($FIO_RUNTIME s per drive)"
    : > "$OUT/fio.txt"
    IFS=',' read -r -a DRV <<< "$DRIVES"
    idx=0
    for d in "${DRV[@]}"; do
        if [ -d "$d" ]; then
            # libaio is required: fio's default sync engine silently caps the
            # queue depth at 1, and its warning line corrupts the JSON on
            # stdout (the gate then parses 0). The per-drive --output file and
            # the grep guard make the parse robust either way.
            fio --name="m2-$(basename "$d")" --directory="$d" --rw=read --bs=4m --direct=1 \
                --ioengine=libaio --iodepth=32 --numjobs=1 --size=8G --runtime="$FIO_RUNTIME" --time_based \
                --group_reporting --output-format=json --output="$OUT/fio.$idx.json" >> "$OUT/fio.txt" 2>&1
            kib=$(grep -v '^note:' "$OUT/fio.$idx.json" 2>/dev/null | jq -r '.jobs[0].read.bw // 0' || echo 0)
            gibs=$(awk -v k="$kib" 'BEGIN{printf "%.3f", k*1024/1073741824}')
            FIO_GIBS_JSON=$(jq -c --arg k "$idx" --argjson v "$gibs" '. + {($k): $v}' <<< "$FIO_GIBS_JSON")
            [ "$gibs" = "0.000" ] && say "WARNING: fio reported 0 GiB/s for $d (see $OUT/fio.txt); the gate will be null"
        else
            say "fio: skip missing dir $d"
        fi
        idx=$((idx+1))
    done
    say "fio per-drive GiB/s: $FIO_GIBS_JSON"
else
    say "fio skipped (not installed or FIO_RUNTIME=0); the gate will be null"
fi

# ---- the sweep + A/B ---------------------------------------------------------
for c in $CACHE_LIST; do
    bench "cache_$c" --cache "$c"
done
# A/B at AB_CACHE. io_uring may fall back to pread if the build lacks liburing;
# the JSON records the backend actually used, so a fallback is visible.
bench "ab_io_pread"   --cache "$AB_CACHE" --io pread
bench "ab_io_uring"   --cache "$AB_CACHE" --io io_uring
bench "ab_direct_no"  --cache "$AB_CACHE" --direct no
bench "ab_direct_yes" --cache "$AB_CACHE" --direct yes

# ---- summary.json ------------------------------------------------------------
# One object per bench run: the raw counters plus derived bandwidth. gate is the
# min over drives of (drive GiB/s / fio GiB/s); null when fio is unavailable.
say "writing summary.json"
{
    echo '['
    first=1
    for f in "$OUT"/bench/*.json; do
        [ -e "$f" ] || continue
        name=$(basename "$f" .json)
        if [ "$first" = 1 ]; then first=0; else echo ','; fi
        jq -c --arg name "$name" --argjson fio "$FIO_GIBS_JSON" '
            def gibs($b; $ns): if $ns > 0 then ($b * 8) / ($ns * 1e-9) / 1073741824 else 0 end;
            . as $s
            | ($s.io.drives | map(.busy_ns) | add // 0) as $busy
            | ($s.io.drives | map(.bytes)   | add // 0) as $bytes
            | (if ($s.decode_tok_s // 0) > 0 and ($s.generated // 0) > 1
               then ($s.generated - 1) / $s.decode_tok_s else 0 end) as $dec_s
            | ($s.io.drives | to_entries
               | map(select(.value.busy_ns > 0)
                     | { b: (.value.bytes * 8 / 1073741824 / (.value.busy_ns * 1e-9)),
                         f: ($fio[(.key | tostring)] // null) })
               | map(select(.f != null and .f > 0) | (.b / .f))
               | if length > 0 then min else null end) as $gate
            | {
                name: $name,
                meta: $s.meta,
                decode_tok_s: $s.decode_tok_s,
                hit_rate: $s.store.hit_rate,
                hits: $s.store.hits,
                misses: $s.store.misses,
                bytes_read: $s.store.bytes_read,
                busy_ns: $busy,
                busy_gibs: gibs($bytes; $busy),
                decode_s: $dec_s,
                eff_gibs: gibs($bytes; $dec_s * 1e9),
                backend: $s.io.backend,
                gate: $gate,
                drives: ($s.io.drives | to_entries | map({ index: .key, reads: .value.reads,
                          bytes: .value.bytes, busy_ns: .value.busy_ns,
                          gibs: gibs(.value.bytes; .value.busy_ns) }))
              }' "$f"
    done
    echo ']'
} | jq . > "$OUT/summary.json"

# ---- table.md ----------------------------------------------------------------
say "writing table.md"

{
    printf '# M2 summary\n\n'
    printf 'store: `%s`\n' "$STORE"
    jq -r '.[0].meta // empty
        | "host: \(.host) — \(.cpu), \(.cores) cores / \(.cpus) cpus, \(.ram_gib) GiB RAM, \(.os) \(.kernel), build \(.version) \(.build), store \(.store.name) (\(.store.profile), sha \(.store.source_sha256[0:12]))"' \
        "$OUT/summary.json"
    printf '\ndecode tok/s and hit rate across the expert-cache sweep, with the per-drive expert-read bandwidth and the gate\n'
    printf '(busy GiB/s / fio GiB/s; needs to be >= 0.85). GiB = 2^30 bytes.\n\n'
    printf '| run | decode tok/s | hit %% | misses | read GiB | busy GiB/s | eff GiB/s | backend | gate |\n'
    printf '|---|---|---|---|---|---|---|---|---|\n'
    jq -r '.[]
        | [.name,
           (.decode_tok_s | tostring),
           ((.hit_rate) | (.*100 | floor / 100 | tostring)),
           (.misses | tostring),
           ((.bytes_read / 1073741824) | (.*100 | floor / 100 | tostring)),
           ((.busy_gibs) | (.*1000 | floor / 1000 | tostring)),
           ((.eff_gibs) | (.*1000 | floor / 1000 | tostring)),
           .backend,
           (if .gate == null then "n/a (no fio)" else (.gate | (.*100 | floor / 100 | tostring)) end)]
        | "| " + join(" | ") + " |"' "$OUT/summary.json"
    printf '\n'
} > "$OUT/table.md"

say "done: $OUT/summary.json and $OUT/table.md"
