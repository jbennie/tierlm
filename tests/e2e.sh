#!/bin/sh
# tests/e2e.sh - end-to-end check of the two stages (research/03-two-stage-design.md):
#   synth -> compile exact over 3 drive dirs -> run with a tiny and a large
#   cache (outputs must be equal) -> synth+compile+run a dense-prefix model
#   (first_k_dense_replace) -> recompile (resume reuses passes 2-3) ->
#   compile balanced with pruning + q5_k plan -> report -> atlas -> run ->
#   compile aggressive with a content exclusion -> config show/check ->
#   missing-drive refusal.
# Honours BUILD (default: build). Uses only a temporary directory.
set -eu

BUILD=${BUILD:-build}
B=$BUILD/bin
for x in tlmc tierlm; do
    [ -x "$B/$x" ] || { echo "e2e: $B/$x not built (make BUILD=$BUILD bins)" >&2; exit 1; }
done

T=$(mktemp -d "${TMPDIR:-/tmp}/tlm-e2e.XXXXXX")
trap 'rm -rf "$T"' EXIT INT TERM
mkdir -p "$T/etc"
TIERLM_ETC=$T/etc          # never read the machine's /etc/tierlm
export TIERLM_ETC

fail() { echo "e2e: FAIL: $*" >&2; exit 1; }
step() { echo "e2e: $*"; }
run_logged() {             # run_logged LOG cmd... : keep output, show it on failure
    log=$1; shift
    if ! "$@" >"$log" 2>&1; then cat "$log" >&2; fail "$*"; fi
}
tokens_only() {             # keep only the printed tokens (drop stats, warns, drive lines)
    grep -v -e '^--' -e '^[a-z]' -e '^  drive' -e 'tierlm warn' "$1"
}

step "synth a tiny Qwen3-MoE-like checkpoint"
run_logged "$T/synth.log" "$B/tlmc" synth --out "$T/src" --layers 3 --experts 16 --top-k 2 --shared 1 --seed 7
[ -s "$T/src/model.safetensors" ] && [ -s "$T/src/config.json" ] || fail "synth wrote no model"

step "compile (exact) over 3 drive dirs"
D3="$T/nvme0,$T/nvme1,$T/nvme2"
run_logged "$T/exact.log" "$B/tlmc" compile "$T/src" --out "$T/models/exact.tlmm" --drives "$D3" \
    --profile exact --replicas auto
grep -q "bit-identical logits yes, greedy free run identical yes" "$T/exact.log" || fail "exact not identical to pass 2"
for d in 0 1 2; do [ -s "$T/nvme$d/exact.tlmp" ] || fail "drive $d has no packet file"; done
[ -s "$T/models/exact.work/ref/exact.tlmp" ] || fail "no pass-2 scratch store"
[ -s "$T/models/exact.work/decisions.jsonl" ] || fail "no decision audit log"
grep -q "replicated (warm middle)" "$T/exact.log" || fail "no placement summary"

step "run: tiny cache (top_k slots) vs large cache must match"
P="Hello, tiered world"
run_logged "$T/tiny.out" "$B/tierlm" run "$T/models/exact.tlmm" --prompt "$P" -n 24 --cache 1 --print-tokens --stats
run_logged "$T/large.out" "$B/tierlm" run "$T/models/exact.tlmm" --prompt "$P" -n 24 --cache 1G --threads 1 \
    --prefetch off --print-tokens
tokens_only "$T/tiny.out" > "$T/tiny.tok"
tokens_only "$T/large.out" > "$T/large.tok"
cmp -s "$T/tiny.tok" "$T/large.tok" || { cat "$T/tiny.tok" "$T/large.tok" >&2; fail "tiny and large cache differ"; }
grep -q "hit rate" "$T/tiny.out" || fail "--stats printed nothing"
run_logged "$T/scalar.out" "$B/tierlm" run "$T/models/exact.tlmm" --prompt "$P" -n 24 --kernel scalar \
    --io io_uring --direct yes --print-tokens
tokens_only "$T/scalar.out" > "$T/scalar.tok"
cmp -s "$T/scalar.tok" "$T/large.tok" || fail "scalar kernel tier / io_uring request changed the tokens"
run_logged "$T/tok.out" "$B/tierlm" run "$T/models/exact.tlmm" --tokens 72,101,108 -n 4 --print-tokens
[ "$(wc -w < "$T/tok.out")" -eq 4 ] || fail "--tokens run"
run_logged "$T/bench.out" "$B/tierlm" bench "$T/models/exact.tlmm" --prompt-len 16 -n 16
grep -q "tok/s" "$T/bench.out" || fail "bench"
grep -Eq '^run +host .*, cpu .*, ram [0-9]+ GiB, os .*, build [^,]+, store .*, sha [0-9a-f]{64}\)$' "$T/bench.out" \
    || fail "bench run stamp (host/cpu/ram/os/build/store) missing"
run_logged "$T/bench.json" "$B/tierlm" bench "$T/models/exact.tlmm" --prompt-len 16 -n 16 --stats-json
grep -Eq '"meta":\{"host":"[^"]*"' "$T/bench.json" || fail "bench --stats-json meta stamp missing"
grep -Eq '"source_sha256":"[0-9a-f]{64}"' "$T/bench.json" || fail "bench --stats-json store sha missing"
step "oracle: per-position teacher-forced top-k dump (P-121)"
run_logged "$T/oracle.out" "$B/tierlm" oracle "$T/models/exact.tlmm" --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
[ "$(grep -c '^{"pos"' "$T/oracle.out")" -eq 16 ] || fail "oracle emitted $(grep -c '^{"pos"' "$T/oracle.out") positions, want 16"
[ "$(grep -c '"topk": \[' "$T/oracle.out")" -eq 16 ] || fail "oracle lines lack a topk array"
grep -q '"pos": 0' "$T/oracle.out" || fail "oracle missing pos 0"
grep -q '"pos": 15' "$T/oracle.out" || fail "oracle missing pos 15 (last)"

step "synth --fp8: block-scale FP8 checkpoint (d_expert 320 = partial 128 block), compile + run"
run_logged "$T/fp8s.log" "$B/tlmc" synth --out "$T/src-fp8" --layers 2 --experts 8 --top-k 2 --shared 1 --d-expert 320 --seed 3 --fp8
grep -q "FP8 E4M3 with 128x128 block scales" "$T/fp8s.log" || fail "synth --fp8 flag ignored"
run_logged "$T/fp8c.log" "$B/tlmc" compile "$T/src-fp8" --out "$T/models/fp8.tlmm" --drives "$D3" --profile exact
grep -q "bit-identical logits yes, greedy free run identical yes" "$T/fp8c.log" || fail "fp8 store not internally consistent"
run_logged "$T/fp8t.out" "$B/tierlm" run "$T/models/fp8.tlmm" --prompt "$P" -n 16 --cache 1 --print-tokens
run_logged "$T/fp8l.out" "$B/tierlm" run "$T/models/fp8.tlmm" --prompt "$P" -n 16 --print-tokens
tokens_only "$T/fp8t.out" > "$T/fp8t.tok"
tokens_only "$T/fp8l.out" > "$T/fp8l.tok"
[ "$(wc -w < "$T/fp8l.tok")" -eq 16 ] || fail "expected 16 tokens from the fp8 store"
cmp -s "$T/fp8t.tok" "$T/fp8l.tok" || { cat "$T/fp8t.tok" "$T/fp8l.tok" >&2; fail "fp8 store: tiny and large cache differ"; }

step "synth + compile + run a dense-prefix model (first_k_dense_replace)"
run_logged "$T/synthd.log" "$B/tlmc" synth --out "$T/srcd" --layers 3 --experts 16 --top-k 2 --shared 1 \
    --seed 7 --dense-layers 2 --dense-ffn 128
grep -q '"first_k_dense_replace": 2' "$T/srcd/config.json" || fail "synth did not emit first_k_dense_replace"
grep -q '"intermediate_size": 128' "$T/srcd/config.json" || fail "synth did not emit intermediate_size"
run_logged "$T/exactd.log" "$B/tlmc" compile "$T/srcd" --out "$T/models/dense.tlmm" --drives "$D3" \
    --profile exact --replicas auto
grep -q "bit-identical logits yes, greedy free run identical yes" "$T/exactd.log" || fail "dense exact not identical to pass 2"
# 3 DENSE packets + 1 routed layer x 16 experts + META + EMBED = 21 packets
grep -q "21 packets on 3 drives" "$T/exactd.log" || fail "dense store has the wrong packet count"
run_logged "$T/dtiny.out" "$B/tierlm" run "$T/models/dense.tlmm" --prompt "$P" -n 24 --cache 1 --print-tokens --stats
run_logged "$T/dlarge.out" "$B/tierlm" run "$T/models/dense.tlmm" --prompt "$P" -n 24 --cache 1G --threads 1 \
    --prefetch off --print-tokens
tokens_only "$T/dtiny.out" > "$T/dtiny.tok"
tokens_only "$T/dlarge.out" > "$T/dlarge.tok"
[ "$(wc -w < "$T/dlarge.tok")" -eq 24 ] || fail "dense run gave $(wc -w < "$T/dlarge.tok") tokens, expected 24"
cmp -s "$T/dtiny.tok" "$T/dlarge.tok" || { cat "$T/dtiny.tok" "$T/dlarge.tok" >&2; fail "dense: tiny and large cache differ"; }

step "recompile (exact): passes 2 and 3 resume from the work dir"
run_logged "$T/exact2.log" "$B/tlmc" compile "$T/src" --out "$T/models/exact.tlmm" --drives "$D3" --profile exact
grep -q "reference store .*\[reused\]" "$T/exact2.log" || fail "pass 2 not reused"
grep -q "calibrated on .*\[reused\]" "$T/exact2.log" || fail "pass 3 not reused"
run_logged "$T/again.out" "$B/tierlm" run "$T/models/exact.tlmm" --prompt "$P" -n 24 --print-tokens
tokens_only "$T/again.out" > "$T/again.tok"
cmp -s "$T/again.tok" "$T/large.tok" || fail "recompiled store gives different tokens"

step "compile (balanced): REAP pruning on a text corpus, gated"
for f in docs/DISTILLED.md research/03-two-stage-design.md research/04-decision-engine.md; do
    [ -f "$f" ] && cat "$f"
done > "$T/corpus.txt" || true
[ -s "$T/corpus.txt" ] || "$B/tlmc" --help > "$T/corpus.txt" 2>&1 || true
mkdir -p "$T/etc-bal"
printf 'experts.protect = 0:1, 2:3\n' > "$T/etc-bal/tierlm.conf"
run_logged "$T/bal.log" "$B/tlmc" compile "$T/src" --out "$T/models/bal.tlmm" --drives "$D3" \
    --profile balanced --corpus "$T/corpus.txt" --calib-tokens 3072 --eval-tokens 768 --etc "$T/etc-bal" --report
grep -q '"profile":"balanced"' "$T/bal.log" || fail "--report did not print the report"
NP=$(sed -n 's/^prune  *\([0-9][0-9]*\) of .*/\1/p' "$T/bal.log")
[ -n "$NP" ] && [ "$NP" -gt 0 ] || fail "balanced compile pruned nothing"
grep -q "attempt .*-> pass" "$T/bal.log" || fail "no passing gate attempt"

step "compile with a BPE tokenizer.json: corpus tokenised with the model's own BPE"
run_logged "$T/synthb.log" "$B/tlmc" synth --out "$T/src-bpe" --layers 2 --experts 8 --top-k 2 --shared 1 --seed 4
cp tests/fixtures/tiny_bpe_tokenizer.json "$T/src-bpe/tokenizer.json"
printf 'aa YY aa YY end of line\n' > "$T/bpe-corpus.txt"
run_logged "$T/bpe.log" "$B/tlmc" compile "$T/src-bpe" --out "$T/models/bpe.tlmm" --drives "$T/nvme0" \
    --profile exact --replicas auto --corpus "$T/bpe-corpus.txt" --calib-tokens 4096
run_logged "$T/bpe.json" "$B/tlmc" report "$T/models/bpe.tlmm" --json
grep -q '"tokenizer":"bpe"' "$T/bpe.json" || fail "BPE tokenizer.json was not used for the corpus"
run_logged "$T/byte.log" "$B/tlmc" compile "$T/src" --out "$T/models/byte.tlmm" --drives "$T/nvme0" \
    --profile exact --replicas auto --corpus "$T/bpe-corpus.txt" --calib-tokens 4096
run_logged "$T/byte.json" "$B/tlmc" report "$T/models/byte.tlmm" --json
grep -q '"tokenizer":"byte"' "$T/byte.json" || fail "synthetic model (no tokenizer.json) should stay byte-level"
BT=$(sed -n 's/.*"calibration":{"tokens":\([0-9]*\).*/\1/p' "$T/bpe.json")
YT=$(sed -n 's/.*"calibration":{"tokens":\([0-9]*\).*/\1/p' "$T/byte.json")
[ -n "$BT" ] && [ -n "$YT" ] && [ "$BT" -lt "$YT" ] || fail "BPE corpus ($BT tokens) should be fewer than byte-level ($YT tokens)"

step "report + atlas"
run_logged "$T/report.out" "$B/tlmc" report "$T/models/bal.tlmm"
grep -q "^prune  *$NP of" "$T/report.out" || fail "report disagrees with compile"
grep -q "^validate" "$T/report.out" || fail "report has no validation"
run_logged "$T/report.json" "$B/tlmc" report "$T/models/bal.tlmm" --json
grep -q '"experts":\[\[' "$T/report.json" || fail "report JSON has no expert atlas"
run_logged "$T/atlas.out" "$B/tlmc" atlas "$T/models/bal.tlmm" --sort heat --top 8
[ "$(grep -c '^[0-9]*:[0-9]' "$T/atlas.out")" -eq 8 ] || fail "atlas --top 8"
run_logged "$T/atlas1.out" "$B/tlmc" atlas "$T/models/bal.tlmm" --layer 1
[ "$(grep -c '^1:[0-9]' "$T/atlas1.out")" -eq 16 ] || fail "atlas --layer 1"
run_logged "$T/atlas-all.out" "$B/tlmc" atlas "$T/models/bal.tlmm"
[ "$(grep -c 'pruned$' "$T/atlas-all.out")" -eq "$NP" ] || fail "atlas pruned count differs from the report"
grep -q -e "^0:1 .*pruned" -e "^2:3 .*pruned" "$T/atlas-all.out" && fail "a protected expert was pruned" || true

step "run the pruned store (tiny vs large cache)"
run_logged "$T/bt.out" "$B/tierlm" run "$T/models/bal.tlmm" --prompt "$P" -n 16 --cache 1 --print-tokens
run_logged "$T/bl.out" "$B/tierlm" run "$T/models/bal.tlmm" --prompt "$P" -n 16 --print-tokens --stats
tokens_only "$T/bt.out" > "$T/bt.tok"
tokens_only "$T/bl.out" > "$T/bl.tok"
[ "$(wc -w < "$T/bl.tok")" -eq 16 ] || fail "expected 16 tokens from the pruned store"
cmp -s "$T/bt.tok" "$T/bl.tok" || fail "pruned store: tiny and large cache differ"

step "compile (aggressive): content exclusion + q5_k plan, 2 drives, no replicas"
mkdir -p "$T/etc-agg"
printf 'content.default = exclude\nprune.max_ratio = 0.4\n' > "$T/etc-agg/tierlm.conf"
# probed drive bandwidth (tierlm-setup --probe facts): nvme1 is 3x faster
printf 'hardware.drive.count = 2\nhardware.drive.0.path = %s\nhardware.drive.0.direct_gbs = 1.0\nhardware.drive.1.path = %s\nhardware.drive.1.direct_gbs = 3.0\n' \
    "$T/nvme0" "$T/nvme1" > "$T/etc-agg/hardware.conf"
run_logged "$T/agg.log" "$B/tlmc" compile "$T/src" --out "$T/models/agg.tlmm" --drives "$T/nvme0,$T/nvme1" \
    --profile aggressive --replicas none --etc "$T/etc-agg" --dense f16
grep -q "0 replicated" "$T/agg.log" || fail "--replicas none still replicated"
grep -q -e "experts demoted to q5_k" -e "q5_k plan rolled back by the gate" "$T/agg.log" || fail "no q5_k precision plan"
grep -q "^precision   dense f16" "$T/agg.log" || fail "--dense f16 ignored"
S0=$(sed -n 's/^  drive 0 .* \([0-9][0-9]*\)% of expected expert reads$/\1/p' "$T/agg.log")
S1=$(sed -n 's/^  drive 1 .* \([0-9][0-9]*\)% of expected expert reads$/\1/p' "$T/agg.log")
[ -n "$S0" ] && [ -n "$S1" ] && [ "$S1" -gt "$S0" ] || fail "placement ignored probed bandwidth ($S0% vs $S1%)"
run_logged "$T/agg.out" "$B/tierlm" run "$T/models/agg.tlmm" --prompt "$P" -n 8 --print-tokens
[ "$(tokens_only "$T/agg.out" | wc -w)" -eq 8 ] || fail "aggressive run"

step "config show / check"
printf 'cpu.threads = 2\nio.prefetch = off\n' > "$T/etc/tierlm.conf"
run_logged "$T/cfg.out" "$B/tierlm" config show --origin
grep -q "^cpu.threads  *= 2 .*# file" "$T/cfg.out" || fail "config show --origin"
run_logged "$T/chk.out" "$B/tierlm" config check
printf 'cpu.threads = many\n' > "$T/etc/tierlm.conf"
if "$B/tierlm" config check > "$T/chk2.out" 2>&1; then fail "config check accepted a bad value"; fi
rm -f "$T/etc/tierlm.conf"

step "a missing drive without replicas is refused"
mv "$T/nvme2/exact.tlmp" "$T/nvme2/exact.tlmp.away"
if "$B/tierlm" run "$T/models/exact.tlmm" -n 2 > "$T/miss.out" 2>&1; then fail "ran with a missing drive"; fi
grep -q "missing" "$T/miss.out" || { cat "$T/miss.out" >&2; fail "unclear missing-drive error"; }
mv "$T/nvme2/exact.tlmp.away" "$T/nvme2/exact.tlmp"

step "synth + compile + run a YaRN rope_scaling model"
run_logged "$T/yarn.log" "$B/tlmc" synth --out "$T/src-yarn" --layers 3 --experts 16 --top-k 2 --shared 1 --seed 7 --yarn
grep -q '"rope_type": "yarn"' "$T/src-yarn/config.json" || fail "synth --yarn wrote no rope_scaling"
run_logged "$T/yc.log" "$B/tlmc" compile "$T/src-yarn" --out "$T/models/yarn.tlmm" --drives "$T/nvme0" --profile exact
run_logged "$T/yr.out" "$B/tlmc" report "$T/models/yarn.tlmm"
grep -q "yarn x4" "$T/yr.out" || fail "report has no rope line"
run_logged "$T/yr.json" "$B/tlmc" report "$T/models/yarn.tlmm" --json
grep -q '"rope":{"type":"yarn-truncate"' "$T/yr.json" || fail "report JSON lost the rope geometry"
run_logged "$T/yt.out" "$B/tierlm" run "$T/models/yarn.tlmm" --prompt "$P" -n 16 --cache 1 --print-tokens
run_logged "$T/yl.out" "$B/tierlm" run "$T/models/yarn.tlmm" --prompt "$P" -n 16 --print-tokens
tokens_only "$T/yt.out" > "$T/yt.tok"
tokens_only "$T/yl.out" > "$T/yl.tok"
[ "$(wc -w < "$T/yl.tok")" -eq 16 ] || fail "yarn run"
cmp -s "$T/yt.tok" "$T/yl.tok" || fail "yarn store: tiny and large cache differ"
cp -r "$T/src-yarn" "$T/src-bad"
sed 's/"rope_type": "yarn"/"rope_type": "linear"/' "$T/src-yarn/config.json" > "$T/src-bad/config.json"
if "$B/tlmc" compile "$T/src-bad" --out "$T/models/bad.tlmm" --drives "$T/nvme0" --profile exact > "$T/bad.out" 2>&1; then
    fail "compile accepted an unsupported rope type"
fi
grep -q "not supported" "$T/bad.out" || { cat "$T/bad.out" >&2; fail "no rope rejection message"; }

step "chat: multi-turn REPL with /reset and /think (Qwen3 tokenizer)"
mkdir -p "$T/chat-tok"
cp tests/fixtures/qwen3_chat_tokenizer.json "$T/chat-tok/tokenizer.json"
if ! "$B/tierlm" chat "$T/models/exact.tlmm" --tokenizer "$T/chat-tok" --system "sys" -n 16 --seed 7 > "$T/chat.out" 2>&1 <<'CHATEOF'
turn one
turn two
/think 1
turn three
/reset
turn four
/quit
CHATEOF
then
    cat "$T/chat.out" >&2
    fail "chat REPL exited nonzero"
fi
grep -q "turn failed" "$T/chat.out" && { cat "$T/chat.out" >&2; fail "chat: a turn failed"; }
[ "$(grep -c '^you>' "$T/chat.out")" -ge 5 ] || fail "chat: expected >=5 prompts, got $(grep -c '^you>' "$T/chat.out")"

step "serve: OpenAI-compatible HTTP (health, completion, 404, 400)"
PORT=18321
"$B/tierlm" serve "$T/models/exact.tlmm" --tokenizer "$T/chat-tok" --system "sys" -n 16 --seed 7 --port "$PORT" > "$T/serve.out" 2>&1 &
SRV=$!
tries=0
until curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; do
    tries=$((tries + 1))
    [ "$tries" -gt 100 ] && { kill "$SRV" 2>/dev/null; cat "$T/serve.out" >&2; fail "serve did not start"; }
    sleep 0.1
done
curl -sf "http://127.0.0.1:$PORT/health" > "$T/health.json" || { kill "$SRV" 2>/dev/null; fail "GET /health"; }
grep -q '"status":"ok"' "$T/health.json" || { kill "$SRV" 2>/dev/null; fail "health body wrong"; }
curl -sf -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"t","messages":[{"role":"user","content":"hi"}]}' > "$T/comp.json" || { kill "$SRV" 2>/dev/null; cat "$T/comp.json" >&2; fail "POST completion"; }
grep -q '"object":"chat.completion"' "$T/comp.json" || { kill "$SRV" 2>/dev/null; cat "$T/comp.json" >&2; fail "completion response shape"; }
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/nope")" = "404" ] || { kill "$SRV" 2>/dev/null; fail "expected 404 on unknown path"; }
[ "$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -d '{bad')" = "400" ] || { kill "$SRV" 2>/dev/null; fail "expected 400 on bad json"; }
kill "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null || true

echo "e2e: ok"
