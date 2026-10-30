#!/usr/bin/env bash
# oracle.sh MODEL MANIFEST TOKENS [TOPK] [PY]
#
# P-121 teacher-forced oracle: run tierlm's per-position top-k dumper and the
# HF reference dumper on the SAME id sequence, then compare per-position
# argmax (the M1 real-model bring-up gate).
#
#   MODEL    HF checkpoint dir (e.g. ~/models/Qwen3-30B-A3B)
#   MANIFEST a compiled .tlmm store (from: tlmc compile ...)
#   TOKENS   comma-separated raw token ids (e.g. 151643,32,151643,...)
#   TOPK     top-k to dump/compare (default 10)
#   PY       python with torch+transformers+bitsandbytes (default the venv)
#
# The reference is 4-bit on CPU (fits ~22GB). The metric is per-position
# argmax + top-k id-set overlap, which is scale-invariant, so the 4-bit
# reference is a valid proxy for bring-up. The strict BF16 gate runs on the
# same box: `oracle_ref.py ... --dtype bf16` (full BF16 forward, ~90 s).
set -euo pipefail

MODEL="${1:?usage: oracle.sh MODEL MANIFEST TOKENS [TOPK] [PY]}"
MANIFEST="${2:?missing MANIFEST}"
TOKENS="${3:?missing TOKENS (comma-separated ids)}"
TOPK="${4:-10}"
PY="${5:-$HOME/venvs/tierlm-oracle/bin/python}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIERLM="$ROOT/build/bin/tierlm"

[ -x "$TIERLM" ] || { echo "oracle: $TIERLM not built; run 'make' first" >&2; exit 2; }
[ -x "$PY" ] || { echo "oracle: $PY not found; build the venv or pass PY" >&2; exit 2; }

T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT

echo "oracle: tierlm oracle  $MANIFEST  (${#TOKENS} bytes of ids)"
"$TIERLM" oracle "$MANIFEST" --tokens "$TOKENS" > "$T/tierlm.jsonl"

echo "oracle: HF reference   $MODEL  (4-bit CPU)"
"$PY" "$ROOT/tools/oracle_ref.py" "$MODEL" --tokens "$TOKENS" --topk "$TOPK" > "$T/ref.jsonl"

echo "oracle: comparing"
"$PY" "$ROOT/tools/oracle_cmp.py" "$T/tierlm.jsonl" "$T/ref.jsonl"
