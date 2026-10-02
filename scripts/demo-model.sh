#!/bin/sh
# scripts/demo-model.sh - build a small synthetic model under ./scratch for
# trying and debugging tierlm (used by the VS Code tasks and launch configs).
#
#   scripts/demo-model.sh [PROFILE]     PROFILE = exact (default) | balanced | aggressive
#
# Produces scratch/demo.tlmm, packet files in scratch/nvme0..3 and the work dir
# scratch/demo.work (decision log, reference store). scratch/etc is an empty
# config dir so nothing from /etc/tierlm leaks in. scratch/ is git-ignored.
set -eu
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build}
PROFILE=${1:-exact}
make -s BUILD="$BUILD" bins
S=scratch
mkdir -p "$S/etc"
export TIERLM_ETC="$PWD/$S/etc"
if [ ! -s "$S/src/model.safetensors" ]; then
    "$BUILD/bin/tlmc" synth --out "$S/src" --layers 4 --experts 32 --top-k 4 --shared 1 --seed 1
fi
"$BUILD/bin/tlmc" compile "$S/src" --out "$S/demo.tlmm" \
    --drives "$S/nvme0,$S/nvme1,$S/nvme2,$S/nvme3" --profile "$PROFILE"
echo
echo "demo model: $S/demo.tlmm"
echo "try:  TIERLM_ETC=$S/etc $BUILD/bin/tierlm run $S/demo.tlmm --prompt Hello -n 32 --stats"
