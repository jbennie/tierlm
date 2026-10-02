#!/bin/sh
# tests/setup.sh - end-to-end checks of tierlm-setup against a temporary etc
# directory: --auto writes tierlm.conf and hardware.conf, --show reads them
# back (exit 0), plus re-runs, dry runs, answers files, conf.d and environment
# precedence, --probe, --apply-system, --drive and the shipped etc/ files.
#
#   BUILD=build sh tests/setup.sh        (BUILD defaults to "build")
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN=${BUILD:-build}/bin/tierlm-setup
case $BIN in /*) ;; *) [ -x "$BIN" ] || BIN=$ROOT/$BIN ;; esac
[ -x "$BIN" ] || { echo "setup.sh: $BIN is not built" >&2; exit 1; }

# a clean environment: no TIERLM_* from the caller
for v in $(env | sed -n 's/^\(TIERLM_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done

TMP=$(mktemp -d "${TMPDIR:-/tmp}/tlm-setup-test-XXXXXX")
trap 'rm -rf "$TMP"' EXIT INT TERM
ETC=$TMP/etc
N=0
fail() { echo "setup.sh: FAIL: $*" >&2; [ -f "$TMP/out" ] && sed 's/^/  | /' "$TMP/out" | tail -40 >&2; exit 1; }
ok() { N=$((N + 1)); }
run() { "$BIN" "$@" </dev/null >"$TMP/out" 2>&1; }

# 1. the required check: --auto writes both files and --show exits 0
run --etc "$ETC" --auto --probe-seconds 1 || fail "--auto exited non-zero"
[ -f "$ETC/tierlm.conf" ] || fail "tierlm.conf missing"
[ -f "$ETC/hardware.conf" ] || fail "hardware.conf missing"
"$BIN" --etc "$ETC" --show >"$TMP/show" 2>"$TMP/out" || fail "--show exited non-zero"
"$BIN" --etc "$ETC" --show --strict >/dev/null 2>"$TMP/out" || fail "--show --strict exited non-zero"
ok

# 2. contents
grep -q '^cpu\.threads  *= [0-9][0-9]*' "$ETC/tierlm.conf" || fail "cpu.threads not numeric"
grep -q '^io\.backend  *= pread' "$ETC/tierlm.conf" || grep -q '^io\.backend  *= io_uring' "$ETC/tierlm.conf" || fail "io.backend"
grep -q '^include conf\.d/\*\.conf$' "$ETC/tierlm.conf" || fail "conf.d include line"
grep -q '^hardware\.cpu\.cores  *= [1-9]' "$ETC/hardware.conf" || fail "hardware.cpu.cores"
grep -q '^hardware\.mem\.read_gbs ' "$ETC/hardware.conf" || fail "hardware.mem.read_gbs"
grep -q '^hardware\.io_uring\.kernel  *= \(yes\|no\)' "$ETC/hardware.conf" || fail "hardware.io_uring.kernel"
grep -q '^cpu\.threads .* # tierlm\.conf:[0-9]' "$TMP/show" || fail "--show origin of cpu.threads"
grep -q '^hardware\.cpu\.cores .* # hardware\.conf:[0-9]' "$TMP/show" || fail "--show origin of hardware.cpu.cores"
cmp -s "$ETC/content-taxonomy.conf" "$ROOT/etc/content-taxonomy.conf" || fail "installed taxonomy differs from etc/"
ok

# 3. a second --auto run changes nothing in tierlm.conf (no backup); hardware.conf is re-measured
cp "$ETC/tierlm.conf" "$TMP/conf.1"
run --etc "$ETC" --auto --probe-seconds 1 || fail "second --auto"
grep -q 'tierlm.conf: no changes' "$TMP/out" || fail "second run should not change tierlm.conf"
cmp -s "$ETC/tierlm.conf" "$TMP/conf.1" || fail "tierlm.conf rewritten without changes"
ls "$ETC"/tierlm.conf.bak.* >/dev/null 2>&1 && fail "unexpected tierlm.conf backup"
ls "$ETC"/hardware.conf.bak.* >/dev/null 2>&1 || fail "hardware.conf backup missing"
ok

# 4. no terminal and no --auto/--answers behaves like --dry-run; --dry-run writes nothing
cp "$ETC/hardware.conf" "$TMP/hw.1"
run --etc "$ETC" --probe-seconds 1 || fail "non-tty run"
grep -q 'stdin is not a terminal' "$TMP/out" || fail "missing non-tty note"
grep -q 'dry run: nothing written' "$TMP/out" || fail "non-tty run should be a dry run"
run --etc "$ETC" --auto --dry-run --section content || fail "--dry-run"
cmp -s "$ETC/tierlm.conf" "$TMP/conf.1" && cmp -s "$ETC/hardware.conf" "$TMP/hw.1" || fail "dry run wrote files"
ok

# 5. answers: content section only (no probe), inherit removes a depth
printf 'content.law = exclude\ncontent.code = Deep\ncompile.profile = balanced\nexperts.protect = 12:178, 17:44\n' >"$TMP/a1.conf"
run --etc "$ETC" --answers "$TMP/a1.conf" --section content || fail "--answers content"
grep -q '^content\.law  *= exclude' "$ETC/tierlm.conf" || fail "content.law not written"
grep -q '^content\.code  *= deep' "$ETC/tierlm.conf" || fail "content.code not canonical"
grep -q '^experts\.protect  *= 12:178, 17:44' "$ETC/tierlm.conf" || fail "experts.protect"
grep -q 'compile.profile ignored (section engine not selected)' "$TMP/out" || fail "engine answer should be ignored"
grep -q '^compile\.profile  *= exact' "$ETC/tierlm.conf" || fail "engine keys must be kept as they were"
grep -q 'compile.profile = exact: exclusions only change calibration' "$TMP/out" || fail "missing exact-profile note"
ls "$ETC"/tierlm.conf.bak.* >/dev/null 2>&1 || fail "tierlm.conf backup missing"
printf 'content.law = inherit\n' >"$TMP/a2.conf"
run --etc "$ETC" --answers "$TMP/a2.conf" --section content || fail "--answers inherit"
grep -q '^content\.law' "$ETC/tierlm.conf" && fail "content.law should be removed"
grep -q -- '- content.law (was exclude)' "$TMP/out" || fail "removal not reported"
ok

# 6. answers: engine section; invalid answers are rejected and nothing is written
printf 'cpu.threads = 3\nkv.dtype = f8_e4m3\n' >"$TMP/a3.conf"
run --etc "$ETC" --answers "$TMP/a3.conf" --section engine --probe-seconds 1 || fail "--answers engine"
grep -q '^cpu\.threads  *= 3  *# from the answers file' "$ETC/tierlm.conf" || fail "cpu.threads answer"
grep -q '^kv\.dtype  *= f8_e4m3' "$ETC/tierlm.conf" || fail "kv.dtype answer"
cp "$ETC/tierlm.conf" "$TMP/conf.2"
printf 'kv.vram_frac = 0.5\nno.such.key = 1\ndecide.ask_threshold = 0.95\n' >"$TMP/bad.conf"
run --etc "$ETC" --answers "$TMP/bad.conf" --probe-seconds 1 && fail "bad answers accepted"
grep -q 'bad.conf:1: kv.vram_frac: 0.5 is out of range' "$TMP/out" || fail "range error not reported"
grep -q "bad.conf:2: unknown key 'no.such.key'" "$TMP/out" || fail "unknown key not reported"
grep -q 'must not exceed decide.auto_threshold' "$TMP/out" || fail "threshold order not checked"
cmp -s "$ETC/tierlm.conf" "$TMP/conf.2" || fail "bad answers changed tierlm.conf"
ok

# 7. precedence as --show reports it: conf.d over tierlm.conf, env over both
mkdir -p "$ETC/conf.d"
printf 'cpu.threads = 6\n' >"$ETC/conf.d/50-local.conf"
"$BIN" --etc "$ETC" --show >"$TMP/show" 2>"$TMP/out" || fail "--show with conf.d"
grep -q '^cpu\.threads  *= 6  *# conf\.d/50-local\.conf:1' "$TMP/show" || fail "conf.d origin"
TIERLM_ETC=$ETC TIERLM_CPU_THREADS=9 "$BIN" --show >"$TMP/show" 2>"$TMP/out" || fail "--show via TIERLM_ETC"
grep -q '^cpu\.threads  *= 9  *# env TIERLM_CPU_THREADS' "$TMP/show" || fail "env origin"
printf 'this is not a config line\n' >"$ETC/conf.d/60-broken.conf"
"$BIN" --etc "$ETC" --show >/dev/null 2>"$TMP/out" && fail "--show accepted a syntax error"
grep -q '60-broken.conf:1' "$TMP/out" || fail "syntax error location"
rm -f "$ETC/conf.d/60-broken.conf"
ok

# 8. --probe refreshes hardware.conf only; --probe with --auto is a usage error
cp "$ETC/tierlm.conf" "$TMP/conf.3"
run --etc "$ETC" --probe --probe-seconds 1 || fail "--probe"
cmp -s "$ETC/tierlm.conf" "$TMP/conf.3" || fail "--probe changed tierlm.conf"
grep -q 'hardware.conf' "$TMP/out" || fail "--probe output"
set +e; run --etc "$ETC" --probe --auto; rc=$?; set -e
[ "$rc" -eq 2 ] || fail "--probe --auto should exit 2 (got $rc)"
set +e; run --etc "$ETC" --no-such-flag; rc=$?; set -e
[ "$rc" -eq 2 ] || fail "unknown flag should exit 2 (got $rc)"
ok

# 9. --drive measures a directory and becomes io.drives; --apply-system writes only <etc>/system
mkdir -p "$TMP/drive0"
run --etc "$ETC" --auto --probe-seconds 1 --drive "$TMP/drive0" --apply-system || fail "--drive --apply-system"
grep -q "^hardware\.drive\.0\.path  *= $TMP/drive0" "$ETC/hardware.conf" || fail "drive path in hardware.conf"
grep -q '^hardware\.drive\.0\.buffered_gbs ' "$ETC/hardware.conf" || grep -q '^hardware\.drive\.0\.note ' "$ETC/hardware.conf" \
    || fail "drive result in hardware.conf"
grep -q "^io\.drives  *= $TMP/drive0" "$ETC/tierlm.conf" || fail "io.drives from --drive"
grep -q '^io\.direct  *= \(yes\|no\)  *# ' "$ETC/tierlm.conf" || fail "io.direct recommendation from the drive test"
for f in 90-tierlm-sysctl.conf 90-tierlm-limits.conf tuned-tierlm.conf; do
    [ -s "$ETC/system/$f" ] || fail "system snippet $f missing"
done
ls -A "$TMP/drive0" | grep -q . && fail "drive test left files behind"
grep -q 'install -m 0644 .*/etc/sysctl.d/90-tierlm.conf' "$TMP/out" || fail "install commands not printed"
ok

# 10. shipped files: built-in taxonomy == etc/content-taxonomy.conf; the example parses
#     strictly and documents every schema key
"$BIN" --print-default-taxonomy >"$TMP/tax" || fail "--print-default-taxonomy"
cmp -s "$TMP/tax" "$ROOT/etc/content-taxonomy.conf" || fail "built-in taxonomy differs from etc/content-taxonomy.conf"
mkdir -p "$TMP/ex"
cp "$ROOT/etc/tierlm.conf.example" "$TMP/ex/tierlm.conf"
"$BIN" --etc "$TMP/ex" --show --strict >"$TMP/show" 2>"$TMP/out" || fail "tierlm.conf.example does not load strictly"
"$BIN" --etc "$TMP/none" --show >"$TMP/keys" 2>"$TMP/out" || fail "--show of an empty etc dir"
for k in $(sed -n 's/^\([a-z][a-z0-9_.]*\) .*=.*/\1/p; s/^# \([a-z][a-z0-9_.]*\) is unset$/\1/p' "$TMP/keys"); do
    grep -q "^#\{0,1\} *$k  *=" "$ROOT/etc/tierlm.conf.example" || fail "tierlm.conf.example lacks $k"
done
ok

echo "setup.sh: $N groups passed"
