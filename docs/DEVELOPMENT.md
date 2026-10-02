# Development

## Toolchain

- C11 compiler: GCC 12+ or Clang 15+. GCC 14 is the RHEL 10 default and knows `-march=znver5`.
- GNU make, POSIX sh, Linux (uses `pread`, `O_DIRECT`, `sched_setaffinity`, sysfs topology).
- Optional: `liburing-devel` for `make URING=1`; `gdb` for the VS Code launch configs.
- Nothing else. No CMake, no Python at build time, no third-party C libraries.

On RHEL 10:

```sh
sudo dnf install gcc make gdb liburing-devel
```

## Build trees

`BUILD=dir` puts everything (objects, `libtlm.a`, binaries, test binaries) under `dir`. Different
configurations therefore never mix:

| Command | Tree | Use |
| --- | --- | --- |
| `make` / `make test` | `build/` | default, `-O2 -g` |
| `make BUILD=build-debug OPTFLAGS=-O0` | `build-debug/` | stepping in gdb (the VS Code launch configs use it) |
| `make PROFILE=wrx90 BUILD=build-wrx90` | `build-wrx90/` | `-O3 -march=znver5`, io_uring |
| `make asan` | `build-asan/` | AddressSanitizer + UBSan, whole suite |
| `make tsan` | `build-tsan/` | ThreadSanitizer, whole suite |

Kernel files get their own ISA flags (`tlm_kern_avx2.c`: `-mavx2 -mfma -mf16c`;
`tlm_kern_avx512.c`: AVX-512 F/BW/VL/VNNI/DQ), so the default binary runs on any x86-64. The
tier is picked at run time, and you can force it with `--kernel scalar|avx2|avx512` or
`TIERLM_KERNEL`.

Never add `-ffast-math`, `-ffp-contract=fast` or `-funsafe-math-optimizations`. They break the
bit-exactness tests (see the determinism contract in `ARCHITECTURE.md`).

## Tests

```sh
make test                   # every tests/test_*.c, then tests/setup.sh, then tests/e2e.sh
make asan                   # same under ASan/UBSan; run before every push that touches src/lib
make tsan                   # threads: pool, io, store
make build/tests/test_store && build/tests/test_store     # one unit test
make bins && sh tests/e2e.sh                               # end to end only
```

- **Unit tests** use `tests/tlm_test.h` (`CHECK`, `CHECK_EQ_INT`, `CHECK_NEAR`, `RUN`,
  `TEST_MAIN_END`, xorshift RNG). A new file `tests/test_<x>.c` is picked up by the Makefile
  automatically.
- **`tests/setup.sh`** drives `tierlm-setup` with `--etc` in a temp dir: answers files, precedence,
  content sections, `--show`, strict mode.
- **`tests/e2e.sh`** runs the whole pipeline on a synthetic model:
  1. synth
  2. compile exact on 3 dirs
  3. tiny vs large cache must give identical tokens
  4. scalar + io_uring + O_DIRECT must match
  5. resume
  6. balanced prune
  7. report and atlas
  8. aggressive with content exclusion
  9. config show/check
  10. a missing drive is refused

Every test sets `TIERLM_ETC` to a temp dir, so `/etc/tierlm` on the host is never read. Keep it
that way.

## A model to play with

```sh
scripts/demo-model.sh            # or: scripts/demo-model.sh balanced
TIERLM_ETC=scratch/etc build/bin/tierlm run scratch/demo.tlmm --prompt Hello -n 32 --stats
```

`scratch/` is git-ignored. `tlmc synth` options control the geometry (`--layers`, `--experts`,
`--top-k`, `--d-model`, `--router sigmoid`, `--qk-norm no`, and so on).

## VS Code

Open the repository folder. The recommended extensions are C/C++ (`ms-vscode.cpptools`), Makefile
Tools, ShellCheck, markdownlint and Claude Code. Then:

- **Build:** `Ctrl+Shift+B` runs `make -j`. The task list also has wrx90, debug, asan, tsan and
  clean builds.
- **Test:** Tasks → `test`, or `test: one unit test` (it asks you to pick the test).
- **Debug:** Run and Debug offers the following configs:
  - `tierlm run (demo model)`
  - `tierlm run, tiny cache, scalar kernels`
  - `tlmc compile (demo, balanced)`
  - `tlmc report`
  - `tierlm-setup --dry-run`
  - `unit test (pick)`

  Each builds `build-debug/` first. Run `demo model (scratch/)` once before the demo configs.
- **IntelliSense:** `.vscode/c_cpp_properties.json` sets `-I src/lib`, `_GNU_SOURCE` and the
  AVX-512 flags, so the intrinsics in the kernel files resolve. clangd users get the same from
  `compile_flags.txt`.

For a remote host (the WRX90), use Remote-SSH: open the folder on the host and install the
extensions there. Build with `make PROFILE=wrx90 BUILD=build-wrx90`.

## Debugging tips

- `TIERLM_LOG_LEVEL=info` (or `debug`), or the same `log.level` in the config, logs I/O
  fallbacks, pinning and backend choice. This applies to `tierlm`.
- `tierlm run --stats` prints per-drive reads and busy time, store hits, misses, evictions,
  prefetch issued/used/dropped, and wait time.
- To prove a change kept the output identical, compare
  `tierlm run ... --print-tokens --cache 1 --kernel scalar` with
  `--cache 1G --kernel avx512 --threads 8`. The outputs must be byte-identical.
- `tlmc compile` prints its decision summary. `scratch/demo.work/decisions.jsonl` has every
  question, its features, the confidence and the outcome.
- When a kernel tier is first used, the library checks each SIMD kernel bit for bit against the
  scalar reference on fixed data. If a kernel fails, the library logs a warning that names the
  tier and the format, and falls back to the slower tier for that format. `test_quant` asserts
  that there were no such failures.

## Coding conventions

- C11 plus `_GNU_SOURCE`. One header per module in `src/lib/`. The public API is the header, and
  each header ends with a `notes` block that records contract details the implementation
  settled.
- Functions return `0` / `TLM_E*`. Set the message with `tlm_seterr()`, and read it with
  `tlm_errmsg()`.
- Allocate once at open and never inside the token loop. The slot arena, the KV and scratch
  buffers are all sized up front.
- Every module header names the `P-xx` pattern ids it implements. When you add behaviour, cite
  the pattern or add a fragment (see `CONTRIBUTING.md`).
- Four spaces, no tabs (except in the Makefile). Lines up to about 100 columns.
