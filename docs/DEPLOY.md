# Deploying and running tierlm

A plain-English guide to standing up a clean machine for **M2 testing** (measuring tierlm on
the two M2 hosts, `drogo` and `doris`) and to using the software day to day. This is the
**CPU-first path — no GPU is needed.** (GPU KV is M4, a separate `dlopen` ROCm backend; it is
not part of this guide.)

The repo builds to three binaries, all under `build/bin/`:

| Binary | What it is | When it runs |
| --- | --- | --- |
| `tlmc` | the model compiler | once, offline, to turn a checkpoint into a store |
| `tierlm` | the runtime | every time you run or benchmark a model |
| `tierlm-setup` | the machine prober | once per machine, to write `/etc/tierlm/` |

## 1. Hardware

### Minimum (to run tierlm at all)

| Component | Minimum |
| --- | --- |
| CPU | 6 cores, AVX2 (x86-64) |
| RAM | 32 GB |
| Disk | one PCIe NVMe, ≥ 3 GB/s sequential read (consumer Gen3 class is fine — the M2 gate is a ratio against that drive's own `fio` baseline) |

### The M2 test hosts (drogo, doris)

Two modest AVX2 hosts — doris on Ubuntu 26.04, drogo on RHEL 10.3. The GPUs are not part of
M2 — the CPU path is what is measured (drogo's 2080 Ti is removed to make room for the 4-way
NVMe adapter).

| | drogo | doris |
| --- | --- | --- |
| CPU | AMD Ryzen 5 4500G — 6C/12T Zen 2, AVX2 (no VNNI, no AVX-512) | Intel Core i7-6700HQ — 4C/8T Skylake, AVX2 (no VNNI, no AVX-512) |
| RAM | 32 GB DDR4 UDIMM → cache sweep caps at ~16 G | 24 GB DDR4 → sweep to 12 G |
| Disk | **4× 1 TB NVMe** on a 4-way PCIe adapter | 2× 256 GB NVMe (Gen3) |
| GPU | (removed) | WX3100 8 GB — unused in M2 |

**Both hosts are multi-drive**: drogo (4 drives) exercises drive sharding, parallel reads and
replica queue-wins at full width; doris (2 drives) covers the same path at two-way. Together
they give a CPU-generation and RAM comparison (Zen 2 dual-channel, 32 G vs Skylake, 24 G
cache).

drogo caveats:

- AM4 is **PCIe 3.0 only**, and the 4-way adapter needs the slot to bifurcate x4/x4/x4/x4 in the
  BIOS. Some AM4 boards only bifurcate x8/x8 or not at all — check the board manual/BIOS before
  relying on four drives. With 2× x8 everything below still works with a two-drive list.
- Expect ~3 GB/s sequential read per drive (consumer Gen3), not workstation Gen4/5 numbers.

The original M2 plan targeted a WRX90 workstation (24 Zen 5 cores, 128 GB DDR5, 4 NVMe); that
machine is not available, so M2 runs on these two boxes. The cache sweep is RAM-capped per host
and every number is consumer-class, not workstation-class.

Two things set the speed, in this order: **RAM for the expert cache is the biggest lever, disk
bandwidth is second.** Total parameter count barely matters — a MoE touches only a few percent of
its weights per token. Plan the cache and the drives around that.

## 2. Prerequisites — RHEL 10.3 / Ubuntu 26.04, no GPU

On a clean install the only build dependencies are a C compiler and `make` (plus optional
`liburing`). There is **no** CMake, **no** Python at build or run time, and **no** third-party
C library. Install:

```sh
# RHEL 10.3 (drogo)
sudo dnf install gcc make git
sudo dnf install liburing-devel        # optional: enables the io_uring backend (make URING=1)

# Ubuntu 26.04 (doris)
sudo apt install -y gcc make git
sudo apt install -y liburing-dev        # optional: enables the io_uring backend (make URING=1)
```

- `gcc` on RHEL 10 is GCC 14; the portable build needs nothing newer (no tuned `-march`).
- No GPU driver, no ROCm, no CUDA, no Python. (Python is needed *only* for the optional HF
  oracle gate in `tools/`, not for compiling or running.)

### io_uring (RHEL 10)

io_uring is a tech preview and **off by default** in RHEL 10 (RHEL-65347; verify on your exact
release). tierlm runs fine without it — the pread pool is the default backend and io_uring is
only *measured* in M2, not required to run. To enable it:

```sh
sudo grubby --update-kernel=ALL --args="kernel.io_uring_disabled=0"
sudo reboot
# verify: cat /proc/sys/kernel/io_uring_disabled    # want 0
```

On Ubuntu (doris) io_uring is on by default — verify with the same `cat
/proc/sys/kernel/io_uring_disabled` (want `0`) and skip the grubby step.

## 3. Clone and build

```sh
git clone https://github.com/jbennie/tierlm.git
cd tierlm
make -j"$(nproc)"        # portable build: scalar + AVX2 + AVX-512, dispatched at run time
make test                # unit + end-to-end tests; sanity-checks the build on this box
```

Both M2 hosts use the portable build as-is — the **identical build on both boxes** keeps the
numbers comparable. With `liburing` installed, add `URING=1` so the io_uring A/B in §6 is a real
A/B (without it the harness records a pread fallback for the io_uring runs):

```sh
make URING=1 -j"$(nproc)"
make URING=1 test
```

Useful extras:

```sh
make asan                # whole test suite under AddressSanitizer + UBSan
```

`make test` runs every `tests/test_*.c`, then `tests/setup.sh` (tierlm-setup), then
`tests/e2e.sh` (the full synth → compile → run pipeline on a synthetic model). A green `make
test` means the clone is good.

## 4. Recompile a model into a store

A **store** is the compiled model: one manifest (`.tlmm`) plus the packet drives it points at.
You compile once; after that the runtime opens the manifest with **no layout work at start-up**.
The store is large and lives on disk (it is git-ignored), so keep it on the NVMe drives, not in
the repo.

The M2 model is **Qwen3-30B-A3B** at Q6_K experts / Q8_0 dense (~28 GB store). (The original
plan was Qwen3-235B-A22B at Q6_K, ~200 GB store; the 30B compiles on-box in a fraction of the
time and exercises the same I/O path. 235B stays possible later — drogo's 4 TB holds its ~470 GB
source.) Download the HF checkpoint first (57 GB), then compile per host:

```sh
# drogo — four drives, in manifest order
./build/bin/tlmc compile /path/to/Qwen3-30B-A3B \
    --out  ~/stores/q30b.tlmm \
    --drives /mnt/nvme0/q30b,/mnt/nvme1/q30b,/mnt/nvme2/q30b,/mnt/nvme3/q30b \
    --profile exact --experts q6_k --dense q8_0 --replicas auto

# doris — two drives
./build/bin/tlmc compile /path/to/Qwen3-30B-A3B \
    --out  ~/stores/q30b.tlmm \
    --drives /mnt/nvme0/q30b,/mnt/nvme1/q30b \
    --profile exact --experts q6_k --dense q8_0 --replicas auto
```

- `--drives` is a comma list of **one directory per NVMe drive**; the compiler places packets
  across them by expected read heat. Each dir gets a `*.tlmp` packet file.
- `--experts` is the routing tier: `q6_k` (default), `q5_k` (smaller), `q8_0` (highest). If a
  model's output is sensitive to expert precision, use `q8_0` — this is exactly what fixed the
  Qwen3-30B-A3B pos0 oracle mismatch (see `STATUS.md`).
- `--dense q8_0` keeps attention/shared/embedding dense (it always does; the flag sets the tier).
- `--profile exact` keeps every expert (no pruning). `balanced`/`aggressive` prune cold experts,
  gated on a quality eval — opt-in, and not what you want for a clean M2 bandwidth measurement.
- `--replicas auto` copies warm experts to a second drive on drogo (reads then split between the
  two copies — shorter queue wins); with one drive (doris) it is a no-op.

The compile prints a summary (geometry, precision, placement, a 512-token self-eval, and per-pass
timings). `tlmc report ~/stores/q30b.tlmm` reprints it later; the full decision log is in
the `<store>.work/decisions.jsonl`.

**Smoke-test the pipeline without a real checkpoint:**

```sh
scripts/demo-model.sh                 # builds a 4-layer synthetic MoE into scratch/
TIERLM_ETC=scratch/etc ./build/bin/tierlm run scratch/demo.tlmm --prompt Hello -n 32 --stats
```

`scratch/` is git-ignored. Use this to confirm the whole toolchain works before compiling the
real model (~20–45 min for 30B on these hosts; run it in the background).

## 5. Using the software (plain English)

### First, on the machine: `tierlm-setup`

`tierlm-setup` probes the box (CPU, RAM, DRAM bandwidth, drives) and writes recommendations to
`/etc/tierlm/`. It is the "doctor" step — **run it before any measurement.** As root:

```sh
sudo ./build/bin/tierlm-setup                       # interactive: it asks, you answer
sudo ./build/bin/tierlm-setup --auto                # accept every recommendation
sudo ./build/bin/tierlm-setup --drive /mnt/nvme0 --drive /mnt/nvme1 ...   # measure specific drives
```

It sets CPU thread count (1 per physical core, SMT siblings left for I/O), the expert-cache size
(RAM − reserve − dense weights), the pin fraction, and the I/O backend. Inspect what it would do
without writing with `--show`.

### Run the model: `tierlm run`

```sh
./build/bin/tierlm run ~/stores/q30b.tlmm --tokenizer ~/models/Qwen3-30B-A3B --prompt "Hello" -n 64 --stats
```

- `--tokenizer DIR` — the checkpoint directory (its `tokenizer.json`); **required for text** —
  without it the prompt is read as raw bytes and the output shows `<id>` placeholders.
- `--prompt TEXT` — a text prompt (BPE-encoded). Or `--tokens 1,2,3` for raw ids from an
  external tokenizer.
- `-n 64` — how many tokens to generate after the prompt.
- `--stats` — per-drive reads, store hits/misses/evictions, prefetch issued/used/dropped, wait
  time. **This is where the M2 bandwidth numbers come from.**
- `--print-tokens` — print the raw token ids instead of text (for bit-exactness checks).

### The knobs — what they do

Every flag below changes **speed only, never the output**. That is the core guarantee of the
engine, and it is what makes the M2 sweeps meaningful.
The one exception is sampling: `--temp`, `--top-p`, `--min-p` and `--seed` change the output.
The default is greedy, which loops on long generations (observed: paragraph repeated verbatim
past ~256 tokens) — use `--temp 0.8` for longer outputs.

| Flag | Meaning |
| --- | --- |
| `--cache SIZE` | expert-cache RAM (e.g. `1G`, `32G`). Bigger = fewer disk misses. |
| `--threads N` | compute threads. The setup default is 1 per physical core. |
| `--io pread\|io_uring` | I/O backend. |
| `--direct yes\|no` | O_DIRECT on/off (bypass the OS page cache). |
| `--prefetch router\|off` | router-lookahead prefetch of the next layer's experts. |
| `--lookahead-k N` | experts the router lookahead hints per layer (default: the model's top_k). Widen it to prefetch past the hard top-k; the runtime caps it at 4x top_k. |
| `--interest F` | "this question needs" (0..1): how boldly to widen prefetch toward the interest; 0 = legacy fixed breadth. The instinct vector (learned) and model heat (calibrated) temper it. |
| `--pin-frac F` | share of the cache held for the pinned hot set. |
| `--kernel scalar\|avx2\|avx512` | force a SIMD tier (default: auto-picked at run time). |
| `--warm FILE` | warm-start sidecar: bulk-load the hot set from a previous run. |

**Prove the guarantee for yourself** — these two runs must be byte-identical:

```sh
./build/bin/tierlm run STORE --print-tokens --cache 1G  --kernel scalar  > /tmp/a
./build/bin/tierlm run STORE --print-tokens --cache 32G --kernel avx512 --threads 8 > /tmp/b
diff /tmp/a /tmp/b        # empty = identical output
```

### Benchmark: `tierlm bench`

```sh
./build/bin/tierlm bench ~/stores/q30b.tlmm --prompt-len 512 -n 128 --stats
```

This is the M2 measurement driver. It runs a fixed prompt length and a fixed number of decode
tokens, and `--stats` gives the hit rate and per-drive bandwidth. Every stats block (text and
`--stats-json`) starts with a **run stamp** — `host, cpu, cores, RAM, OS, kernel, build (git
revision), store name/profile/geometry/source-sha` — so each metric file is self-describing:
you can always tell which box, with what hardware and which build, ran which store.

### Config: `tierlm config`

```sh
./build/bin/tierlm config show     # the effective config, every key with its origin
./build/bin/tierlm config check    # validate the config
```

All keys are documented in `etc/tierlm.conf.example`.

## 6. M2 measurement plan

The **M2 gate is ≥ 85% of `fio` sequential-read bandwidth on expert misses**, with decode tok/s
recorded at the hit rate. The gate is answered by the missiest run (smallest cache): per-drive
aggregate delivery during decode (bytes / decode wall time) against that drive's fio baseline.
The whole sweep is one command — the harness runs the probe, the per-drive `fio` baseline, the
cache-size sweep, and both A/B tests, then writes the gate:

```sh
# drogo — four drives; 32 GB RAM, so the sweep stops at 16G
CACHE_LIST="1G 4G 8G 16G" AB_CACHE=8G \
    tools/m2_eval.sh ~/stores/q30b.tlmm \
    /mnt/nvme0/q30b,/mnt/nvme1/q30b,/mnt/nvme2/q30b,/mnt/nvme3/q30b m2out-drogo

# doris — two drives; 24 GB RAM, sweep to 12G
CACHE_LIST="1G 4G 8G 12G" AB_CACHE=8G \
    tools/m2_eval.sh ~/stores/q30b.tlmm /mnt/nvme0/q30b,/mnt/nvme1/q30b m2out-doris
```

The drive list must be in the store's manifest order (the gate indexes the fio baseline to the
bench's drive index). Outputs land in the third argument:

- `summary.json` — one object per run: the run-stamp `meta` (host, CPU, RAM, OS, build, store
  identity), raw counters, per-drive wall GiB/s (bytes delivered during decode) and stream
  GiB/s (per-read latency figure), the fio baseline, and `gate` = the minimum over drives of
  (wall GiB/s / fio GiB/s). **`gate >= 0.85` passes**; the missiest run answers it.
- `table.md` — the sweep as a markdown table (tok/s, hit rate, per-drive bandwidth), headed by
  the host/CPU/build line so the file says where it came from.
- `fio.txt`, `probe.txt`, `report.txt`, `bench/*.json` — the raw inputs, for re-deriving anything.

Environment knobs (all optional): `CACHE_LIST` (default `1G 4G 8G 16G 32G 64G`), `AB_CACHE`
(default `16G`), `PLEN`/`GEN` (512/128), `FIO_RUNTIME` (seconds per drive, default 30; `0`
skips fio and leaves `gate` null), `NO_PROBE=1`. The harness needs `jq`; `fio` is optional but
required for the gate. If io_uring is not available in the build, the A/B run records the
backend it actually used, so a pread fallback is visible in the JSON.

To do it by hand instead, the steps are: baseline each drive with `fio` at the expert-read size
(4–8 MB, deep queue); sweep `tierlm bench --cache 1G`, `4G`, `8G`, … and record tok/s + hit rate;
A/B `--io pread` vs `--io io_uring`; A/B `--direct yes` vs `--direct no`; compare the miss
bandwidth from `--stats-json` against the fio baseline.

Write the measured numbers back into `docs/STATUS.md` (the M2 milestone row), and keep the
`m2out-*/summary.json` files as the machine-readable record. The research notes hold pre-M2
engineering estimates.

## Notes

- **No GPU.** Everything here is the CPU path. Do not install ROCm/CUDA for M2.
- **The store.** Qwen3-30B-A3B at Q6_K experts / Q8_0 dense is ~28 GB (~7 GB per drive on
  drogo, ~14 GB per drive on doris). Put it on the NVMe, never in the repo.
- **One drive per shard.** `--drives` expects independent devices; no RAID layer. `tierlm-setup
  --drive` measures each so the placement is honest.
- **Recompile is slow** (ingest + quantise + calibrate + place + validate) — ~20 min for the 30B
  model on the 12-core dev box; expect 30–60 min on the 6–8-core M2 hosts. Run it in the
  background.
