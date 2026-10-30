# Deploying and running tierlm

A plain-English guide to standing up a clean machine for **M2 testing** (measuring tierlm on
the WRX90) and to using the software day to day. This is the **CPU-first path — no GPU is
needed.** (GPU KV is M4, a separate `dlopen` ROCm backend; it is not part of this guide.)

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
| CPU | 8 cores, AVX2 (x86-64) |
| RAM | 32 GB |
| Disk | one PCIe Gen4 NVMe, ≥ 5 GB/s sequential read |

### The M2 test box (WRX90)

| Component | Spec |
| --- | --- |
| CPU | AMD Threadripper PRO 9965WX — 24 Zen 5 cores, full-width AVX-512 (VNNI/BF16) |
| RAM | 128 GB DDR5-6400, 8-channel (real bandwidth ~200–250 GB/s, not the spec-sheet 409.6) |
| Disk | **4 independent NVMe drives**, separate from the OS drive, read in parallel |
| GPU | none |

Two things set the speed, in this order: **RAM for the expert cache is the biggest lever, disk
bandwidth is second.** Total parameter count barely matters — a MoE touches only a few percent of
its weights per token. Plan the cache and the drives around that.

## 2. Prerequisites — RHEL 10.3, no GPU

On a clean RHEL 10.3 install the only build dependencies are a C compiler and `make` (plus
optional `liburing`). There is **no** CMake, **no** Python at build or run time, and **no**
third-party C library. Install:

```sh
sudo dnf install gcc make git
sudo dnf install liburing-devel        # optional: enables the io_uring backend (make URING=1)
```

- `gcc` on RHEL 10 is GCC 14, which knows `-march=znver5`. (Older compilers fall back to
  `znver4` — the same AVX-512 subset tierlm uses — so the build still works.)
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

## 3. Clone and build

```sh
git clone https://github.com/jbennie/tierlm.git
cd tierlm
make -j"$(nproc)"        # portable build: scalar + AVX2 + AVX-512, dispatched at run time
make test                # unit + end-to-end tests; sanity-checks the build on this box
```

On the WRX90, build the tuned profile instead (or in addition):

```sh
make PROFILE=wrx90 -j"$(nproc)"   # -O3 -march=znver5, io_uring on by default
make PROFILE=wrx90 test           # test the tuned build
```

Useful extras:

```sh
make asan                # whole test suite under AddressSanitizer + UBSan
make URING=1             # force the io_uring backend into a portable build
```

`make test` runs every `tests/test_*.c`, then `tests/setup.sh` (tierlm-setup), then
`tests/e2e.sh` (the full synth → compile → run pipeline on a synthetic model). A green `make
test` means the clone is good.

## 4. Recompile a model into a store

A **store** is the compiled model: one manifest (`.tlmm`) plus the packet drives it points at.
You compile once; after that the runtime opens the manifest with **no layout work at start-up**.
The store is large and lives on disk (it is git-ignored), so keep it on the NVMe drives, not in
the repo.

The M2 model is Qwen3-235B-A22B at Q6_K. Download the HF checkpoint first, then:

```sh
./build/bin/tlmc compile /path/to/Qwen3-235B-A22B \
    --out  ~/stores/qwen3-235b.tlmm \
    --drives /mnt/nvme0/q325,/mnt/nvme1/q325,/mnt/nvme2/q325,/mnt/nvme3/q325 \
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
- `--replicas auto` copies warm experts to a second drive so a single-drive miss still hits.

The compile prints a summary (geometry, precision, placement, a 512-token self-eval, and per-pass
timings). `tlmc report ~/stores/qwen3-235b.tlmm` reprints it later; the full decision log is in
the `<store>.work/decisions.jsonl`.

**Smoke-test the pipeline without a real checkpoint:**

```sh
scripts/demo-model.sh                 # builds a 4-layer synthetic MoE into scratch/
TIERLM_ETC=scratch/etc ./build/bin/tierlm run scratch/demo.tlmm --prompt Hello -n 32 --stats
```

`scratch/` is git-ignored. Use this to confirm the whole toolchain works before spending an hour
compiling the real 235B model.

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
./build/bin/tierlm run ~/stores/qwen3-235b.tlmm --prompt "Hello" -n 64 --stats
```

- `--prompt TEXT` — a text prompt (uses the built-in BPE tokenizer). Or `--tokens 1,2,3` for raw
  ids from an external tokenizer.
- `-n 64` — how many tokens to generate after the prompt.
- `--stats` — per-drive reads, store hits/misses/evictions, prefetch issued/used/dropped, wait
  time. **This is where the M2 bandwidth numbers come from.**
- `--print-tokens` — print the raw token ids instead of text (for bit-exactness checks).

### The knobs — what they do

Every flag below changes **speed only, never the output**. That is the core guarantee of the
engine, and it is what makes the M2 sweeps meaningful.

| Flag | Meaning |
| --- | --- |
| `--cache SIZE` | expert-cache RAM (e.g. `1G`, `32G`). Bigger = fewer disk misses. |
| `--threads N` | compute threads. The setup default is 1 per physical core. |
| `--io pread\|io_uring` | I/O backend. |
| `--direct yes\|no` | O_DIRECT on/off (bypass the OS page cache). |
| `--prefetch router\|off` | router-lookahead prefetch of the next layer's experts. |
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
./build/bin/tierlm bench ~/stores/qwen3-235b.tlmm --prompt-len 512 -n 128 --stats
```

This is the M2 measurement driver. It runs a fixed prompt length and a fixed number of decode
tokens, and `--stats` gives the hit rate and per-drive bandwidth.

### Config: `tierlm config`

```sh
./build/bin/tierlm config show     # the effective config, every key with its origin
./build/bin/tierlm config check    # validate the config
```

All keys are documented in `etc/tierlm.conf.example`.

## 6. M2 measurement plan

The **M2 gate is ≥ 85% of `fio` sequential-read bandwidth on expert misses**, with decode tok/s
recorded at the hit rate. The steps:

1. **Baseline the drives with `fio`** at the expert-read size (4–8 MB, deep queue):
   ```sh
   fio --name=read --rw=read --bs=4m --direct=1 --iodepth=32 --numjobs=1 \
       --filename=/mnt/nvme0/fiofile --size=4G --runtime=60
   ```
   Record per-drive and aggregate sequential-read bandwidth.
2. **Sweep the expert cache** with `tierlm bench --cache 1G`, `4G`, `8G`, … and record tok/s +
   hit rate at each point. This is the cache-size curve.
3. **A/B the I/O backend:** `--io pread` vs `--io io_uring`.
4. **A/B O_DIRECT:** `--direct yes` vs `--direct no`.
5. **Compare** the miss bandwidth from `--stats` against the `fio` baseline. The gate is ≥ 85%.

Write the measured numbers back into `research/02-wrx90-build-profile.md` — it currently holds
engineering estimates that M2 replaces with real measurements.

## Notes

- **No GPU.** Everything here is the CPU path. Do not install ROCm/CUDA for M2.
- **The store is big.** Qwen3-235B-A22B at Q6_K is on the order of 200 GB across the drives
  (larger still with `--replicas`). Put it on the NVMe drives, never in the repo.
- **One drive per shard.** `--drives` expects independent devices; no RAID layer. `tierlm-setup
  --drive` measures each so the placement is honest.
- **Recompile is slow** (ingest + quantise + calibrate + place + validate) — for a 30B-class
  model it ran ~15–30 min on the dev box; a 235B model takes longer. Run it in the background.
