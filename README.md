# tierlm

An experimental CPU-first inference engine, written in C, for very large Mixture-of-Experts models.
The dense weights live in RAM. The routed experts live on NVMe and are streamed into a RAM cache
as the router asks for them.

tierlm has two stages:

1. **`tlmc`, the model compiler.** It runs offline and does every expensive decision once. It
   takes a checkpoint, quantises it, calibrates routing heat and expert saliency, and can prune or
   re-tier experts through an embedded decision engine (gated on quality). It then cuts the model
   into CRC-checked 4 KiB-aligned packets and places them across dedicated drives.
2. **`tierlm`, the runtime.** It opens the compiled manifest, loads the dense packets, pins the
   hot experts the compiler picked, and streams the rest through a frequency-aware slot cache with
   router-lookahead prefetch. It does no layout work at start-up. Cache size, prefetch, thread
   count, I/O backend and SIMD tier change speed only. The output is bit-identical across all of them.

A third tool, **`tierlm-setup`**, probes the machine and writes `key = value` preferences to
`/etc/tierlm/`. It includes a section for choosing which experts and content types to keep, demote or
exclude.

The library was distilled clean-room from eight open MoE engines. We read them, wrote the designs
down as pseudocode fragments with their origin tagged, scored which patterns several projects
converge on, and built only from those. No reference code is copied (see `NOTICE` and
`fragments/`).

> Status: early. The full pipeline works end to end on synthetic Qwen3-MoE-like models and is
> covered by unit, end-to-end and sanitizer tests. It does not yet run a real model: there is no
> real tokenizer, and MTP speculation, GPU KV and the cluster mode are not built yet. See
> [`docs/STATUS.md`](docs/STATUS.md).

## Build

Requirements: a C11 compiler (GCC ≥ 12 or Clang ≥ 15), GNU make and Linux. There are no other
dependencies. liburing is optional.

```sh
make                      # portable build: scalar + AVX2 + AVX-512 kernels, picked at run time
make PROFILE=wrx90        # Threadripper PRO 9000 (Zen 5) tuned: -O3 -march=znver5, io_uring on
make URING=1              # add the io_uring backend (dnf install liburing-devel)
make test                 # unit tests + tests/setup.sh + tests/e2e.sh
make asan                 # same suite under AddressSanitizer + UBSan (build-asan/)
make tsan                 # ThreadSanitizer (build-tsan/)
```

The binaries land in `build/bin/`: `tlmc`, `tierlm` and `tierlm-setup`. The static library is
`build/libtlm.a`.

On RHEL 10, io_uring is off by default. To use `io.backend = io_uring`, set
`kernel.io_uring_disabled = 0`, or set it to `1` together with `kernel.io_uring_group`
(`tierlm-setup --apply-system` writes the snippet for you).

## Quick start (synthetic model)

```sh
make -j
T=$(mktemp -d)

# 1. a tiny random Qwen3-MoE-like checkpoint (config.json + model.safetensors)
build/bin/tlmc synth --out $T/src --layers 4 --experts 32 --top-k 4 --shared 1

# 2. compile it across three "drives" (any directories work for testing)
build/bin/tlmc compile $T/src --out $T/model.tlmm \
    --drives $T/nvme0,$T/nvme1,$T/nvme2 --profile exact      # add --report for the full JSON

# 3. run it with a deliberately tiny expert cache; the tokens match any cache size
build/bin/tierlm run $T/model.tlmm --prompt "Hello" -n 32 --cache 1 --stats --print-tokens
build/bin/tierlm bench $T/model.tlmm --prompt-len 64 -n 64

# 4. inspect what the compiler decided
build/bin/tlmc report $T/model.tlmm
build/bin/tlmc atlas  $T/model.tlmm --sort heat --top 20
```

`--profile balanced` and `--profile aggressive` add REAP-based pruning and heat-based Q6_K→Q5_K
demotion. Both are accepted only if a held-out top-1 agreement gate passes (97% / 93%, with a
5-point back-off). Every decision is logged as JSONL in the work directory next to the manifest
(`model.work/decisions.jsonl` for `model.tlmm`). That directory also lets a recompile resume.

## Configure a host

```sh
sudo build/bin/tierlm-setup                 # interactive: probe, recommend, explain, write
sudo build/bin/tierlm-setup --auto --drive /srv/tlm/nvme0 --drive /srv/tlm/nvme1 \
                            --drive /srv/tlm/nvme2 --drive /srv/tlm/nvme3
build/bin/tierlm-setup --show               # every effective value and where it came from
build/bin/tierlm config check
```

The configuration is layered, and later layers win:

1. built-in defaults
2. `/etc/tierlm/hardware.conf`
3. `/etc/tierlm/tierlm.conf`
4. `/etc/tierlm/conf.d/*.conf`
5. `TIERLM_<KEY>` environment variables
6. command-line flags

`etc/tierlm.conf.example` documents every key. `etc/content-taxonomy.conf` holds the content types
that the Experts-and-content section offers. Set `TIERLM_ETC=DIR` (or pass `--etc DIR`) to work
without touching `/etc`.

## Repository layout

| Path | What it holds |
| --- | --- |
| `src/lib/` | `libtlm`: platform, config, JSON, safetensors, quant kernels (scalar/AVX2/AVX-512), packet store, async I/O, expert cache, worker pool, MoE decoder, decision engine |
| `src/bin/` | `tlmc` (compiler), `tierlm` (runtime), `tierlm-setup` (host setup) |
| `tests/` | unit tests (`test_*.c`), `setup.sh`, `e2e.sh`, the tiny `tlm_test.h` harness |
| `etc/` | example config and the content taxonomy |
| `profiles/` | per-machine make profiles (`wrx90.mk`) |
| `research/` | the research reports this project started from, plus `SOURCES.md` |
| `fragments/` | clean-room pseudocode fragments with origin tags, and `CONSENSUS.md` scoring |
| `docs/` | how fragments became modules, current status, architecture, the cluster and swarm designs |

## Where to read next

- [`docs/STATUS.md`](docs/STATUS.md): what works, known gaps and the next milestones
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): module layering, data flow and the invariants
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md): building, testing, debugging, VS Code
- [`research/README.md`](research/README.md): the research reports, in reading order
- [`fragments/README.md`](fragments/README.md): how the clean-room library was derived
- [`AGENTS.md`](AGENTS.md): working rules for coding agents and contributors

## Licence

Apache-2.0 (see `LICENSE` and `NOTICE`). This is a research project for experimentation. Some of
the projects studied are licensed for non-commercial use only. `NOTICE` explains how their ideas
were handled.
