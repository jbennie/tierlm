# Working on tierlm (agents and contributors)

Read `docs/STATUS.md` first: it lists what works, the known gaps and the next milestone.
`docs/ARCHITECTURE.md` explains how the modules fit together.

## Hard rules

1. **No copied code.** Never paste or closely paraphrase code from another engine (llama.cpp,
   colibri, ds4, pulsar, monolite, slipstream, ktransformers, apus, flash-moe or any other).
   The workflow is:
   1. Read the reference.
   2. Write or extend a pseudocode fragment in `fragments/`, in your own words, with an
      `Origin:` tag.
   3. Score it in `fragments/CONSENSUS.md`.
   4. Implement from the fragment.

   Public format specifications (GGUF block layouts, safetensors, OCP MX/FP8) may be followed
   exactly.
2. **Keep output deterministic.** Tokens and logits must not depend on any of these:
   - cache size
   - prefetch
   - thread count
   - I/O backend or O_DIRECT
   - replica choice
   - kernel tier

   Integer SIMD kernels must equal the scalar kernel bit for bit. Never add `-ffast-math` or
   FMA contraction. Never sum floats across threads. Experts are combined in ascending top-k
   order. If a change needs to break this, stop and ask.
3. **Tests must pass before you commit:** `make test`, and `make asan` for anything in `src/lib/`.
   Add a test with every behaviour change: unit tests in `tests/test_<module>.c`, pipeline checks
   in `tests/e2e.sh`.
4. **Never touch the host's `/etc/tierlm`** from tests or tools. Use `TIERLM_ETC=<tmpdir>` or
   `--etc`.
5. **Use your own build tree** when several agents work at once: `make BUILD=build-<name> test`.
   `build*/` is git-ignored.
6. **Do not assign issues to Copilot or other bots** without the maintainer's explicit OK.

## Where things go

| Change | Files |
| --- | --- |
| New config key | `src/lib/tlm_cfg_schema.c` (type, default, range, help), `etc/tierlm.conf.example`, `tierlm-setup` recommendation if it is machine-dependent |
| New quant format | `tlm_quant.h` dtype, scalar reference first, then AVX2/AVX-512 behind the self-test; bit-exact test in `test_quant.c` |
| New packet type or manifest field | `tlm_packet.h` (bump `TLM_STORE_VERSION` if the layout changes), reader validation, `test_packet.c` |
| New model family | ingest mapping in `tlmc.c`, geometry in `tlm_model_geom`, forward pass in `tlm_model.c`; a synthetic fixture in `tlmc synth` and an e2e step |
| New compiler decision | a typed question through `tlm_decide` (never an ad-hoc threshold), a rule, a safe default, audit-log fields |
| New research or design | `research/` (reports), `docs/design/` (proposals), `research/SOURCES.md` for every URL |

## Commands

```sh
make -j && make -j test          # build + full suite (~40 s)
make -j asan                     # sanitizers
scripts/demo-model.sh            # synthetic model in scratch/
TIERLM_ETC=scratch/etc build/bin/tierlm run scratch/demo.tlmm --prompt Hi -n 32 --stats
```

## Style

- C11 with `_GNU_SOURCE`.
- Return `0` or `TLM_E*` and set the message with `tlm_seterr`.
- No allocation in the per-token path.
- Each module header cites the `P-xx` ids it implements and ends with a notes block that records
  contract details.
- Four spaces, about 100 columns.
- Prose in docs: plain English, short sentences, numbers with units, and say which numbers are
  measured and which are estimated.
