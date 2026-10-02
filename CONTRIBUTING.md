# Contributing

tierlm is an experimental research engine. Issues and pull requests are welcome. Agents and people
follow the same rules, which are in [AGENTS.md](AGENTS.md).

## Adding knowledge: the fragment workflow

The library is built only from patterns that are written down and scored:

1. **Read** a reference engine, paper or measurement.
2. **Write a fragment** in `fragments/`, using the template in `fragments/FORMAT.md`:
   - a short pseudocode description in your own words
   - an `Origin:` line with the project tag(s) and a `Licence-of-origin:` line
   - the measured result if there is one, including negative results
3. **Score it** in `fragments/CONSENSUS.md`:
   - Add the fragment to an existing `P-xx` pattern, or propose a new one.
   - The Count is the number of independent projects that converge on it. Lineage
     (one project derived from another) is marked †.
   - The verdict follows the rules in `fragments/README.md`: `adopt`, `adopt-tested`,
     `experimental` or `reject`.
   - Measured failures also go in the negative-results ledger.
4. **Implement** from the fragment, citing the `P-xx` id in the module header. Build `adopt`
   patterns by default. `adopt-tested` patterns go behind a switch with a benchmark.
   `experimental` patterns get a hook only. Never build `reject` patterns.

## Pull requests

- `make test` and `make asan` pass.
- New behaviour has a test, and determinism holds (see `docs/ARCHITECTURE.md`).
- Docs are updated where the change is visible: `docs/STATUS.md`, `etc/tierlm.conf.example`, and
  the README if a command changed.
- Benchmarks: say what was measured, on which machine, with what cache size and hit rate.
