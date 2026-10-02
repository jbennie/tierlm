# CLAUDE.md

The project rules for every coding agent live in AGENTS.md (imported below), so that Claude Code,
Copilot and others follow the same instructions.

@AGENTS.md

## Claude-specific notes

- For multi-module work, split it by module and give each sub-agent its own `BUILD=build-<name>`
  tree. Only merge when `make test` passes in every tree.
- Before committing, run the no-copy check if reference sources are on disk: compare normalised
  4-line windows of `src/` against them. Only `#include` runs and `extern "C"` guards may match.
- End commit messages with the attribution trailer the session asks for.
