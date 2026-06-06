# Claude Code — Project Instructions

## Branch Naming

Do **not** use a `claude/` prefix when creating branches. Use descriptive names
that match the work (e.g., `feat/async-control-session`, `fix/mypy-strict`,
`chore/release-0.1.0a4`).

## Attribution

Do **not** append "Co-authored-by: Claude" or any Anthropic/Claude session URL
to commit messages or PR descriptions. Commits should appear solely under the
user's git identity.

## Quality Gates

Before committing or pushing:

```bash
python -m ruff check src/ benchmarks/ experiments/
python -m mypy src/agentprop
python -m pytest tests/ -q
```

All three must pass. Never use `--no-verify` to skip hooks.

## Key Conventions

- See [AGENTS.md](AGENTS.md) for the full agent guide (directory layout, CLI
  commands, data-commit policy).
- Label benchmark results as **early signal**, **directional**, or **benchmark
  result** — never overclaim from single-task runs.
- Sanitized benchmark artifacts go under `docs/results/` only.
