# Repository Layout

Where things live, what belongs in git, and which copies are intentional.

## Root (minimal)

| Path | Purpose |
| --- | --- |
| `README.md` | GitHub and PyPI entry — links into `docs/` |
| `pyproject.toml` | Package metadata and tool config |
| `LICENSE` | Apache 2.0 |
| `uv.lock` | Locked dependency graph for `uv` |
| `Makefile` | Thin wrapper → [scripts/Makefile](../scripts/Makefile) |
| `CONTRIBUTING.md` | Pointer → [docs/project/CONTRIBUTING.md](project/CONTRIBUTING.md) |
| `SECURITY.md` | GitHub security policy |
| `AGENTS.md` | Pointer → [docs/project/AGENTS.md](project/AGENTS.md) |

## Library and CLI

| Path | Purpose |
| --- | --- |
| `src/agentprop/` | Python package — graph core, algorithms, runtime, CLI |
| `configs/schemas/workflow.json` | Machine-readable workflow contract |
| `configs/.env.example` | Example environment variables |
| `benchmarks/` | Workflow fixtures, task packs, perf microbenchmarks |
| `configs/experiment_suites/` | Experiment suite definitions |
| `configs/sweeps/` | ML/RL sweep configs |
| `scripts/Makefile` | `make test`, `make lint`, and other dev targets |

## Documentation

| Path | Purpose |
| --- | --- |
| `docs/index.md` | Documentation hub |
| `docs/overview.md` | Core ideas, performance, research context |
| `docs/project/` | Contributing, changelog, and project meta |
| `docs/local/` | **Gitignored** working notes (never commit) |

## Runnable trees

| Path | Purpose |
| --- | --- |
| `examples/` | Integration templates — [README](../examples/README.md) |
| `experiments/` | Repro scripts — [README](../experiments/README.md) |

Local output (gitignored): `results/`, `reports/`, `benchmark-results/`.

## Public artifacts

| Path | Purpose |
| --- | --- |
| `docs/results/` | Sanitized finalized benchmarks — [ARTIFACTS.md](results/ARTIFACTS.md) |

Do not commit raw LLM runs, Harbor bundles, or in-progress checkpoints here.

## Distribution (editor agents)

Three surfaces, one canonical skill source:

| Path | Role |
| --- | --- |
| `skills/agentprop-workflow-optimizer/` | **Canonical** skill for skills.sh — edit here first |
| `plugins/agentprop/` | Same-repo Codex + Claude plugin bundle (synced copy of skill + MCP config) |
| `integrations/claude-code/` | Thin wrapper for legacy Claude paths — points to canonical skill |
| `integrations/codex/` | Codex agent brief template |

When changing skill content, update `skills/` then sync into
`plugins/agentprop/skills/`. See [plugin distribution](plugin_distribution.md).

## Local-only (gitignored)

- `.venv/`, `.agentprop_sessions/`, `.local_pkgs/`

## Quality gates

```bash
make dev test lint typecheck
agentprop doctor --tier graph
agentprop readiness --json
```
