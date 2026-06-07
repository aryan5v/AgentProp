# Repository Layout

Where things live, what belongs in git, and which copies are intentional.

## Root (minimal)

| Path | Purpose |
| --- | --- |
| `README.md` | GitHub and PyPI entry — links into `docs/` |
| `pyproject.toml` | Package metadata and tool config |
| `LICENSE` | Apache 2.0 |
| `uv.lock` | Locked dependency graph for `uv` |
| `Makefile` | Thin wrapper → [dev/scripts/Makefile](../dev/scripts/Makefile) |
| `CONTRIBUTING.md` | Pointer → [docs/project/CONTRIBUTING.md](project/CONTRIBUTING.md) |
| `SECURITY.md` | GitHub security policy |
| `AGENTS.md` | Pointer → [docs/project/AGENTS.md](project/AGENTS.md) |
| `src/` | Installable Python package |
| `tests/` | Pytest suite |
| `docs/` | Documentation hub |
| [`dev/`](../dev/README.md) | Benchmarks, configs, experiments, examples, dev scripts |
| [`distribution/`](../distribution/README.md) | Editor-agent plugins, skills, and wrappers |
| `.agents/`, `.claude-plugin/` | Marketplace metadata (required at repo root) |

## Library and CLI

| Path | Purpose |
| --- | --- |
| `src/agentprop/` | Python package — graph core, algorithms, runtime, CLI |
| `dev/configs/schemas/workflow.json` | Machine-readable workflow contract |
| `dev/configs/.env.example` | Example environment variables |
| `dev/benchmarks/` | Workflow fixtures, task packs, perf microbenchmarks |
| `dev/configs/experiment_suites/` | Experiment suite definitions |
| `dev/configs/sweeps/` | ML/RL sweep configs |
| `dev/scripts/Makefile` | `make test`, `make lint`, and other dev targets |

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
| `dev/examples/` | Integration templates — [README](../dev/examples/README.md) |
| `dev/experiments/` | Repro scripts — [README](../dev/experiments/README.md) |

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
| `distribution/skills/agentprop-workflow-optimizer/` | **Canonical** skill for skills.sh — edit here first |
| `distribution/plugins/agentprop/` | Same-repo Codex + Claude plugin bundle (synced copy of skill + MCP config) |
| `distribution/wrappers/claude-code/` | Thin wrapper for legacy Claude paths — points to canonical skill |
| `distribution/wrappers/codex/` | Codex agent brief template |

When changing skill content, update `distribution/skills/` then sync into
`distribution/plugins/agentprop/skills/`. See [plugin distribution](plugin_distribution.md).

## Local-only (gitignored)

- `.venv/`, `.agentprop_sessions/`, `.local_pkgs/`

## Quality gates

```bash
make dev test lint typecheck
agentprop doctor --tier graph
agentprop readiness --json
```
