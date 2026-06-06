# AgentProp — Agent Guide

Quick map for coding agents working in this repository.

## What this project is

AgentProp models multi-agent workflows as directed weighted graphs and provides
graph analysis, propagation simulation, verifier placement, and runtime control
(`ControlSession`). It wraps existing agent loops; it does not replace them.

## Start here

- [README.md](README.md) — install, quickstart, status
- [docs/index.md](docs/index.md) — documentation map
- [CONTRIBUTING.md](CONTRIBUTING.md) — quality gates and data policy
- [examples/quickstart.py](examples/quickstart.py) — minimal Python usage

## Common commands

```bash
python -m pip install -e ".[dev]"
ruff check . && mypy src && pytest
agentprop analyze planner_coder_tester_reviewer
agentprop control-demo --demo terminal --out-dir reports/control-demo
```

## Do not commit

- API keys, Harbor/Modal credentials, or `.env` files
- Local run output: `results/`, `reports/`, `benchmark-results/`
- In-progress artifacts: `PARTIAL_REPORT.md`, `results_partial.json`, `checkpoint.json`
- Raw LLM prompts from real paid runs
- Terminal-Bench launch bundles (generate locally with `agentprop terminal-bench prepare`)

Sanitized finalized benchmarks belong only under `docs/results/` per
[docs/results/ARTIFACTS.md](docs/results/ARTIFACTS.md).

## Key directories

| Path | Role |
| --- | --- |
| `src/agentprop/` | Library and CLI |
| `benchmarks/workflows/` | JSON workflow fixtures |
| `experiments/` | Reproducible research scripts |
| `docs/` | Public documentation |
| `docs/local/` | Gitignored working notes (see `docs/local/README.example.md`) |
| `skills/agentprop-workflow-optimizer/` | Canonical agent skill (skills.sh) |
| `integrations/claude-code/` | Claude Code install path + `agents/openai.yaml` |

## Public claims

Label results as **early signal**, **directional**, or **benchmark result**.
Link to saved artifacts in `docs/results/` or document reproduction commands.
Do not overclaim from single-task or partial runs.
