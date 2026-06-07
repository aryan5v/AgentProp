# AgentProp — Agent Guide

Quick map for coding agents working in this repository.

## What this project is

**AgentProp** = Agent + **Propagation** (not "properties", not an orchestrator).

It models multi-agent workflows as directed weighted graphs and provides graph
analysis, propagation simulation, metric-dimension verifier placement, and
runtime control via `ControlSession`. It wraps existing agent loops; it does
not replace them.

## Five-minute path

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
agentprop doctor --tier graph
agentprop analyze planner_coder_tester_reviewer --json
agentprop control-demo --demo terminal --out-dir reports/control-demo
agentprop workflows list
```

No API keys required for the steps above.

## Start here

- [README.md](../../README.md) — install, quickstart, status
- [Architecture](../ARCHITECTURE.md) — how modules connect
- [Documentation index](../index.md) — full doc map
- [Environment setup](../environment.md) — env vars and tiers
- [CONTRIBUTING.md](CONTRIBUTING.md) — quality gates and data policy
- [dev/examples/README.md](../../dev/examples/README.md) — integration learning path
- [dev/experiments/README.md](../../dev/experiments/README.md) — repro script catalog
- [Repository layout](../repository_layout.md) — where artifacts belong

## Common commands

```bash
make test                    # pytest via dev extra
agentprop workflows list
agentprop optimize planner_coder_tester_reviewer --budget 2
agentprop ingest-trace path/to/trace.json --out-brief reports/brief.md
agentprop readiness --json
```

## Custom workflows and runtime events

1. Define a workflow JSON per [workflow schema](../workflow_schema.md)
   or use a built-in name from `agentprop workflows list`.
2. Wrap your harness with `ControlSession` — emit one `ExecutionEvent` per step.
3. Act on decisions: `CONTINUE`, `FORCE_VERIFY`, `SWITCH_STRATEGY`, `FINALIZE`.

See [control layer quickstart](../control_layer_quickstart.md).

## MCP and skills

- MCP server: `pip install "agentprop[mcp]"` then `agentprop-mcp`
- Skill: `npx skills add https://github.com/aryan5v/AgentProp --skill agentprop-workflow-optimizer`

## Do not commit

- API keys, Harbor/Modal credentials, or `.env` files
- Local run output: `results/`, `reports/`, `benchmark-results/`
- In-progress artifacts: `PARTIAL_REPORT.md`, `results_partial.json`, `checkpoint.json`
- Raw LLM prompts from real paid runs
- Terminal-Bench launch bundles (generate locally with `agentprop terminal-bench prepare`)

Sanitized finalized benchmarks belong only under `docs/results/` per
[ARTIFACTS.md](../results/ARTIFACTS.md).

## Key directories

| Path | Role |
| --- | --- |
| `src/agentprop/` | Library and CLI |
| `dev/benchmarks/workflows/` | JSON workflow fixtures |
| `dev/experiments/` | Reproducible research scripts |
| `docs/` | Public documentation |
| `docs/local/` | Gitignored working notes; never commit local research or ops notes |
| `distribution/skills/agentprop-workflow-optimizer/` | Canonical skill — edit here first |
| `distribution/plugins/agentprop/` | Codex/Claude plugin bundle (sync skill from `distribution/skills/`) |
| `distribution/wrappers/claude-code/` | Thin legacy wrapper — do not fork skill content here |

## Public claims

Label results as **early signal**, **directional**, or **benchmark result**.
Link to saved artifacts in `docs/results/` or document reproduction commands.
Do not overclaim from single-task or partial runs.
