# Public Benchmark Artifacts

This directory holds **sanitized, finalized** benchmark outputs that support
reproducibility and public claims. It is not a dump of local experiment runs.

## What belongs here

| Artifact | Purpose |
| --- | --- |
| `REPORT.md` | Human-readable summary with limitations stated |
| `results.json` | Machine-readable aggregate metrics |
| `outputs.jsonl` | Per-task rows with questions, pass/fail, token counts — no API keys or raw provider prompts |
| `README.md` | How the run was produced and how to reproduce |

## What must stay local

Never commit these under `docs/results/` or anywhere else in the repo:

- `results/`, `reports/`, `benchmark-results/` from local runs
- `PARTIAL_REPORT.md`, `results_partial.json`, `checkpoint.json` (in-progress checkpoints)
- Terminal-Bench launch bundles (`RUNBOOK.md`, `manifest.json`, `run_with_watchdog.sh`) — generate with `agentprop terminal-bench prepare` into a local output directory
- Raw LLM prompts, API keys, Harbor/Modal credentials, or provider traces with secrets
- Maintainer-only registry snapshots
- Internal working notes (`docs/local/`: paper drafts, roadmaps, publishing ops,
  regression postmortems, launch runbooks)

See [CONTRIBUTING.md](../../CONTRIBUTING.md#data-secrets-and-benchmark-artifacts) for the full policy.
Working notes use the gitignored [`docs/local/`](../local/README.example.md) layout.

## Current public artifacts

| Directory | Status | Notes |
| --- | --- | --- |
| [v1/](v1/README.md) | Final | Synthetic workflow benchmark |
| [gaia_benchmark/](gaia_benchmark/REPORT.md) | Final | GAIA-style QA routing study |
| [real_routing_case_study/](real_routing_case_study/REPORT.md) | Final | Code-generation routing study |
| [terminal_bench_guided/](terminal_bench_guided/README.md) | Final | Directional Terminal-Bench snapshot |

Before adding a new directory, confirm the report labels limitations clearly and
that `outputs.jsonl` contains no secrets or private prompts.
