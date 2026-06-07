# Experiments

Reproducible research scripts. Run from the repo root after `pip install -e ".[dev]"`.

**Output policy:** finalized, sanitized artifacts go under `docs/results/` per
[ARTIFACTS.md](../docs/results/ARTIFACTS.md). Local runs use gitignored
`results/`, `reports/`, or `benchmark-results/`.

## Quick repro (no API keys)

| Script | Output | Public artifact |
| --- | --- | --- |
| `failure_localization_study.py` | `docs/results/failure_localization/` | [README](../docs/results/failure_localization/README.md) |
| `quality_cascade_vs_ic.py` | `docs/results/quality_cascade_vs_ic/` | [README](../docs/results/quality_cascade_vs_ic/README.md) |
| `verifier_placement_evidence.py` | stdout + local | — |
| `rzf_scaling_study.py` | stdout + local | — |
| `run_evidence_harness.py` | `docs/results/scale_quality_evidence/` | [README](../docs/results/scale_quality_evidence/README.md) |

CLI equivalent for the evidence matrix:

```bash
agentprop run-evidence --tasks-per-arm 30 --repeats 3 \
  --out-dir docs/results/scale_quality_evidence
```

## Graph benchmarks and baselines

| Script | Tier | Notes |
| --- | --- | --- |
| `run_benchmark.py` | graph | JSON/CSV benchmark tables |
| `evaluate_routing_baselines.py` | graph | Classical vs ML vs RL baselines |
| `run_experiment_suite.py` | graph | Config-driven suites in `dev/configs/experiment_suites/` |
| `run_ml_rl_sweep.py` | graph | Sweeps in `dev/configs/sweeps/` |

## Case studies and real LLM validation

| Script | Tier | Output default |
| --- | --- | --- |
| `run_case_study.py` | LLM | `results/case_study_offline/` (local) |
| `analyze_case_study.py` | — | Reads `--results` path you pass |
| `run_real_routing_case_study.py` | LLM | `results/real_routing_case_study/` |
| `run_gaia_style_benchmark.py` | LLM | `results/gaia_benchmark/` |

Published real-routing artifact:
[docs/results/real_routing_case_study/](../docs/results/real_routing_case_study/REPORT.md).

## Terminal-Bench and Harbor

| Script | Tier | Notes |
| --- | --- | --- |
| `prepare_terminal_bench_21.py` | Terminal-Bench | Local launch bundle only — do not commit |
| `summarize_harbor_results.py` | Terminal-Bench | Summarize saved Harbor `result.json` trees |
| `run_with_watchdog.py` | — | Wall-clock watchdog for long external commands |

Protocol for multi-task A0 vs A2:
[docs/results/terminal_bench_multi/](../docs/results/terminal_bench_multi/README.md).

## ML / RL training

| Script | Tier | Notes |
| --- | --- | --- |
| `train_seed_scorer.py` | graph | Lightweight node-policy scorer |
| `train_edge_pruning_scorer.py` | graph | Edge-pruning scorer |
| `train_torch_gnn.py` | graph + torch | Optional GNN seed scorer |
| `train_learned_propagation.py` | graph | Fit propagation from traces |
| `build_empirical_rows.py` | graph | Build training rows from traces |
| `run_rl_routing.py` | graph | Q-learning routing |
| `replay_rl_trajectory.py` | graph | Replay exported trajectories |
| `run_bandit_routing.py` | graph | Category-conditioned bandit |
| `evaluate_ml_generalization.py` | graph | Held-out workflow evaluation |

## Credential tiers

See [docs/environment.md](../docs/environment.md):

- **graph** — editable install only
- **dev** — pytest, ruff, mypy
- **LLM** — `OPENAI_API_KEY` or compatible endpoint for case studies
- **Terminal-Bench** — Harbor + dataset access for live agent benchmarks
