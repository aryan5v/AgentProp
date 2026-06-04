# AgentProp 0.1.0-alpha.2 Release Notes

Release date: 2026-06-04

AgentProp `0.1.0-alpha.2` is a research-facing public alpha refresh. It
sharpens the project around graph observability, propagation control, and live
agent-runtime governance.

## Highlights

- New README and docs index centered on:
  - metric-dimension verifier placement,
  - Quality Cascade propagation,
  - Randomized Zero Forcing scaling,
  - runtime stop/retry/verify control.
- Public references page for the graph theory, influence-maximization, and
  multi-agent topology papers that inspired AgentProp.
- Metric-dimension verifier placement with resolving coverage and
  fault-tolerant resolving coverage.
- Quality Cascade propagation model and quality-decay benchmarking transform.
- Runtime controller improvements for terminal-agent loops:
  - local-pass distrust,
  - verifier forcing,
  - stale-verifier avoidance,
  - deferred command handling,
  - safer pass-preserving finalization.
- Category-conditioned bandit policy and safer reward shaping for learned
  runtime control.
- Early Codex CLI A0-vs-A2 smoke signal: on `terminal-bench/regex-log`, both
  arms passed, while A2 used 33.8% fewer tokens, 41.0% lower reported cost, and
  14.8% less wall time.

## Validation Scope

This is still public alpha research software. The Codex CLI result is a
single-task early signal, not a benchmark claim. Strong claims require repeated
matched runs across larger held-out task sets.

## Suggested Checks Before Tagging

```bash
python -m ruff check .
python -m mypy src
python -m pytest
PYTHONPATH=src:. python experiments/verifier_placement_evidence.py
PYTHONPATH=src:. python experiments/rzf_scaling_study.py
```
