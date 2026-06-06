# Scale / Quality Evidence Artifacts

Sanitized synthetic routing matrix comparing broadcast, greedy-family, RZF,
quality-aware, IMM, and degree baselines across expanded workflow templates.

## Command

```bash
agentprop run-evidence --tasks-per-arm 30 --repeats 3 \
  --out-dir docs/results/scale_quality_evidence
```

Paper-grade reproduction (N=30/arm, 3 repeats) uses the defaults above.
For a quick smoke check, pass `--tasks-per-arm 5 --repeats 2`.

## Files

- [REPORT.md](REPORT.md) — human-readable table
- [results.json](results.json) — machine-readable aggregates (no secrets)
- [outputs.jsonl](outputs.jsonl) — per-task rows (coverage, savings, trial seed)
