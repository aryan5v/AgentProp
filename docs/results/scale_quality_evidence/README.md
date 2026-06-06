# Scale / Quality Evidence Artifacts

Sanitized synthetic routing matrix comparing broadcast, greedy-family, RZF,
quality-aware, and IMM arms across expanded workflow templates.

## Command

```bash
PYTHONPATH=src:. python experiments/run_evidence_harness.py \
  --tasks-per-arm 5 --repeats 2 \
  --out-dir docs/results/scale_quality_evidence
```

Paper-grade reproduction (N=30/arm, 3 repeats) uses the defaults above.
For a quick smoke check, pass `--tasks-per-arm 5 --repeats 2`.

## Files

- [REPORT.md](REPORT.md) — human-readable table
- [results.json](results.json) — machine-readable aggregates (no secrets)
