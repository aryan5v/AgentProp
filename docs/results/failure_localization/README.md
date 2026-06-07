# Failure Localization Study

Synthetic study measuring whether verifier-distance signatures uniquely
localize injected faults under different placement methods.

## Reproduce

```bash
pip install -e ".[dev]"
python experiments/failure_localization_study.py
```

No API keys required. Writes `results.json` in this directory.

## Methods

| Method | Placement heuristic |
| --- | --- |
| `metric_dim` | Metric-dimension (resolving set) placement |
| `risk_aware` | Risk-aware verifier placement |
| `betweenness` | Betweenness-centrality placement |

For each built-in workflow template, the script sweeps verifier budget `k ∈ {1…5}`,
computes **collision rate** (fraction of nodes sharing a signature) and
**resolving coverage**.

## Files

- [REPORT.md](REPORT.md) — summary and limitations
- [results.json](results.json) — per-workflow, per-method, per-k rows
