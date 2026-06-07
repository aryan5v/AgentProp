# Development and research assets

Everything outside the installable Python package that supports benchmarks,
repro scripts, and integration examples.

| Path | Purpose |
| --- | --- |
| [benchmarks/](benchmarks/) | Workflow JSON fixtures, task packs, perf microbenchmarks |
| [configs/](configs/) | Env template, workflow schema, experiment suites and sweeps |
| [experiments/](experiments/) | Reproducible research scripts — [catalog](experiments/README.md) |
| [examples/](examples/) | Integration templates — [catalog](examples/README.md) |
| [scripts/](scripts/Makefile) | Dev Makefile (`make test`, `make lint`, …) |

Local output (gitignored): `results/`, `reports/`, `benchmark-results/`.

```bash
make dev test lint
python dev/examples/minimal_control_loop.py
python dev/experiments/failure_localization_study.py
```
