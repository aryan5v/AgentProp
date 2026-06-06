# AgentProp v1 Benchmark Results

This directory contains the first committed AgentProp synthetic workflow
benchmark artifacts.

## Command

```bash
PYTHONPATH=src:. python experiments/run_benchmark.py --budget 2 --trials 20 --out-dir docs/results/v1
```

## Artifacts

- [results.json](results.json): full benchmark payload.
- [results.csv](results.csv): table-ready benchmark export.
- [savings_by_algorithm.svg](savings_by_algorithm.svg): first plot of average estimated savings by algorithm.

## Scope

The benchmark covers all built-in workflow templates, all current seed-selection
algorithms, and all current propagation models, including deterministic zero
forcing.
