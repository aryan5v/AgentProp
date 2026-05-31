# AgentProp 0.1.0-alpha.1 Release Notes

Release date: 2026-05-31

AgentProp `0.1.0-alpha.1` is the first private alpha release of the framework.

## Highlights

- Directed weighted graph abstraction for multi-agent LLM workflows.
- CLI commands for analysis, optimization, benchmarking, reports, traces, and
  visualization.
- Propagation models: Independent Cascade, Linear Threshold, Bootstrap
  Percolation, Randomized Zero Forcing, and deterministic Zero Forcing.
- Seed-selection baselines: random, degree, PageRank-style, betweenness, greedy,
  CELF, and cost-aware greedy.
- Diagnostics for bottlenecks, pruning, verifier placement, and observability.
- Saved v1 benchmark table and first SVG plot.
- Optional torch GNN seed scorers for GCN, GraphSAGE, and GAT.
- Q-learning policy for sequential routing beyond greedy lookahead.
- PRD, literature review, tutorial, docs index, case-study protocol, and paper
  outline.

## Validation

Verified before tagging:

```bash
python -m pytest
python -m ruff check .
python -m mypy src
PYTHONPATH=src:. python experiments/run_rl_routing.py --policy q-learning --episodes 10 --trials 3 --out /tmp/agentprop-q-routing.json
```

Optional torch smoke test:

```bash
PYTHONPATH=src:. python experiments/train_torch_gnn.py --architecture graphsage --epochs 2 --hidden-dim 8 --trials 3 --out /tmp/agentprop-dl-smoke.json
```

## Known Limitations

- Real LLM case-study results are not yet attached.
- Learned policies are v1 baselines, not tuned production optimizers.
- The public-release recommendation is to stay private until the first real LLM
  case study is complete, or publish as alpha with this limitation explicit.
