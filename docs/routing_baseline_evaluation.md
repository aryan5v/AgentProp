# Routing Baseline Evaluation

AgentProp can compare broadcast routing, classical graph heuristics,
dependency-light ML scorers, and RL policies on the same workflow templates.

Run a quick comparison:

```bash
PYTHONPATH=src:. python experiments/evaluate_routing_baselines.py \
  --workflows chain,star,tree,generic_dag \
  --budget 2 \
  --trials 20 \
  --episodes 40 \
  --epochs 40 \
  --out results/rl/routing_baseline_comparison.json
```

The experiment writes:

- `rows`: per-workflow, per-policy coverage, cost, latency, propagation time,
  savings, and efficiency metrics.
- `summary`: mean coverage, total cost, savings, and efficiency grouped by
  policy.

Included policy families:

- `broadcast`: full-context routing reference cost.
- Classical graph baselines: `random`, `degree`, `pagerank`, `betweenness`,
  `closeness`, `k_core`, `greedy`, and `celf`.
- ML baselines: `mlp`, `message_passing_gnn`, `pairwise_ranker`, and
  `marginal_gain_regressor`, trained with leave-one-workflow-out greedy labels,
  seed preference pairs, or marginal utility targets.
- RL baselines: `q_learning` and `reinforce`.

The `message_passing_gnn` baseline is dependency-light and CPU-only. It is not a
torch model; it is meant to provide a stable GNN-style comparison path in core
CI before optional torch GCN, GraphSAGE, GAT, and GIN experiments are run.
