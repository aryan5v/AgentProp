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
  `closeness`, `k_core`, `pure-greedy`, role-critical `greedy`, and `celf`.
- ML baselines: `mlp`, `message_passing_gnn`, `pairwise_ranker`, and
  `marginal_gain_regressor`, trained with leave-one-workflow-out greedy labels,
  seed preference pairs, or marginal utility targets.
- RL baselines: `q_learning`, `reinforce`, and `ppo`.

`pure-greedy` is the theory-preserving influence-maximization baseline: it does
not pre-seed high-importance roles or multiply scores by role criticality. The
role-critical `greedy` variant is empirical and product-oriented; it protects
context-sensitive nodes such as coders and verifiers, but it should not be used
when claiming the classical `(1 - 1/e)` greedy approximation guarantee.

Heuristic ML examples are behavior-cloning baselines. They are useful for
checking whether small scorers can approximate classical graph policies, but
they are not evidence that learned routing beats the teacher. For that, train
with empirical outcome rows from routed case studies or benchmark artifacts via
`experiments/train_seed_scorer.py --empirical-results ...`. Edge-pruning
scorers can likewise train from empirical `pruned_edges` rows with
`experiments/train_edge_pruning_scorer.py --empirical-results ...`, which
separates low-cost edges from edges actually removed without hurting task
success.

The `message_passing_gnn` baseline is dependency-light and CPU-only. It is not a
torch model; it is meant to provide a stable GNN-style comparison path in core
CI before optional torch GCN, GraphSAGE, GAT, and GIN experiments are run.
