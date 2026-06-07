# Routing Baseline Evaluation

AgentProp can compare broadcast routing, classical graph heuristics,
dependency-light ML scorers, and RL policies on the same workflow templates.

Run a quick comparison:

```bash
PYTHONPATH=src:. python dev/experiments/evaluate_routing_baselines.py \
  --workflows chain,star,tree,generic_dag \
  --budget 2 \
  --trials 20 \
  --episodes 40 \
  --epochs 40 \
  --out results/rl/routing_baseline_comparison.json
```

The experiment writes:

- `rows`: per-workflow, per-policy coverage, cost, latency, propagation time,
  savings, and efficiency metrics. Rows also include `constrained_savings`
  (savings credited only when all critical nodes are reached),
  `critical_coverage` (fraction of OUTPUT and VERIFIER nodes activated), and
  `cost_per_coverage` (total cost normalized by coverage).
- `summary`: mean coverage, total cost, savings, and efficiency grouped by
  policy.

Included policy families:

- `broadcast`: full-context routing reference cost.
- Classical graph baselines: `random`, `degree`, `pagerank`, `betweenness`,
  `closeness`, `k_core`, `pure-greedy`, role-critical `greedy`, and `celf`.
- ML baselines: `mlp`, `message_passing_gnn`, `pairwise_ranker`, and
  `marginal_gain_regressor`, trained with leave-one-workflow-out greedy labels,
  seed preference pairs, or marginal utility targets.
- Learned-scorer routing policies: `mlp_routing_policy`,
  `message_passing_gnn_routing_policy`, `pairwise_ranker_routing_policy`, and
  `marginal_gain_regressor_routing_policy`, which run the learned node scores
  through the same sequential `AgentRoutingEnv` rollout used by RL policies.
- RL baselines: `q_learning`, `reinforce`, `ppo`, and `feature_policy`.

`pure-greedy` is the theory-preserving influence-maximization baseline: it does
not pre-seed high-importance roles or multiply scores by role criticality. The
role-critical `greedy` variant is empirical and product-oriented; it protects
context-sensitive nodes such as coders and verifiers, but it should not be used
when claiming the classical `(1 - 1/e)` greedy approximation guarantee.

Heuristic ML examples are behavior-cloning baselines. They are useful for
checking whether small scorers can approximate classical graph policies, but
they are not evidence that learned routing beats the teacher. For that, train
with empirical outcome rows from routed case studies or benchmark artifacts via
`dev/experiments/train_seed_scorer.py --empirical-results ...`. The same script
supports empirical verifier placement with `--task verifier` when rows include
observed verifier activations. Edge-pruning scorers can likewise train from
empirical `pruned_edges` rows with
`dev/experiments/train_edge_pruning_scorer.py --empirical-results ...`, which
separates low-cost edges from edges actually removed without hurting task
success.

The `message_passing_gnn` baseline is dependency-light and CPU-only. It is not a
torch model; it is meant to provide a stable GNN-style comparison path in core
CI before optional torch GCN, GraphSAGE, GAT, and GIN experiments are run.
Its `_routing_policy` row uses the same node scores as an environment policy,
so learned graph scorers can be audited as sequential routing decisions rather
than only as static top-k seed sets.

The `feature_policy` RL baseline is also dependency-light. It learns linear
weights over reusable graph/node/state features, so it can transfer a routing
preference shape across workflow templates instead of memorizing one graph's
stringified tabular states.
