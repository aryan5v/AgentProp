# Reinforcement Learning Routing

AgentProp includes a small sequential routing environment and four policies:

- `GreedyCoveragePolicy`: one-step lookahead baseline.
- `TabularQPolicy`: trained over full seed-selection episodes with Q-learning.
- `ReinforcePolicy`: trained with a dependency-light episodic policy-gradient
  loop over learned state-action preferences.
- `PPOPolicy`: trained with a dependency-light clipped policy-gradient update
  and a tabular value baseline. This is a research baseline, not a neural PPO
  implementation with function approximation.
- `GraphFeaturePolicy`: trained with policy-gradient updates over reusable
  graph/node/state features instead of stringified state-action tables. This is
  the dependency-light bridge between the ML feature stack and RL routing; it is
  transferable across workflow graphs but is still a linear policy, not a neural
  GNN policy.

Run the trainable policy:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py --policy q-learning --episodes 100 --out results/rl/routing_policy.json
```

Run the policy-gradient baseline:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py --policy reinforce --episodes 100 --out results/rl/reinforce_policy.json
```

Run the clipped policy-gradient baseline:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py --policy ppo --episodes 100 --out results/rl/ppo_policy.json
```

Run the graph-feature-conditioned policy:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py --policy feature-policy --episodes 100 --out results/rl/feature_policy.json
```

The Q-learning, REINFORCE, PPO, and graph-feature policy loops optimize over
complete routing trajectories. The default action set selects context seed nodes
or stops. The graph-feature policy currently uses the seed-selection action set
and scores each candidate node with normalized graph features plus compact state
features such as remaining budget and current coverage. The expanded action set
for tabular policies supports:

- `SEND_CONTEXT(node)`
- `ACTIVATE_VERIFIER(node)`
- `SEND_MESSAGE(edge)`
- `PRUNE_EDGE(edge)`
- `CALL_TOOL(node)`
- `REQUEST_SUMMARY(node)`
- `STOP`

Run expanded-action training:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py --policy ppo --expanded-actions --episodes 100
```

Trained dependency-light RL policies can be saved as JSON checkpoints:

```python
from agentprop.rl import load_rl_policy, save_rl_policy

save_rl_policy(policy, "results/models/ppo_routing_policy.json", metadata={"policy": "ppo"})
checkpoint = load_rl_policy("results/models/ppo_routing_policy.json")
policy = checkpoint.policy
```

`GraphFeaturePolicy` checkpoints store feature names and linear weights, which
keeps runs inspectable and compatible with the model registry.

Compare RL with broadcast, classical graph algorithms, and ML/GNN-style
baselines:

```bash
PYTHONPATH=src:. python experiments/evaluate_routing_baselines.py --workflows chain,star,tree --episodes 40
```

The comparison report now includes both seed-only policies (`q_learning`,
`reinforce`, `ppo`) and expanded-control policies (`q_learning_expanded`,
`reinforce_expanded`, `ppo_expanded`). Expanded rows preserve the chosen action
trace, reward trace, activated verifiers, pruned edges, tool calls, and summary
requests so RL decisions can be audited against greedy and GNN-style routing
baselines.

Expanded-action rewards keep the base coverage/cost/time score and add small
interpretable terms for verifier risk coverage, safe pruning savings, risky
pruning exposure, tool reliability, and summary token savings.

By default those weights are fixed for reproducible synthetic comparisons. For
real routed tasks, calibrate cost and latency penalties from empirical rows that
include pass/fail or quality labels:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py \
  --policy ppo \
  --reward-calibration-rows results/case_study/results.json \
  --out results/rl/ppo_empirical_reward.json
```

The exported trajectory records the reward profile source and weights. This
keeps default RL baselines reproducible while allowing benchmark/case-study
runs to learn a cost frontier from real outcomes.

When empirical rows also include routing context, such as `context_allocations`,
`context_ratios`, or `selected_seeds`, the same command also fits an
`ExpectedSuccessProfile`. The environment then uses estimated task success as
the quality term in the RL reward instead of raw propagation coverage:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py \
  --policy feature-policy \
  --reward-calibration-rows results/case_study/results.json \
  --out results/rl/feature_policy_expected_success.json
```

Cost-only rows still calibrate token, message, and latency penalties without
switching the reward target away from coverage. This prevents the RL loop from
pretending that task success is learned when the rows do not identify which
nodes received full or compressed context.

Replay an exported trajectory with a deterministic propagation seed:

```bash
PYTHONPATH=src:. python experiments/replay_rl_trajectory.py \
  --trajectory results/rl/routing_policy.json \
  --workflow planner_coder_tester_reviewer \
  --policy ppo \
  --seed 0 \
  --out results/rl/replayed_trajectory.json
```

The environment also exposes `reset_gymnasium()` and `step_gymnasium()` methods
that return Gymnasium-style observations without making Gymnasium a required
core dependency.
