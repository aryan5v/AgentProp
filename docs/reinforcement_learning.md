# Reinforcement Learning Routing

AgentProp includes a small sequential routing environment and two policies:

- `GreedyCoveragePolicy`: one-step lookahead baseline.
- `TabularQPolicy`: trained over full seed-selection episodes with Q-learning.

Run the trainable policy:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py --policy q-learning --episodes 100 --out results/rl/routing_policy.json
```

The Q-learning loop optimizes over complete routing trajectories. Each state is
the currently selected seed set plus the remaining budget, and each action
selects another seed node or stops. This keeps v1 dependency-light while proving
that AgentProp can train policies beyond immediate greedy lookahead.
