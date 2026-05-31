# Reinforcement Learning Routing

AgentProp includes a small sequential routing environment and three policies:

- `GreedyCoveragePolicy`: one-step lookahead baseline.
- `TabularQPolicy`: trained over full seed-selection episodes with Q-learning.
- `ReinforcePolicy`: trained with a dependency-light episodic policy-gradient
  loop over learned state-action preferences.

Run the trainable policy:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py --policy q-learning --episodes 100 --out results/rl/routing_policy.json
```

Run the policy-gradient baseline:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py --policy reinforce --episodes 100 --out results/rl/reinforce_policy.json
```

The Q-learning and REINFORCE loops optimize over complete routing trajectories.
The default action set selects context seed nodes or stops. The expanded action
set supports:

- `SEND_CONTEXT(node)`
- `ACTIVATE_VERIFIER(node)`
- `SEND_MESSAGE(edge)`
- `PRUNE_EDGE(edge)`
- `CALL_TOOL(node)`
- `REQUEST_SUMMARY(node)`
- `STOP`

Run expanded-action training:

```bash
PYTHONPATH=src:. python experiments/run_rl_routing.py --policy reinforce --expanded-actions --episodes 100
```

The environment also exposes `reset_gymnasium()` and `step_gymnasium()` methods
that return Gymnasium-style observations without making Gymnasium a required
core dependency.
