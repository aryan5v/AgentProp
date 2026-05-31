"""Run sequential routing policies on built-in workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.rl import (
    AgentRoutingEnv,
    GreedyCoveragePolicy,
    QLearningConfig,
    RoutingAction,
    train_q_policy,
)
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AgentProp sequential routing policies.")
    parser.add_argument("--policy", choices=["q-learning", "greedy"], default="q-learning")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=Path("results/rl/routing_policy.json"))
    args = parser.parse_args(argv)

    rows = []
    for workflow_name, builder in WORKFLOW_TEMPLATES.items():
        env = AgentRoutingEnv(builder(), budget=args.budget, trials=args.trials)
        training = None
        if args.policy == "q-learning":
            policy, training = train_q_policy(
                env,
                config=QLearningConfig(
                    episodes=args.episodes,
                    learning_rate=args.learning_rate,
                    epsilon=args.epsilon,
                ),
            )
            env.reset()
        else:
            policy = GreedyCoveragePolicy()

        trajectory = []
        done = False
        while not done:
            action = policy.act(env)
            if action == RoutingAction.STOP.value:
                state, reward, done, info = env.step(action)
            else:
                state, reward, done, info = env.step(action)
            trajectory.append(
                {
                    "action": action,
                    "reward": reward,
                    "coverage": state.coverage,
                    "token_cost": state.token_cost,
                    "message_cost": state.message_cost,
                    "done": done,
                    "info": dict(info),
                }
            )
        row = {"workflow": workflow_name, "policy": args.policy, "trajectory": trajectory}
        if training is not None:
            row["training"] = {
                "episodes": training.episodes,
                "episode_rewards": training.episode_rewards,
                "q_value_count": training.q_value_count,
            }
            row["q_values"] = policy.to_dict()
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
