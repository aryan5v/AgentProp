"""Run sequential routing policies on built-in workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.rl import (
    AgentRoutingEnv,
    GreedyCoveragePolicy,
    PPOConfig,
    QLearningConfig,
    ReinforceConfig,
    RoutingAction,
    train_ppo_policy,
    train_q_policy,
    train_reinforce_policy,
)
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AgentProp sequential routing policies.")
    parser.add_argument(
        "--policy",
        choices=["q-learning", "reinforce", "ppo", "greedy"],
        default="q-learning",
    )
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--expanded-actions", action="store_true")
    parser.add_argument("--max-steps", type=int, default=20)
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
                    expanded_actions=args.expanded_actions,
                ),
            )
            env.reset()
        elif args.policy == "reinforce":
            policy, training = train_reinforce_policy(
                env,
                config=ReinforceConfig(
                    episodes=args.episodes,
                    learning_rate=args.learning_rate,
                    expanded_actions=args.expanded_actions,
                    max_steps=args.max_steps,
                ),
            )
            env.reset()
        elif args.policy == "ppo":
            policy, training = train_ppo_policy(
                env,
                config=PPOConfig(
                    episodes=args.episodes,
                    learning_rate=args.learning_rate,
                    clip_epsilon=args.clip_epsilon,
                    expanded_actions=args.expanded_actions,
                    max_steps=args.max_steps,
                ),
            )
            env.reset()
        else:
            policy = GreedyCoveragePolicy()

        trajectory = []
        done = False
        steps = 0
        while not done and steps < args.max_steps:
            action = policy.act(env)
            if action == RoutingAction.STOP.value:
                state, reward, done, info = env.step(action)
            else:
                state, reward, done, info = env.step(action)
            steps += 1
            trajectory.append(
                {
                    "action": action,
                    "reward": reward,
                    "coverage": state.coverage,
                    "token_cost": state.token_cost,
                    "message_cost": state.message_cost,
                    "activated_verifiers": list(state.activated_verifiers),
                    "used_edges": [list(edge) for edge in state.used_edges],
                    "pruned_edges": [list(edge) for edge in state.pruned_edges],
                    "called_tools": list(state.called_tools),
                    "summary_nodes": list(state.summary_nodes),
                    "done": done,
                    "info": dict(info),
                }
            )
        row = {"workflow": workflow_name, "policy": args.policy, "trajectory": trajectory}
        if training is not None:
            row["training"] = {
                "episodes": training.episodes,
                "episode_rewards": training.episode_rewards,
            }
            if args.policy == "q-learning":
                row["training"]["q_value_count"] = training.q_value_count
                row["q_values"] = policy.to_dict()
            if args.policy == "reinforce":
                row["training"]["preference_count"] = training.preference_count
                row["training"]["truncated_episodes"] = training.truncated_episodes
                row["preferences"] = policy.to_dict()
            if args.policy == "ppo":
                row["training"]["preference_count"] = training.preference_count
                row["training"]["value_count"] = training.value_count
                row["training"]["truncated_episodes"] = training.truncated_episodes
                row["preferences"] = policy.to_dict()
                row["values"] = policy.values_to_dict()
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
