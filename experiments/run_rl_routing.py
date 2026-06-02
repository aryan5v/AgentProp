"""Run sequential routing policies on built-in workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeAlias

from agentprop.evaluation import quality_cost_summary, register_artifact, safe_artifact_id
from agentprop.rl import (
    AgentRoutingEnv,
    FeaturePolicyConfig,
    FeaturePolicyTrainingResult,
    GraphFeaturePolicy,
    GreedyCoveragePolicy,
    PPOConfig,
    PPOPolicy,
    PPOTrainingResult,
    QLearningConfig,
    QLearningTrainingResult,
    ReinforceConfig,
    ReinforcePolicy,
    ReinforceTrainingResult,
    RoutingAction,
    RoutingRewardProfile,
    RoutingState,
    TabularQPolicy,
    calibrate_routing_reward_profile,
    save_rl_policy,
    train_feature_policy,
    train_ppo_policy,
    train_q_policy,
    train_reinforce_policy,
)
from agentprop.workflows import WORKFLOW_TEMPLATES

RoutingPolicy: TypeAlias = (
    TabularQPolicy | ReinforcePolicy | PPOPolicy | GraphFeaturePolicy | GreedyCoveragePolicy
)
TrainingResult: TypeAlias = (
    QLearningTrainingResult
    | ReinforceTrainingResult
    | PPOTrainingResult
    | FeaturePolicyTrainingResult
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AgentProp sequential routing policies.")
    parser.add_argument(
        "--policy",
        choices=["q-learning", "reinforce", "ppo", "feature-policy", "greedy"],
        default="q-learning",
    )
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--expanded-actions", action="store_true")
    parser.add_argument("--reward-calibration-rows", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("results/rl/routing_policy.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    reward_profile = (
        calibrate_routing_reward_profile(_load_empirical_rows(args.reward_calibration_rows))
        if args.reward_calibration_rows is not None
        else RoutingRewardProfile()
    )
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None and args.registry_root is not None:
        checkpoint_dir = args.registry_root / "checkpoints"
    for workflow_name, builder in WORKFLOW_TEMPLATES.items():
        env = AgentRoutingEnv(
            builder(),
            budget=args.budget,
            trials=args.trials,
            reward_profile=reward_profile,
        )
        policy: RoutingPolicy
        training: TrainingResult | None = None
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
        elif args.policy == "feature-policy":
            policy, training = train_feature_policy(
                env,
                config=FeaturePolicyConfig(
                    episodes=args.episodes,
                    learning_rate=args.learning_rate,
                    epsilon=args.epsilon,
                    max_steps=args.max_steps,
                ),
            )
            env.reset()
        else:
            policy = GreedyCoveragePolicy()

        trajectory: list[dict[str, Any]] = []
        done = False
        steps = 0
        total_reward = 0.0
        while not done and steps < args.max_steps:
            action = policy.act(env)
            if action == RoutingAction.STOP.value:
                state, reward, done, info = env.step(action)
            else:
                state, reward, done, info = env.step(action)
            steps += 1
            total_reward += reward
            trajectory.append(
                {
                    "action": action,
                    "reward": reward,
                    "cumulative_reward": total_reward,
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
        final_state = env.state
        row: dict[str, Any] = {
            "workflow": workflow_name,
            "policy": args.policy,
            "trajectory": trajectory,
            "summary": _trajectory_summary(final_state, total_reward),
            "reward_profile": reward_profile.to_dict(),
        }
        if training is not None:
            row["training"] = {
                "episodes": training.episodes,
                "episode_rewards": training.episode_rewards,
            }
            if isinstance(training, QLearningTrainingResult) and isinstance(policy, TabularQPolicy):
                row["training"]["q_value_count"] = training.q_value_count
                row["q_values"] = policy.to_dict()
            if isinstance(training, ReinforceTrainingResult) and isinstance(
                policy, ReinforcePolicy
            ):
                row["training"]["preference_count"] = training.preference_count
                row["training"]["truncated_episodes"] = training.truncated_episodes
                row["preferences"] = policy.to_dict()
            if isinstance(training, PPOTrainingResult) and isinstance(policy, PPOPolicy):
                row["training"]["preference_count"] = training.preference_count
                row["training"]["value_count"] = training.value_count
                row["training"]["truncated_episodes"] = training.truncated_episodes
                row["preferences"] = policy.to_dict()
                row["values"] = policy.values_to_dict()
            if isinstance(training, FeaturePolicyTrainingResult) and isinstance(
                policy, GraphFeaturePolicy
            ):
                row["training"]["feature_count"] = training.feature_count
                row["training"]["truncated_episodes"] = training.truncated_episodes
                row["feature_policy"] = policy.to_dict()
            if checkpoint_dir is not None and isinstance(
                policy, TabularQPolicy | ReinforcePolicy | PPOPolicy | GraphFeaturePolicy
            ):
                artifact_id = safe_artifact_id(
                    f"{args.run_id}-{workflow_name}"
                    if args.run_id
                    else f"{workflow_name}-{args.policy}-routing-policy"
                )
                checkpoint_path = save_rl_policy(
                    policy,
                    checkpoint_dir / f"{artifact_id}.json",
                    metadata={
                        "workflow": workflow_name,
                        "policy": args.policy,
                        "budget": args.budget,
                        "trials": args.trials,
                        "episodes": args.episodes,
                        "learning_rate": args.learning_rate,
                        "expanded_actions": args.expanded_actions,
                        "max_steps": args.max_steps,
                        "reward_profile": reward_profile.to_dict(),
                    },
                )
                row["checkpoint_path"] = str(checkpoint_path)
                if args.registry_root is not None:
                    register_artifact(
                        args.registry_root,
                        artifact_id=artifact_id,
                        kind="rl-policy",
                        path=checkpoint_path,
                        source="experiments.run_rl_routing",
                        metrics_path=args.out,
                        tags=("routing-policy", args.policy, workflow_name),
                        metadata={
                            "workflow": workflow_name,
                            "budget": args.budget,
                            "trials": args.trials,
                            "episodes": args.episodes,
                            "expanded_actions": args.expanded_actions,
                            "final_coverage": row["summary"]["final_coverage"],
                            "efficiency_score": row["summary"]["efficiency_score"],
                            "reward_source": reward_profile.source,
                        },
                    )
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


def _trajectory_summary(state: RoutingState, total_reward: float) -> dict[str, float]:
    token_cost = state.token_cost
    message_cost = state.message_cost
    total_cost = token_cost + message_cost
    quality = quality_cost_summary(
        success_rate=state.coverage,
        token_cost=total_cost,
        latency=state.propagation_time,
    )
    return {
        "total_reward": total_reward,
        "final_coverage": quality.success_rate,
        "final_token_cost": token_cost,
        "final_message_cost": message_cost,
        "final_total_cost": total_cost,
        "final_propagation_time": quality.latency,
        "proxy_success_rate": quality.success_rate,
        "cost_adjusted_success": quality.cost_adjusted_success,
        "efficiency_score": quality.efficiency_score,
    }


def _load_empirical_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "tasks", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)]
    raise ValueError("Reward calibration rows must be a list or contain rows/tasks/results")


if __name__ == "__main__":
    raise SystemExit(main())
