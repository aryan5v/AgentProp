"""Dependency-light PPO-style training for sequential routing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from agentprop.rl.env import AgentRoutingEnv, RoutingAction, RoutingState


@dataclass(slots=True)
class PPOConfig:
    """Training parameters for tabular clipped policy optimization."""

    episodes: int = 100
    learning_rate: float = 0.05
    value_learning_rate: float = 0.1
    discount: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 2
    seed: int = 0
    expanded_actions: bool = False
    max_steps: int = 20


@dataclass(slots=True)
class PPOTrainingResult:
    """Summary of a PPO-style training run."""

    episodes: int
    episode_rewards: list[float]
    preference_count: int
    value_count: int
    truncated_episodes: int


@dataclass(slots=True)
class PPOPolicy:
    """Policy backed by clipped policy-optimization preferences."""

    preferences: dict[tuple[str, str], float] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    expanded_actions: bool = False

    def act(self, env: AgentRoutingEnv) -> str:
        """Choose the highest-preference action for the current environment state."""

        actions = _routing_actions(env, expanded=self.expanded_actions)
        if not actions:
            return RoutingAction.STOP.value
        state_key = _state_key(env.state)
        return max(
            sorted(actions),
            key=lambda action: self.preferences.get((state_key, action), 0.0),
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize learned preferences with stable string keys."""

        return {
            f"{state_key} -> {action}": value
            for (state_key, action), value in sorted(self.preferences.items())
        }

    def values_to_dict(self) -> dict[str, float]:
        """Serialize learned state values."""

        return dict(sorted(self.values.items()))


@dataclass(frozen=True, slots=True)
class _TrajectoryStep:
    state_key: str
    action: str
    actions: tuple[str, ...]
    reward: float
    old_probability: float


def train_ppo_policy(
    env: AgentRoutingEnv,
    *,
    config: PPOConfig | None = None,
) -> tuple[PPOPolicy, PPOTrainingResult]:
    """Train a routing policy with a tabular PPO-style clipped objective."""

    cfg = config or PPOConfig()
    if cfg.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if cfg.update_epochs < 1:
        raise ValueError("update_epochs must be at least 1")
    if cfg.clip_epsilon < 0:
        raise ValueError("clip_epsilon must be non-negative")

    rng = random.Random(cfg.seed)
    policy = PPOPolicy(expanded_actions=cfg.expanded_actions)
    episode_rewards: list[float] = []
    truncated_episodes = 0

    for _ in range(cfg.episodes):
        trajectory, total_reward, done = _collect_trajectory(env, policy, cfg, rng)
        if not done:
            truncated_episodes += 1
        _update_policy(policy, trajectory, cfg)
        episode_rewards.append(total_reward)

    return policy, PPOTrainingResult(
        episodes=cfg.episodes,
        episode_rewards=episode_rewards,
        preference_count=len(policy.preferences),
        value_count=len(policy.values),
        truncated_episodes=truncated_episodes,
    )


def _collect_trajectory(
    env: AgentRoutingEnv,
    policy: PPOPolicy,
    cfg: PPOConfig,
    rng: random.Random,
) -> tuple[list[_TrajectoryStep], float, bool]:
    env.reset()
    done = False
    total_reward = 0.0
    trajectory: list[_TrajectoryStep] = []

    for _step in range(cfg.max_steps):
        actions = _routing_actions(env, expanded=cfg.expanded_actions)
        if not actions:
            break
        state_key = _state_key(env.state)
        probabilities = _softmax(policy, state_key, actions)
        action = _sample_action(actions, probabilities, rng)
        old_probability = probabilities[actions.index(action)]
        _, reward, done, _ = env.step(action)
        trajectory.append(
            _TrajectoryStep(
                state_key=state_key,
                action=action,
                actions=tuple(actions),
                reward=reward,
                old_probability=old_probability,
            )
        )
        total_reward += reward
        if done:
            break

    return trajectory, total_reward, done


def _update_policy(
    policy: PPOPolicy,
    trajectory: list[_TrajectoryStep],
    cfg: PPOConfig,
) -> None:
    returns = _discounted_returns([step.reward for step in trajectory], cfg.discount)
    for step, episode_return in zip(trajectory, returns, strict=True):
        old_value = policy.values.get(step.state_key, 0.0)
        policy.values[step.state_key] = old_value + cfg.value_learning_rate * (
            episode_return - old_value
        )

    for _ in range(cfg.update_epochs):
        for step, episode_return in zip(trajectory, returns, strict=True):
            baseline = policy.values.get(step.state_key, 0.0)
            advantage = episode_return - baseline
            if advantage == 0.0:
                continue
            actions = list(step.actions)
            probabilities = _softmax(policy, step.state_key, actions)
            action_index = actions.index(step.action)
            new_probability = probabilities[action_index]
            ratio = new_probability / max(step.old_probability, 1e-12)
            if _is_clipped(ratio, advantage, cfg.clip_epsilon):
                continue
            gradient_scale = advantage * ratio
            for action, probability in zip(actions, probabilities, strict=True):
                gradient = (1.0 if action == step.action else 0.0) - probability
                key = (step.state_key, action)
                old_preference = policy.preferences.get(key, 0.0)
                policy.preferences[key] = (
                    old_preference + cfg.learning_rate * gradient_scale * gradient
                )


def _is_clipped(ratio: float, advantage: float, clip_epsilon: float) -> bool:
    if advantage > 0:
        return ratio > 1.0 + clip_epsilon
    return ratio < 1.0 - clip_epsilon


def _sample_action(
    actions: list[str],
    probabilities: list[float],
    rng: random.Random,
) -> str:
    draw = rng.random()
    cumulative = 0.0
    for action, probability in zip(actions, probabilities, strict=True):
        cumulative += probability
        if draw <= cumulative:
            return action
    return actions[-1]


def _discounted_returns(rewards: list[float], discount: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = rewards[index] + discount * running_return
        returns[index] = running_return
    return returns


def _softmax(policy: PPOPolicy, state_key: str, actions: list[str]) -> list[float]:
    if not actions:
        return []
    values = [policy.preferences.get((state_key, action), 0.0) for action in actions]
    max_value = max(values)
    weights = [math.exp(value - max_value) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def _routing_actions(env: AgentRoutingEnv, *, expanded: bool = False) -> list[str]:
    if expanded:
        return env.available_control_actions()
    seed_actions = env.available_seed_actions()
    return seed_actions if seed_actions else [RoutingAction.STOP.value]


def _state_key(state: RoutingState) -> str:
    seeds = ",".join(state.selected_seeds) if state.selected_seeds else "<none>"
    verifiers = (
        ",".join(state.activated_verifiers) if state.activated_verifiers else "<none>"
    )
    pruned = (
        ",".join(f"{source}->{target}" for source, target in state.pruned_edges)
        if state.pruned_edges
        else "<none>"
    )
    return (
        f"seeds={seeds};verifiers={verifiers};pruned={pruned};"
        f"remaining={state.remaining_budget}"
    )
