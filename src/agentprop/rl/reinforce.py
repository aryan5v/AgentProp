"""Dependency-light REINFORCE training for sequential routing.

Pedagogical baseline on small tabular state spaces. The production
routing path is ContextualThompsonSamplingPolicy (rl/contextual_thompson.py)
plus off-policy evaluation (rl/ope.py); see docs/reinforcement_learning.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from agentprop.rl.env import AgentRoutingEnv, RoutingAction, RoutingState


@dataclass(slots=True)
class ReinforceConfig:
    """Training parameters for tabular REINFORCE routing."""

    episodes: int = 100
    learning_rate: float = 0.05
    discount: float = 0.95
    seed: int = 0
    expanded_actions: bool = False
    max_steps: int = 20


@dataclass(slots=True)
class ReinforceTrainingResult:
    """Summary of a REINFORCE run."""

    episodes: int
    episode_rewards: list[float]
    preference_count: int
    truncated_episodes: int


@dataclass(slots=True)
class ReinforcePolicy:
    """Policy backed by learned state-action preferences."""

    preferences: dict[tuple[str, str], float] = field(default_factory=dict)
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


@dataclass(frozen=True, slots=True)
class _TrajectoryStep:
    state_key: str
    action: str
    actions: tuple[str, ...]
    reward: float


def train_reinforce_policy(
    env: AgentRoutingEnv,
    *,
    config: ReinforceConfig | None = None,
) -> tuple[ReinforcePolicy, ReinforceTrainingResult]:
    """Train a routing policy with the episodic REINFORCE gradient estimator."""

    cfg = config or ReinforceConfig()
    if cfg.max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    rng = random.Random(cfg.seed)
    policy = ReinforcePolicy(expanded_actions=cfg.expanded_actions)
    episode_rewards: list[float] = []
    truncated_episodes = 0

    for _ in range(cfg.episodes):
        env.reset()
        done = False
        total_reward = 0.0
        trajectory: list[_TrajectoryStep] = []

        for _step in range(cfg.max_steps):
            actions = _routing_actions(env, expanded=cfg.expanded_actions)
            if not actions:
                break
            state_key = _state_key(env.state)
            action = _sample_action(policy, state_key, actions, rng)
            _, reward, done, _ = env.step(action)
            trajectory.append(
                _TrajectoryStep(
                    state_key=state_key,
                    action=action,
                    actions=tuple(actions),
                    reward=reward,
                )
            )
            total_reward += reward
            if done:
                break

        if not done:
            truncated_episodes += 1
        _update_preferences(policy, trajectory, cfg)
        episode_rewards.append(total_reward)

    return policy, ReinforceTrainingResult(
        episodes=cfg.episodes,
        episode_rewards=episode_rewards,
        preference_count=len(policy.preferences),
        truncated_episodes=truncated_episodes,
    )


def _sample_action(
    policy: ReinforcePolicy,
    state_key: str,
    actions: list[str],
    rng: random.Random,
) -> str:
    probabilities = _softmax(policy, state_key, actions)
    draw = rng.random()
    cumulative = 0.0
    for action, probability in zip(actions, probabilities, strict=True):
        cumulative += probability
        if draw <= cumulative:
            return action
    return actions[-1]


def _update_preferences(
    policy: ReinforcePolicy,
    trajectory: list[_TrajectoryStep],
    cfg: ReinforceConfig,
) -> None:
    returns = _discounted_returns([step.reward for step in trajectory], cfg.discount)
    for step, episode_return in zip(trajectory, returns, strict=True):
        probabilities = _softmax(policy, step.state_key, list(step.actions))
        for action, probability in zip(step.actions, probabilities, strict=True):
            gradient = (1.0 if action == step.action else 0.0) - probability
            key = (step.state_key, action)
            old_value = policy.preferences.get(key, 0.0)
            policy.preferences[key] = old_value + cfg.learning_rate * episode_return * gradient


def _discounted_returns(rewards: list[float], discount: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = rewards[index] + discount * running_return
        returns[index] = running_return
    return returns


def _softmax(policy: ReinforcePolicy, state_key: str, actions: list[str]) -> list[float]:
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
