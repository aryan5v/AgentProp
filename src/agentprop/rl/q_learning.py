"""Tabular Q-learning for sequential routing.

Pedagogical baseline on small tabular state spaces. The production
routing path is ContextualThompsonSamplingPolicy (rl/contextual_thompson.py)
plus off-policy evaluation (rl/ope.py); see docs/reinforcement_learning.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from agentprop.rl.env import AgentRoutingEnv, RoutingAction, RoutingState


@dataclass(slots=True)
class QLearningConfig:
    """Training parameters for tabular routing Q-learning."""

    episodes: int = 100
    learning_rate: float = 0.3
    discount: float = 0.9
    epsilon: float = 0.2
    seed: int = 0
    expanded_actions: bool = False


@dataclass(slots=True)
class QLearningTrainingResult:
    """Summary of a Q-learning run."""

    episodes: int
    episode_rewards: list[float]
    q_value_count: int


@dataclass(slots=True)
class TabularQPolicy:
    """Policy backed by learned state-action values."""

    q_values: dict[tuple[str, str], float] = field(default_factory=dict)
    expanded_actions: bool = False

    def act(self, env: AgentRoutingEnv) -> str:
        """Choose the highest-value action for the current environment state."""

        return self._best_action(env.state, _routing_actions(env, expanded=self.expanded_actions))

    def to_dict(self) -> dict[str, float]:
        """Serialize q-values with stable string keys."""

        return {
            f"{state_key} -> {action}": value
            for (state_key, action), value in sorted(self.q_values.items())
        }

    def _best_action(self, state: RoutingState, actions: list[str]) -> str:
        if not actions:
            return RoutingAction.STOP.value
        state_key = _state_key(state)
        return max(
            sorted(actions),
            key=lambda action: self.q_values.get((state_key, action), 0.0),
        )


def train_q_policy(
    env: AgentRoutingEnv,
    *,
    config: QLearningConfig | None = None,
) -> tuple[TabularQPolicy, QLearningTrainingResult]:
    """Train a routing policy over complete selection episodes."""

    cfg = config or QLearningConfig()
    rng = random.Random(cfg.seed)
    policy = TabularQPolicy(expanded_actions=cfg.expanded_actions)
    episode_rewards: list[float] = []

    for _ in range(cfg.episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            actions = _routing_actions(env, expanded=cfg.expanded_actions)
            action = _epsilon_greedy(policy, state, actions, cfg.epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            state_key = _state_key(state)
            old_value = policy.q_values.get((state_key, action), 0.0)
            next_actions = _routing_actions(env, expanded=cfg.expanded_actions)
            next_value = 0.0 if done else _max_q(policy, next_state, next_actions)
            target = reward + cfg.discount * next_value
            policy.q_values[(state_key, action)] = old_value + cfg.learning_rate * (
                target - old_value
            )
            state = next_state

        episode_rewards.append(total_reward)

    return policy, QLearningTrainingResult(
        episodes=cfg.episodes,
        episode_rewards=episode_rewards,
        q_value_count=len(policy.q_values),
    )


def _epsilon_greedy(
    policy: TabularQPolicy,
    state: RoutingState,
    actions: list[str],
    epsilon: float,
    rng: random.Random,
) -> str:
    if not actions:
        return RoutingAction.STOP.value
    if rng.random() < epsilon:
        return rng.choice(actions)
    return policy._best_action(state, actions)


def _max_q(policy: TabularQPolicy, state: RoutingState, actions: list[str]) -> float:
    if not actions:
        return 0.0
    state_key = _state_key(state)
    return max(policy.q_values.get((state_key, action), 0.0) for action in actions)


def _routing_actions(env: AgentRoutingEnv, *, expanded: bool = False) -> list[str]:
    if expanded:
        return env.available_control_actions()
    seed_actions = env.available_seed_actions()
    return seed_actions if seed_actions else [RoutingAction.STOP.value]


def _state_key(state: RoutingState) -> str:
    seeds = ",".join(state.selected_seeds) if state.selected_seeds else "<none>"
    return f"seeds={seeds};remaining={state.remaining_budget}"
