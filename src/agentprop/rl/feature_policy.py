"""Graph-feature-conditioned policy-gradient routing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import cast

from agentprop.core import AgentGraph
from agentprop.ml.features import GraphFeatures, extract_graph_features
from agentprop.rl.env import AgentRoutingEnv, RoutingAction

_LAST_GRAPH: AgentGraph | None = None
_LAST_FEATURES: GraphFeatures | None = None


@dataclass(slots=True)
class FeaturePolicyConfig:
    """Training parameters for graph-feature-conditioned routing."""

    episodes: int = 100
    learning_rate: float = 0.05
    discount: float = 0.95
    epsilon: float = 0.1
    seed: int = 0
    max_steps: int = 20


@dataclass(slots=True)
class FeaturePolicyTrainingResult:
    """Summary of a graph-feature policy training run."""

    episodes: int
    episode_rewards: list[float]
    feature_count: int
    truncated_episodes: int


@dataclass(slots=True)
class GraphFeaturePolicy:
    """Linear policy over reusable graph/action features."""

    weights: list[float]
    feature_names: list[str] = field(default_factory=list)

    @classmethod
    def initialize(cls, feature_names: list[str]) -> GraphFeaturePolicy:
        """Create a zero-initialized feature policy."""

        return cls(weights=[0.0] * len(feature_names), feature_names=feature_names)

    def act(self, env: AgentRoutingEnv) -> str:
        """Choose the highest-scoring seed action for the current graph state."""

        actions = env.available_seed_actions()
        if not actions:
            return RoutingAction.STOP.value
        features = _seed_action_features(env, actions)
        return max(sorted(actions), key=lambda action: _dot(self.weights, features[action]))

    def to_dict(self) -> dict[str, object]:
        """Serialize feature weights and names."""

        return {
            "weights": self.weights,
            "feature_names": self.feature_names,
        }


@dataclass(frozen=True, slots=True)
class _TrajectoryStep:
    action: str
    action_features: dict[str, list[float]]
    reward: float


def train_feature_policy(
    env: AgentRoutingEnv,
    *,
    config: FeaturePolicyConfig | None = None,
) -> tuple[GraphFeaturePolicy, FeaturePolicyTrainingResult]:
    """Train a transferable seed-routing policy over graph features."""

    cfg = config or FeaturePolicyConfig()
    if cfg.max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not 0.0 <= cfg.epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    if cfg.discount < 0.0:
        raise ValueError("discount must be non-negative")

    rng = random.Random(cfg.seed)
    feature_names = _seed_action_feature_names(env)
    policy = GraphFeaturePolicy.initialize(feature_names)
    episode_rewards: list[float] = []
    truncated_episodes = 0

    for _ in range(cfg.episodes):
        env.reset()
        done = False
        total_reward = 0.0
        trajectory: list[_TrajectoryStep] = []

        for _step in range(cfg.max_steps):
            actions = env.available_seed_actions()
            if not actions:
                action = RoutingAction.STOP.value
                _, reward, done, _ = env.step(action)
                total_reward += reward
                break
            action_features = _seed_action_features(env, actions)
            action = _sample_action(policy, action_features, rng, epsilon=cfg.epsilon)
            _, reward, done, _ = env.step(action)
            total_reward += reward
            trajectory.append(
                _TrajectoryStep(
                    action=action,
                    action_features=action_features,
                    reward=reward,
                )
            )
            if done:
                break

        if not done:
            truncated_episodes += 1
        _update_policy(policy, trajectory, cfg)
        episode_rewards.append(total_reward)

    return policy, FeaturePolicyTrainingResult(
        episodes=cfg.episodes,
        episode_rewards=episode_rewards,
        feature_count=len(policy.weights),
        truncated_episodes=truncated_episodes,
    )


def _update_policy(
    policy: GraphFeaturePolicy,
    trajectory: list[_TrajectoryStep],
    cfg: FeaturePolicyConfig,
) -> None:
    returns = _discounted_returns([step.reward for step in trajectory], cfg.discount)
    weight_updates = [0.0] * len(policy.weights)
    for step, episode_return in zip(trajectory, returns, strict=True):
        probabilities = _softmax(policy, step.action_features)
        for action, probability in probabilities.items():
            direction = (1.0 if action == step.action else 0.0) - probability
            values = step.action_features[action]
            for index, value in enumerate(values):
                weight_updates[index] += cfg.learning_rate * episode_return * direction * value
    for index, update in enumerate(weight_updates):
        policy.weights[index] += update


def _sample_action(
    policy: GraphFeaturePolicy,
    action_features: dict[str, list[float]],
    rng: random.Random,
    *,
    epsilon: float,
) -> str:
    actions = sorted(action_features)
    if rng.random() < epsilon:
        return rng.choice(actions)
    probabilities = _softmax(policy, action_features)
    draw = rng.random()
    cumulative = 0.0
    for action in actions:
        cumulative += probabilities[action]
        if draw <= cumulative:
            return action
    return actions[-1]


def _softmax(
    policy: GraphFeaturePolicy,
    action_features: dict[str, list[float]],
) -> dict[str, float]:
    scores = {
        action: _dot(policy.weights, features)
        for action, features in action_features.items()
    }
    max_score = max(scores.values(), default=0.0)
    weights = {action: math.exp(score - max_score) for action, score in scores.items()}
    total = sum(weights.values()) or 1.0
    return {action: weight / total for action, weight in weights.items()}


def _seed_action_features(env: AgentRoutingEnv, actions: list[str]) -> dict[str, list[float]]:
    graph_features = _get_graph_features(env.graph)
    state = env.state
    remaining_ratio = state.remaining_budget / max(env.budget, 1)
    selected_ratio = len(state.selected_seeds) / max(env.budget, 1)
    current_coverage = max(0.0, min(1.0, state.coverage))
    return {
        action: [
            *graph_features.node_features[action],
            remaining_ratio,
            selected_ratio,
            current_coverage,
            1.0,
        ]
        for action in actions
    }


def _seed_action_feature_names(env: AgentRoutingEnv) -> list[str]:
    return [
        *_get_graph_features(env.graph).feature_names,
        "remaining_budget_ratio",
        "selected_seed_ratio",
        "current_coverage",
        "bias",
    ]


def _get_graph_features(graph: AgentGraph) -> GraphFeatures:
    global _LAST_FEATURES, _LAST_GRAPH
    if graph is not _LAST_GRAPH:
        _LAST_GRAPH = graph
        _LAST_FEATURES = extract_graph_features(graph)
    return cast(GraphFeatures, _LAST_FEATURES)


def _discounted_returns(rewards: list[float], discount: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = rewards[index] + discount * running_return
        returns[index] = running_return
    return returns


def _dot(weights: list[float], values: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, values, strict=True))
