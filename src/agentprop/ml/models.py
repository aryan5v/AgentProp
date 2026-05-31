"""Dependency-light learned node scorers."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from agentprop.ml.datasets import SeedSelectionExample
from agentprop.ml.features import GraphFeatures


@dataclass(slots=True)
class LinearNodeScorer:
    """Tiny logistic node scorer for imitation-learning baselines."""

    weights: list[float]
    bias: float = 0.0

    @classmethod
    def initialize(cls, feature_count: int) -> LinearNodeScorer:
        """Create a zero-initialized scorer."""

        return cls(weights=[0.0] * feature_count)

    def score_nodes(self, features: GraphFeatures) -> dict[str, float]:
        """Return probability-like node scores."""

        return {
            node_id: _sigmoid(_dot(self.weights, values) + self.bias)
            for node_id, values in features.node_features.items()
        }

    def train(
        self,
        examples: list[SeedSelectionExample],
        *,
        epochs: int = 200,
        learning_rate: float = 0.1,
    ) -> None:
        """Train with logistic loss using simple gradient descent."""

        for _ in range(epochs):
            for example in examples:
                for node_id, values in example.features.node_features.items():
                    label = example.labels[node_id]
                    prediction = _sigmoid(_dot(self.weights, values) + self.bias)
                    error = prediction - label
                    for index, value in enumerate(values):
                        self.weights[index] -= learning_rate * error * value
                    self.bias -= learning_rate * error


@dataclass(slots=True)
class MessagePassingNodeScorer:
    """Simple graph-neighborhood scorer that mimics one GNN message-passing layer."""

    base_scorer: LinearNodeScorer
    neighbor_weight: float = 0.35

    def score_nodes(
        self,
        features: GraphFeatures,
        neighbors: dict[str, list[str]],
    ) -> dict[str, float]:
        """Blend local node scores with neighboring scores."""

        base_scores = self.base_scorer.score_nodes(features)
        smoothed: dict[str, float] = {}
        for node_id, score in base_scores.items():
            node_neighbors = neighbors.get(node_id, [])
            if not node_neighbors:
                smoothed[node_id] = score
                continue
            neighbor_score = sum(base_scores.get(neighbor, 0.0) for neighbor in node_neighbors)
            neighbor_score /= len(node_neighbors)
            smoothed[node_id] = (1.0 - self.neighbor_weight) * score
            smoothed[node_id] += self.neighbor_weight * neighbor_score
        return smoothed


def _dot(weights: list[float], values: list[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, values, strict=True))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)
