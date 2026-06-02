"""Dependency-light learned node scorers."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from agentprop.ml.datasets import (
    EdgePruningExample,
    EmpiricalEdgePruningExample,
    EmpiricalRoutingExample,
    EmpiricalVerifierPlacementExample,
    SeedRankingExample,
    SeedSelectionExample,
    VerifierPlacementExample,
)
from agentprop.ml.features import EdgeFeatures, GraphFeatures


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
        examples: list[
            SeedSelectionExample
            | VerifierPlacementExample
            | EmpiricalRoutingExample
            | EmpiricalVerifierPlacementExample
        ],
        *,
        epochs: int = 200,
        learning_rate: float = 0.1,
        l2_penalty: float = 0.0,
    ) -> None:
        """Train with logistic loss using simple gradient descent."""

        for _ in range(epochs):
            for example in examples:
                sample_weight = _example_weight(example)
                for node_id, values in example.features.node_features.items():
                    label = example.labels[node_id]
                    prediction = _sigmoid(_dot(self.weights, values) + self.bias)
                    error = sample_weight * (prediction - label)
                    for index, value in enumerate(values):
                        gradient = error * value + l2_penalty * self.weights[index]
                        self.weights[index] -= learning_rate * gradient
                    self.bias -= learning_rate * error


@dataclass(slots=True)
class MLPNodeScorer:
    """Small dependency-light one-hidden-layer MLP node scorer."""

    input_weights: list[list[float]]
    output_weights: list[float]
    hidden_bias: list[float]
    output_bias: float = 0.0

    @classmethod
    def initialize(cls, feature_count: int, *, hidden_dim: int = 8) -> MLPNodeScorer:
        """Create a deterministic small MLP scorer."""

        input_weights = [
            [0.01 * ((row + column) % 5 - 2) for column in range(feature_count)]
            for row in range(hidden_dim)
        ]
        output_weights = [0.01 * ((index % 5) - 2) for index in range(hidden_dim)]
        return cls(
            input_weights=input_weights,
            output_weights=output_weights,
            hidden_bias=[0.0] * hidden_dim,
        )

    def score_nodes(self, features: GraphFeatures) -> dict[str, float]:
        """Return probability-like node scores."""

        return {
            node_id: _sigmoid(_dot(self.output_weights, self._hidden(values)) + self.output_bias)
            for node_id, values in features.node_features.items()
        }

    def train(
        self,
        examples: list[
            SeedSelectionExample
            | VerifierPlacementExample
            | EmpiricalRoutingExample
            | EmpiricalVerifierPlacementExample
        ],
        *,
        epochs: int = 200,
        learning_rate: float = 0.05,
        l2_penalty: float = 0.0,
    ) -> None:
        """Train both layers with logistic loss and backpropagation."""

        for _ in range(epochs):
            for example in examples:
                sample_weight = _example_weight(example)
                for node_id, values in example.features.node_features.items():
                    label = example.labels[node_id]
                    hidden = self._hidden(values)
                    prediction = _sigmoid(_dot(self.output_weights, hidden) + self.output_bias)
                    error = sample_weight * (prediction - label)
                    previous_output_weights = self.output_weights.copy()
                    for index, value in enumerate(hidden):
                        gradient = error * value + l2_penalty * self.output_weights[index]
                        self.output_weights[index] -= learning_rate * gradient
                    self.output_bias -= learning_rate * error
                    for hidden_index, hidden_value in enumerate(hidden):
                        hidden_gradient = (
                            error
                            * previous_output_weights[hidden_index]
                            * _relu_derivative(hidden_value)
                        )
                        for input_index, value in enumerate(values):
                            gradient = (
                                hidden_gradient * value
                                + l2_penalty * self.input_weights[hidden_index][input_index]
                            )
                            self.input_weights[hidden_index][input_index] -= (
                                learning_rate * gradient
                            )
                        self.hidden_bias[hidden_index] -= learning_rate * hidden_gradient

    def _hidden(self, values: list[float]) -> list[float]:
        return [
            _relu(_dot(weights, values) + bias)
            for weights, bias in zip(self.input_weights, self.hidden_bias, strict=True)
        ]


@dataclass(slots=True)
class PairwiseNodeRanker:
    """Linear node ranker trained from pairwise seed preferences."""

    weights: list[float]

    @classmethod
    def initialize(cls, feature_count: int) -> PairwiseNodeRanker:
        """Create a zero-initialized pairwise ranker."""

        return cls(weights=[0.0] * feature_count)

    def score_nodes(self, features: GraphFeatures) -> dict[str, float]:
        """Return unbounded node ranking scores."""

        return {
            node_id: _dot(self.weights, values)
            for node_id, values in features.node_features.items()
        }

    def train(
        self,
        examples: list[SeedRankingExample],
        *,
        epochs: int = 200,
        learning_rate: float = 0.05,
        l2_penalty: float = 0.0,
    ) -> None:
        """Train with a logistic pairwise ranking loss."""

        for _ in range(epochs):
            for example in examples:
                for winner, loser in example.preference_pairs:
                    winner_values = example.features.node_features[winner]
                    loser_values = example.features.node_features[loser]
                    difference = _subtract(winner_values, loser_values)
                    probability = _sigmoid(_dot(self.weights, difference))
                    gradient_scale = 1.0 - probability
                    for index, value in enumerate(difference):
                        gradient = gradient_scale * value - l2_penalty * self.weights[index]
                        self.weights[index] += learning_rate * gradient


@dataclass(slots=True)
class LinearNodeRegressor:
    """Linear node scorer trained to predict marginal seed utility."""

    weights: list[float]
    bias: float = 0.0

    @classmethod
    def initialize(cls, feature_count: int) -> LinearNodeRegressor:
        """Create a zero-initialized regressor."""

        return cls(weights=[0.0] * feature_count)

    def score_nodes(self, features: GraphFeatures) -> dict[str, float]:
        """Return predicted marginal utility for each node."""

        return {
            node_id: _dot(self.weights, values) + self.bias
            for node_id, values in features.node_features.items()
        }

    def train(
        self,
        examples: list[SeedRankingExample],
        *,
        epochs: int = 200,
        learning_rate: float = 0.05,
        l2_penalty: float = 0.0,
    ) -> None:
        """Train with squared error on marginal-utility targets."""

        for _ in range(epochs):
            for example in examples:
                for node_id in example.seed_candidates:
                    values = example.features.node_features[node_id]
                    target = example.utility_targets[node_id]
                    prediction = _dot(self.weights, values) + self.bias
                    error = prediction - target
                    for index, value in enumerate(values):
                        gradient = error * value + l2_penalty * self.weights[index]
                        self.weights[index] -= learning_rate * gradient
                    self.bias -= learning_rate * error


@dataclass(slots=True)
class LinearEdgeScorer:
    """Logistic edge scorer for pruning policies."""

    weights: list[float]
    bias: float = 0.0

    @classmethod
    def initialize(cls, feature_count: int) -> LinearEdgeScorer:
        """Create a zero-initialized edge scorer."""

        return cls(weights=[0.0] * feature_count)

    def score_edges(self, features: EdgeFeatures) -> dict[tuple[str, str], float]:
        """Return probability-like edge scores."""

        return {
            edge_id: _sigmoid(_dot(self.weights, values) + self.bias)
            for edge_id, values in features.edge_features.items()
        }

    def train(
        self,
        examples: list[EdgePruningExample | EmpiricalEdgePruningExample],
        *,
        epochs: int = 200,
        learning_rate: float = 0.1,
        l2_penalty: float = 0.0,
    ) -> None:
        """Train from examples with edge features and labels."""

        for _ in range(epochs):
            for example in examples:
                sample_weight = _example_weight(example)
                features = example.features
                labels = example.labels
                for edge_id, values in features.edge_features.items():
                    label = labels[edge_id]
                    prediction = _sigmoid(_dot(self.weights, values) + self.bias)
                    error = sample_weight * (prediction - label)
                    for index, value in enumerate(values):
                        gradient = error * value + l2_penalty * self.weights[index]
                        self.weights[index] -= learning_rate * gradient
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


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [left_value - right_value for left_value, right_value in zip(left, right, strict=True)]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def _relu(value: float) -> float:
    return max(value, 0.0)


def _relu_derivative(hidden_value: float) -> float:
    return 1.0 if hidden_value > 0.0 else 0.0


def _example_weight(example: object) -> float:
    weight = getattr(example, "sample_weight", 1.0)
    if isinstance(weight, int | float):
        return max(0.0, float(weight))
    return 1.0
