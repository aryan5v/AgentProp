"""Feature-conditioned propagation that transfers across workflows.

``LearnedPropagation`` memorizes per-edge probabilities from one workflow's
traces, so it cannot say anything about an edge (or graph) it has never seen.
This model instead fits a logistic function from *edge features* to activation
probability, so anything trained on one set of workflows produces calibrated
probabilities on structurally new graphs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from agentprop.core import AgentGraph
from agentprop.propagation.base import PropagationResult
from agentprop.propagation.independent_cascade import IndependentCascade

EdgeKey = tuple[str, str]


@dataclass(slots=True)
class FeatureCalibratedPropagation:
    """IC-style propagation with logistic edge probabilities over features.

    Train with ``fit`` on (graph, edge -> activation outcome) observations
    pooled from any number of workflows; ``simulate`` then works on graphs the
    model has never seen, because probabilities come from features rather than
    edge identity.
    """

    name = "feature-calibrated"

    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    seed: int | None = None
    learning_rate: float = 0.5
    epochs: int = 200
    l2: float = 1e-3

    def fit(
        self,
        observations: list[tuple[AgentGraph, dict[EdgeKey, bool]]],
    ) -> FeatureCalibratedPropagation:
        """Fit logistic weights from pooled edge-activation observations."""

        rows: list[tuple[dict[str, float], bool]] = []
        for graph, outcomes in observations:
            features = _edge_feature_rows(graph)
            for edge_key, activated in outcomes.items():
                if edge_key in features:
                    rows.append((features[edge_key], activated))
        if not rows:
            raise ValueError("fit requires at least one edge observation")
        names = sorted(rows[0][0])
        weights = dict.fromkeys(names, 0.0)
        bias = 0.0
        n = len(rows)
        for _ in range(self.epochs):
            gradient = dict.fromkeys(names, 0.0)
            bias_gradient = 0.0
            for feature_row, activated in rows:
                z = bias + sum(weights[name] * feature_row.get(name, 0.0) for name in names)
                error = _sigmoid(z) - (1.0 if activated else 0.0)
                for name in names:
                    gradient[name] += error * feature_row.get(name, 0.0)
                bias_gradient += error
            for name in names:
                weights[name] -= self.learning_rate * (
                    gradient[name] / n + self.l2 * weights[name]
                )
            bias -= self.learning_rate * bias_gradient / n
        self.weights = weights
        self.bias = bias
        return self

    def edge_probability(self, graph: AgentGraph, source: str, target: str) -> float:
        """Predicted activation probability for one edge of any graph."""

        if not self.weights:
            raise RuntimeError("model is not fitted; call fit() or load()")
        features = _edge_feature_rows(graph).get((source, target))
        if features is None:
            raise ValueError(f"unknown edge: {source} -> {target}")
        z = self.bias + sum(
            weight * features.get(name, 0.0) for name, weight in self.weights.items()
        )
        return _sigmoid(z)

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 100,
    ) -> PropagationResult:
        """Monte Carlo propagation using feature-predicted edge probabilities."""

        probabilities = {
            (edge.source, edge.target): self.edge_probability(graph, edge.source, edge.target)
            for edge in graph.edges()
        }
        shadow = _graph_with_probabilities(graph, probabilities)
        return IndependentCascade(seed=self.seed).simulate(shadow, seeds, trials=trials)

    def to_dict(self) -> dict[str, object]:
        """Serialize fitted parameters."""

        return {"weights": dict(self.weights), "bias": self.bias}

    def save(self, path: str | Path) -> Path:
        """Persist fitted parameters as JSON."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, path: str | Path, *, seed: int | None = None) -> FeatureCalibratedPropagation:
        """Load fitted parameters from JSON."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(seed=seed)
        model.weights = {str(k): float(v) for k, v in payload["weights"].items()}
        model.bias = float(payload["bias"])
        return model


def observations_from_trace_dicts(
    graph: AgentGraph,
    traces: list[dict[str, object]],
) -> dict[EdgeKey, bool]:
    """Convert trace rows with ``source``/``target``/``activated`` into outcomes."""

    outcomes: dict[EdgeKey, bool] = {}
    for row in traces:
        source = row.get("source")
        target = row.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        outcomes[(source, target)] = bool(row.get("activated", True))
    return outcomes


def _edge_feature_rows(graph: AgentGraph) -> dict[EdgeKey, dict[str, float]]:
    # Imported lazily: agentprop.ml.features pulls in agentprop.algorithms,
    # which imports the propagation package this module belongs to.
    from agentprop.ml.features import extract_edge_features

    extracted = extract_edge_features(graph)
    return {
        edge_key: dict(zip(extracted.feature_names, values, strict=True))
        for edge_key, values in extracted.edge_features.items()
    }


def _graph_with_probabilities(
    graph: AgentGraph,
    probabilities: dict[EdgeKey, float],
) -> AgentGraph:
    shadow = AgentGraph()
    for node in graph.nodes():
        shadow.add_node(
            node.id,
            type=node.type,
            name=node.name,
            role=node.role,
            token_cost=node.token_cost,
            latency=node.latency,
            reliability=node.reliability,
            error_rate=node.error_rate,
        )
    for edge in graph.edges():
        shadow.add_edge(
            edge.source,
            edge.target,
            message_cost=edge.message_cost,
            latency=edge.latency,
            # IC multiplies activation_probability * reliability * relevance, so
            # both are neutralized to make the predicted probability authoritative.
            relevance=1.0,
            reliability=1.0,
            activation_probability=probabilities[(edge.source, edge.target)],
            weight=edge.weight,
        )
    return shadow


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    expz = math.exp(z)
    return expz / (1.0 + expz)

