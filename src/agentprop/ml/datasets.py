"""Dataset builders for learned AgentProp policies."""

from __future__ import annotations

from dataclasses import dataclass

from agentprop.algorithms import greedy_seed_selection
from agentprop.core import AgentGraph
from agentprop.ml.features import GraphFeatures, extract_graph_features
from agentprop.propagation import IndependentCascade, PropagationModel


@dataclass(slots=True)
class SeedSelectionExample:
    """Supervised example for node-scoring seed policies."""

    features: GraphFeatures
    labels: dict[str, float]
    positive_seeds: list[str]
    budget: int


def build_seed_selection_example(
    graph: AgentGraph,
    *,
    budget: int,
    propagation_model: PropagationModel | None = None,
    trials: int = 50,
) -> SeedSelectionExample:
    """Label nodes by whether greedy influence maximization selected them."""

    model = propagation_model or IndependentCascade(seed=0)
    positive_seeds = greedy_seed_selection(
        graph,
        budget,
        propagation_model=model,
        trials=trials,
    )
    positives = set(positive_seeds)
    features = extract_graph_features(graph)
    labels = {node_id: 1.0 if node_id in positives else 0.0 for node_id in features.node_features}
    return SeedSelectionExample(
        features=features,
        labels=labels,
        positive_seeds=positive_seeds,
        budget=budget,
    )
