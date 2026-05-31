"""Dataset builders for learned AgentProp policies."""

from __future__ import annotations

from dataclasses import dataclass

from agentprop.algorithms import (
    greedy_seed_selection,
    low_weight_edges,
    risk_aware_verifier_placement,
)
from agentprop.core import AgentGraph
from agentprop.ml.features import (
    EdgeFeatures,
    GraphFeatures,
    extract_edge_features,
    extract_graph_features,
)
from agentprop.propagation import IndependentCascade, PropagationModel


@dataclass(slots=True)
class SeedSelectionExample:
    """Supervised example for node-scoring seed policies."""

    features: GraphFeatures
    labels: dict[str, float]
    positive_seeds: list[str]
    budget: int
    neighbors: dict[str, list[str]]


@dataclass(slots=True)
class EdgePruningExample:
    """Supervised example for edge-pruning policies."""

    features: EdgeFeatures
    labels: dict[tuple[str, str], float]
    positive_edges: list[tuple[str, str]]


@dataclass(slots=True)
class VerifierPlacementExample:
    """Supervised example for verifier-placement policies."""

    features: GraphFeatures
    labels: dict[str, float]
    positive_verifiers: list[str]
    budget: int
    neighbors: dict[str, list[str]]


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
    neighbors = {
        node_id: sorted({*graph.predecessors(node_id), *graph.successors(node_id)})
        for node_id in features.node_features
    }
    return SeedSelectionExample(
        features=features,
        labels=labels,
        positive_seeds=positive_seeds,
        budget=budget,
        neighbors=neighbors,
    )


def build_edge_pruning_example(
    graph: AgentGraph,
    *,
    fraction: float = 0.2,
) -> EdgePruningExample:
    """Label low-weight edges as pruning candidates."""

    positive_edges = low_weight_edges(graph, fraction=fraction)
    positives = set(positive_edges)
    features = extract_edge_features(graph)
    labels = {
        edge_id: 1.0 if edge_id in positives else 0.0
        for edge_id in features.edge_features
    }
    return EdgePruningExample(
        features=features,
        labels=labels,
        positive_edges=positive_edges,
    )


def build_verifier_placement_example(
    graph: AgentGraph,
    *,
    budget: int,
) -> VerifierPlacementExample:
    """Label nodes selected by risk-aware verifier placement."""

    positive_verifiers = risk_aware_verifier_placement(graph, budget)
    positives = set(positive_verifiers)
    features = extract_graph_features(graph)
    labels = {
        node_id: 1.0 if node_id in positives else 0.0
        for node_id in features.node_features
    }
    neighbors = {
        node_id: sorted({*graph.predecessors(node_id), *graph.successors(node_id)})
        for node_id in features.node_features
    }
    return VerifierPlacementExample(
        features=features,
        labels=labels,
        positive_verifiers=positive_verifiers,
        budget=budget,
        neighbors=neighbors,
    )
