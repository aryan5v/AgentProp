"""Feature extraction for learned graph policies."""

from __future__ import annotations

from dataclasses import dataclass

from agentprop.algorithms.seed_selection import centrality_scores
from agentprop.core import AgentGraph


@dataclass(slots=True)
class GraphFeatures:
    """Feature matrix keyed by node id."""

    node_features: dict[str, list[float]]
    feature_names: list[str]


@dataclass(slots=True)
class EdgeFeatures:
    """Feature matrix keyed by directed edge."""

    edge_features: dict[tuple[str, str], list[float]]
    feature_names: list[str]


def extract_graph_features(graph: AgentGraph) -> GraphFeatures:
    """Extract normalized node features for seed/pruning/verifier policies."""

    centrality = centrality_scores(graph)
    max_token_cost = max((node.token_cost for node in graph.nodes()), default=1.0) or 1.0
    max_latency = max((node.latency for node in graph.nodes()), default=1.0) or 1.0
    max_degree = max((centrality["degree"].values()), default=1.0) or 1.0

    feature_names = [
        "token_cost_norm",
        "latency_norm",
        "reliability",
        "error_rate",
        "degree_norm",
        "pagerank",
        "betweenness",
        "out_degree_norm",
        "in_degree_norm",
    ]
    nx_graph = graph.to_networkx()
    node_features: dict[str, list[float]] = {}

    for node in graph.nodes():
        node_features[node.id] = [
            node.token_cost / max_token_cost,
            node.latency / max_latency,
            node.reliability,
            node.error_rate,
            centrality["degree"].get(node.id, 0.0) / max_degree,
            centrality["pagerank"].get(node.id, 0.0),
            centrality["betweenness"].get(node.id, 0.0),
            float(nx_graph.out_degree(node.id)) / max_degree,
            float(nx_graph.in_degree(node.id)) / max_degree,
        ]

    return GraphFeatures(node_features=node_features, feature_names=feature_names)


def extract_edge_features(graph: AgentGraph) -> EdgeFeatures:
    """Extract normalized edge features for pruning policies."""

    max_message_cost = max((edge.message_cost for edge in graph.edges()), default=1.0) or 1.0
    max_latency = max((edge.latency for edge in graph.edges()), default=1.0) or 1.0
    max_weight = max((edge.weight for edge in graph.edges()), default=1.0) or 1.0
    nx_graph = graph.to_networkx()
    feature_names = [
        "message_cost_norm",
        "latency_norm",
        "relevance",
        "reliability",
        "activation_probability",
        "dependency_strength",
        "weight_norm",
        "source_out_degree",
        "target_in_degree",
    ]
    edge_features: dict[tuple[str, str], list[float]] = {}
    for edge in graph.edges():
        edge_features[(edge.source, edge.target)] = [
            edge.message_cost / max_message_cost,
            edge.latency / max_latency,
            edge.relevance,
            edge.reliability,
            edge.activation_probability,
            edge.dependency_strength,
            edge.weight / max_weight,
            float(nx_graph.out_degree(edge.source)) / max(graph.edge_count, 1),
            float(nx_graph.in_degree(edge.target)) / max(graph.edge_count, 1),
        ]
    return EdgeFeatures(edge_features=edge_features, feature_names=feature_names)
