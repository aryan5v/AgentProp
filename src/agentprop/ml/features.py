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
