"""Verifier-placement heuristics."""

from __future__ import annotations

import networkx as nx

from agentprop.core import AgentGraph


def risk_aware_verifier_placement(graph: AgentGraph, k: int) -> list[str]:
    """Rank nodes by error risk, graph centrality, and downstream influence."""

    if k < 1:
        raise ValueError("k must be at least 1")
    nx_graph = graph.to_networkx()
    betweenness = nx.betweenness_centrality(nx_graph, weight="weight") if graph.node_count else {}
    scores: dict[str, float] = {}

    for node in graph.nodes():
        out_degree = nx_graph.out_degree(node.id)
        scores[node.id] = (
            0.50 * node.error_rate
            + 0.25 * (1.0 - node.reliability)
            + 0.15 * float(betweenness.get(node.id, 0.0))
            + 0.10 * float(out_degree)
        )

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [node_id for node_id, _ in ranked[:k]]
