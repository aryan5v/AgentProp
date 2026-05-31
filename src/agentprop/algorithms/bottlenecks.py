"""Workflow bottleneck detection."""

from __future__ import annotations

import networkx as nx

from agentprop.core import AgentGraph


def bottleneck_nodes(graph: AgentGraph, *, limit: int = 5) -> list[tuple[str, float]]:
    """Return nodes that look structurally important or operationally risky."""

    nx_graph = graph.to_networkx()
    if not nx_graph:
        return []

    betweenness = nx.betweenness_centrality(nx_graph, weight="weight")
    scores: dict[str, float] = {}
    for node in graph.nodes():
        scores[node.id] = (
            float(betweenness.get(node.id, 0.0))
            + 0.05 * float(nx_graph.out_degree(node.id))
            + 0.05 * float(nx_graph.in_degree(node.id))
            + 0.25 * (1.0 - node.reliability)
            + 0.25 * node.error_rate
        )

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
