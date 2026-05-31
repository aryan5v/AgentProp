"""Observability and verifier-placement metrics."""

from __future__ import annotations

import networkx as nx

from agentprop.core import AgentGraph


def observability_scores(graph: AgentGraph) -> dict[str, float]:
    """Score nodes by usefulness for observing failures and corrections."""

    nx_graph = graph.to_networkx()
    if graph.node_count == 0:
        return {}

    betweenness = nx.betweenness_centrality(nx_graph, weight="weight")
    scores: dict[str, float] = {}
    max_reachable = max(graph.node_count - 1, 1)

    for node in graph.nodes():
        descendants = nx.descendants(nx_graph, node.id)
        ancestors = nx.ancestors(nx_graph, node.id)
        reach_score = len(descendants) / max_reachable
        attribution_score = len(ancestors) / max_reachable
        reliability_risk = 1.0 - node.reliability + node.error_rate
        scores[node.id] = (
            0.35 * reach_score
            + 0.25 * attribution_score
            + 0.25 * float(betweenness.get(node.id, 0.0))
            + 0.15 * reliability_risk
        )

    return scores


def observability_coverage(graph: AgentGraph, observers: list[str]) -> float:
    """Return fraction of nodes observable by the selected observer nodes."""

    nx_graph = graph.to_networkx()
    observed = set(observers)
    for observer in observers:
        if observer in nx_graph:
            observed.update(str(node_id) for node_id in nx.ancestors(nx_graph, observer))
            observed.update(str(node_id) for node_id in nx.descendants(nx_graph, observer))
    return len(observed) / max(graph.node_count, 1)


def verifier_observability_placement(graph: AgentGraph, k: int) -> list[str]:
    """Choose verifier/logging nodes by observability score."""

    if k < 1:
        raise ValueError("k must be at least 1")
    scores = observability_scores(graph)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [node_id for node_id, _ in ranked[:k]]
