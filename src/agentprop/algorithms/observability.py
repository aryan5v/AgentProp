"""Observability and verifier-placement metrics."""

from __future__ import annotations

import networkx as nx

from agentprop.core import AgentGraph


def observability_scores(graph: AgentGraph) -> dict[str, float]:
    """Score nodes by usefulness for observing failures and corrections.

    Now uses the AgentGraph centrality + reachability closure cache
    (phase1-centrality-cache) to avoid repeated to_networkx + betweenness
    + ancestor/descendant walks on every call.
    """

    if graph.node_count == 0:
        return {}

    betweenness = graph.get_betweenness_centrality()
    max_reachable = max(graph.node_count - 1, 1)

    scores: dict[str, float] = {}
    for node in graph.nodes():
        nid = node.id
        descendants = graph.get_descendants(nid)
        ancestors = graph.get_ancestors(nid)
        reach_score = len(descendants) / max_reachable
        attribution_score = len(ancestors) / max_reachable
        reliability_risk = 1.0 - node.reliability + node.error_rate
        scores[nid] = (
            0.35 * reach_score
            + 0.25 * attribution_score
            + 0.25 * float(betweenness.get(nid, 0.0))
            + 0.15 * reliability_risk
        )

    return scores


def observability_coverage(graph: AgentGraph, observers: list[str]) -> float:
    """Return fraction of nodes observable by the selected observer nodes.

    Uses cached ancestor/descendant closures.
    """

    observed = set(observers)
    for observer in observers:
        observed.update(graph.get_ancestors(observer))
        observed.update(graph.get_descendants(observer))
    return len(observed) / max(graph.node_count, 1)


def verifier_observability_placement(graph: AgentGraph, k: int) -> list[str]:
    """Choose verifier/logging nodes by observability score."""

    if k < 1:
        raise ValueError("k must be at least 1")
    scores = observability_scores(graph)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [node_id for node_id, _ in ranked[:k]]
