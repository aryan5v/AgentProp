"""Verifier-placement heuristics."""

from __future__ import annotations

import networkx as nx

from agentprop.algorithms.seed_selection import centrality_scores
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


def betweenness_verifier_placement(graph: AgentGraph, k: int) -> list[str]:
    """Place verifiers on high-betweenness nodes."""

    _validate_k(k)
    nx_graph = graph.to_networkx()
    scores = nx.betweenness_centrality(nx_graph, weight="weight") if graph.node_count else {}
    return _rank_scores({str(node): float(score) for node, score in scores.items()}, k)


def pagerank_verifier_placement(graph: AgentGraph, k: int) -> list[str]:
    """Place verifiers on nodes that receive or aggregate important context."""

    _validate_k(k)
    if graph.node_count == 0:
        return []
    return _rank_scores(centrality_scores(graph)["pagerank"], k)


def error_propagation_centrality(graph: AgentGraph) -> dict[str, float]:
    """Score nodes by likelihood that local errors propagate downstream."""

    nx_graph = graph.to_networkx()
    if not nx_graph:
        return {}
    scores: dict[str, float] = {}
    max_reachable = max(graph.node_count - 1, 1)
    for node in graph.nodes():
        downstream = nx.descendants(nx_graph, node.id)
        downstream_risk = sum(
            1.0 - graph.node(str(target)).reliability + graph.node(str(target)).error_rate
            for target in downstream
        )
        reach = len(downstream) / max_reachable
        local_risk = 1.0 - node.reliability + node.error_rate
        scores[node.id] = local_risk + 0.5 * reach + 0.1 * downstream_risk
    return scores


def error_propagation_verifier_placement(graph: AgentGraph, k: int) -> list[str]:
    """Place verifiers where errors are likely to spread widely."""

    _validate_k(k)
    return _rank_scores(error_propagation_centrality(graph), k)


def greedy_correction_coverage_placement(graph: AgentGraph, k: int) -> list[str]:
    """Greedily place verifiers to maximize observed ancestors and descendants."""

    _validate_k(k)
    nx_graph = graph.to_networkx()
    selected: list[str] = []
    observed: set[str] = set()
    candidates = [node.id for node in graph.nodes()]

    while len(selected) < min(k, len(candidates)):
        best_node = None
        best_gain = float("-inf")
        for node_id in candidates:
            if node_id in selected:
                continue
            coverage = _observable_nodes(nx_graph, node_id)
            gain = len(coverage - observed)
            if gain > best_gain:
                best_gain = float(gain)
                best_node = node_id
        if best_node is None:
            break
        selected.append(best_node)
        observed.update(_observable_nodes(nx_graph, best_node))

    return selected


def _observable_nodes(nx_graph: nx.DiGraph, node_id: str) -> set[str]:
    if node_id not in nx_graph:
        return set()
    return {
        node_id,
        *(str(node) for node in nx.ancestors(nx_graph, node_id)),
        *(str(node) for node in nx.descendants(nx_graph, node_id)),
    }


def _rank_scores(scores: dict[str, float], k: int) -> list[str]:
    return [
        node_id
        for node_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    ]


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")
