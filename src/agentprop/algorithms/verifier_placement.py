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


def context_sensitive_verifier_placement(
    graph: AgentGraph,
    context_ratios: dict[str, float],
    k: int,
) -> list[str]:
    """Place verifiers downstream of high-sensitivity compressed nodes."""

    _validate_k(k)
    scores = error_propagation_centrality(graph)
    for node in graph.nodes():
        importance = node.importance_score if node.importance_score is not None else 0.5
        compression_risk = max(0.0, 1.0 - context_ratios.get(node.id, 1.0)) * importance
        if compression_risk == 0:
            continue
        downstream = graph.successors(node.id)
        if downstream:
            for target in downstream:
                scores[target] = scores.get(target, 0.0) + compression_risk
        else:
            scores[node.id] = scores.get(node.id, 0.0) + compression_risk
    return _rank_scores(scores, k)


def metric_dimension_verifier_placement(
    graph: AgentGraph,
    k: int,
    *,
    fault_tolerant: bool = False,
) -> list[str]:
    """Place verifiers as a resolving set grounded in metric dimension theory.

    A resolving set S guarantees that every distinct failure mode (node pair)
    has a unique distance-vector signature to S, making any failure uniquely
    localizable. When fault_tolerant=True, the set remains resolving even if
    any single verifier fails (fault-tolerant metric dimension, Geneson 2026).

    Uses undirected shortest-path distances for reachability across all node
    pairs, matching the standard metric dimension definition.
    """

    _validate_k(k)
    if graph.node_count == 0:
        return []

    nx_ug = graph.to_networkx().to_undirected()
    node_ids = sorted(str(n) for n in nx_ug.nodes())
    distances = dict(nx.all_pairs_shortest_path_length(nx_ug))

    verifiers: list[str] = []
    _extend_to_resolving(verifiers, node_ids, distances, k)

    if fault_tolerant:
        budget_remaining = k - len(verifiers)
        while budget_remaining > 0:
            if _is_fault_tolerant_resolving(verifiers, node_ids, distances):
                break
            candidates = [n for n in node_ids if n not in verifiers]
            if not candidates:
                break
            best = _best_fault_tolerant_candidate(verifiers, candidates, node_ids, distances)
            if best is None:
                break
            verifiers.append(best)
            budget_remaining -= 1

    return verifiers


def resolving_coverage(graph: AgentGraph, verifiers: list[str]) -> float:
    """Fraction of node pairs uniquely resolved by the verifier set.

    A pair (u, v) is resolved if there exists a verifier w such that
    d(u, w) != d(v, w). Returns 1.0 when the verifier set is a full
    resolving set (metric dimension guarantee satisfied).
    """

    if graph.node_count < 2:
        return 1.0
    nx_ug = graph.to_networkx().to_undirected()
    node_ids = sorted(str(n) for n in nx_ug.nodes())
    distances = dict(nx.all_pairs_shortest_path_length(nx_ug))
    verifier_set = [v for v in verifiers if v in nx_ug]

    resolved = 0
    total = 0
    for i, u in enumerate(node_ids):
        for v in node_ids[i + 1 :]:
            total += 1
            if _is_resolved(u, v, verifier_set, distances):
                resolved += 1

    return resolved / max(total, 1)


def _extend_to_resolving(
    verifiers: list[str],
    node_ids: list[str],
    distances: dict[str, dict[str, int]],
    budget: int,
) -> None:
    while len(verifiers) < budget:
        candidates = [n for n in node_ids if n not in verifiers]
        if not candidates:
            break
        best = _best_resolving_candidate(verifiers, candidates, node_ids, distances)
        if best is None:
            break
        verifiers.append(best)


def _best_resolving_candidate(
    verifiers: list[str],
    candidates: list[str],
    node_ids: list[str],
    distances: dict[str, dict[str, int]],
) -> str | None:
    unresolved_pairs = [
        (u, v)
        for i, u in enumerate(node_ids)
        for v in node_ids[i + 1 :]
        if not _is_resolved(u, v, verifiers, distances)
    ]
    if not unresolved_pairs:
        return None

    best_node = None
    best_gain = -1
    for candidate in candidates:
        gain = sum(
            1
            for u, v in unresolved_pairs
            if distances.get(u, {}).get(candidate) != distances.get(v, {}).get(candidate)
        )
        if gain > best_gain or (gain == best_gain and (best_node is None or candidate < best_node)):
            best_gain = gain
            best_node = candidate
    return best_node if best_gain > 0 else None


def _is_resolved(
    u: str,
    v: str,
    verifiers: list[str],
    distances: dict[str, dict[str, int]],
) -> bool:
    for w in verifiers:
        d_uw = distances.get(u, {}).get(w)
        d_vw = distances.get(v, {}).get(w)
        if d_uw != d_vw:
            return True
    return False


def _best_fault_tolerant_candidate(
    verifiers: list[str],
    candidates: list[str],
    node_ids: list[str],
    distances: dict[str, dict[str, int]],
) -> str | None:
    """Find the candidate that backs up the most singly-covered node pairs.

    A pair is "singly covered" if exactly one current verifier resolves it;
    adding a candidate that resolves it provides fault-tolerant redundancy.
    """
    singly_covered: list[tuple[str, str]] = []
    for j, u in enumerate(node_ids):
        for v in node_ids[j + 1 :]:
            coverage_count = sum(
                1
                for w in verifiers
                if distances.get(u, {}).get(w) != distances.get(v, {}).get(w)
            )
            if coverage_count == 1:
                singly_covered.append((u, v))

    if not singly_covered:
        return None

    best_node = None
    best_gain = -1
    for candidate in candidates:
        gain = sum(
            1
            for u, v in singly_covered
            if distances.get(u, {}).get(candidate) != distances.get(v, {}).get(candidate)
        )
        if gain > best_gain or (gain == best_gain and (best_node is None or candidate < best_node)):
            best_gain = gain
            best_node = candidate
    return best_node if best_gain > 0 else None


def _is_fault_tolerant_resolving(
    verifiers: list[str],
    node_ids: list[str],
    distances: dict[str, dict[str, int]],
) -> bool:
    for i, _removed in enumerate(verifiers):
        remaining = verifiers[:i] + verifiers[i + 1 :]
        for j, u in enumerate(node_ids):
            for v in node_ids[j + 1 :]:
                if not _is_resolved(u, v, remaining, distances):
                    return False
    return True


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
