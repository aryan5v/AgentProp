"""Training-free seed-selection algorithms."""

from __future__ import annotations

import random
from collections.abc import Callable

import networkx as nx

from agentprop.core import AgentGraph, NodeType
from agentprop.propagation import IndependentCascade, PropagationModel

ScoreMap = dict[str, float]


def random_seed_selection(graph: AgentGraph, k: int, *, seed: int | None = None) -> list[str]:
    """Select `k` random nodes as a baseline."""

    _validate_budget(k)
    rng = random.Random(seed)
    nodes = _seed_eligible_nodes(graph)
    rng.shuffle(nodes)
    return nodes[:k]


def degree_seed_selection(graph: AgentGraph, k: int, *, direction: str = "total") -> list[str]:
    """Select high-degree nodes."""

    _validate_budget(k)
    nx_graph = graph.to_networkx()
    if direction == "in":
        scores = {str(node): float(degree) for node, degree in nx_graph.in_degree()}
    elif direction == "out":
        scores = {str(node): float(degree) for node, degree in nx_graph.out_degree()}
    elif direction == "total":
        scores = {str(node): float(degree) for node, degree in nx_graph.degree()}
    else:
        raise ValueError("direction must be one of: total, in, out")
    scores = _filter_scores_to_seed_eligible_nodes(graph, scores)
    return _top_k(scores, k)


def pagerank_seed_selection(graph: AgentGraph, k: int) -> list[str]:
    """Select high-PageRank nodes."""

    _validate_budget(k)
    scores = _filter_scores_to_seed_eligible_nodes(graph, _pagerank_scores(graph, reverse=True))
    return _top_k(scores, k)


def betweenness_seed_selection(graph: AgentGraph, k: int) -> list[str]:
    """Select high-betweenness nodes."""

    _validate_budget(k)
    scores = nx.betweenness_centrality(graph.to_networkx(), weight="weight")
    filtered_scores = _filter_scores_to_seed_eligible_nodes(
        graph,
        {str(node): score for node, score in scores.items()},
    )
    return _top_k(filtered_scores, k)


def greedy_seed_selection(
    graph: AgentGraph,
    k: int,
    *,
    propagation_model: PropagationModel | None = None,
    trials: int = 100,
    objective: Callable[[float, float], float] | None = None,
) -> list[str]:
    """Greedily choose seeds that maximize propagation utility."""

    _validate_budget(k)
    model = propagation_model or IndependentCascade(seed=0)
    score_fn = objective or _default_objective
    selected: list[str] = []
    candidates = _seed_eligible_nodes(graph)

    while len(selected) < min(k, len(candidates)):
        best_node = None
        best_score = float("-inf")
        for node_id in candidates:
            if node_id in selected:
                continue
            result = model.simulate(graph, [*selected, node_id], trials=trials)
            propagation_time = result.expected_propagation_time or result.propagation_time
            score = score_fn(result.coverage, propagation_time)
            if score > best_score:
                best_score = score
                best_node = node_id
        if best_node is None:
            break
        selected.append(best_node)

    return selected


def centrality_scores(graph: AgentGraph) -> dict[str, ScoreMap]:
    """Return common centrality scores for report generation."""

    nx_graph = graph.to_networkx()
    betweenness = nx.betweenness_centrality(nx_graph, weight="weight") if graph.node_count else {}
    degree = dict(nx_graph.degree())
    return {
        "degree": {str(node): float(score) for node, score in degree.items()},
        "pagerank": _pagerank_scores(graph, reverse=True),
        "betweenness": {str(node): float(score) for node, score in betweenness.items()},
    }


def _validate_budget(k: int) -> None:
    if k < 1:
        raise ValueError("k must be at least 1")


def _top_k(scores: ScoreMap, k: int) -> list[str]:
    return [
        node_id
        for node_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    ]


def _default_objective(coverage: float, propagation_time: float) -> float:
    return coverage - 0.02 * propagation_time


def _seed_eligible_nodes(graph: AgentGraph) -> list[str]:
    excluded = {NodeType.OUTPUT}
    return [node.id for node in graph.nodes() if node.type not in excluded]


def _filter_scores_to_seed_eligible_nodes(graph: AgentGraph, scores: ScoreMap) -> ScoreMap:
    eligible = set(_seed_eligible_nodes(graph))
    return {node_id: score for node_id, score in scores.items() if node_id in eligible}


def _pagerank_scores(
    graph: AgentGraph,
    *,
    reverse: bool = False,
    damping: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> ScoreMap:
    nodes = [node.id for node in graph.nodes()]
    if not nodes:
        return {}

    node_count = len(nodes)
    scores = {node_id: 1.0 / node_count for node_id in nodes}
    base_score = (1.0 - damping) / node_count

    for _ in range(max_iter):
        next_scores = {node_id: base_score for node_id in nodes}
        dangling_score = 0.0

        for source in nodes:
            successors = graph.predecessors(source) if reverse else graph.successors(source)
            if not successors:
                dangling_score += scores[source] / node_count
                continue
            total_weight = sum(
                _pagerank_edge_weight(graph, source, target, reverse=reverse)
                for target in successors
            )
            if total_weight == 0:
                share = scores[source] / len(successors)
                for target in successors:
                    next_scores[target] += damping * share
                continue
            for target in successors:
                edge_weight = _pagerank_edge_weight(graph, source, target, reverse=reverse)
                weight_share = edge_weight / total_weight
                next_scores[target] += damping * scores[source] * weight_share

        for node_id in nodes:
            next_scores[node_id] += damping * dangling_score

        delta = sum(abs(next_scores[node_id] - scores[node_id]) for node_id in nodes)
        scores = next_scores
        if delta < tolerance:
            break

    return scores


def _pagerank_edge_weight(
    graph: AgentGraph,
    source: str,
    target: str,
    *,
    reverse: bool,
) -> float:
    if reverse:
        return max(graph.edge(target, source).weight, 0.0)
    return max(graph.edge(source, target).weight, 0.0)
