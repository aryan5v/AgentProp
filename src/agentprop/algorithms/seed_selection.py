"""Training-free seed-selection algorithms."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx

from agentprop.core import AgentGraph, NodeType
from agentprop.propagation import IndependentCascade, PropagationModel, RandomizedZeroForcing

ScoreMap = dict[str, float]


@dataclass(slots=True)
class SeedSelectionTrace:
    """Debug trace for greedy-style seed selection."""

    selected: list[str]
    marginal_gains: dict[str, float]


@dataclass(slots=True)
class _CelfQueueItem:
    node: str
    gain: float
    last_updated: int


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


def closeness_seed_selection(graph: AgentGraph, k: int) -> list[str]:
    """Select nodes with high downstream closeness.

    NetworkX computes directed closeness over incoming paths by default. Agent
    workflows propagate along outgoing edges, so this uses the reversed graph to
    score how close each seed is to the nodes it can reach downstream.
    """

    _validate_budget(k)
    nx_graph = graph.to_networkx()
    scores = nx.closeness_centrality(nx_graph.reverse(copy=True)) if graph.node_count else {}
    filtered_scores = _filter_scores_to_seed_eligible_nodes(
        graph,
        {str(node): float(score) for node, score in scores.items()},
    )
    return _top_k(filtered_scores, k)


def k_core_seed_selection(graph: AgentGraph, k: int) -> list[str]:
    """Select nodes from the highest undirected core numbers."""

    _validate_budget(k)
    nx_graph = graph.to_networkx().to_undirected()
    if not graph.node_count:
        return []
    core_numbers = nx.core_number(nx_graph)
    scores = _filter_scores_to_seed_eligible_nodes(
        graph,
        {str(node): float(score) for node, score in core_numbers.items()},
    )
    return _top_k(scores, k)


def greedy_seed_selection(
    graph: AgentGraph,
    k: int,
    *,
    propagation_model: PropagationModel | None = None,
    trials: int = 100,
    objective: Callable[[float, float], float] | None = None,
    importance_weight: float = 1.0,
    protect_critical_nodes: bool = True,
    critical_importance_threshold: float = 0.80,
) -> list[str]:
    """Greedily choose seeds that maximize propagation and role-critical utility.

    Nodes with high ``importance_score`` are context-sensitive: starving them of
    full context is likely to hurt task quality even when topology coverage looks
    good. By default, critical nodes are protected before the propagation-only
    greedy loop runs.
    """

    _validate_budget(k)
    if importance_weight < 0:
        raise ValueError("importance_weight must be non-negative")
    if not 0 <= critical_importance_threshold <= 1:
        raise ValueError("critical_importance_threshold must be between 0 and 1")
    model = propagation_model or IndependentCascade(seed=0)
    score_fn = objective or _default_objective
    candidates = _seed_eligible_nodes(graph)
    selected: list[str] = []

    if protect_critical_nodes:
        selected.extend(
            _critical_seed_candidates(
                graph,
                candidates,
                budget=k,
                threshold=critical_importance_threshold,
            )
        )

    while len(selected) < min(k, len(candidates)):
        best_node = None
        best_score = float("-inf")
        for node_id in candidates:
            if node_id in selected:
                continue
            result = model.simulate(graph, [*selected, node_id], trials=trials)
            propagation_time = result.expected_propagation_time or result.propagation_time
            score = score_fn(result.coverage, propagation_time)
            score *= 1.0 + importance_weight * _node_importance(graph, node_id)
            if score > best_score:
                best_score = score
                best_node = node_id
        if best_node is None:
            break
        selected.append(best_node)

    return selected


def pure_greedy_seed_selection(
    graph: AgentGraph,
    k: int,
    *,
    propagation_model: PropagationModel | None = None,
    trials: int = 100,
) -> list[str]:
    """Greedily maximize propagation coverage/time without role-critical reweighting.

    This is the theory-preserving influence-maximization baseline. Under the
    standard IC/LT assumptions, the plain expected-coverage objective is the one
    associated with the classical greedy approximation guarantee. Role-critical
    pre-seeding and importance reweighting are intentionally disabled here.
    """

    return greedy_seed_selection(
        graph,
        k,
        propagation_model=propagation_model,
        trials=trials,
        importance_weight=0.0,
        protect_critical_nodes=False,
    )


def quality_aware_greedy_seed_selection(
    graph: AgentGraph,
    k: int,
    *,
    propagation_model: PropagationModel | None = None,
    trials: int = 100,
    cost_weight: float = 0.001,
    quality_weight: float = 1.0,
    importance_weight: float = 1.0,
) -> list[str]:
    """Select seeds for expected task success minus token cost.

    This is the public bridge between topology-first influence maximization and
    empirical quality-aware routing. It uses existing node metadata now and can be
    swapped to learned expected-success estimators as trace data accumulates.
    """

    def objective(coverage: float, propagation_time: float) -> float:
        return quality_weight * coverage - 0.02 * propagation_time

    return greedy_seed_selection(
        graph,
        k,
        propagation_model=propagation_model,
        trials=trials,
        objective=objective,
        importance_weight=importance_weight + cost_weight,
        protect_critical_nodes=True,
    )


def cost_aware_greedy_seed_selection(
    graph: AgentGraph,
    k: int,
    *,
    propagation_model: PropagationModel | None = None,
    trials: int = 100,
    cost_weight: float = 0.001,
) -> list[str]:
    """Greedily select seeds with a token-cost penalty."""

    return greedy_seed_selection(
        graph,
        k,
        propagation_model=propagation_model,
        trials=trials,
        objective=lambda coverage, time: coverage - 0.02 * time,
    ) if cost_weight == 0 else _cost_aware_greedy(
        graph,
        k,
        propagation_model=propagation_model,
        trials=trials,
        cost_weight=cost_weight,
    )


def celf_seed_selection(
    graph: AgentGraph,
    k: int,
    *,
    propagation_model: PropagationModel | None = None,
    trials: int = 100,
) -> list[str]:
    """Cost-effective lazy forward selection for influence maximization."""

    _validate_budget(k)
    model = propagation_model or IndependentCascade(seed=0)
    candidates = _seed_eligible_nodes(graph)
    selected: list[str] = []
    queue = [
        _CelfQueueItem(
            node=node_id,
            gain=_seed_set_score(graph, [node_id], model, trials),
            last_updated=0,
        )
        for node_id in candidates
    ]

    while len(selected) < min(k, len(candidates)) and queue:
        queue.sort(key=lambda item: (-item.gain, item.node))
        best = queue.pop(0)
        if best.last_updated == len(selected):
            selected.append(best.node)
            queue = [item for item in queue if item.node != best.node]
            continue

        candidate = best.node
        current_score = _seed_set_score(graph, selected, model, trials)
        updated_score = _seed_set_score(graph, [*selected, candidate], model, trials)
        best.gain = updated_score - current_score
        best.last_updated = len(selected)
        queue.append(best)

    return selected


def centrality_scores(graph: AgentGraph) -> dict[str, ScoreMap]:
    """Return common centrality scores for report generation."""

    nx_graph = graph.to_networkx()
    betweenness = nx.betweenness_centrality(nx_graph, weight="weight") if graph.node_count else {}
    degree = dict(nx_graph.degree())
    return {
        "degree": {str(node): float(score) for node, score in degree.items()},
        "in_degree": {str(node): float(score) for node, score in nx_graph.in_degree()},
        "out_degree": {str(node): float(score) for node, score in nx_graph.out_degree()},
        "pagerank": _pagerank_scores(graph, reverse=True),
        "betweenness": {str(node): float(score) for node, score in betweenness.items()},
        "closeness": {
            str(node): float(score)
            for node, score in nx.closeness_centrality(nx_graph.reverse(copy=True)).items()
        } if graph.node_count else {},
        "k_core": {
            str(node): float(score)
            for node, score in nx.core_number(nx_graph.to_undirected()).items()
        } if graph.node_count else {},
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


def _critical_seed_candidates(
    graph: AgentGraph,
    candidates: list[str],
    *,
    budget: int,
    threshold: float,
) -> list[str]:
    candidate_set = set(candidates)
    ranked = sorted(
        (
            (node.id, _node_importance(graph, node.id))
            for node in graph.nodes()
            if node.id in candidate_set
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return [node_id for node_id, score in ranked if score >= threshold][:budget]


def _node_importance(graph: AgentGraph, node_id: str) -> float:
    node = graph.node(node_id)
    if node.importance_score is not None:
        return max(0.0, min(1.0, float(node.importance_score)))
    role = (node.role or node.id).lower()
    if node.type == NodeType.EXECUTOR or "coder" in role or "implement" in role:
        return 0.90
    if node.type == NodeType.VERIFIER or "test" in role or "verif" in role:
        return 0.80
    if node.type == NodeType.REVIEWER:
        return 0.65
    if node.type == NodeType.PLANNER:
        return 0.55
    return 0.35


def _cost_aware_greedy(
    graph: AgentGraph,
    k: int,
    *,
    propagation_model: PropagationModel | None,
    trials: int,
    cost_weight: float,
) -> list[str]:
    _validate_budget(k)
    model = propagation_model or IndependentCascade(seed=0)
    selected: list[str] = []
    candidates = _seed_eligible_nodes(graph)

    while len(selected) < min(k, len(candidates)):
        best_node = None
        best_score = float("-inf")
        for node_id in candidates:
            if node_id in selected:
                continue
            node = graph.node(node_id)
            score = _seed_set_score(graph, [*selected, node_id], model, trials)
            score -= cost_weight * node.token_cost
            if score > best_score:
                best_score = score
                best_node = node_id
        if best_node is None:
            break
        selected.append(best_node)

    return selected


def _seed_set_score(
    graph: AgentGraph,
    seeds: list[str],
    model: PropagationModel,
    trials: int,
) -> float:
    result = model.simulate(graph, seeds, trials=trials)
    propagation_time = result.expected_propagation_time or result.propagation_time
    return _default_objective(result.coverage, propagation_time)


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


def rzf_centrality_seed_selection(
    graph: AgentGraph,
    k: int,
    *,
    trials: int = 50,
    seed: int = 0,
) -> list[str]:
    """Select seeds by Randomized Zero Forcing process-based centrality.

    Scores each candidate by simulating RZF from that node alone and computing
    coverage / expected_propagation_time — a dynamic centrality measure native
    to the weighted directed propagation structure (Geneson 2026). This is
    O(n * trials) vs O(k * n * trials) for greedy, making it ~k times cheaper.

    For tree-structured graphs the expected propagation time has an exact
    Markov-chain closed form (Geneson 2022); Monte Carlo is used here for
    generality across all workflow topologies.
    """

    _validate_budget(k)
    candidates = _seed_eligible_nodes(graph)
    if not candidates:
        return []

    model = RandomizedZeroForcing(seed=seed)
    scores: ScoreMap = {}
    for node_id in candidates:
        result = model.simulate(graph, [node_id], trials=trials)
        propagation_time = result.expected_propagation_time or float(result.propagation_time)
        scores[node_id] = result.coverage / max(1.0, propagation_time)

    return sorted(candidates, key=lambda n: (-scores[n], n))[:k]
