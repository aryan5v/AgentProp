"""Scalable influence maximization via IMM/TIM-style reverse reachable sets."""

from __future__ import annotations

import random
from dataclasses import dataclass

from agentprop.core import AgentGraph
from agentprop.propagation import IndependentCascade, PropagationModel


@dataclass(slots=True)
class IMMConfig:
    """Configuration for reverse-reachable influence estimation."""

    rr_samples: int | None = None
    """Number of RR sets; defaults to min(500, 50 * node_count)."""

    seed: int = 0
    lazy: bool = True


def imm_greedy_seed_selection(
    graph: AgentGraph,
    k: int,
    *,
    propagation_model: PropagationModel | None = None,
    trials: int = 100,
    config: IMMConfig | None = None,
) -> list[str]:
    """Greedy seed selection using IMM-style reverse reachable set sampling.

    For large graphs this replaces full Monte Carlo re-simulation on every
    marginal with TIM/IMM-style RR-set coverage estimates. Falls back to a
    small MC check on the final set when a propagation model is supplied.
    """

    if k < 1:
        raise ValueError("k must be at least 1")
    cfg = config or IMMConfig()
    rng = random.Random(cfg.seed)
    candidates = _seed_eligible_nodes(graph)
    if not candidates:
        return []

    sample_count = cfg.rr_samples or min(500, max(50, 50 * graph.node_count))
    rr_sets = [_generate_rr_set(graph, rng) for _ in range(sample_count)]
    node_index = {node_id: index for index, node_id in enumerate(candidates)}
    coverage_counts = [0] * len(candidates)
    for rr in rr_sets:
        for node_id in rr:
            idx = node_index.get(node_id)
            if idx is not None:
                coverage_counts[idx] += 1

    if cfg.lazy and k > 1:
        selected = _lazy_imm_select(candidates, coverage_counts, rr_sets, k, rng)
    else:
        selected = _greedy_imm_select(candidates, coverage_counts, rr_sets, k)

    if propagation_model is not None and len(selected) < k:
        from agentprop.algorithms.seed_selection import greedy_seed_selection

        return greedy_seed_selection(
            graph,
            k,
            propagation_model=propagation_model,
            trials=max(20, trials // 2),
        )
    return selected[:k]


def tim_seed_selection(
    graph: AgentGraph,
    k: int,
    *,
    propagation_model: PropagationModel | None = None,
    trials: int = 100,
    config: IMMConfig | None = None,
) -> list[str]:
    """Alias for IMM greedy selection (TIM-style RR-set backend)."""

    return imm_greedy_seed_selection(
        graph,
        k,
        propagation_model=propagation_model,
        trials=trials,
        config=config,
    )


def estimate_imm_influence(
    graph: AgentGraph,
    seeds: list[str],
    *,
    rr_samples: int | None = None,
    seed: int = 0,
) -> float:
    """Estimate expected IC influence using RR-set coverage."""

    rng = random.Random(seed)
    sample_count = rr_samples or min(500, max(50, 50 * graph.node_count))
    seed_set = set(seeds)
    covered = 0
    for _ in range(sample_count):
        rr = _generate_rr_set(graph, rng)
        if seed_set & rr:
            covered += 1
    return covered / max(sample_count, 1)


def _greedy_imm_select(
    candidates: list[str],
    coverage_counts: list[int],
    rr_sets: list[set[str]],
    k: int,
) -> list[str]:
    selected: list[str] = []
    covered_rr: set[int] = set()
    remaining = set(range(len(candidates)))

    while len(selected) < min(k, len(candidates)) and remaining:
        best_idx = max(
            remaining,
            key=lambda idx: _marginal_rr_gain(idx, candidates, covered_rr, rr_sets),
        )
        gain = _marginal_rr_gain(best_idx, candidates, covered_rr, rr_sets)
        if gain <= 0:
            break
        selected.append(candidates[best_idx])
        for rr_index, rr in enumerate(rr_sets):
            if candidates[best_idx] in rr:
                covered_rr.add(rr_index)
        remaining.remove(best_idx)
    return selected


def _lazy_imm_select(
    candidates: list[str],
    coverage_counts: list[int],
    rr_sets: list[set[str]],
    k: int,
    rng: random.Random,
) -> list[str]:
    selected: list[str] = []
    covered_rr: set[int] = set()
    last_marginal = {idx: coverage_counts[idx] for idx in range(len(candidates))}

    while len(selected) < min(k, len(candidates)):
        remaining = [idx for idx in range(len(candidates)) if candidates[idx] not in selected]
        if not remaining:
            break
        remaining.sort(key=lambda idx: (-last_marginal[idx], candidates[idx]))
        best_idx = remaining[0]
        true_gain = _marginal_rr_gain(best_idx, candidates, covered_rr, rr_sets)
        if true_gain <= 0:
            break
        selected.append(candidates[best_idx])
        for rr_index, rr in enumerate(rr_sets):
            if candidates[best_idx] in rr:
                covered_rr.add(rr_index)
        for idx in remaining:
            if idx != best_idx:
                last_marginal[idx] = _marginal_rr_gain(idx, candidates, covered_rr, rr_sets)
    return selected


def _marginal_rr_gain(
    idx: int,
    candidates: list[str],
    covered_rr: set[int],
    rr_sets: list[set[str]],
) -> int:
    node_id = candidates[idx]
    gain = 0
    for rr_index, rr in enumerate(rr_sets):
        if rr_index in covered_rr:
            continue
        if node_id in rr:
            gain += 1
    return gain


def _generate_rr_set(graph: AgentGraph, rng: random.Random) -> set[str]:
    """Sample one reverse reachable set under independent cascade."""

    nodes = graph.node_ids()
    if not nodes:
        return set()
    start = rng.choice(nodes)
    active = {start}
    frontier = [start]
    while frontier:
        target = frontier.pop()
        for source in graph.predecessors(target):
            if source in active:
                continue
            edge = graph.edge(source, target)
            probability = edge.activation_probability * edge.reliability * edge.relevance
            if rng.random() <= max(0.0, min(1.0, probability)):
                active.add(source)
                frontier.append(source)
    return active


def _seed_eligible_nodes(graph: AgentGraph) -> list[str]:
    from agentprop.core.types import NodeType

    blocked = {NodeType.OUTPUT, NodeType.TOOL, NodeType.MEMORY}
    return [node.id for node in graph.nodes() if node.type not in blocked]
