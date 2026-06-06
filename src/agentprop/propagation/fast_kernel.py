"""Optional numpy-accelerated propagation kernels with pure-Python fallback."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING

from agentprop.core.propagation_index import PropagationGraphIndex
from agentprop.propagation.base import PropagationResult, deterministic_result

if TYPE_CHECKING:
    from agentprop.core import AgentGraph

_NUMPY_AVAILABLE = False
try:
    import numpy as np  # noqa: F401

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    pass


def get_propagation_index(graph: AgentGraph) -> PropagationGraphIndex:
    """Return a cached propagation index for ``graph``."""

    index: PropagationGraphIndex = graph.get_propagation_index()
    return index


def ic_simulate_once_indexed(
    index: PropagationGraphIndex,
    seeds: list[str],
    rng: random.Random,
) -> tuple[dict[str, int], list[float]]:
    """Single IC trial using integer-indexed adjacency (no NetworkX copy)."""

    active = [False] * index.node_count
    activation_rounds: dict[str, int] = {}
    frontier: list[int] = []
    for seed in seeds:
        idx = index.node_index.get(seed)
        if idx is None or active[idx]:
            continue
        active[idx] = True
        activation_rounds[index.node_ids[idx]] = 0
        frontier.append(idx)

    coverage_by_round = [_coverage(sum(active), index.node_count)]
    round_number = 0

    while frontier:
        round_number += 1
        next_frontier: list[int] = []
        for source_idx in frontier:
            for target_idx, probability in index.ic_successors[source_idx]:
                if active[target_idx]:
                    continue
                if rng.random() <= probability:
                    active[target_idx] = True
                    activation_rounds[index.node_ids[target_idx]] = round_number
                    next_frontier.append(target_idx)
        frontier = next_frontier
        if frontier:
            coverage_by_round.append(_coverage(sum(active), index.node_count))

    return activation_rounds, coverage_by_round


def rzf_simulate_once_indexed(
    index: PropagationGraphIndex,
    seeds: list[str],
    rng: random.Random,
    *,
    max_rounds: int | None = None,
) -> tuple[dict[str, int], list[float]]:
    """Single RZF trial using integer-indexed predecessor weights."""

    active = [False] * index.node_count
    activation_rounds: dict[str, int] = {}
    for seed in seeds:
        idx = index.node_index.get(seed)
        if idx is not None:
            active[idx] = True
            activation_rounds[index.node_ids[idx]] = 0

    coverage_by_round = [_coverage(sum(active), index.node_count)]
    limit = max_rounds or max(1, index.node_count * 4)

    for round_number in range(1, limit + 1):
        changed = False
        for target_idx, predecessors in enumerate(index.rzf_predecessors):
            if active[target_idx] or not predecessors:
                continue
            total_weight = sum(weight for _, weight in predecessors)
            if total_weight <= 0.0:
                continue
            active_weight = sum(
                weight for source_idx, weight in predecessors if active[source_idx]
            )
            probability = active_weight / total_weight
            if probability > 0.0 and rng.random() <= probability:
                active[target_idx] = True
                activation_rounds[index.node_ids[target_idx]] = round_number
                changed = True
        coverage_by_round.append(_coverage(sum(active), index.node_count))
        if not changed or sum(active) == index.node_count:
            break

    return activation_rounds, coverage_by_round


def aggregate_trial_results(
    graph: AgentGraph,
    trial_rounds: list[dict[str, int]],
    trial_coverages: list[list[float]],
    *,
    trials: int,
) -> PropagationResult:
    """Combine per-trial activation maps into a standard PropagationResult."""

    activation_counts: dict[str, int] = defaultdict(int)
    round_totals: dict[str, int] = defaultdict(int)
    coverage_by_round_totals: list[float] = []
    full_activations = 0
    propagation_time_total = 0

    for activation_rounds, coverage_by_round in zip(trial_rounds, trial_coverages, strict=True):
        for node_id, round_number in activation_rounds.items():
            activation_counts[node_id] += 1
            round_totals[node_id] += round_number
        coverage_by_round_totals = _sum_round_coverages(
            coverage_by_round_totals,
            coverage_by_round,
        )
        propagation_time = max(activation_rounds.values(), default=0)
        propagation_time_total += propagation_time
        if len(activation_rounds) == graph.node_count:
            full_activations += 1

    representative_rounds = {
        node_id: round(round_totals[node_id] / activation_counts[node_id])
        for node_id in activation_counts
    }
    averaged_coverage = [value / trials for value in coverage_by_round_totals]

    return deterministic_result(
        graph,
        representative_rounds,
        trials=trials,
        full_activation_probability=full_activations / trials,
        expected_propagation_time=propagation_time_total / trials,
        coverage_by_round=averaged_coverage,
    )


def simulate_batch_ic(
    graph: AgentGraph,
    seed_sets: list[list[str]],
    *,
    trials: int = 100,
    seed: int | None = None,
) -> list[PropagationResult]:
    """Run IC simulation for multiple seed sets (shared graph index)."""

    if trials < 1:
        raise ValueError("trials must be at least 1")
    index = get_propagation_index(graph)
    rng = random.Random(seed)
    results: list[PropagationResult] = []
    for seeds in seed_sets:
        trial_rounds: list[dict[str, int]] = []
        trial_coverages: list[list[float]] = []
        for _ in range(trials):
            rounds, coverages = ic_simulate_once_indexed(index, seeds, rng)
            trial_rounds.append(rounds)
            trial_coverages.append(coverages)
        results.append(
            aggregate_trial_results(
                graph,
                trial_rounds,
                trial_coverages,
                trials=trials,
            )
        )
    return results


def simulate_batch_rzf(
    graph: AgentGraph,
    seed_sets: list[list[str]],
    *,
    trials: int = 100,
    seed: int | None = None,
    max_rounds: int | None = None,
) -> list[PropagationResult]:
    """Run RZF simulation for multiple seed sets (shared graph index)."""

    if trials < 1:
        raise ValueError("trials must be at least 1")
    index = get_propagation_index(graph)
    rng = random.Random(seed)
    results: list[PropagationResult] = []
    for seeds in seed_sets:
        trial_rounds: list[dict[str, int]] = []
        trial_coverages: list[list[float]] = []
        for _ in range(trials):
            rounds, coverages = rzf_simulate_once_indexed(
                index,
                seeds,
                rng,
                max_rounds=max_rounds,
            )
            trial_rounds.append(rounds)
            trial_coverages.append(coverages)
        results.append(
            aggregate_trial_results(
                graph,
                trial_rounds,
                trial_coverages,
                trials=trials,
            )
        )
    return results


def numpy_ic_batch_available() -> bool:
    """True when numpy is installed (reserved for future vectorized kernels)."""

    return _NUMPY_AVAILABLE


def _coverage(active_count: int, node_count: int) -> float:
    return active_count / max(node_count, 1)


def _sum_round_coverages(existing: list[float], new_values: list[float]) -> list[float]:
    if len(existing) < len(new_values):
        existing.extend([existing[-1] if existing else 0.0] * (len(new_values) - len(existing)))
    for index, value in enumerate(new_values):
        existing[index] += value
    if new_values:
        final_value = new_values[-1]
        for index in range(len(new_values), len(existing)):
            existing[index] += final_value
    return existing
