"""Independent Cascade propagation model."""

from __future__ import annotations

import random
from collections import defaultdict

from agentprop.core import AgentGraph
from agentprop.propagation.base import PropagationResult, deterministic_result


class IndependentCascade:
    """Probabilistic diffusion where active nodes get one chance to activate neighbors."""

    name = "independent-cascade"

    def __init__(self, *, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 100,
    ) -> PropagationResult:
        """Run Monte Carlo Independent Cascade simulation."""

        if trials < 1:
            raise ValueError("trials must be at least 1")

        activation_counts: dict[str, int] = defaultdict(int)
        round_totals: dict[str, int] = defaultdict(int)
        coverage_by_round_totals: list[float] = []
        full_activations = 0
        propagation_time_total = 0

        for _ in range(trials):
            activation_rounds, coverage_by_round = self._simulate_once(graph, seeds)
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

    def _simulate_once(
        self,
        graph: AgentGraph,
        seeds: list[str],
    ) -> tuple[dict[str, int], list[float]]:
        nx_graph = graph.to_networkx()
        active = set(seeds)
        activation_rounds = {seed: 0 for seed in seeds if seed in nx_graph}
        frontier = set(activation_rounds)
        coverage_by_round = [_coverage(len(active), graph.node_count)]
        round_number = 0

        while frontier:
            round_number += 1
            next_frontier: set[str] = set()
            for source in frontier:
                for target in nx_graph.successors(source):
                    if target in active:
                        continue
                    edge = graph.edge(source, target)
                    probability = edge.activation_probability * edge.reliability * edge.relevance
                    if self._random.random() <= max(0.0, min(1.0, probability)):
                        active.add(target)
                        activation_rounds[target] = round_number
                        next_frontier.add(target)
            frontier = next_frontier
            if frontier:
                coverage_by_round.append(_coverage(len(active), graph.node_count))

        return activation_rounds, coverage_by_round


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
