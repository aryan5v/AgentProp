"""Randomized zero forcing on directed weighted graphs."""

from __future__ import annotations

import random
from collections import defaultdict

from agentprop.core import AgentGraph
from agentprop.propagation.base import PropagationResult, deterministic_result


class RandomizedZeroForcing:
    """Activate a node with probability equal to active incoming weight share."""

    name = "randomized-zero-forcing"

    def __init__(self, *, seed: int | None = None, max_rounds: int | None = None) -> None:
        self._random = random.Random(seed)
        self.max_rounds = max_rounds

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 100,
    ) -> PropagationResult:
        """Run Monte Carlo randomized zero forcing."""

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

        return deterministic_result(
            graph,
            representative_rounds,
            trials=trials,
            full_activation_probability=full_activations / trials,
            expected_propagation_time=propagation_time_total / trials,
            coverage_by_round=[value / trials for value in coverage_by_round_totals],
        )

    def _simulate_once(
        self,
        graph: AgentGraph,
        seeds: list[str],
    ) -> tuple[dict[str, int], list[float]]:
        nx_graph = graph.to_networkx()
        active = {seed for seed in seeds if seed in nx_graph}
        activation_rounds = {seed: 0 for seed in active}
        coverage_by_round = [_coverage(len(active), graph.node_count)]
        max_rounds = self.max_rounds or max(1, graph.node_count * 4)

        for round_number in range(1, max_rounds + 1):
            next_active = set(active)
            for node_id in nx_graph.nodes:
                if node_id in active:
                    continue
                probability = self._active_incoming_weight_share(graph, str(node_id), active)
                if probability > 0 and self._random.random() <= probability:
                    next_active.add(str(node_id))
                    activation_rounds[str(node_id)] = round_number

            if next_active == active:
                break
            active = next_active
            coverage_by_round.append(_coverage(len(active), graph.node_count))
            if len(active) == graph.node_count:
                break

        return activation_rounds, coverage_by_round

    def _active_incoming_weight_share(
        self,
        graph: AgentGraph,
        node_id: str,
        active: set[str],
    ) -> float:
        incoming = graph.predecessors(node_id)
        if not incoming:
            return 0.0

        total_weight = 0.0
        active_weight = 0.0
        for source in incoming:
            edge = graph.edge(source, node_id)
            weight = max(edge.weight, 0.0)
            total_weight += weight
            if source in active:
                active_weight += weight

        if total_weight == 0:
            return 0.0
        return active_weight / total_weight


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
