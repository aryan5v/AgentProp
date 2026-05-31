"""Trace-calibrated learned propagation model."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from agentprop.core import AgentGraph
from agentprop.propagation.base import PropagationResult, deterministic_result

EdgeKey = tuple[str, str]


@dataclass(slots=True)
class LearnedPropagationFit:
    """Learned edge probabilities and training counts."""

    edge_probabilities: dict[EdgeKey, float]
    edge_counts: dict[EdgeKey, float]
    source_counts: dict[str, float]


class LearnedPropagation:
    """Independent-cascade-style propagation calibrated from workflow traces."""

    name = "learned"

    def __init__(
        self,
        *,
        edge_probabilities: dict[EdgeKey, float] | None = None,
        fallback_probability: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.edge_probabilities = edge_probabilities or {}
        self.fallback_probability = fallback_probability
        self._random = random.Random(seed)

    @classmethod
    def fit_from_trace_dicts(
        cls,
        traces: list[dict[str, Any]],
        *,
        smoothing: float = 0.5,
        seed: int | None = None,
    ) -> LearnedPropagation:
        """Fit edge probabilities from one or more trace dictionaries."""

        fit = fit_learned_propagation_from_trace_dicts(traces, smoothing=smoothing)
        return cls(edge_probabilities=fit.edge_probabilities, seed=seed)

    @classmethod
    def fit_from_graph(
        cls,
        graph: AgentGraph,
        *,
        smoothing: float = 0.5,
        seed: int | None = None,
    ) -> LearnedPropagation:
        """Fit edge probabilities from trace metadata on an AgentGraph."""

        fit = fit_learned_propagation_from_graph(graph, smoothing=smoothing)
        return cls(edge_probabilities=fit.edge_probabilities, seed=seed)

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 100,
    ) -> PropagationResult:
        """Run Monte Carlo propagation using learned edge probabilities."""

        if trials < 1:
            raise ValueError("trials must be at least 1")
        if not self.edge_probabilities and _graph_has_trace_metadata(graph):
            self.edge_probabilities = fit_learned_propagation_from_graph(graph).edge_probabilities

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
                    probability = self._edge_probability(graph, str(source), str(target))
                    if self._random.random() <= probability:
                        active.add(str(target))
                        activation_rounds[str(target)] = round_number
                        next_frontier.add(str(target))
            frontier = next_frontier
            if frontier:
                coverage_by_round.append(_coverage(len(active), graph.node_count))

        return activation_rounds, coverage_by_round

    def _edge_probability(self, graph: AgentGraph, source: str, target: str) -> float:
        learned = self.edge_probabilities.get((source, target))
        if learned is not None:
            return _clamp_probability(learned)

        edge = graph.edge(source, target)
        trace_count = float(edge.metadata.get("trace_message_count", 0.0))
        if trace_count > 0:
            return _clamp_probability(edge.reliability * edge.activation_probability)
        return _clamp_probability(
            edge.activation_probability
            * edge.reliability
            * edge.relevance
            * self.fallback_probability
        )


def fit_learned_propagation_from_trace_dicts(
    traces: list[dict[str, Any]],
    *,
    smoothing: float = 0.5,
) -> LearnedPropagationFit:
    """Estimate conditional edge activation probabilities from trace dictionaries."""

    edge_successes: dict[EdgeKey, float] = defaultdict(float)
    edge_counts: dict[EdgeKey, float] = defaultdict(float)
    source_counts: dict[str, float] = defaultdict(float)
    source_targets: dict[str, set[str]] = defaultdict(set)

    for trace in traces:
        events = trace.get("events", trace.get("messages", []))
        if not isinstance(events, list):
            raise ValueError("trace must contain an events or messages list")
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("trace event must be an object")
            source = _required_str(event, "source")
            target = _required_str(event, "target")
            key = (source, target)
            source_counts[source] += 1.0
            source_targets[source].add(target)
            edge_counts[key] += 1.0
            if bool(event.get("success", True)):
                edge_successes[key] += 1.0

    probabilities = _smoothed_probabilities(
        edge_successes,
        edge_counts,
        source_counts,
        source_targets,
        smoothing=smoothing,
    )
    return LearnedPropagationFit(
        edge_probabilities=probabilities,
        edge_counts=dict(edge_counts),
        source_counts=dict(source_counts),
    )


def fit_learned_propagation_from_graph(
    graph: AgentGraph,
    *,
    smoothing: float = 0.5,
) -> LearnedPropagationFit:
    """Estimate edge activation probabilities from trace metadata on a graph."""

    edge_successes: dict[EdgeKey, float] = defaultdict(float)
    edge_counts: dict[EdgeKey, float] = defaultdict(float)
    source_counts: dict[str, float] = defaultdict(float)
    source_targets: dict[str, set[str]] = defaultdict(set)

    for edge in graph.edges():
        count = float(edge.metadata.get("trace_message_count", edge.weight))
        successes = float(edge.metadata.get("trace_success_count", count * edge.reliability))
        key = (edge.source, edge.target)
        edge_counts[key] += count
        edge_successes[key] += successes
        source_counts[edge.source] += count
        source_targets[edge.source].add(edge.target)

    probabilities = _smoothed_probabilities(
        edge_successes,
        edge_counts,
        source_counts,
        source_targets,
        smoothing=smoothing,
    )
    return LearnedPropagationFit(
        edge_probabilities=probabilities,
        edge_counts=dict(edge_counts),
        source_counts=dict(source_counts),
    )


def _smoothed_probabilities(
    edge_successes: dict[EdgeKey, float],
    edge_counts: dict[EdgeKey, float],
    source_counts: dict[str, float],
    source_targets: dict[str, set[str]],
    *,
    smoothing: float,
) -> dict[EdgeKey, float]:
    probabilities = {}
    for key, count in edge_counts.items():
        source, _ = key
        target_count = max(len(source_targets[source]), 1)
        denominator = source_counts[source] + smoothing * target_count
        probabilities[key] = (edge_successes[key] + smoothing) / max(denominator, 1.0)
        if count == 0:
            probabilities[key] = 0.0
    return probabilities


def _required_str(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"trace event missing non-empty {key}")
    return value


def _graph_has_trace_metadata(graph: AgentGraph) -> bool:
    return any("trace_message_count" in edge.metadata for edge in graph.edges())


def _coverage(active_count: int, node_count: int) -> float:
    return active_count / max(node_count, 1)


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


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
