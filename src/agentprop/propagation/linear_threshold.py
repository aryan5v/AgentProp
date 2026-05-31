"""Linear Threshold propagation model."""

from __future__ import annotations

from agentprop.core import AgentGraph
from agentprop.propagation.base import PropagationResult, deterministic_result


class LinearThreshold:
    """Activate nodes when incoming active influence crosses a threshold."""

    name = "linear-threshold"

    def __init__(self, *, threshold: float = 0.5, max_rounds: int | None = None) -> None:
        self.threshold = threshold
        self.max_rounds = max_rounds

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 1,
    ) -> PropagationResult:
        """Run deterministic Linear Threshold propagation."""

        nx_graph = graph.to_networkx()
        active = {seed for seed in seeds if seed in nx_graph}
        activation_rounds = {seed: 0 for seed in active}
        max_rounds = self.max_rounds or max(1, graph.node_count)
        coverage_by_round = [len(active) / max(graph.node_count, 1)]

        for round_number in range(1, max_rounds + 1):
            next_active = set(active)
            for node_id in nx_graph.nodes:
                node_id = str(node_id)
                if node_id in active:
                    continue
                if self._incoming_share(graph, node_id, active) >= self.threshold:
                    next_active.add(node_id)
                    activation_rounds[node_id] = round_number
            if next_active == active:
                break
            active = next_active
            coverage_by_round.append(len(active) / max(graph.node_count, 1))

        return deterministic_result(
            graph,
            activation_rounds,
            trials=trials,
            full_activation_probability=1.0 if len(active) == graph.node_count else 0.0,
            expected_propagation_time=float(max(activation_rounds.values(), default=0)),
            coverage_by_round=coverage_by_round,
        )

    def _incoming_share(self, graph: AgentGraph, node_id: str, active: set[str]) -> float:
        incoming = graph.predecessors(node_id)
        total = sum(max(graph.edge(source, node_id).weight, 0.0) for source in incoming)
        if total == 0:
            return 0.0
        active_total = sum(
            max(graph.edge(source, node_id).weight, 0.0) for source in incoming if source in active
        )
        return active_total / total
