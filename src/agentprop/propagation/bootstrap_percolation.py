"""Bootstrap percolation propagation model."""

from __future__ import annotations

from agentprop.core import AgentGraph
from agentprop.propagation.base import PropagationResult, deterministic_result


class BootstrapPercolation:
    """Activate a node after enough incoming neighbors are active."""

    name = "bootstrap-percolation"

    def __init__(self, *, threshold: int = 2, max_rounds: int | None = None) -> None:
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        self.threshold = threshold
        self.max_rounds = max_rounds

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 1,
    ) -> PropagationResult:
        """Run deterministic bootstrap percolation."""

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
                active_incoming = sum(
                    1 for source in graph.predecessors(node_id) if source in active
                )
                if active_incoming >= self.threshold:
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
