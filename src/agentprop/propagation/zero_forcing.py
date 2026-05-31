"""Deterministic classical zero forcing propagation."""

from __future__ import annotations

from agentprop.core import AgentGraph
from agentprop.propagation.base import PropagationResult, deterministic_result


class ZeroForcing:
    """Classical color-change zero forcing on directed workflow graphs.

    A blue node forces its only white outgoing neighbor. For directed agent
    workflows, outgoing neighbors are used because edges represent information
    flow direction.
    """

    name = "zero-forcing"

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 1,
    ) -> PropagationResult:
        """Run deterministic zero forcing from seed nodes."""

        nx_graph = graph.to_networkx()
        blue = {seed for seed in seeds if seed in nx_graph}
        activation_rounds = {seed: 0 for seed in blue}
        coverage_by_round = [len(blue) / max(graph.node_count, 1)]
        round_number = 0

        while True:
            round_number += 1
            forced: set[str] = set()
            for node_id in list(blue):
                white_successors = [
                    str(target)
                    for target in nx_graph.successors(node_id)
                    if str(target) not in blue
                ]
                if len(white_successors) == 1:
                    forced.add(white_successors[0])

            if not forced:
                break

            for node_id in sorted(forced):
                blue.add(node_id)
                activation_rounds[node_id] = round_number
            coverage_by_round.append(len(blue) / max(graph.node_count, 1))

            if len(blue) == graph.node_count:
                break

        return deterministic_result(
            graph,
            activation_rounds,
            trials=trials,
            full_activation_probability=1.0 if len(blue) == graph.node_count else 0.0,
            expected_propagation_time=float(max(activation_rounds.values(), default=0)),
            coverage_by_round=coverage_by_round,
        )
