"""Base protocol and helpers for graph propagation simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentprop.core import AgentGraph


@dataclass(slots=True)
class PropagationResult:
    """Summary metrics from a propagation simulation."""

    activated_nodes: set[str]
    propagation_time: int
    coverage: float
    activation_rounds: dict[str, int]
    trials: int = 1
    full_activation_probability: float | None = None
    expected_propagation_time: float | None = None
    coverage_by_round: list[float] | None = None


class PropagationModel(Protocol):
    """Interface all propagation models should implement."""

    name: str

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 1,
    ) -> PropagationResult:
        """Run propagation from seed nodes."""


def deterministic_result(
    graph: AgentGraph,
    activation_rounds: dict[str, int],
    *,
    trials: int = 1,
    full_activation_probability: float | None = None,
    expected_propagation_time: float | None = None,
    coverage_by_round: list[float] | None = None,
) -> PropagationResult:
    """Build a standard result object from activation rounds."""

    node_count = max(graph.node_count, 1)
    activated_nodes = set(activation_rounds)
    propagation_time = max(activation_rounds.values(), default=0)
    coverage = len(activated_nodes) / node_count
    return PropagationResult(
        activated_nodes=activated_nodes,
        propagation_time=propagation_time,
        coverage=coverage,
        activation_rounds=activation_rounds,
        trials=trials,
        full_activation_probability=full_activation_probability,
        expected_propagation_time=expected_propagation_time,
        coverage_by_round=coverage_by_round,
    )
