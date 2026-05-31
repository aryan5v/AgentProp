"""Base protocol for graph propagation simulations."""

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
    full_activation_probability: float | None = None
    expected_propagation_time: float | None = None


class PropagationModel(Protocol):
    """Interface all propagation models should implement."""

    def simulate(self, graph: AgentGraph, seeds: list[str], *, trials: int = 1) -> PropagationResult:
        """Run propagation from seed nodes."""
