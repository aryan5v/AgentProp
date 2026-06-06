"""Precomputed adjacency indices for fast propagation simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentprop.core.graph import AgentGraph


@dataclass(slots=True)
class PropagationGraphIndex:
    """Integer-indexed adjacency for Monte Carlo propagation kernels.

    Built once per graph fingerprint and reused across simulate / simulate_batch
    calls to avoid repeated ``to_networkx()`` copies and per-trial object churn.
    """

    node_ids: list[str]
    node_index: dict[str, int]
    # IC: (target_idx, activation_probability) per source
    ic_successors: list[list[tuple[int, float]]]
    # RZF: (predecessor_idx, edge_weight) per target
    rzf_predecessors: list[list[tuple[int, float]]]
    node_count: int


def build_propagation_index(graph: AgentGraph) -> PropagationGraphIndex:  # type: ignore[name-defined]
    """Materialize adjacency lists with precomputed edge probabilities."""

    node_ids = graph.node_ids()
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    ic_successors: list[list[tuple[int, float]]] = [[] for _ in node_ids]
    rzf_predecessors: list[list[tuple[int, float]]] = [[] for _ in node_ids]

    for edge in graph.edges():
        source_idx = node_index.get(edge.source)
        target_idx = node_index.get(edge.target)
        if source_idx is None or target_idx is None:
            continue
        probability = edge.activation_probability * edge.reliability * edge.relevance
        bounded = max(0.0, min(1.0, probability))
        ic_successors[source_idx].append((target_idx, bounded))
        weight = max(edge.weight, 0.0)
        rzf_predecessors[target_idx].append((source_idx, weight))

    return PropagationGraphIndex(
        node_ids=node_ids,
        node_index=node_index,
        ic_successors=ic_successors,
        rzf_predecessors=rzf_predecessors,
        node_count=len(node_ids),
    )
