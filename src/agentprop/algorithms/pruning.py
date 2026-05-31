"""Edge pruning heuristics."""

from __future__ import annotations

from agentprop.core import AgentGraph


def low_weight_edges(graph: AgentGraph, *, fraction: float = 0.2) -> list[tuple[str, str]]:
    """Return the lowest-weight edges as pruning candidates."""

    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    edges = sorted(graph.edges(), key=lambda edge: (edge.weight, edge.relevance, edge.reliability))
    keep_count = int(round(len(edges) * fraction))
    return [(edge.source, edge.target) for edge in edges[:keep_count]]


def high_cost_low_relevance_edges(graph: AgentGraph, *, limit: int = 5) -> list[tuple[str, str]]:
    """Return expensive edges with low relevance/reliability."""

    scored = []
    for edge in graph.edges():
        quality = max(edge.relevance * edge.reliability, 0.01)
        score = edge.message_cost / quality
        scored.append((score, edge.source, edge.target))
    return [(source, target) for _, source, target in sorted(scored, reverse=True)[:limit]]
