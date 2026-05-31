"""Cost and propagation metrics for workflow recommendations."""

from __future__ import annotations

from dataclasses import dataclass

from agentprop.core import AgentGraph
from agentprop.propagation import PropagationResult


@dataclass(slots=True)
class CostSummary:
    """Token/message/latency cost estimate for a routing strategy."""

    token_cost: float
    message_cost: float
    latency: float
    message_count: int

    @property
    def total_cost(self) -> float:
        """Combined token and message cost."""

        return self.token_cost + self.message_cost


@dataclass(slots=True)
class RecommendationReport:
    """Single optimization report for CLI and library users."""

    seeds: list[str]
    propagation_model: str
    propagation: PropagationResult
    optimized_cost: CostSummary
    broadcast_cost: CostSummary
    estimated_savings: float
    bottlenecks: list[tuple[str, float]]
    pruning_candidates: list[tuple[str, str]]
    verifier_candidates: list[str]


def broadcast_cost(graph: AgentGraph) -> CostSummary:
    """Estimate cost for sending full context through every node and edge."""

    token_cost = sum(node.token_cost for node in graph.nodes())
    message_cost = sum(edge.message_cost for edge in graph.edges())
    node_latency = sum(node.latency for node in graph.nodes())
    edge_latency = sum(edge.latency for edge in graph.edges())
    latency = node_latency + edge_latency
    return CostSummary(
        token_cost=token_cost,
        message_cost=message_cost,
        latency=latency,
        message_count=graph.edge_count,
    )


def seeded_routing_cost(
    graph: AgentGraph,
    seeds: list[str],
    activated_nodes: set[str],
) -> CostSummary:
    """Estimate cost for giving full context to seeds and routing only activated edges."""

    seed_set = set(seeds)
    token_cost = 0.0
    latency = 0.0
    for node in graph.nodes():
        if node.id in seed_set:
            token_cost += node.token_cost
            latency += node.latency
        elif node.id in activated_nodes:
            token_cost += node.token_cost * 0.35
            latency += node.latency * 0.5

    message_cost = 0.0
    message_count = 0
    for edge in graph.edges():
        if edge.source in activated_nodes and edge.target in activated_nodes:
            message_cost += edge.message_cost
            latency += edge.latency
            message_count += 1

    return CostSummary(
        token_cost=token_cost,
        message_cost=message_cost,
        latency=latency,
        message_count=message_count,
    )


def compare_routing(
    graph: AgentGraph,
    seeds: list[str],
    propagation_model: str,
    propagation: PropagationResult,
    *,
    bottlenecks: list[tuple[str, float]] | None = None,
    pruning_candidates: list[tuple[str, str]] | None = None,
    verifier_candidates: list[str] | None = None,
) -> RecommendationReport:
    """Compare broadcast routing with a seed-based selective routing plan."""

    broadcast = broadcast_cost(graph)
    optimized = seeded_routing_cost(graph, seeds, propagation.activated_nodes)
    estimated_savings = 0.0
    if broadcast.total_cost > 0:
        estimated_savings = (broadcast.total_cost - optimized.total_cost) / broadcast.total_cost

    return RecommendationReport(
        seeds=seeds,
        propagation_model=propagation_model,
        propagation=propagation,
        optimized_cost=optimized,
        broadcast_cost=broadcast,
        estimated_savings=estimated_savings,
        bottlenecks=bottlenecks or [],
        pruning_candidates=pruning_candidates or [],
        verifier_candidates=verifier_candidates or [],
    )
