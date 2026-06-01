"""Cost and propagation metrics for workflow recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

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
    robustness: RobustnessSummary | None = None
    pruning_risk: PruningRiskSummary | None = None


@dataclass(slots=True)
class RobustnessSummary:
    """Connectivity robustness under simulated graph failures."""

    baseline_reachable_pairs: float
    average_node_failure_loss: float
    worst_node_failure_loss: float
    average_edge_failure_loss: float
    worst_edge_failure_loss: float


@dataclass(slots=True)
class PruningRiskSummary:
    """Risk and target-fit metrics for a proposed pruning plan."""

    target_cost_reduction: float
    achieved_cost_reduction: float
    coverage_delta: float
    coverage_loss: float
    target_gap: float
    risk_score: float


@dataclass(slots=True)
class QualityCostSummary:
    """Task-quality metrics normalized by cost."""

    success_rate: float
    token_cost: float
    latency: float
    cost_adjusted_success: float
    efficiency_score: float


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


def robustness_under_failures(graph: AgentGraph) -> RobustnessSummary:
    """Estimate reachability loss under single-node and single-edge failures."""

    nx_graph = graph.to_networkx()
    baseline_pairs = _reachable_pair_count(nx_graph)
    if baseline_pairs == 0:
        return RobustnessSummary(0.0, 0.0, 0.0, 0.0, 0.0)

    node_losses = []
    for node_id in nx_graph.nodes:
        reduced = nx_graph.copy()
        reduced.remove_node(node_id)
        node_losses.append((baseline_pairs - _reachable_pair_count(reduced)) / baseline_pairs)

    edge_losses = []
    for source, target in nx_graph.edges:
        reduced = nx_graph.copy()
        reduced.remove_edge(source, target)
        edge_losses.append((baseline_pairs - _reachable_pair_count(reduced)) / baseline_pairs)

    return RobustnessSummary(
        baseline_reachable_pairs=baseline_pairs,
        average_node_failure_loss=_mean(node_losses),
        worst_node_failure_loss=max(node_losses, default=0.0),
        average_edge_failure_loss=_mean(edge_losses),
        worst_edge_failure_loss=max(edge_losses, default=0.0),
    )


def quality_cost_summary(
    *,
    success_rate: float,
    token_cost: float,
    latency: float,
    token_cost_weight: float = 0.0001,
    latency_weight: float = 0.01,
) -> QualityCostSummary:
    """Compute cost-adjusted success and efficiency-style metrics."""

    if not 0 <= success_rate <= 1:
        raise ValueError("success_rate must be between 0 and 1")
    normalized_cost = max(token_cost, 1.0)
    return QualityCostSummary(
        success_rate=success_rate,
        token_cost=token_cost,
        latency=latency,
        cost_adjusted_success=success_rate / normalized_cost,
        efficiency_score=success_rate - token_cost_weight * token_cost - latency_weight * latency,
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
    robustness: RobustnessSummary | None = None,
    pruning_risk: PruningRiskSummary | None = None,
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
        robustness=robustness if robustness is not None else robustness_under_failures(graph),
        pruning_risk=pruning_risk,
    )


def _reachable_pair_count(nx_graph: nx.DiGraph) -> float:
    total = 0
    for source in nx_graph.nodes:
        total += len(nx.descendants(nx_graph, source))
    return float(total)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
