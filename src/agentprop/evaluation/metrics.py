"""Cost and propagation metrics for workflow recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from agentprop.core import AgentGraph, NodeType
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
class WhatIfKEntry:
    """Coverage/cost snapshot for a seed-budget what-if analysis."""

    k: int
    seeds: list[str]
    coverage: float
    coverage_std: float
    estimated_savings: float
    quality_objective_score: float | None = None


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
    context_allocations: dict[str, float] = field(default_factory=dict)
    routing_risks: list[Any] = field(default_factory=list)
    quality_objective_score: float | None = None
    what_if_k: list[WhatIfKEntry] = field(default_factory=list)
    coverage_uncertainty: float | None = None
    algorithm_path: str = "exact"


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


@dataclass(slots=True)
class CoverageConstrainedCost:
    """Cost metrics that reward reaching critical nodes, not minimizing activity.

    The plain ``estimated_savings`` metric is confounded with coverage: a
    strategy that activates fewer nodes always looks cheaper, so random seeding
    can "win" by reaching almost nothing. These metrics close that loophole by
    treating OUTPUT/VERIFIER nodes as a goal that must be reached: savings are
    only credited when every critical node is activated, and cost is normalized
    by coverage so doing nothing is penalized rather than rewarded.
    """

    critical_coverage: float
    """Fraction of critical (OUTPUT/VERIFIER) nodes that are activated."""

    cost_per_coverage: float
    """Optimized total cost divided by overall coverage (lower is better)."""

    constrained_savings: float
    """Estimated savings, credited only when all critical nodes are reached."""

    reached_goal: bool
    """True when every critical node is activated."""


CRITICAL_NODE_TYPES: frozenset[NodeType] = frozenset({NodeType.OUTPUT, NodeType.VERIFIER})


def coverage_constrained_cost(
    graph: AgentGraph,
    propagation: PropagationResult,
    *,
    optimized_cost: CostSummary,
    broadcast_cost_summary: CostSummary,
) -> CoverageConstrainedCost:
    """Compute goal-aware cost metrics that 'do nothing' cannot win.

    Critical nodes are OUTPUT and VERIFIER nodes — the workflow only succeeds
    if these are reached. ``constrained_savings`` credits cost reduction only
    when all critical nodes are activated; ``cost_per_coverage`` normalizes the
    optimized cost by coverage so low-coverage strategies are penalized.
    """

    critical = {node.id for node in graph.nodes() if node.type in CRITICAL_NODE_TYPES}
    activated = set(propagation.activated_nodes)
    if critical:
        critical_coverage = len(critical & activated) / len(critical)
    else:
        # No explicit goal nodes: fall back to overall coverage as the target.
        critical_coverage = propagation.coverage
    reached_goal = critical_coverage >= 1.0 - 1e-9

    coverage = max(propagation.coverage, 1e-9)
    cost_per_coverage = optimized_cost.total_cost / coverage

    estimated_savings = 0.0
    if broadcast_cost_summary.total_cost > 0:
        estimated_savings = (
            broadcast_cost_summary.total_cost - optimized_cost.total_cost
        ) / broadcast_cost_summary.total_cost
    constrained_savings = estimated_savings if reached_goal else 0.0

    return CoverageConstrainedCost(
        critical_coverage=critical_coverage,
        cost_per_coverage=cost_per_coverage,
        constrained_savings=constrained_savings,
        reached_goal=reached_goal,
    )


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

    baseline_pairs = graph.get_reachable_pair_count()
    if baseline_pairs == 0:
        return RobustnessSummary(0.0, 0.0, 0.0, 0.0, 0.0)
    nx_graph = graph.to_networkx()

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
    *,
    context_ratios: dict[str, float] | None = None,
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
            ratio = context_ratios.get(node.id, 0.35) if context_ratios is not None else 0.35
            ratio = max(0.0, min(1.0, ratio))
            token_cost += node.token_cost * ratio
            latency += node.latency * max(0.25, ratio)

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


def build_what_if_k_curve(
    graph: AgentGraph,
    *,
    model,
    candidate_seeds: list[str],
    max_k: int,
    trials: int = 50,
) -> list[WhatIfKEntry]:
    """Simulate coverage/cost for k=1..max_k using greedy seed augmentation."""

    from agentprop.evaluation.routing import (
        QualityAwareRoutingObjective,
        graded_context_allocations,
    )

    if max_k < 1 or not candidate_seeds:
        return []
    ordered = list(dict.fromkeys(candidate_seeds))
    limit = min(max_k, len(ordered))
    selected: list[str] = []
    entries: list[WhatIfKEntry] = []
    broadcast = broadcast_cost(graph)
    remaining = list(ordered)
    for k in range(1, limit + 1):
        if k == 1:
            best = max(remaining, key=lambda seed: model.simulate(graph, [seed], trials=trials).coverage)
            selected = [best]
            remaining.remove(best)
        else:
            best = max(
                remaining,
                key=lambda seed: model.simulate(graph, selected + [seed], trials=trials).coverage,
            )
            selected.append(best)
            remaining.remove(best)
        propagation = model.simulate(graph, selected, trials=trials)
        allocations = graded_context_allocations(
            graph,
            seeds=selected,
            activated_nodes=propagation.activated_nodes,
        )
        optimized = seeded_routing_cost(
            graph,
            selected,
            propagation.activated_nodes,
            context_ratios=allocations,
        )
        savings = 0.0
        if broadcast.total_cost > 0:
            savings = (broadcast.total_cost - optimized.total_cost) / broadcast.total_cost
        uncertainty = 0.0
        if propagation.full_activation_probability is not None:
            uncertainty = max(0.0, 1.0 - propagation.full_activation_probability)
        entries.append(
            WhatIfKEntry(
                k=k,
                seeds=list(selected),
                coverage=propagation.coverage,
                coverage_std=uncertainty,
                estimated_savings=savings,
                quality_objective_score=QualityAwareRoutingObjective().score(
                    graph,
                    seeds=selected,
                    activated_nodes=propagation.activated_nodes,
                    cost=optimized,
                    context_ratios=allocations,
                ),
            )
        )
    return entries


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

    from agentprop.evaluation.routing import (
        QualityAwareRoutingObjective,
        graded_context_allocations,
        routing_risks,
    )  # noqa: PLC0415 — local import avoids cycle with routing helpers

    context_allocations = graded_context_allocations(
        graph,
        seeds=seeds,
        activated_nodes=propagation.activated_nodes,
    )
    broadcast = broadcast_cost(graph)
    optimized = seeded_routing_cost(
        graph,
        seeds,
        propagation.activated_nodes,
        context_ratios=context_allocations,
    )
    estimated_savings = 0.0
    if broadcast.total_cost > 0:
        estimated_savings = (broadcast.total_cost - optimized.total_cost) / broadcast.total_cost
    quality_score = QualityAwareRoutingObjective().score(
        graph,
        seeds=seeds,
        activated_nodes=propagation.activated_nodes,
        cost=optimized,
        context_ratios=context_allocations,
    )

    uncertainty = None
    if propagation.full_activation_probability is not None:
        uncertainty = max(0.0, 1.0 - propagation.full_activation_probability)

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
        context_allocations=context_allocations,
        routing_risks=routing_risks(graph, context_ratios=context_allocations),
        quality_objective_score=quality_score,
        coverage_uncertainty=uncertainty,
        algorithm_path="fast" if graph.node_count > 15 else "exact",
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
