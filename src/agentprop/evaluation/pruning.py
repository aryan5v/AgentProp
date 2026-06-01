"""Evaluate edge-pruning decisions against propagation behavior."""

from __future__ import annotations

from dataclasses import dataclass

from agentprop.core import AgentGraph
from agentprop.evaluation.metrics import PruningRiskSummary
from agentprop.propagation import IndependentCascade, PropagationModel


@dataclass(slots=True)
class PruningEvaluation:
    """Effect of removing candidate edges."""

    removed_edges: list[tuple[str, str]]
    baseline_coverage: float
    pruned_coverage: float
    coverage_delta: float
    baseline_cost: float
    pruned_cost: float
    cost_delta: float


def evaluate_pruning(
    graph: AgentGraph,
    edges_to_remove: list[tuple[str, str]],
    *,
    seeds: list[str],
    propagation_model: PropagationModel | None = None,
    trials: int = 50,
) -> PruningEvaluation:
    """Compare propagation and message cost before/after removing edges."""

    model = propagation_model or IndependentCascade(seed=0)
    baseline = model.simulate(graph, seeds, trials=trials)
    pruned = _without_edges(graph, edges_to_remove)
    pruned_result = model.simulate(pruned, seeds, trials=trials)
    baseline_cost = sum(edge.message_cost for edge in graph.edges())
    pruned_cost = sum(edge.message_cost for edge in pruned.edges())
    return PruningEvaluation(
        removed_edges=edges_to_remove,
        baseline_coverage=baseline.coverage,
        pruned_coverage=pruned_result.coverage,
        coverage_delta=pruned_result.coverage - baseline.coverage,
        baseline_cost=baseline_cost,
        pruned_cost=pruned_cost,
        cost_delta=pruned_cost - baseline_cost,
    )


def summarize_pruning_risk(
    evaluation: PruningEvaluation,
    *,
    target_cost_reduction: float,
) -> PruningRiskSummary:
    """Summarize pruning risk as coverage harm plus target miss."""

    if not 0 <= target_cost_reduction <= 1:
        raise ValueError("target_cost_reduction must be between 0 and 1")
    achieved_cost_reduction = (
        0.0
        if evaluation.baseline_cost == 0
        else (evaluation.baseline_cost - evaluation.pruned_cost) / evaluation.baseline_cost
    )
    coverage_loss = max(0.0, -evaluation.coverage_delta)
    target_gap = max(0.0, target_cost_reduction - achieved_cost_reduction)
    return PruningRiskSummary(
        target_cost_reduction=target_cost_reduction,
        achieved_cost_reduction=achieved_cost_reduction,
        coverage_delta=evaluation.coverage_delta,
        coverage_loss=coverage_loss,
        target_gap=target_gap,
        risk_score=coverage_loss + target_gap,
    )


def _without_edges(graph: AgentGraph, edges_to_remove: list[tuple[str, str]]) -> AgentGraph:
    data = graph.to_dict()
    blocked = set(edges_to_remove)
    data["edges"] = [
        edge
        for edge in data["edges"]
        if (str(edge["source"]), str(edge["target"])) not in blocked
    ]
    return AgentGraph.from_dict(data)
