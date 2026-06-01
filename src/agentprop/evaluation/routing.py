"""Quality-aware routing utilities for context-sensitive workflows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from agentprop.core import AgentGraph, NodeType
from agentprop.evaluation.metrics import CostSummary


@dataclass(frozen=True, slots=True)
class ContextCompressionProfile:
    """Measured context-compression ratios from routed workflow traces."""

    default_ratio: float = 0.35
    node_ratios: dict[str, float] = field(default_factory=dict)

    def ratio_for(self, node_id: str) -> float:
        """Return a bounded compression ratio for a node."""

        ratio = self.node_ratios.get(node_id, self.default_ratio)
        return max(0.0, min(1.0, ratio))


@dataclass(frozen=True, slots=True)
class RoutingRisk:
    """A risk signal attached to a routing recommendation."""

    node_id: str
    severity: str
    risk_score: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "node": self.node_id,
            "severity": self.severity,
            "risk_score": self.risk_score,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class QualityAwareRoutingObjective:
    """Expected-success objective for balancing quality and routing cost."""

    token_penalty: float = 0.0001
    context_penalty: float = 0.35

    def score(
        self,
        graph: AgentGraph,
        *,
        seeds: list[str],
        activated_nodes: set[str],
        cost: CostSummary,
        context_ratios: dict[str, float] | None = None,
    ) -> float:
        """Score a routing plan as expected success minus token/message cost."""

        ratios = context_ratios or graded_context_allocations(
            graph,
            seeds=seeds,
            activated_nodes=activated_nodes,
        )
        expected_success = estimate_expected_success(
            graph,
            context_ratios=ratios,
            context_penalty=self.context_penalty,
        )
        return expected_success - self.token_penalty * cost.total_cost


def calibrate_context_compression(
    rows: list[dict[str, Any]],
    *,
    default_ratio: float = 0.35,
) -> ContextCompressionProfile:
    """Fit per-node compression ratios from measured full vs compressed calls.

    Rows may come from the real-routing harness. The function accepts either
    ``stage_prompt_tokens`` or ``stage_tokens`` plus ``stage_full_context``.
    """

    full_costs: dict[str, list[float]] = defaultdict(list)
    compressed_costs: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        stage_full_context = row.get("stage_full_context")
        stage_costs = row.get("stage_prompt_tokens") or row.get("stage_tokens")
        if not isinstance(stage_full_context, dict) or not isinstance(stage_costs, dict):
            continue
        for node_id, raw_cost in stage_costs.items():
            try:
                cost = float(raw_cost)
            except (TypeError, ValueError):
                continue
            if bool(stage_full_context.get(node_id)):
                full_costs[str(node_id)].append(cost)
            else:
                compressed_costs[str(node_id)].append(cost)

    ratios: dict[str, float] = {}
    for node_id, compressed in compressed_costs.items():
        full = full_costs.get(node_id)
        if full:
            ratios[node_id] = max(0.0, min(1.0, mean(compressed) / max(mean(full), 1.0)))

    return ContextCompressionProfile(default_ratio=default_ratio, node_ratios=ratios)


def graded_context_allocations(
    graph: AgentGraph,
    *,
    seeds: list[str],
    activated_nodes: set[str] | None = None,
    profile: ContextCompressionProfile | None = None,
    min_ratio: float = 0.25,
    max_non_seed_ratio: float = 0.85,
) -> dict[str, float]:
    """Assign each node a context ratio instead of binary full-or-summary routing."""

    seed_set = set(seeds)
    active = activated_nodes or {node.id for node in graph.nodes()}
    profile = profile or ContextCompressionProfile(default_ratio=min_ratio)
    allocations: dict[str, float] = {}

    for node in graph.nodes():
        if node.type == NodeType.OUTPUT:
            allocations[node.id] = 0.0
        elif node.id in seed_set:
            allocations[node.id] = 1.0
        elif node.id not in active:
            allocations[node.id] = 0.0
        else:
            importance = _importance(node.importance_score, node.type, node.id, node.role)
            relevance = _incoming_relevance(graph, node.id)
            measured = profile.ratio_for(node.id)
            graded = min_ratio + (1.0 - min_ratio) * relevance * importance
            allocations[node.id] = max(measured, min(max_non_seed_ratio, graded))

    return allocations


def estimate_expected_success(
    graph: AgentGraph,
    *,
    context_ratios: dict[str, float],
    context_penalty: float = 0.35,
) -> float:
    """Estimate workflow success from reliability and context starvation risk."""

    node_scores = []
    for node in graph.nodes():
        if node.type == NodeType.OUTPUT:
            continue
        importance = _importance(node.importance_score, node.type, node.id, node.role)
        context_ratio = context_ratios.get(node.id, 0.0)
        starvation = max(0.0, 1.0 - context_ratio) * importance
        base = max(0.0, min(1.0, node.reliability * (1.0 - node.error_rate)))
        node_scores.append(max(0.0, base - context_penalty * starvation))
    return mean(node_scores) if node_scores else 1.0


def routing_risks(
    graph: AgentGraph,
    *,
    context_ratios: dict[str, float],
    high_risk_threshold: float = 0.40,
) -> list[RoutingRisk]:
    """Surface risk when high-sensitivity nodes receive compressed context."""

    risks: list[RoutingRisk] = []
    for node in graph.nodes():
        if node.type == NodeType.OUTPUT:
            continue
        importance = _importance(node.importance_score, node.type, node.id, node.role)
        ratio = context_ratios.get(node.id, 0.0)
        if ratio <= 0.0:
            continue
        risk_score = importance * max(0.0, 1.0 - ratio)
        if risk_score <= 0:
            continue
        if risk_score >= high_risk_threshold:
            severity = "high"
        elif risk_score >= high_risk_threshold / 2:
            severity = "medium"
        else:
            severity = "low"
        if severity == "low":
            continue
        risks.append(
            RoutingRisk(
                node_id=node.id,
                severity=severity,
                risk_score=risk_score,
                reason=(
                    f"{node.id} has context sensitivity {importance:.2f} "
                    f"but receives {ratio:.0%} of full context"
                ),
            )
        )
    return sorted(risks, key=lambda risk: (-risk.risk_score, risk.node_id))


def _incoming_relevance(graph: AgentGraph, node_id: str) -> float:
    incoming = [
        graph.edge(source, node_id).relevance
        for source in graph.predecessors(node_id)
        if source != node_id
    ]
    return max(incoming, default=1.0)


def _importance(
    explicit: float | None,
    node_type: NodeType,
    node_id: str,
    role: str | None,
) -> float:
    if explicit is not None:
        return max(0.0, min(1.0, explicit))
    label = f"{node_id} {role or ''}".lower()
    if node_type == NodeType.EXECUTOR or "coder" in label or "implement" in label:
        return 0.90
    if node_type == NodeType.VERIFIER or "test" in label or "verif" in label:
        return 0.80
    if node_type == NodeType.REVIEWER:
        return 0.65
    if node_type == NodeType.PLANNER:
        return 0.55
    return 0.35
