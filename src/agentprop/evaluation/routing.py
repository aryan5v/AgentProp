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
class ExpectedSuccessProfile:
    """Empirical expected-success penalties learned from routed task rows."""

    default_success: float
    node_context_penalties: dict[str, float] = field(default_factory=dict)
    high_context_threshold: float = 0.80
    source: str = "empirical"
    example_count: int = 0

    def estimate(
        self,
        graph: AgentGraph,
        *,
        context_ratios: dict[str, float],
    ) -> float:
        """Estimate task success from observed context sensitivity."""

        penalties = []
        for node in graph.nodes():
            if node.type == NodeType.OUTPUT:
                continue
            penalty = self.node_context_penalties.get(node.id, 0.0)
            if penalty <= 0.0:
                continue
            ratio = max(0.0, min(1.0, context_ratios.get(node.id, 0.0)))
            shortage = max(0.0, self.high_context_threshold - ratio)
            penalties.append(penalty * shortage / max(self.high_context_threshold, 1e-12))
        return max(0.0, min(1.0, self.default_success - sum(penalties)))

    def to_dict(self) -> dict[str, object]:
        """Serialize profile metadata and learned penalties."""

        return {
            "default_success": self.default_success,
            "node_context_penalties": dict(sorted(self.node_context_penalties.items())),
            "high_context_threshold": self.high_context_threshold,
            "source": self.source,
            "example_count": self.example_count,
        }


@dataclass(frozen=True, slots=True)
class QualityAwareRoutingObjective:
    """Expected-success objective for balancing quality and routing cost."""

    token_penalty: float = 0.0001
    context_penalty: float = 0.35
    success_profile: ExpectedSuccessProfile | None = None

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
            profile=self.success_profile,
        )
        return expected_success - self.token_penalty * cost.total_cost


def calibrate_expected_success(
    rows: list[dict[str, Any]],
    *,
    high_context_threshold: float = 0.80,
    default_success: float = 0.5,
) -> ExpectedSuccessProfile:
    """Fit expected-success penalties from empirical routed task rows.

    Rows should include a task outcome (`verification_passed`, `quality_passed`,
    `quality_score`, or `passed`) plus `context_allocations` or
    `context_ratios`. Retry-recommended infra rows are ignored.
    """

    outcomes = []
    high_context_outcomes: dict[str, list[float]] = defaultdict(list)
    low_context_outcomes: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        if bool(row.get("retry_recommended")):
            continue
        outcome = _row_outcome(row)
        if outcome is None:
            continue
        outcomes.append(outcome)

        context_ratios = _row_context_ratios(row)
        for seed in _string_list(row.get("selected_seeds")):
            context_ratios.setdefault(seed, 1.0)
        for node_id, ratio in context_ratios.items():
            if ratio >= high_context_threshold:
                high_context_outcomes[node_id].append(outcome)
            else:
                low_context_outcomes[node_id].append(outcome)

    if not outcomes:
        return ExpectedSuccessProfile(
            default_success=default_success,
            high_context_threshold=high_context_threshold,
            source="default",
            example_count=0,
        )

    baseline = mean(outcomes)
    penalties = {}
    for node_id in sorted(set(high_context_outcomes) | set(low_context_outcomes)):
        high = high_context_outcomes.get(node_id)
        low = low_context_outcomes.get(node_id)
        if not high or not low:
            continue
        penalty = mean(high) - mean(low)
        if penalty > 0.0:
            penalties[node_id] = max(0.0, min(1.0, penalty))

    return ExpectedSuccessProfile(
        default_success=max(0.0, min(1.0, baseline)),
        node_context_penalties=penalties,
        high_context_threshold=high_context_threshold,
        example_count=len(outcomes),
    )


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
    profile: ExpectedSuccessProfile | None = None,
) -> float:
    """Estimate workflow success from empirical data or heuristic fallback."""

    if profile is not None:
        return profile.estimate(graph, context_ratios=context_ratios)

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


def _row_context_ratios(row: dict[str, Any]) -> dict[str, float]:
    raw = row.get("context_allocations") or row.get("context_ratios")
    if not isinstance(raw, dict):
        return {}
    ratios = {}
    for node_id, value in raw.items():
        if isinstance(value, int | float):
            ratios[str(node_id)] = max(0.0, min(1.0, float(value)))
    return ratios


def _row_outcome(row: dict[str, Any]) -> float | None:
    verification = row.get("verification_passed")
    if isinstance(verification, bool):
        return 1.0 if verification else 0.0
    quality_passed = row.get("quality_passed")
    if isinstance(quality_passed, bool):
        return 1.0 if quality_passed else 0.0
    quality_score = row.get("quality_score")
    if isinstance(quality_score, int | float):
        return max(0.0, min(1.0, float(quality_score)))
    passed = row.get("passed")
    if isinstance(passed, bool):
        return 1.0 if passed else 0.0
    return None


def _string_list(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str)]
    return []
