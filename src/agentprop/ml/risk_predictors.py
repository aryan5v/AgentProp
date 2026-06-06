"""Lightweight calibrated risk predictors for adaptive routing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from agentprop.core import AgentGraph
from agentprop.evaluation.metrics import CostSummary
from agentprop.evaluation.routing import ExpectedSuccessProfile, estimate_expected_success
from agentprop.propagation.base import PropagationResult


@dataclass(slots=True)
class TimeoutRiskPredictor:
    """Predict timeout risk from graph latency + node error rates."""

    latency_weight: float = 0.02
    error_weight: float = 0.35
    budget_penalty: float = 0.15

    def predict(
        self,
        graph: AgentGraph,
        *,
        activated_nodes: set[str],
        wall_time_budget_s: float | None = None,
        observed_elapsed_s: float = 0.0,
    ) -> float:
        """Return timeout risk in [0, 1]."""

        if not activated_nodes:
            return 0.0
        latency = 0.0
        error_signal = 0.0
        for node in graph.nodes():
            if node.id not in activated_nodes:
                continue
            latency += node.latency
            error_signal = max(error_signal, node.error_rate)
        risk = self.latency_weight * latency + self.error_weight * error_signal
        if wall_time_budget_s is not None and wall_time_budget_s > 0:
            utilization = observed_elapsed_s / wall_time_budget_s
            if utilization > 0.7:
                risk += self.budget_penalty * min(1.0, utilization)
        return max(0.0, min(1.0, risk))


@dataclass(slots=True)
class AdaptiveRoutingScorer:
    """Combine expected success, timeout risk, and cost for seed/verifier choices."""

    success_profile: ExpectedSuccessProfile | None = None
    timeout_predictor: TimeoutRiskPredictor = field(default_factory=TimeoutRiskPredictor)
    alpha_token_savings: float = 0.10
    beta_quality_loss: float = 0.55
    gamma_timeout_risk: float = 0.25

    def score_plan(
        self,
        graph: AgentGraph,
        *,
        seeds: list[str],
        propagation: PropagationResult,
        cost: CostSummary,
        context_ratios: dict[str, float],
        broadcast_cost: CostSummary,
        wall_time_budget_s: float | None = None,
    ) -> float:
        """Higher is better: success minus quality/timeout penalties plus savings."""

        activated = set(propagation.activated_nodes)
        expected_success = estimate_expected_success(
            graph,
            context_ratios=context_ratios,
            profile=self.success_profile,
        )
        timeout_risk = self.timeout_predictor.predict(
            graph,
            activated_nodes=activated,
            wall_time_budget_s=wall_time_budget_s,
        )
        quality_loss = max(0.0, 1.0 - expected_success)
        token_savings = 0.0
        if broadcast_cost.total_cost > 0:
            token_savings = (broadcast_cost.total_cost - cost.total_cost) / broadcast_cost.total_cost
        return (
            expected_success
            + self.alpha_token_savings * max(-1.0, min(1.0, token_savings))
            - self.beta_quality_loss * quality_loss
            - self.gamma_timeout_risk * timeout_risk
        )


def min_k_for_quality_floor(
    graph: AgentGraph,
    *,
    quality_floor: float,
    candidate_seeds: list[str],
    simulate,
    trials: int = 50,
    max_k: int | None = None,
) -> tuple[int, list[str], float]:
    """Find minimal seed budget k such that predicted quality >= quality_floor.

    ``simulate`` is a callable(graph, seeds, trials=...) -> PropagationResult.
    Uses a greedy augmentation starting from the best single seed.
    """

    if not candidate_seeds:
        return 0, [], 0.0
    limit = max_k or min(len(candidate_seeds), graph.node_count)
    ordered = list(candidate_seeds)
    best_single = max(
        ordered,
        key=lambda seed: simulate(graph, [seed], trials=trials).coverage,
    )
    selected = [best_single]
    best_coverage = simulate(graph, selected, trials=trials).coverage
    if best_coverage >= quality_floor:
        return 1, selected, best_coverage

    remaining = [seed for seed in ordered if seed not in selected]
    while len(selected) < limit and remaining:
        best_candidate = max(
            remaining,
            key=lambda seed: simulate(graph, selected + [seed], trials=trials).coverage,
        )
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_coverage = simulate(graph, selected, trials=trials).coverage
        if best_coverage >= quality_floor:
            break

    return len(selected), selected, best_coverage


@dataclass(slots=True)
class LearnedRiskState:
    """Persisted scorer/bandit auxiliary state for continual learning."""

    category_timeout_bias: dict[str, float] = field(default_factory=dict)
    category_quality_bias: dict[str, float] = field(default_factory=dict)
    example_count: int = 0

    def update_from_outcome(
        self,
        *,
        category: str,
        passed: bool,
        quality_score: float | None,
        elapsed_s: float,
        wall_time_budget_s: float | None,
    ) -> None:
        self.example_count += 1
        quality = quality_score if quality_score is not None else (1.0 if passed else 0.0)
        self.category_quality_bias[category] = _ema(
            self.category_quality_bias.get(category, 0.5),
            quality,
        )
        if wall_time_budget_s and wall_time_budget_s > 0:
            timeout_signal = min(1.0, elapsed_s / wall_time_budget_s)
            self.category_timeout_bias[category] = _ema(
                self.category_timeout_bias.get(category, 0.0),
                timeout_signal,
            )

    def timeout_adjustment(self, category: str) -> float:
        return self.category_timeout_bias.get(category, 0.0)

    def quality_adjustment(self, category: str) -> float:
        return self.category_quality_bias.get(category, 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category_timeout_bias": dict(sorted(self.category_timeout_bias.items())),
            "category_quality_bias": dict(sorted(self.category_quality_bias.items())),
            "example_count": self.example_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedRiskState:
        return cls(
            category_timeout_bias={
                str(k): float(v)
                for k, v in dict(data.get("category_timeout_bias", {})).items()
            },
            category_quality_bias={
                str(k): float(v)
                for k, v in dict(data.get("category_quality_bias", {})).items()
            },
            example_count=int(data.get("example_count", 0)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> LearnedRiskState:
        return cls.from_dict(json.loads(Path(path).read_text()))


def calibrate_timeout_risk_from_rows(
    rows: list[dict[str, Any]],
    *,
    base: TimeoutRiskPredictor | None = None,
) -> TimeoutRiskPredictor:
    """Fit timeout weights from empirical elapsed/budget rows."""

    predictor = base or TimeoutRiskPredictor()
    usable = [
        row
        for row in rows
        if not bool(row.get("retry_recommended"))
        and isinstance(row.get("elapsed_s"), int | float)
    ]
    if len(usable) < 3:
        return predictor
    latencies = [float(row.get("latency", 0.0) or 0.0) for row in usable]
    elapsed = [float(row["elapsed_s"]) for row in usable]
    typical_latency = mean(latencies) if latencies else 1.0
    typical_elapsed = mean(elapsed) if elapsed else 1.0
    return TimeoutRiskPredictor(
        latency_weight=max(0.005, 0.15 / max(typical_latency, 1.0)),
        error_weight=predictor.error_weight,
        budget_penalty=max(0.05, min(0.4, typical_elapsed / max(typical_latency * 10, 1.0))),
    )


def _ema(previous: float, observation: float, *, alpha: float = 0.2) -> float:
    return (1.0 - alpha) * previous + alpha * observation
