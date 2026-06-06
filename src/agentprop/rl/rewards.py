"""Reward functions for routing environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoutingRewardProfile:
    """Reward weights used by sequential routing environments."""

    coverage_weight: float = 10.0
    token_cost_weight: float = 0.001
    message_cost_weight: float = 0.001
    time_weight: float = 0.05
    verifier_weight: float = 1.5
    safe_pruning_weight: float = 0.002
    risky_pruning_weight: float = 2.0
    tool_weight: float = 0.5
    summary_weight: float = 0.0005
    source: str = "default"
    example_count: int = 0

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "coverage_weight": self.coverage_weight,
            "token_cost_weight": self.token_cost_weight,
            "message_cost_weight": self.message_cost_weight,
            "time_weight": self.time_weight,
            "verifier_weight": self.verifier_weight,
            "safe_pruning_weight": self.safe_pruning_weight,
            "risky_pruning_weight": self.risky_pruning_weight,
            "tool_weight": self.tool_weight,
            "summary_weight": self.summary_weight,
            "source": self.source,
            "example_count": self.example_count,
        }


@dataclass(frozen=True, slots=True)
class WorkflowControlReward:
    """Action-level reward terms for expanded routing policies."""

    verifier_bonus: float = 0.0
    safe_pruning_bonus: float = 0.0
    risky_pruning_penalty: float = 0.0
    tool_bonus: float = 0.0
    summary_bonus: float = 0.0

    @property
    def total(self) -> float:
        """Return the combined action-shaping reward."""

        return (
            self.verifier_bonus
            + self.safe_pruning_bonus
            - self.risky_pruning_penalty
            + self.tool_bonus
            + self.summary_bonus
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize reward terms for trajectory/debug output."""

        return {
            "verifier_bonus": self.verifier_bonus,
            "safe_pruning_bonus": self.safe_pruning_bonus,
            "risky_pruning_penalty": self.risky_pruning_penalty,
            "tool_bonus": self.tool_bonus,
            "summary_bonus": self.summary_bonus,
            "total": self.total,
        }


def shaped_routing_reward(
    *,
    token_savings: float,
    quality_score: float,
    quality_loss: float | None = None,
    timeout_risk: float = 0.0,
    regression_risk: float = 0.0,
    alpha: float = 0.10,
    beta: float = 0.55,
    gamma: float = 0.25,
    delta: float = 0.50,
) -> float:
    """R = quality + α·token_savings − β·quality_loss − γ·timeout_risk − δ·regression_risk."""

    bounded_savings = max(-1.0, min(1.0, token_savings))
    loss = quality_loss if quality_loss is not None else max(0.0, 1.0 - quality_score)
    return (
        quality_score
        + alpha * bounded_savings
        - beta * max(0.0, min(1.0, loss))
        - gamma * max(0.0, min(1.0, timeout_risk))
        - delta * max(0.0, min(1.0, regression_risk))
    )


def propagation_reward(
    *,
    coverage: float,
    token_cost: float,
    message_cost: float,
    propagation_time: float,
    coverage_weight: float = 10.0,
    token_cost_weight: float = 0.001,
    message_cost_weight: float = 0.001,
    time_weight: float = 0.05,
) -> float:
    """Reward high coverage while penalizing cost and slow propagation."""

    return (
        coverage_weight * coverage
        - token_cost_weight * token_cost
        - message_cost_weight * message_cost
        - time_weight * propagation_time
    )


def calibrate_routing_reward_profile(
    rows: list[dict[str, object]],
    *,
    base_coverage_weight: float = 10.0,
    cost_penalty_fraction: float = 0.15,
    time_penalty_fraction: float = 0.05,
) -> RoutingRewardProfile:
    """Fit reward penalties from empirical outcome/cost rows.

    This keeps the interpretation stable: full coverage is worth roughly
    ``base_coverage_weight`` reward units, while typical observed cost and
    latency consume bounded fractions of that reward.
    """

    usable = [row for row in rows if not bool(row.get("retry_recommended"))]
    if not usable:
        return RoutingRewardProfile()

    token_costs = [
        _row_float(row, "token_cost", "total_llm_tokens", "input_tokens")
        for row in usable
    ]
    message_costs = [_row_float(row, "message_cost", "message_count") for row in usable]
    latencies = [_row_float(row, "latency", "elapsed_time_s", "duration_s") for row in usable]
    outcomes = [_row_outcome(row) for row in usable]
    mean_outcome = _mean([outcome for outcome in outcomes if outcome is not None], default=0.5)
    reward_scale = base_coverage_weight * max(mean_outcome, 0.25)

    return RoutingRewardProfile(
        coverage_weight=base_coverage_weight,
        token_cost_weight=_penalty_weight(
            token_costs,
            reward_scale=reward_scale,
            penalty_fraction=cost_penalty_fraction,
        ),
        message_cost_weight=_penalty_weight(
            message_costs,
            reward_scale=reward_scale,
            penalty_fraction=cost_penalty_fraction / 2.0,
        ),
        time_weight=_penalty_weight(
            latencies,
            reward_scale=reward_scale,
            penalty_fraction=time_penalty_fraction,
        ),
        source="empirical",
        example_count=len(usable),
    )


def workflow_control_reward(
    *,
    activated_verifier_risk: float = 0.0,
    safe_pruning_savings: float = 0.0,
    risky_pruning_exposure: float = 0.0,
    tool_reliability_gain: float = 0.0,
    summary_token_savings: float = 0.0,
    verifier_weight: float = 1.5,
    safe_pruning_weight: float = 0.002,
    risky_pruning_weight: float = 2.0,
    tool_weight: float = 0.5,
    summary_weight: float = 0.0005,
) -> WorkflowControlReward:
    """Score expanded workflow-control actions with small interpretable terms."""

    return WorkflowControlReward(
        verifier_bonus=verifier_weight * activated_verifier_risk,
        safe_pruning_bonus=safe_pruning_weight * safe_pruning_savings,
        risky_pruning_penalty=risky_pruning_weight * risky_pruning_exposure,
        tool_bonus=tool_weight * tool_reliability_gain,
        summary_bonus=summary_weight * summary_token_savings,
    )


def _penalty_weight(
    values: list[float],
    *,
    reward_scale: float,
    penalty_fraction: float,
) -> float:
    positive = [value for value in values if value > 0.0]
    if not positive:
        return 0.0
    typical = _mean(positive, default=1.0)
    return max(0.0, penalty_fraction * reward_scale / max(typical, 1.0))


def _row_float(row: dict[str, object], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int | float):
            return float(value)
    return 0.0


def _row_outcome(row: dict[str, object]) -> float | None:
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


def _mean(values: list[float], *, default: float) -> float:
    return sum(values) / len(values) if values else default
