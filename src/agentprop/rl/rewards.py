"""Reward functions for routing environments."""

from __future__ import annotations

from dataclasses import dataclass


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
