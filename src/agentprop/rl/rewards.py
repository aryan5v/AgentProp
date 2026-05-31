"""Reward functions for routing environments."""

from __future__ import annotations


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
