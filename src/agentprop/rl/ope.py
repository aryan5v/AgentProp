"""Off-policy evaluation over logged bandit reward records.

Answers "what would this policy have earned on historical traffic?" without
redeploying: weighted importance sampling (WIS) and a doubly-robust (DR)
estimator over the JSONL rows written by ``RuntimeRewardLogger``. Both are
dependency-light and return bootstrap confidence intervals.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentprop.evaluation.intervals import ConfidenceInterval, bootstrap_mean_interval


@dataclass(frozen=True, slots=True)
class LoggedDecision:
    """One historical routing decision with its observed reward."""

    category: str
    arm: str
    reward: float
    behavior_probability: float
    """Probability the logging policy assigned to the chosen arm."""

    @classmethod
    def from_reward_row(
        cls,
        row: dict[str, Any],
        *,
        behavior_probability: float,
        reward_key: str = "token_savings",
    ) -> LoggedDecision:
        """Build from a ``RuntimeRewardLogger`` JSONL row."""

        return cls(
            category=str(row["category"]),
            arm=str(row.get("strategy", row.get("arm", ""))),
            reward=float(row.get(reward_key) or 0.0),
            behavior_probability=behavior_probability,
        )


@dataclass(frozen=True, slots=True)
class OPEResult:
    """Estimated mean reward of the target policy on logged traffic."""

    estimate: ConfidenceInterval
    method: str
    decisions: int
    effective_sample_size: float

    def to_dict(self) -> dict[str, object]:
        """Serialize for reports."""

        return {
            "estimate": self.estimate.to_dict(),
            "method": self.method,
            "decisions": self.decisions,
            "effective_sample_size": self.effective_sample_size,
        }


TargetPolicy = Callable[[str], dict[str, float]]
"""Maps a category to the target policy's arm probabilities."""


def weighted_importance_sampling(
    decisions: Sequence[LoggedDecision],
    target_policy: TargetPolicy,
    *,
    max_weight: float = 20.0,
    seed: int = 0,
) -> OPEResult:
    """WIS estimate of the target policy's mean reward on logged decisions."""

    weights, rewards = _weights_and_rewards(decisions, target_policy, max_weight)
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("target policy never selects any logged arm")
    weighted = [
        w * r * len(weights) / total_weight for w, r in zip(weights, rewards, strict=True)
    ]
    return OPEResult(
        estimate=bootstrap_mean_interval(weighted, seed=seed),
        method="weighted-importance-sampling",
        decisions=len(decisions),
        effective_sample_size=_effective_sample_size(weights),
    )


def doubly_robust(
    decisions: Sequence[LoggedDecision],
    target_policy: TargetPolicy,
    *,
    reward_model: Callable[[str, str], float] | None = None,
    max_weight: float = 20.0,
    seed: int = 0,
) -> OPEResult:
    """Doubly-robust estimate combining a reward model with importance weights.

    ``reward_model(category, arm)`` predicts reward; when omitted, per-
    (category, arm) empirical means from the logged data are used (a simple
    direct-method baseline). The estimator stays consistent if either the
    reward model or the behavior probabilities are accurate.
    """

    model = reward_model or _empirical_reward_model(decisions)
    weights, _ = _weights_and_rewards(decisions, target_policy, max_weight)
    contributions: list[float] = []
    for decision, weight in zip(decisions, weights, strict=True):
        target_probs = target_policy(decision.category)
        baseline = sum(
            prob * model(decision.category, arm) for arm, prob in target_probs.items()
        )
        correction = weight * (decision.reward - model(decision.category, decision.arm))
        contributions.append(baseline + correction)
    return OPEResult(
        estimate=bootstrap_mean_interval(contributions, seed=seed),
        method="doubly-robust",
        decisions=len(decisions),
        effective_sample_size=_effective_sample_size(weights),
    )


def load_logged_decisions(
    path: str | Path,
    *,
    behavior_probability: Callable[[dict[str, Any]], float],
    reward_key: str = "token_savings",
) -> list[LoggedDecision]:
    """Load reward-record JSONL rows into logged decisions."""

    decisions: list[LoggedDecision] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "category" not in row:
            continue
        decisions.append(
            LoggedDecision.from_reward_row(
                row,
                behavior_probability=behavior_probability(row),
                reward_key=reward_key,
            )
        )
    return decisions


def _weights_and_rewards(
    decisions: Sequence[LoggedDecision],
    target_policy: TargetPolicy,
    max_weight: float,
) -> tuple[list[float], list[float]]:
    if not decisions:
        raise ValueError("off-policy evaluation requires at least one logged decision")
    weights: list[float] = []
    rewards: list[float] = []
    for decision in decisions:
        if decision.behavior_probability <= 0:
            raise ValueError("behavior probability must be positive for logged arms")
        target_prob = target_policy(decision.category).get(decision.arm, 0.0)
        weights.append(min(target_prob / decision.behavior_probability, max_weight))
        rewards.append(decision.reward)
    return weights, rewards


def _effective_sample_size(weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return 0.0
    return total * total / sum(w * w for w in weights)


def _empirical_reward_model(
    decisions: Sequence[LoggedDecision],
) -> Callable[[str, str], float]:
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for decision in decisions:
        key = (decision.category, decision.arm)
        sums[key] = sums.get(key, 0.0) + decision.reward
        counts[key] = counts.get(key, 0) + 1

    def model(category: str, arm: str) -> float:
        key = (category, arm)
        if key not in counts:
            return 0.0
        return sums[key] / counts[key]

    return model
