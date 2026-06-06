"""Category-conditioned online bandit for routing-policy selection."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(slots=True)
class BanditArmStats:
    """Online reward statistics for one policy arm."""

    count: int = 0
    value: float = 0.0

    def update(self, reward: float) -> None:
        self.count += 1
        self.value += (reward - self.value) / self.count


@dataclass(slots=True)
class CategoryBanditRoutingPolicy:
    """Choose routing policies per task category and learn from pass/fail outcomes."""

    arms: tuple[str, ...] = ("broadcast", "quality-aware-greedy", "cost-aware-greedy")
    epsilon: float = 0.10
    seed: int = 0
    default_arm: str | None = None
    """Safe fallback for a category with no observations yet.

    With little training data, many eval categories are never seen. Falling back to
    a known-strong default (rather than an arbitrary index tiebreak) stops the policy
    from regressing baseline tasks it has learned nothing about. Defaults to the
    first arm when unset."""
    cost_weight: float = 0.10
    quality_loss_weight: float = 0.55
    timeout_risk_weight: float = 0.25
    regression_risk_weight: float = 0.50
    stats: dict[str, dict[str, BanditArmStats]] = field(default_factory=dict)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("arms must not be empty")
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        if self.default_arm is not None and self.default_arm not in self.arms:
            raise ValueError(f"default_arm must be one of arms: {self.default_arm!r}")
        if self.cost_weight < 0:
            raise ValueError("cost_weight must be non-negative")
        self._rng = random.Random(self.seed)

    def choose(self, category: str, *, explore: bool = True) -> str:
        """Choose a routing policy for a task category.

        Set ``explore=False`` when scoring held-out tasks so a graded eval never
        regresses on a random exploration pick."""

        category_stats = self._category_stats(category)
        if all(stats.count == 0 for stats in category_stats.values()):
            # Cold start: no evidence for this category — use the safe default.
            return self.default_arm or self.arms[0]
        if explore and self._rng.random() < self.epsilon:
            return self._rng.choice(self.arms)
        return max(
            self.arms,
            key=lambda arm: (category_stats[arm].value, -self.arms.index(arm)),
        )

    def exploit(self, category: str) -> str:
        """Greedy, exploration-free selection for scoring/eval."""

        return self.choose(category, explore=False)

    def update(
        self,
        category: str,
        arm: str,
        *,
        passed: bool,
        token_savings: float,
        quality_score: float | None = None,
        regression_risk: float = 0.0,
        timeout_risk: float = 0.0,
        quality_loss: float | None = None,
    ) -> None:
        """Update an arm using shaped reward: savings − quality_loss − timeout − regression."""

        if arm not in self.arms:
            raise ValueError(f"unknown arm: {arm}")
        quality = quality_score if quality_score is not None else (1.0 if passed else 0.0)
        bounded_savings = max(-1.0, min(1.0, token_savings))
        loss = quality_loss if quality_loss is not None else max(0.0, 1.0 - quality)
        timeout_pen = max(0.0, min(1.0, timeout_risk))
        risk_pen = max(0.0, min(1.0, regression_risk))
        reward = (
            quality
            + self.cost_weight * bounded_savings
            - self.quality_loss_weight * loss
            - self.timeout_risk_weight * timeout_pen
            - self.regression_risk_weight * risk_pen
        )
        if not passed:
            reward -= 1.0
        self._category_stats(category)[arm].update(reward)

    def values(self, category: str) -> dict[str, float]:
        """Return current arm values for a category."""

        return {arm: stats.value for arm, stats in self._category_stats(category).items()}

    def _category_stats(self, category: str) -> dict[str, BanditArmStats]:
        if category not in self.stats:
            self.stats[category] = {arm: BanditArmStats() for arm in self.arms}
        return self.stats[category]
