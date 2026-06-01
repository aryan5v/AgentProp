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
    stats: dict[str, dict[str, BanditArmStats]] = field(default_factory=dict)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("arms must not be empty")
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        self._rng = random.Random(self.seed)

    def choose(self, category: str) -> str:
        """Choose a routing policy for a task category."""

        category_stats = self._category_stats(category)
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.arms)
        return max(
            self.arms,
            key=lambda arm: (category_stats[arm].value, -self.arms.index(arm)),
        )

    def update(
        self,
        category: str,
        arm: str,
        *,
        passed: bool,
        token_savings: float,
        quality_score: float | None = None,
    ) -> None:
        """Update an arm using real success and cost feedback."""

        if arm not in self.arms:
            raise ValueError(f"unknown arm: {arm}")
        quality = quality_score if quality_score is not None else (1.0 if passed else 0.0)
        reward = quality + 0.25 * token_savings
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
