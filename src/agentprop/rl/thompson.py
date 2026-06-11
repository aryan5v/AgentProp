"""Thompson-sampling routing policy with auto-decaying exploration.

Drop-in alternative to :class:`CategoryBanditRoutingPolicy`'s epsilon-greedy
exploration. Each (category, arm) keeps a Gaussian posterior over the shaped
reward; choosing samples from the posteriors, so exploration decays
automatically as evidence accumulates instead of staying at a fixed epsilon.
A circuit breaker falls back to ``default_arm`` until a category has enough
successful observations to trust the posterior.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass(slots=True)
class GaussianArmPosterior:
    """Normal-Normal posterior over one arm's mean reward (known noise)."""

    prior_mean: float = 0.0
    prior_variance: float = 1.0
    noise_variance: float = 0.25
    count: int = 0
    success_count: int = 0
    sum_rewards: float = 0.0

    @property
    def posterior_mean(self) -> float:
        precision = 1.0 / self.prior_variance + self.count / self.noise_variance
        weighted = self.prior_mean / self.prior_variance + self.sum_rewards / self.noise_variance
        return weighted / precision

    @property
    def posterior_variance(self) -> float:
        return 1.0 / (1.0 / self.prior_variance + self.count / self.noise_variance)

    def update(self, reward: float, *, passed: bool) -> None:
        self.count += 1
        self.sum_rewards += reward
        if passed:
            self.success_count += 1

    def sample(self, rng: random.Random) -> float:
        return rng.gauss(self.posterior_mean, math.sqrt(self.posterior_variance))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "prior_mean": self.prior_mean,
            "prior_variance": self.prior_variance,
            "noise_variance": self.noise_variance,
            "count": self.count,
            "success_count": self.success_count,
            "sum_rewards": self.sum_rewards,
        }


@dataclass(slots=True)
class ThompsonSamplingRoutingPolicy:
    """Per-category Thompson sampling over routing-strategy arms.

    ``choose(category)`` samples each arm's posterior and picks the argmax;
    ``choose(category, explore=False)`` uses posterior means for serving.
    Categories with fewer than ``min_successes`` successful outcomes route to
    ``default_arm`` (circuit breaker against cold-start regressions).

    Priors can be seeded per category (e.g. from quality-cascade simulations
    or a structurally similar workflow's history) via :meth:`seed_prior`.
    """

    arms: tuple[str, ...] = ("broadcast", "quality-aware-greedy", "cost-aware-greedy")
    seed: int = 0
    default_arm: str | None = None
    min_successes: int = 5
    prior_mean: float = 0.0
    prior_variance: float = 1.0
    noise_variance: float = 0.25
    stats: dict[str, dict[str, GaussianArmPosterior]] = field(default_factory=dict)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("arms must not be empty")
        if self.default_arm is not None and self.default_arm not in self.arms:
            raise ValueError(f"default_arm must be one of arms: {self.default_arm!r}")
        if self.min_successes < 0:
            raise ValueError("min_successes must be non-negative")
        self._rng = random.Random(self.seed)

    def choose(self, category: str, *, explore: bool = True) -> str:
        """Sample posteriors (or use means when serving) and pick the best arm."""

        posteriors = self._category_stats(category)
        successes = sum(p.success_count for p in posteriors.values())
        if successes < self.min_successes and self.default_arm is not None:
            return self.default_arm
        if explore:
            scores = {arm: posteriors[arm].sample(self._rng) for arm in self.arms}
        else:
            scores = {arm: posteriors[arm].posterior_mean for arm in self.arms}
        return max(self.arms, key=lambda arm: (scores[arm], -self.arms.index(arm)))

    def exploit(self, category: str) -> str:
        """Greedy, exploration-free selection for scoring/eval."""

        return self.choose(category, explore=False)

    def update(self, category: str, arm: str, *, reward: float, passed: bool) -> None:
        """Record one shaped reward observation for an arm."""

        if arm not in self.arms:
            raise ValueError(f"unknown arm: {arm}")
        self._category_stats(category)[arm].update(reward, passed=passed)

    def seed_prior(
        self,
        category: str,
        arm: str,
        *,
        mean: float,
        variance: float = 0.5,
    ) -> None:
        """Initialize an arm's prior, e.g. from simulation or a similar workflow.

        Only allowed before real observations arrive for that arm; the prior
        then decays naturally as evidence accumulates.
        """

        posterior = self._category_stats(category)[arm]
        if posterior.count:
            raise ValueError("cannot re-seed a prior after observations exist")
        posterior.prior_mean = mean
        posterior.prior_variance = variance

    def values(self, category: str) -> dict[str, float]:
        """Posterior mean per arm for a category."""

        return {
            arm: posterior.posterior_mean
            for arm, posterior in self._category_stats(category).items()
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize policy state for checkpointing."""

        return {
            "arms": list(self.arms),
            "default_arm": self.default_arm,
            "min_successes": self.min_successes,
            "stats": {
                category: {arm: posterior.to_dict() for arm, posterior in arms.items()}
                for category, arms in self.stats.items()
            },
        }

    def _category_stats(self, category: str) -> dict[str, GaussianArmPosterior]:
        if category not in self.stats:
            self.stats[category] = {
                arm: GaussianArmPosterior(
                    prior_mean=self.prior_mean,
                    prior_variance=self.prior_variance,
                    noise_variance=self.noise_variance,
                )
                for arm in self.arms
            }
        return self.stats[category]
