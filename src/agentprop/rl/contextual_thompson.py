"""Contextual Thompson sampling over graph-position features (LinTS).

Upgrades :class:`ThompsonSamplingRoutingPolicy` from per-category arms to a
Bayesian linear model per arm: reward ~ N(theta_a . x, sigma^2) with a
Gaussian prior on theta_a. Choosing samples theta_a from each posterior and
picks the argmax of theta_a . x, so exploration decays automatically where
data accumulates and persists where the feature space is still uncertain.

The context x comes from ``agentprop.rl.graph_features`` (node position,
workflow embedding), which is exactly what every reward record already logs
(schema v2) — historical logs are directly trainable.

Dependency-light: dense matrices over feature dims ~10-40 in pure Python.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

Matrix = list[list[float]]
Vector = list[float]

_INTERCEPT = "__intercept__"


def shaped_reward(
    *,
    passed: bool,
    token_savings: float,
    alpha: float = 0.10,
) -> float:
    """Reward = success x (1 + alpha * cost_efficiency); zero on failure.

    Cost credit only applies when the task actually passed, closing the
    do-nothing loophole where a policy earns savings by failing cheaply.
    """

    if not passed:
        return 0.0
    bounded = max(-1.0, min(1.0, token_savings))
    return 1.0 + alpha * bounded


def rloo_advantages(rewards: Sequence[float]) -> list[float]:
    """REINFORCE leave-one-out baselines: r_i minus mean of the others."""

    n = len(rewards)
    if n < 2:
        return [0.0] * n
    total = sum(rewards)
    return [r - (total - r) / (n - 1) for r in rewards]


@dataclass(slots=True)
class _ArmModel:
    """Bayesian linear regression state for one arm (precision form)."""

    dim: int
    lambda_: float
    precision: Matrix = field(init=False)
    weighted_sum: Vector = field(init=False)
    count: int = 0
    success_count: int = 0

    def __post_init__(self) -> None:
        self.precision = [
            [self.lambda_ if i == j else 0.0 for j in range(self.dim)]
            for i in range(self.dim)
        ]
        self.weighted_sum = [0.0] * self.dim

    def update(self, x: Vector, reward: float, *, passed: bool) -> None:
        for i in range(self.dim):
            for j in range(self.dim):
                self.precision[i][j] += x[i] * x[j]
            self.weighted_sum[i] += reward * x[i]
        self.count += 1
        if passed:
            self.success_count += 1

    def posterior_mean(self) -> Vector:
        return _solve(self.precision, self.weighted_sum)

    def sample_theta(self, rng: random.Random, noise_variance: float) -> Vector:
        mean = self.posterior_mean()
        covariance_factor = _cholesky(_inverse(self.precision))
        z = [rng.gauss(0.0, math.sqrt(noise_variance)) for _ in range(self.dim)]
        offset = _mat_vec(covariance_factor, z)
        return [m + o for m, o in zip(mean, offset, strict=True)]

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "success_count": self.success_count,
            "posterior_mean": self.posterior_mean() if self.count else [0.0] * self.dim,
        }


@dataclass(slots=True)
class ContextualThompsonSamplingPolicy:
    """LinTS routing policy over graph-position feature mappings.

    Features are passed as name->value mappings; the feature space is frozen
    on first use (sorted names + intercept) so historical logs replay
    consistently. The circuit breaker mirrors the per-category policy: until
    ``min_successes`` total successful outcomes are observed, route to
    ``default_arm``.
    """

    arms: tuple[str, ...]
    seed: int = 0
    default_arm: str | None = None
    min_successes: int = 5
    lambda_: float = 1.0
    noise_variance: float = 0.25
    feature_names: tuple[str, ...] | None = None
    _models: dict[str, _ArmModel] = field(default_factory=dict, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("arms must not be empty")
        if self.default_arm is not None and self.default_arm not in self.arms:
            raise ValueError(f"default_arm must be one of arms: {self.default_arm!r}")
        self._rng = random.Random(self.seed)

    def choose(self, features: Mapping[str, float], *, explore: bool = True) -> str:
        """Sample posteriors (or use means) at the given context."""

        x = self._vectorize(features)
        successes = sum(m.success_count for m in self._models.values())
        if successes < self.min_successes and self.default_arm is not None:
            return self.default_arm
        scores: dict[str, float] = {}
        for arm in self.arms:
            model = self._model(arm)
            theta = (
                model.sample_theta(self._rng, self.noise_variance)
                if explore
                else model.posterior_mean()
            )
            scores[arm] = sum(t * xi for t, xi in zip(theta, x, strict=True))
        return max(self.arms, key=lambda arm: (scores[arm], -self.arms.index(arm)))

    def exploit(self, features: Mapping[str, float]) -> str:
        """Greedy, exploration-free selection for serving."""

        return self.choose(features, explore=False)

    def update(
        self,
        arm: str,
        features: Mapping[str, float],
        *,
        reward: float,
        passed: bool,
    ) -> None:
        """Record one observed (context, arm, reward) outcome."""

        if arm not in self.arms:
            raise ValueError(f"unknown arm: {arm}")
        x = self._vectorize(features)
        self._model(arm).update(x, reward, passed=passed)

    def update_batch_rloo(
        self,
        arm: str,
        observations: Sequence[tuple[Mapping[str, float], float, bool]],
    ) -> None:
        """Batch update using leave-one-out advantages as regression targets.

        Centering rewards against the batch baseline reduces variance the
        same way RLOO does for REINFORCE; useful when several runs of the
        same arm land together (e.g. one ablation sweep).
        """

        advantages = rloo_advantages([reward for _, reward, _ in observations])
        for (features, _, passed), advantage in zip(observations, advantages, strict=True):
            self.update(arm, features, reward=advantage, passed=passed)

    def values(self, features: Mapping[str, float]) -> dict[str, float]:
        """Posterior-mean score per arm at the given context."""

        x = self._vectorize(features)
        return {
            arm: sum(t * xi for t, xi in zip(self._model(arm).posterior_mean(), x, strict=True))
            for arm in self.arms
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "arms": list(self.arms),
            "default_arm": self.default_arm,
            "min_successes": self.min_successes,
            "feature_names": list(self.feature_names or ()),
            "models": {arm: model.to_dict() for arm, model in self._models.items()},
        }

    def _vectorize(self, features: Mapping[str, float]) -> Vector:
        if self.feature_names is None:
            names = sorted(str(k) for k in features)
            self.feature_names = (_INTERCEPT, *names)
        return [
            1.0 if name == _INTERCEPT else float(features.get(name, 0.0))
            for name in self.feature_names
        ]

    def _model(self, arm: str) -> _ArmModel:
        if arm not in self._models:
            if self.feature_names is None:
                raise RuntimeError("vectorize a context before accessing models")
            self._models[arm] = _ArmModel(dim=len(self.feature_names), lambda_=self.lambda_)
        return self._models[arm]


# ---------------------------------------------------------------------------
# Small dense linear algebra (pure Python, dims ~10-40)


def _cholesky(matrix: Matrix) -> Matrix:
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                value = matrix[i][i] - s
                lower[i][j] = math.sqrt(max(value, 1e-12))
            else:
                lower[i][j] = (matrix[i][j] - s) / lower[j][j]
    return lower


def _solve(matrix: Matrix, rhs: Vector) -> Vector:
    lower = _cholesky(matrix)
    n = len(rhs)
    y = [0.0] * n
    for i in range(n):
        y[i] = (rhs[i] - sum(lower[i][k] * y[k] for k in range(i))) / lower[i][i]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))) / lower[i][i]
    return x


def _inverse(matrix: Matrix) -> Matrix:
    n = len(matrix)
    columns = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        columns.append(_solve(matrix, e))
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def _mat_vec(matrix: Matrix, vector: Vector) -> Vector:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]
