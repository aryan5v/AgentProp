"""Bootstrap confidence intervals for benchmark and propagation metrics."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Mean with a two-sided bootstrap percentile interval."""

    mean: float
    lower: float
    upper: float
    confidence: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to JSON-compatible data."""

        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "samples": self.samples,
        }

    def __str__(self) -> str:
        pct = int(round(self.confidence * 100))
        return f"{self.mean:.4g} [{pct}% CI {self.lower:.4g}, {self.upper:.4g}]"


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int | None = None,
) -> ConfidenceInterval:
    """Return a percentile bootstrap CI for the mean of ``values``.

    With fewer than two observations the interval collapses to the point
    estimate so callers can render it without special-casing.
    """

    if not values:
        raise ValueError("bootstrap_mean_interval requires at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    mean = fmean(values)
    if len(values) < 2:
        return ConfidenceInterval(mean, mean, mean, confidence, len(values))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(fmean(rng.choices(values, k=n)) for _ in range(resamples))
    alpha = (1.0 - confidence) / 2.0
    lower = means[int(alpha * (resamples - 1))]
    upper = means[int((1.0 - alpha) * (resamples - 1))]
    return ConfidenceInterval(mean, lower, upper, confidence, n)


def bootstrap_difference_interval(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int | None = None,
) -> ConfidenceInterval:
    """Bootstrap CI for mean(treatment) - mean(control) on independent samples."""

    if not treatment or not control:
        raise ValueError("both samples must be non-empty")
    rng = random.Random(seed)
    point = fmean(treatment) - fmean(control)
    if len(treatment) < 2 and len(control) < 2:
        return ConfidenceInterval(point, point, point, confidence, len(treatment) + len(control))
    diffs = sorted(
        fmean(rng.choices(treatment, k=len(treatment)))
        - fmean(rng.choices(control, k=len(control)))
        for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lower = diffs[int(alpha * (resamples - 1))]
    upper = diffs[int((1.0 - alpha) * (resamples - 1))]
    return ConfidenceInterval(point, lower, upper, confidence, len(treatment) + len(control))
