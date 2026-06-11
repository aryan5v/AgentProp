"""Split conformal calibration for risk-gated control decisions.

Wraps any scalar risk score (e.g. :class:`TimeoutRiskPredictor` output or a
caller-supplied ``regression_risk``) with a distribution-free threshold. After
calibrating on ``n`` labeled outcomes, gating on the calibrated threshold
guarantees the expected miss rate on true failures is at most ``alpha``
(finite-sample, assuming exchangeability of calibration and test outcomes).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ConformalCalibrationResult:
    """Calibrated threshold and the data that produced it."""

    threshold: float
    alpha: float
    positives: int
    negatives: int
    empirical_recall: float
    empirical_false_alarm_rate: float

    def to_dict(self) -> dict[str, float | int]:
        """Serialize to JSON-compatible data."""

        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
            "positives": self.positives,
            "negatives": self.negatives,
            "empirical_recall": self.empirical_recall,
            "empirical_false_alarm_rate": self.empirical_false_alarm_rate,
        }


@dataclass(slots=True)
class ConformalRiskGate:
    """Distribution-free FORCE_VERIFY gate over any scalar risk score.

    Calibrate with risk scores observed before known outcomes (``True`` =
    the step actually failed / needed verification). ``should_flag`` then
    fires whenever a new score is at least the conformal threshold, missing
    true failures at rate at most ``alpha``.
    """

    alpha: float = 0.1
    _result: ConformalCalibrationResult | None = field(default=None, repr=False)

    def calibrate(
        self,
        scores: Sequence[float],
        outcomes: Sequence[bool],
    ) -> ConformalCalibrationResult:
        """Fit the threshold from labeled (score, failed?) calibration pairs."""

        if len(scores) != len(outcomes):
            raise ValueError("scores and outcomes must have the same length")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        positive_scores = sorted(
            score for score, failed in zip(scores, outcomes, strict=True) if failed
        )
        negative_scores = [
            score for score, failed in zip(scores, outcomes, strict=True) if not failed
        ]
        if not positive_scores:
            raise ValueError("calibration requires at least one positive (failed) outcome")
        n = len(positive_scores)
        # Conformal quantile: flag when score >= the k-th smallest positive
        # score, where k = floor(alpha * (n + 1)). With k == 0 the guarantee
        # needs the threshold below every positive score.
        k = math.floor(self.alpha * (n + 1))
        if k <= 0:
            threshold = math.nextafter(positive_scores[0], -math.inf)
        else:
            threshold = positive_scores[min(k, n) - 1]
        recall = sum(1 for s in positive_scores if s >= threshold) / n
        false_alarms = (
            sum(1 for s in negative_scores if s >= threshold) / len(negative_scores)
            if negative_scores
            else 0.0
        )
        self._result = ConformalCalibrationResult(
            threshold=threshold,
            alpha=self.alpha,
            positives=n,
            negatives=len(negative_scores),
            empirical_recall=recall,
            empirical_false_alarm_rate=false_alarms,
        )
        return self._result

    @property
    def result(self) -> ConformalCalibrationResult:
        if self._result is None:
            raise RuntimeError("gate is not calibrated; call calibrate() first")
        return self._result

    @property
    def is_calibrated(self) -> bool:
        return self._result is not None

    def should_flag(self, score: float) -> bool:
        """Return True when the score crosses the calibrated threshold."""

        return score >= self.result.threshold

    def save(self, path: str | Path) -> Path:
        """Persist the calibrated gate as JSON."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"alpha": self.alpha, "result": self.result.to_dict()}
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> ConformalRiskGate:
        """Load a calibrated gate from JSON."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        result = payload["result"]
        gate = cls(alpha=float(payload["alpha"]))
        gate._result = ConformalCalibrationResult(
            threshold=float(result["threshold"]),
            alpha=float(result["alpha"]),
            positives=int(result["positives"]),
            negatives=int(result["negatives"]),
            empirical_recall=float(result["empirical_recall"]),
            empirical_false_alarm_rate=float(result["empirical_false_alarm_rate"]),
        )
        return gate
