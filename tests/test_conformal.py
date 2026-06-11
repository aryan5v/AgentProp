"""Tests for the conformal risk gate."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from agentprop.ml import ConformalRiskGate


def _calibration_data(
    n: int, *, seed: int
) -> tuple[list[float], list[bool]]:
    rng = random.Random(seed)
    scores: list[float] = []
    outcomes: list[bool] = []
    for _ in range(n):
        failed = rng.random() < 0.3
        # Failures score higher on average but distributions overlap.
        score = rng.gauss(0.7 if failed else 0.3, 0.15)
        scores.append(min(max(score, 0.0), 1.0))
        outcomes.append(failed)
    return scores, outcomes


def test_calibrated_recall_meets_guarantee() -> None:
    scores, outcomes = _calibration_data(400, seed=1)
    gate = ConformalRiskGate(alpha=0.1)
    result = gate.calibrate(scores, outcomes)
    assert result.empirical_recall >= 1.0 - gate.alpha
    assert 0.0 <= result.empirical_false_alarm_rate <= 1.0


def test_holdout_miss_rate_near_alpha() -> None:
    scores, outcomes = _calibration_data(1000, seed=2)
    gate = ConformalRiskGate(alpha=0.15)
    gate.calibrate(scores, outcomes)
    held_scores, held_outcomes = _calibration_data(1000, seed=3)
    positives = [s for s, o in zip(held_scores, held_outcomes, strict=True) if o]
    misses = sum(1 for s in positives if not gate.should_flag(s))
    assert misses / len(positives) <= gate.alpha + 0.05


def test_requires_positive_outcomes() -> None:
    gate = ConformalRiskGate()
    with pytest.raises(ValueError):
        gate.calibrate([0.1, 0.2], [False, False])


def test_uncalibrated_gate_raises() -> None:
    with pytest.raises(RuntimeError):
        ConformalRiskGate().should_flag(0.5)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    scores, outcomes = _calibration_data(200, seed=4)
    gate = ConformalRiskGate(alpha=0.2)
    gate.calibrate(scores, outcomes)
    path = gate.save(tmp_path / "gate.json")
    loaded = ConformalRiskGate.load(path)
    assert loaded.result.threshold == gate.result.threshold
    assert loaded.should_flag(0.99)


def test_small_alpha_flags_all_positives() -> None:
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    outcomes = [True, True, True, False, False]
    gate = ConformalRiskGate(alpha=0.05)
    gate.calibrate(scores, outcomes)
    assert all(gate.should_flag(s) for s, o in zip(scores, outcomes, strict=True) if o)
