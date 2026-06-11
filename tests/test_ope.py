"""Tests for off-policy evaluation over logged reward records."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from agentprop.rl import (
    LoggedDecision,
    doubly_robust,
    load_logged_decisions,
    weighted_importance_sampling,
)

ARMS = ("good", "bad")
TRUE_MEANS = {"good": 0.8, "bad": 0.2}


def _uniform_logs(n: int, seed: int) -> list[LoggedDecision]:
    rng = random.Random(seed)
    logs = []
    for _ in range(n):
        arm = rng.choice(ARMS)
        logs.append(
            LoggedDecision(
                category="cat",
                arm=arm,
                reward=rng.gauss(TRUE_MEANS[arm], 0.1),
                behavior_probability=0.5,
            )
        )
    return logs


def _target_good(_: str) -> dict[str, float]:
    return {"good": 1.0, "bad": 0.0}


def test_wis_recovers_target_value() -> None:
    logs = _uniform_logs(2000, seed=1)
    result = weighted_importance_sampling(logs, _target_good, seed=1)
    assert result.estimate.lower <= TRUE_MEANS["good"] <= result.estimate.upper
    assert result.method == "weighted-importance-sampling"
    assert 0 < result.effective_sample_size <= len(logs)


def test_doubly_robust_recovers_target_value() -> None:
    logs = _uniform_logs(2000, seed=2)
    result = doubly_robust(logs, _target_good, seed=2)
    assert abs(result.estimate.mean - TRUE_MEANS["good"]) < 0.05


def test_doubly_robust_with_exact_model_has_tight_interval() -> None:
    logs = _uniform_logs(500, seed=3)
    exact = doubly_robust(
        logs, _target_good, reward_model=lambda _c, arm: TRUE_MEANS[arm], seed=3
    )
    baseline = doubly_robust(logs, _target_good, seed=3)
    assert (exact.estimate.upper - exact.estimate.lower) <= (
        baseline.estimate.upper - baseline.estimate.lower
    ) * 1.5


def test_empty_logs_raise() -> None:
    with pytest.raises(ValueError):
        weighted_importance_sampling([], _target_good)


def test_zero_behavior_probability_raises() -> None:
    bad = [LoggedDecision("cat", "good", 0.5, behavior_probability=0.0)]
    with pytest.raises(ValueError):
        weighted_importance_sampling(bad, _target_good)


def test_load_logged_decisions_from_jsonl(tmp_path: Path) -> None:
    rows = [
        {"category": "cat", "strategy": "good", "token_savings": 0.4},
        {"category": "cat", "strategy": "bad", "token_savings": -0.1},
        {"row_type": "analysis"},
    ]
    path = tmp_path / "rewards.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    decisions = load_logged_decisions(path, behavior_probability=lambda _row: 0.5)
    assert len(decisions) == 2
    assert decisions[0].arm == "good"
