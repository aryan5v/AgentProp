"""Tests for contextual Thompson sampling (LinTS)."""

from __future__ import annotations

import random

import pytest

from agentprop.rl.contextual_thompson import (
    ContextualThompsonSamplingPolicy,
    rloo_advantages,
    shaped_reward,
)


def _policy(**kwargs: object) -> ContextualThompsonSamplingPolicy:
    defaults: dict[str, object] = {
        "arms": ("careful", "fast"),
        "min_successes": 0,
        "seed": 3,
    }
    defaults.update(kwargs)
    return ContextualThompsonSamplingPolicy(**defaults)  # type: ignore[arg-type]


def test_learns_context_dependent_routing() -> None:
    """'careful' wins at deep positions, 'fast' wins at shallow ones."""

    policy = _policy()
    rng = random.Random(11)
    for _ in range(400):
        depth = rng.choice([0.0, 1.0])
        features = {"depth": depth, "quality": rng.random()}
        arm = policy.choose(features)
        true_mean = (0.9 if depth > 0.5 else 0.2) if arm == "careful" else (
            0.2 if depth > 0.5 else 0.9
        )
        reward = rng.gauss(true_mean, 0.1)
        policy.update(arm, features, reward=reward, passed=reward > 0.5)
    assert policy.exploit({"depth": 1.0, "quality": 0.5}) == "careful"
    assert policy.exploit({"depth": 0.0, "quality": 0.5}) == "fast"


def test_cold_start_circuit_breaker() -> None:
    policy = _policy(min_successes=5, default_arm="careful")
    assert policy.choose({"depth": 0.3}) == "careful"


def test_unknown_arm_rejected() -> None:
    policy = _policy()
    with pytest.raises(ValueError):
        policy.update("zzz", {"depth": 0.1}, reward=0.5, passed=True)


def test_feature_space_frozen_after_first_use() -> None:
    policy = _policy()
    policy.update("fast", {"a": 1.0, "b": 2.0}, reward=0.5, passed=True)
    # New unseen feature names are ignored, missing ones default to 0.
    arm = policy.exploit({"a": 1.0, "c": 9.0})
    assert arm in policy.arms
    assert policy.feature_names is not None
    assert "c" not in policy.feature_names


def test_shaped_reward_closes_do_nothing_loophole() -> None:
    assert shaped_reward(passed=False, token_savings=1.0) == 0.0
    assert shaped_reward(passed=True, token_savings=0.0) == 1.0
    assert shaped_reward(passed=True, token_savings=1.0) == pytest.approx(1.10)
    assert shaped_reward(passed=True, token_savings=-2.0) == pytest.approx(0.90)


def test_rloo_advantages_center_rewards() -> None:
    advantages = rloo_advantages([1.0, 0.0, 0.5])
    assert sum(advantages) == pytest.approx(0.0, abs=1e-9)
    assert advantages[0] > advantages[1]
    assert rloo_advantages([0.7]) == [0.0]


def test_update_batch_rloo_runs() -> None:
    policy = _policy()
    observations = [
        ({"depth": 1.0}, 0.9, True),
        ({"depth": 0.0}, 0.1, False),
        ({"depth": 0.5}, 0.5, True),
    ]
    policy.update_batch_rloo("careful", observations)
    assert policy.to_dict()["models"]["careful"]["count"] == 3  # type: ignore[index]


def test_exploration_decays_with_data() -> None:
    """Posterior variance shrinks: repeated exploit choices become consistent."""

    policy = _policy(seed=5)
    features = {"depth": 1.0}
    for _ in range(100):
        policy.update("careful", features, reward=0.8, passed=True)
        policy.update("fast", features, reward=0.2, passed=True)
    picks = {policy.choose(features) for _ in range(50)}
    assert picks == {"careful"}
