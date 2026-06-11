"""Tests for the Thompson-sampling routing policy."""

from __future__ import annotations

import random

import pytest

from agentprop.rl import ThompsonSamplingRoutingPolicy


def test_cold_start_uses_default_arm() -> None:
    policy = ThompsonSamplingRoutingPolicy(
        arms=("a", "b"), default_arm="a", min_successes=5, seed=1
    )
    assert policy.choose("cat") == "a"


def test_converges_to_better_arm() -> None:
    policy = ThompsonSamplingRoutingPolicy(
        arms=("good", "bad"), default_arm="good", min_successes=0, seed=7
    )
    rng = random.Random(3)
    for _ in range(200):
        arm = policy.choose("cat")
        reward = rng.gauss(0.8 if arm == "good" else 0.2, 0.1)
        policy.update("cat", arm, reward=reward, passed=reward > 0.5)
    assert policy.exploit("cat") == "good"
    picks = [policy.choose("cat") for _ in range(100)]
    assert picks.count("good") > 90  # exploration has decayed


def test_exploration_decays_with_evidence() -> None:
    policy = ThompsonSamplingRoutingPolicy(arms=("a",), min_successes=0, seed=2)
    fresh_variance = policy._category_stats("cat")["a"].posterior_variance
    for _ in range(50):
        policy.update("cat", "a", reward=0.5, passed=True)
    trained_variance = policy._category_stats("cat")["a"].posterior_variance
    assert trained_variance < fresh_variance / 10


def test_seed_prior_biases_cold_choice_and_locks_after_data() -> None:
    policy = ThompsonSamplingRoutingPolicy(
        arms=("a", "b"), min_successes=0, seed=4
    )
    policy.seed_prior("cat", "b", mean=1.0, variance=0.01)
    assert policy.exploit("cat") == "b"
    policy.update("cat", "b", reward=0.9, passed=True)
    with pytest.raises(ValueError):
        policy.seed_prior("cat", "b", mean=0.0)


def test_unknown_arm_rejected() -> None:
    policy = ThompsonSamplingRoutingPolicy(arms=("a",))
    with pytest.raises(ValueError):
        policy.update("cat", "zzz", reward=0.1, passed=False)


def test_to_dict_roundtrips_counts() -> None:
    policy = ThompsonSamplingRoutingPolicy(arms=("a", "b"), min_successes=0)
    policy.update("cat", "a", reward=0.4, passed=True)
    payload = policy.to_dict()
    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert stats["cat"]["a"]["count"] == 1
