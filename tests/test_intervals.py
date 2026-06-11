"""Tests for bootstrap confidence intervals."""

from __future__ import annotations

import pytest

from agentprop.evaluation import (
    bootstrap_difference_interval,
    bootstrap_mean_interval,
)


def test_mean_interval_brackets_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = bootstrap_mean_interval(values, seed=7)
    assert ci.mean == pytest.approx(3.0)
    assert ci.lower <= ci.mean <= ci.upper
    assert ci.samples == 5


def test_mean_interval_is_deterministic_with_seed() -> None:
    values = [0.2, 0.9, 0.4, 0.7]
    a = bootstrap_mean_interval(values, seed=11)
    b = bootstrap_mean_interval(values, seed=11)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_single_value_collapses_to_point() -> None:
    ci = bootstrap_mean_interval([2.5], seed=1)
    assert ci.lower == ci.upper == ci.mean == 2.5


def test_empty_values_raise() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_interval([])


def test_difference_interval_sign() -> None:
    treatment = [10.0, 11.0, 12.0, 13.0]
    control = [1.0, 2.0, 3.0, 4.0]
    ci = bootstrap_difference_interval(treatment, control, seed=3)
    assert ci.mean == pytest.approx(9.0)
    assert ci.lower > 0


def test_interval_str_mentions_confidence() -> None:
    ci = bootstrap_mean_interval([1.0, 2.0, 3.0], seed=5)
    assert "95% CI" in str(ci)


def test_mcnemar_no_discordant_pairs_is_null() -> None:
    from agentprop.evaluation import mcnemar_exact

    result = mcnemar_exact([True, False], [True, False])
    assert result.discordant == 0
    assert result.p_value == 1.0


def test_mcnemar_detects_one_sided_advantage() -> None:
    from agentprop.evaluation import mcnemar_exact

    treatment = [True] * 10 + [True] * 20
    control = [True] * 10 + [False] * 20
    result = mcnemar_exact(treatment, control)
    assert result.treatment_only_successes == 20
    assert result.control_only_successes == 0
    assert result.p_value < 0.001


def test_mcnemar_requires_paired_lengths() -> None:
    from agentprop.evaluation import mcnemar_exact

    with pytest.raises(ValueError):
        mcnemar_exact([True], [True, False])
