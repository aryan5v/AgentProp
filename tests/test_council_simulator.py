"""Tests for the zero-cost Council economics simulator."""

from __future__ import annotations

from agentprop.council.simulator import (
    SimConfig,
    SimModel,
    simulate_strategies,
)

POOL = [
    SimModel("flash", competence=0.55, price_per_ktok=0.10, tier=1),
    SimModel("kimi", competence=0.58, price_per_ktok=0.15, tier=2),
    SimModel("deepseek", competence=0.60, price_per_ktok=0.20, tier=2),
]
FRONTIER = SimModel("frontier", competence=0.70, price_per_ktok=1.20, tier=3)


def _by_name(outcomes):  # noqa: ANN001
    return {o.strategy: o for o in outcomes}


def test_deterministic_under_seed() -> None:
    a = simulate_strategies(POOL, frontier=FRONTIER, trials=200, seed=7)
    b = simulate_strategies(POOL, frontier=FRONTIER, trials=200, seed=7)
    assert [o.mean_cost_usd for o in a] == [o.mean_cost_usd for o in b]
    assert [o.mean_accuracy for o in a] == [o.mean_accuracy for o in b]


def test_all_strategies_present() -> None:
    names = {o.strategy for o in simulate_strategies(POOL, frontier=FRONTIER, trials=100)}
    assert names == {"single-budget", "single-frontier", "ensemble", "council"}


def test_council_far_cheaper_than_ensemble() -> None:
    out = _by_name(simulate_strategies(POOL, frontier=FRONTIER, trials=300, seed=1))
    assert out["council"].mean_cost_usd < 0.5 * out["ensemble"].mean_cost_usd


def test_frontier_is_most_expensive() -> None:
    out = _by_name(simulate_strategies(POOL, frontier=FRONTIER, trials=200, seed=2))
    assert out["single-frontier"].mean_cost_usd == max(
        o.mean_cost_usd for o in out.values()
    )


def test_council_best_cost_efficiency() -> None:
    out = _by_name(simulate_strategies(POOL, frontier=FRONTIER, trials=400, seed=3))
    ratio = {k: v.mean_cost_usd / max(v.mean_accuracy, 1e-9) for k, v in out.items()}
    assert ratio["council"] == min(ratio.values())


def test_outcomes_carry_confidence_intervals() -> None:
    out = simulate_strategies(POOL, frontier=FRONTIER, trials=150, seed=4)
    for o in out:
        assert o.accuracy.lower <= o.mean_accuracy <= o.accuracy.upper
        assert o.cost_usd.lower <= o.mean_cost_usd <= o.cost_usd.upper


def test_more_subtasks_cost_more_for_council() -> None:
    few = _by_name(
        simulate_strategies(POOL, cfg=SimConfig(subtasks=3), frontier=FRONTIER, seed=5)
    )["council"]
    many = _by_name(
        simulate_strategies(POOL, cfg=SimConfig(subtasks=8), frontier=FRONTIER, seed=5)
    )["council"]
    assert many.mean_cost_usd > few.mean_cost_usd


def test_runs_without_frontier() -> None:
    out = simulate_strategies(POOL, trials=50)
    assert {o.strategy for o in out} == {"single-budget", "ensemble", "council"}
