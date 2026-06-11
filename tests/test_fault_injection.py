"""Tests for the fault-injection GO/NO-GO gate."""

from __future__ import annotations

from agentprop.evaluation.fault_injection import (
    chain,
    classical_placement,
    detection_probability,
    directed_distances,
    directed_js_placement,
    layered_dag,
    localization_accuracy,
    run_gate,
)


def test_directed_distances_follow_edges() -> None:
    graph = chain(4)
    distances = directed_distances(graph)
    assert distances["n0"]["n3"] == 3
    assert "n0" not in distances["n3"] or distances["n3"].get("n0") is None


def test_detection_probability_bounds_and_decay() -> None:
    near = detection_probability(0, noise=0.05)
    far = detection_probability(5, noise=0.05)
    unreachable = detection_probability(None, noise=0.05)
    assert 0.0 < unreachable < far < near < 1.0


def test_placements_return_requested_budget() -> None:
    graph = layered_dag(3, 3, seed=0)
    assert len(classical_placement(graph, 3)) == 3
    assert len(directed_js_placement(graph, 3, noise=0.05)) == 3


def test_localization_beats_chance() -> None:
    graph = chain(8)
    verifiers = directed_js_placement(graph, 3, noise=0.05)
    accuracy = localization_accuracy(graph, verifiers, noise=0.05, trials=300, seed=1)
    assert accuracy > 1.0 / graph.node_count  # better than random guessing


def test_localization_degrades_with_noise() -> None:
    graph = layered_dag(3, 3, seed=0)
    verifiers = classical_placement(graph, 3)
    low = localization_accuracy(graph, verifiers, noise=0.02, trials=300, seed=2)
    high = localization_accuracy(graph, verifiers, noise=0.3, trials=300, seed=2)
    assert low > high


def test_run_gate_smoke_is_deterministic() -> None:
    a = run_gate(noises=(0.05,), budgets=(2,), trials=50, seeds=1)
    b = run_gate(noises=(0.05,), budgets=(2,), trials=50, seeds=1)
    assert a.mean_advantage == b.mean_advantage
    assert len(a.conditions) == 4  # one per graph family
