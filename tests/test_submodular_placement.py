"""Tests for the submodular probabilistic-coverage placement."""

from __future__ import annotations

import random

from agentprop.algorithms.submodular_placement import (
    coverage_objective,
    pair_separation_scores,
    submodular_verifier_placement,
)
from agentprop.algorithms.verifier_placement import resolving_coverage
from agentprop.core import AgentGraph
from agentprop.evaluation.fault_injection import layered_dag, spider
from agentprop.workflows import WORKFLOW_TEMPLATES


def _random_graph(n: int, seed: int) -> AgentGraph:
    rng = random.Random(seed)
    graph = AgentGraph()
    for i in range(n):
        graph.add_node(f"n{i}")
    for i in range(1, n):
        graph.add_edge(f"n{rng.randint(0, i - 1)}", f"n{i}")
    for _ in range(n // 2):
        a, b = rng.sample(range(n), 2)
        graph.add_edge(f"n{a}", f"n{b}")
    return graph


def _objective(graph: AgentGraph, verifiers: list[str], model: str) -> float:
    return coverage_objective(graph, verifiers, model=model)  # type: ignore[arg-type]


def test_objective_is_monotone_and_submodular_on_random_graphs() -> None:
    """Diminishing returns: gain of x given A >= gain of x given B ⊇ A."""

    rng = random.Random(7)
    for seed in range(5):
        graph = _random_graph(9, seed)
        node_ids = [n.id for n in graph.nodes()]
        for model in ("deterministic", "noisy"):
            for _ in range(10):
                a_size = rng.randint(0, 3)
                subset_a = rng.sample(node_ids, a_size)
                extra = [n for n in node_ids if n not in subset_a]
                superset_b = subset_a + rng.sample(extra, min(2, len(extra)))
                candidates = [n for n in node_ids if n not in superset_b]
                if not candidates:
                    continue
                x = rng.choice(candidates)
                base_a = _objective(graph, subset_a, model)
                base_b = _objective(graph, superset_b, model)
                gain_a = _objective(graph, [*subset_a, x], model) - base_a
                gain_b = _objective(graph, [*superset_b, x], model) - base_b
                assert gain_a >= -1e-9  # monotone
                assert gain_a >= gain_b - 1e-9  # submodular


def test_lazy_greedy_matches_naive_greedy() -> None:
    for seed in (1, 2, 3):
        graph = _random_graph(10, seed)
        lazy = submodular_verifier_placement(graph, 3)
        # Naive greedy reference
        node_ids = [n.id for n in graph.nodes()]
        chosen: list[str] = []
        for _ in range(3):
            best = max(
                (n for n in node_ids if n not in chosen),
                key=lambda n: _objective(graph, [*chosen, n], "deterministic"),
            )
            chosen.append(best)
        assert _objective(graph, list(lazy.verifiers), "deterministic") >= (
            _objective(graph, chosen, "deterministic") - 1e-9
        )


def test_deterministic_objective_tracks_resolving_coverage() -> None:
    graph = WORKFLOW_TEMPLATES["planner_coder_tester_reviewer"]()
    placement = submodular_verifier_placement(graph, 2)
    assert placement.objective_fraction > 0.5
    # Full surrogate coverage implies full resolving coverage.
    full = submodular_verifier_placement(graph, graph.node_count)
    if full.objective_fraction == 1.0:
        assert resolving_coverage(graph, list(full.verifiers)) == 1.0


def test_marginal_gains_are_diminishing() -> None:
    graph = layered_dag(3, 3, seed=0)
    placement = submodular_verifier_placement(graph, 4, model="noisy", noise=0.05)
    gains = placement.marginal_gains
    assert all(gains[i] >= gains[i + 1] - 1e-9 for i in range(len(gains) - 1))


def test_handles_trivial_inputs() -> None:
    empty = AgentGraph()
    assert submodular_verifier_placement(empty, 2).verifiers == ()
    graph = spider(2, 2)
    assert len(submodular_verifier_placement(graph, 100).verifiers) == graph.node_count


def test_separation_scores_symmetric_pairs_only_once() -> None:
    graph = _random_graph(6, 4)
    scores = pair_separation_scores(graph)
    pair_count = graph.node_count * (graph.node_count - 1) // 2
    assert all(len(row) == pair_count for row in scores.values())
