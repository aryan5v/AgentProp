"""Tests for IMM/TIM scalable influence maximization."""

from __future__ import annotations

from agentprop.algorithms.influence_maximization import (
    IMMConfig,
    estimate_imm_influence,
    imm_greedy_seed_selection,
)
from agentprop.algorithms.seed_selection import auto_seed_algorithm
from agentprop.core import AgentGraph


def _chain_graph(n: int) -> AgentGraph:
    nodes = [{"id": f"n{i}"} for i in range(n)]
    edges = [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(n - 1)]
    return AgentGraph.from_dict({"nodes": nodes, "edges": edges})


def test_imm_selects_within_budget() -> None:
    graph = _chain_graph(25)
    seeds = imm_greedy_seed_selection(
        graph,
        3,
        config=IMMConfig(rr_samples=80, seed=0),
    )
    assert len(seeds) == 3
    assert len(set(seeds)) == 3


def test_estimate_imm_influence_positive() -> None:
    graph = _chain_graph(20)
    seeds = imm_greedy_seed_selection(graph, 2, config=IMMConfig(rr_samples=60, seed=1))
    score = estimate_imm_influence(graph, seeds, rr_samples=60, seed=1)
    assert 0.0 < score <= 1.0


def test_auto_seed_algorithm_thresholds() -> None:
    small = _chain_graph(10)
    medium = _chain_graph(30)
    large = _chain_graph(70)
    assert auto_seed_algorithm(small, requested="auto") == "greedy"
    assert auto_seed_algorithm(medium, requested="auto") == "rzf-centrality"
    assert auto_seed_algorithm(large, requested="auto") == "imm"
    assert auto_seed_algorithm(large, requested="degree") == "degree"
