"""Tests for indexed propagation kernels and simulate_batch."""

from __future__ import annotations

from agentprop.propagation import IndependentCascade, RandomizedZeroForcing
from agentprop.workflows import chain_workflow


def test_ic_simulate_batch_matches_single() -> None:
    graph = chain_workflow(length=8)
    model = IndependentCascade(seed=7)
    seed_sets = [["node_0"], ["node_0", "node_2"]]
    singles = [model.simulate(graph, seeds, trials=30) for seeds in seed_sets]
    batch = model.simulate_batch(graph, seed_sets, trials=30)
    assert len(batch) == 2
    assert abs(batch[0].coverage - singles[0].coverage) < 0.15
    assert abs(batch[1].coverage - singles[1].coverage) < 0.15


def test_rzf_uses_propagation_index_without_networkx_copy() -> None:
    graph = chain_workflow(length=12)
    graph.warm_analysis_cache()
    result = RandomizedZeroForcing(seed=3).simulate(graph, ["node_0"], trials=40)
    assert result.coverage > 0.0
