"""Tests for feature-conditioned propagation that transfers to unseen graphs."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from agentprop.core import AgentGraph
from agentprop.propagation import FeatureCalibratedPropagation, observations_from_trace_dicts


def _graph(n: int, *, seed: int) -> AgentGraph:
    """Random DAG where high-relevance/high-reliability edges activate."""

    rng = random.Random(seed)
    graph = AgentGraph()
    ids = [f"n{i}" for i in range(n)]
    for node_id in ids:
        graph.add_node(node_id)
    for i in range(n - 1):
        for j in range(i + 1, min(i + 3, n)):
            graph.add_edge(
                ids[i],
                ids[j],
                relevance=rng.random(),
                reliability=rng.random(),
                activation_probability=rng.random(),
            )
    return graph


def _outcomes(graph: AgentGraph, *, seed: int) -> dict[tuple[str, str], bool]:
    """Ground truth: activation depends on relevance (a feature, not identity)."""

    rng = random.Random(seed)
    return {
        (e.source, e.target): rng.random() < e.relevance for e in graph.edges()
    }


def test_fit_learns_relevance_signal_and_transfers() -> None:
    train_graphs = [_graph(12, seed=s) for s in range(4)]
    observations = [(g, _outcomes(g, seed=100 + i)) for i, g in enumerate(train_graphs)]
    model = FeatureCalibratedPropagation(seed=0).fit(observations)

    unseen = _graph(12, seed=99)
    edges = unseen.edges()
    high = max(edges, key=lambda e: e.relevance)
    low = min(edges, key=lambda e: e.relevance)
    p_high = model.edge_probability(unseen, high.source, high.target)
    p_low = model.edge_probability(unseen, low.source, low.target)
    assert p_high > p_low
    assert 0.0 <= p_low <= p_high <= 1.0


def test_simulate_runs_on_unseen_graph() -> None:
    train = _graph(10, seed=1)
    model = FeatureCalibratedPropagation(seed=0).fit([(train, _outcomes(train, seed=5))])
    unseen = _graph(10, seed=2)
    result = model.simulate(unseen, [unseen.nodes()[0].id], trials=50)
    assert 0.0 < result.coverage <= 1.0


def test_unfitted_model_raises() -> None:
    graph = _graph(5, seed=1)
    with pytest.raises(RuntimeError):
        FeatureCalibratedPropagation().edge_probability(graph, "n0", "n1")


def test_fit_requires_observations() -> None:
    with pytest.raises(ValueError):
        FeatureCalibratedPropagation().fit([])


def test_save_load_roundtrip(tmp_path: Path) -> None:
    train = _graph(8, seed=3)
    model = FeatureCalibratedPropagation(seed=0).fit([(train, _outcomes(train, seed=7))])
    path = model.save(tmp_path / "model.json")
    loaded = FeatureCalibratedPropagation.load(path)
    edge = train.edges()[0]
    assert loaded.edge_probability(train, edge.source, edge.target) == pytest.approx(
        model.edge_probability(train, edge.source, edge.target)
    )


def test_observations_from_trace_dicts() -> None:
    graph = _graph(5, seed=4)
    edge = graph.edges()[0]
    rows = [
        {"source": edge.source, "target": edge.target, "activated": True},
        {"row_type": "noise"},
    ]
    outcomes = observations_from_trace_dicts(graph, rows)
    assert outcomes == {(edge.source, edge.target): True}
