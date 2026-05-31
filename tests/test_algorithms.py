from agentprop.algorithms import (
    bottleneck_nodes,
    greedy_seed_selection,
    pagerank_seed_selection,
    risk_aware_verifier_placement,
)
from agentprop.propagation import IndependentCascade
from agentprop.workflows import planner_coder_tester_reviewer


def test_pagerank_seed_selection_returns_budgeted_nodes() -> None:
    graph = planner_coder_tester_reviewer()

    seeds = pagerank_seed_selection(graph, 2)

    assert len(seeds) == 2
    assert all(seed in {node.id for node in graph.nodes()} for seed in seeds)


def test_greedy_seed_selection_uses_propagation_model() -> None:
    graph = planner_coder_tester_reviewer()

    seeds = greedy_seed_selection(
        graph,
        2,
        propagation_model=IndependentCascade(seed=0),
        trials=10,
    )

    assert len(seeds) == 2


def test_diagnostics_return_ranked_candidates() -> None:
    graph = planner_coder_tester_reviewer()

    assert bottleneck_nodes(graph)
    assert risk_aware_verifier_placement(graph, 2)
