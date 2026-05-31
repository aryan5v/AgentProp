from agentprop.algorithms import (
    articulation_bottlenecks,
    bottleneck_nodes,
    bridge_bottlenecks,
    celf_seed_selection,
    closeness_seed_selection,
    cost_aware_greedy_seed_selection,
    degree_seed_selection,
    edge_bottlenecks,
    failure_sensitive_nodes,
    greedy_seed_selection,
    k_core_seed_selection,
    low_reliability_cut_points,
    observability_coverage,
    observability_scores,
    pagerank_seed_selection,
    risk_aware_verifier_placement,
    verifier_observability_placement,
)
from agentprop.propagation import IndependentCascade
from agentprop.workflows import planner_coder_tester_reviewer


def test_pagerank_seed_selection_returns_budgeted_nodes() -> None:
    graph = planner_coder_tester_reviewer()

    seeds = pagerank_seed_selection(graph, 2)

    assert len(seeds) == 2
    assert all(seed in {node.id for node in graph.nodes()} for seed in seeds)


def test_core_centrality_seed_selection_returns_budgeted_nodes() -> None:
    graph = planner_coder_tester_reviewer()

    assert len(degree_seed_selection(graph, 2, direction="in")) == 2
    assert len(degree_seed_selection(graph, 2, direction="out")) == 2
    assert len(closeness_seed_selection(graph, 2)) == 2
    assert len(k_core_seed_selection(graph, 2)) == 2


def test_greedy_seed_selection_uses_propagation_model() -> None:
    graph = planner_coder_tester_reviewer()

    seeds = greedy_seed_selection(
        graph,
        2,
        propagation_model=IndependentCascade(seed=0),
        trials=10,
    )

    assert len(seeds) == 2


def test_celf_and_cost_aware_seed_selection_return_budgeted_nodes() -> None:
    graph = planner_coder_tester_reviewer()
    model = IndependentCascade(seed=0)

    celf_seeds = celf_seed_selection(graph, 2, propagation_model=model, trials=5)
    cost_aware_seeds = cost_aware_greedy_seed_selection(
        graph,
        2,
        propagation_model=model,
        trials=5,
    )

    assert len(celf_seeds) == 2
    assert len(cost_aware_seeds) == 2


def test_diagnostics_return_ranked_candidates() -> None:
    graph = planner_coder_tester_reviewer()

    assert bottleneck_nodes(graph)
    assert edge_bottlenecks(graph)
    assert low_reliability_cut_points(graph)
    assert failure_sensitive_nodes(graph)
    assert risk_aware_verifier_placement(graph, 2)


def test_structural_bottlenecks_detect_chain_cut_points() -> None:
    from agentprop.workflows import chain_workflow

    graph = chain_workflow()

    assert articulation_bottlenecks(graph)
    assert bridge_bottlenecks(graph)


def test_observability_metrics_rank_and_cover_workflow_nodes() -> None:
    graph = planner_coder_tester_reviewer()

    scores = observability_scores(graph)
    observers = verifier_observability_placement(graph, 2)
    coverage = observability_coverage(graph, observers)

    assert set(scores) == {node.id for node in graph.nodes()}
    assert len(observers) == 2
    assert coverage >= 0.8
