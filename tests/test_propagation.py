from agentprop.core import AgentGraph
from agentprop.evaluation import graded_context_allocations
from agentprop.propagation import (
    IndependentCascade,
    LinearThreshold,
    QualityCascade,
    QualityCascadeResult,
    RandomizedZeroForcing,
    ZeroForcing,
)
from agentprop.workflows import planner_coder_tester_reviewer


def test_independent_cascade_reaches_full_pipeline_with_reliable_edges() -> None:
    graph = planner_coder_tester_reviewer()
    for edge in graph.edges():
        graph.add_edge(
            edge.source,
            edge.target,
            message_cost=edge.message_cost,
            latency=edge.latency,
            relevance=edge.relevance,
            reliability=edge.reliability,
            activation_probability=1.0,
            dependency_strength=edge.dependency_strength,
            weight=edge.weight,
            **edge.metadata,
        )
    result = IndependentCascade(seed=0).simulate(graph, ["planner"], trials=20)

    assert result.coverage == 1.0
    assert result.full_activation_probability == 1.0
    assert "final" in result.activated_nodes


def test_randomized_zero_forcing_reports_expected_time() -> None:
    graph = planner_coder_tester_reviewer()
    result = RandomizedZeroForcing(seed=0).simulate(graph, ["planner"], trials=20)

    assert result.coverage > 0.5
    assert result.expected_propagation_time is not None
    assert result.trials == 20


def test_linear_threshold_activates_when_weight_share_is_high_enough() -> None:
    graph = planner_coder_tester_reviewer()
    result = LinearThreshold(threshold=0.5).simulate(graph, ["planner"])

    assert "coder" in result.activated_nodes
    assert result.coverage >= 0.8


def test_zero_forcing_deterministically_forces_path() -> None:
    graph = AgentGraph()
    for node_id in ("a", "b", "c"):
        graph.add_agent(node_id)
    graph.add_edge("a", "b", activation_probability=1.0)
    graph.add_edge("b", "c", activation_probability=1.0)

    result = ZeroForcing().simulate(graph, ["a"])

    assert result.coverage == 1.0
    assert result.full_activation_probability == 1.0
    assert result.expected_propagation_time == 2.0
    assert result.coverage_by_round == [1 / 3, 2 / 3, 1.0]


def test_quality_cascade_seeds_have_full_quality() -> None:
    from agentprop.workflows import chain_workflow

    graph = chain_workflow()
    seed_id = graph.nodes()[0].id

    result = QualityCascade().simulate(graph, [seed_id])

    assert isinstance(result, QualityCascadeResult)
    assert result.node_qualities[seed_id] == 1.0


def test_quality_degrades_monotonically_along_chain() -> None:
    from agentprop.workflows import chain_workflow

    graph = chain_workflow()
    nodes = graph.nodes()
    seed_id = nodes[0].id

    result = QualityCascade().simulate(graph, [seed_id])

    qualities = [result.node_qualities.get(n.id, 0.0) for n in nodes]
    for i in range(len(qualities) - 1):
        assert qualities[i] >= qualities[i + 1]


def test_quality_cascade_drives_context_allocation() -> None:
    from agentprop.workflows import chain_workflow

    graph = chain_workflow()
    seed_id = graph.nodes()[0].id

    result = QualityCascade().simulate(graph, [seed_id])
    ratios = graded_context_allocations(graph, seeds=[seed_id], quality_result=result)

    from agentprop.core import NodeType

    assert ratios[seed_id] == 1.0
    for node in graph.nodes()[1:]:
        if node.type == NodeType.OUTPUT:
            continue
        expected = result.node_qualities.get(node.id, 0.0)
        assert abs(ratios[node.id] - expected) < 1e-9


def test_quality_cascade_mean_output_quality_is_populated() -> None:
    from agentprop.workflows import chain_workflow

    graph = chain_workflow()

    result = QualityCascade().simulate(graph, [graph.nodes()[0].id])

    assert 0.0 <= result.mean_output_quality <= 1.0
