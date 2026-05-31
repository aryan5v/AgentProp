from agentprop.propagation import IndependentCascade, LinearThreshold, RandomizedZeroForcing
from agentprop.workflows import planner_coder_tester_reviewer


def test_independent_cascade_reaches_full_pipeline_with_reliable_edges() -> None:
    graph = planner_coder_tester_reviewer()
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
