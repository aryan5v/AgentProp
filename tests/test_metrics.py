from agentprop.evaluation import quality_cost_summary, robustness_under_failures
from agentprop.workflows import chain_workflow, planner_coder_tester_reviewer


def test_robustness_under_failures_reports_reachability_loss() -> None:
    graph = chain_workflow()

    robustness = robustness_under_failures(graph)

    assert robustness.baseline_reachable_pairs > 0
    assert robustness.worst_node_failure_loss > 0
    assert robustness.worst_edge_failure_loss > 0


def test_quality_cost_summary_penalizes_cost_and_latency() -> None:
    graph = planner_coder_tester_reviewer()
    token_cost = sum(node.token_cost for node in graph.nodes())

    summary = quality_cost_summary(success_rate=0.9, token_cost=token_cost, latency=5.0)

    assert summary.cost_adjusted_success > 0
    assert summary.efficiency_score < summary.success_rate
