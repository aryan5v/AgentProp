from agentprop.evaluation import evaluate_pruning
from agentprop.propagation import ZeroForcing
from agentprop.workflows import planner_coder_tester_reviewer


def test_pruning_evaluation_reports_cost_and_coverage_delta() -> None:
    graph = planner_coder_tester_reviewer()

    evaluation = evaluate_pruning(
        graph,
        [("planner", "reviewer")],
        seeds=["planner"],
        propagation_model=ZeroForcing(),
    )

    assert evaluation.removed_edges == [("planner", "reviewer")]
    assert evaluation.pruned_cost < evaluation.baseline_cost
    assert evaluation.cost_delta < 0
    assert evaluation.coverage_delta == evaluation.pruned_coverage - evaluation.baseline_coverage
