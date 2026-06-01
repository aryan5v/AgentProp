from agentprop.evaluation import evaluate_pruning, summarize_pruning_risk
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

    risk = summarize_pruning_risk(evaluation, target_cost_reduction=0.3)
    assert risk.achieved_cost_reduction > 0
    assert risk.risk_score >= risk.coverage_loss
