from agentprop.evaluation import (
    QualityAwareRoutingObjective,
    calibrate_context_compression,
    graded_context_allocations,
    quality_cost_summary,
    robustness_under_failures,
    routing_risks,
)
from agentprop.evaluation.metrics import CostSummary
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


def test_graded_context_and_risk_surface_coder_starvation() -> None:
    graph = planner_coder_tester_reviewer()

    allocations = graded_context_allocations(
        graph,
        seeds=["planner", "tester"],
        activated_nodes={node.id for node in graph.nodes()},
        min_ratio=0.25,
        max_non_seed_ratio=0.25,
    )
    risks = routing_risks(graph, context_ratios=allocations)

    assert allocations["coder"] < 1.0
    assert risks[0].node_id == "coder"
    assert risks[0].severity == "high"


def test_context_compression_calibration_uses_measured_stage_tokens() -> None:
    profile = calibrate_context_compression(
        [
            {
                "stage_tokens": {"coder": 1000, "tester": 500},
                "stage_full_context": {"coder": True, "tester": True},
            },
            {
                "stage_tokens": {"coder": 400, "tester": 250},
                "stage_full_context": {"coder": False, "tester": False},
            },
        ]
    )

    assert profile.ratio_for("coder") == 0.4
    assert profile.ratio_for("tester") == 0.5


def test_quality_aware_objective_penalizes_context_starvation() -> None:
    graph = planner_coder_tester_reviewer()
    objective = QualityAwareRoutingObjective(token_penalty=0.0)
    full = {node.id: 1.0 for node in graph.nodes()}
    compressed = dict(full)
    compressed["coder"] = 0.25
    cost = CostSummary(token_cost=100.0, message_cost=10.0, latency=1.0, message_count=1)

    full_score = objective.score(
        graph,
        seeds=["coder"],
        activated_nodes=set(full),
        cost=cost,
        context_ratios=full,
    )
    compressed_score = objective.score(
        graph,
        seeds=["planner"],
        activated_nodes=set(full),
        cost=cost,
        context_ratios=compressed,
    )

    assert full_score > compressed_score
