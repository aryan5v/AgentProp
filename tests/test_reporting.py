import json
from pathlib import Path

from agentprop.algorithms import bottleneck_nodes, low_weight_edges, risk_aware_verifier_placement
from agentprop.evaluation import compare_routing
from agentprop.evaluation.reporting import render_markdown_report, write_report
from agentprop.propagation import IndependentCascade
from agentprop.workflows import planner_coder_tester_reviewer


def test_markdown_report_contains_core_recommendation() -> None:
    graph = planner_coder_tester_reviewer()
    model = IndependentCascade(seed=0)
    result = model.simulate(graph, ["planner", "tester"], trials=5)
    report = compare_routing(
        graph,
        ["planner", "tester"],
        model.name,
        result,
        bottlenecks=bottleneck_nodes(graph),
        pruning_candidates=low_weight_edges(graph),
        verifier_candidates=risk_aware_verifier_placement(graph, 2),
    )

    markdown = render_markdown_report(report, workflow_name="demo")

    assert "# AgentProp Optimization Report" in markdown
    assert "planner, tester" in markdown
    assert "Cost Comparison" in markdown


def test_write_report_supports_json(tmp_path: Path) -> None:
    graph = planner_coder_tester_reviewer()
    model = IndependentCascade(seed=0)
    result = model.simulate(graph, ["planner"], trials=5)
    report = compare_routing(graph, ["planner"], model.name, result)
    path = tmp_path / "report.json"

    write_report(report, path, workflow_name="demo")

    payload = json.loads(path.read_text())
    assert payload["seeds"] == ["planner"]
    assert payload["broadcast_cost"]["total_cost"] > 0
