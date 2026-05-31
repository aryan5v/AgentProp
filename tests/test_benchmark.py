from agentprop.evaluation.runner import run_benchmark
from agentprop.workflows import WORKFLOW_TEMPLATES, planner_coder_tester_reviewer


def test_builtin_workflow_registry_includes_research_fixtures() -> None:
    assert "planner_coder_tester_reviewer" in WORKFLOW_TEMPLATES
    assert "rag_pipeline" in WORKFLOW_TEMPLATES
    assert "hub_and_spoke_supervisor" in WORKFLOW_TEMPLATES


def test_run_benchmark_compares_algorithms_and_models() -> None:
    graph = planner_coder_tester_reviewer()

    rows = run_benchmark(
        graph,
        workflow_name="planner_coder_tester_reviewer",
        algorithms=["degree", "greedy"],
        models=["independent-cascade", "rzf"],
        budget=2,
        trials=5,
    )

    assert len(rows) == 4
    assert {row.algorithm for row in rows} == {"degree", "greedy"}
    assert all(row.coverage > 0 for row in rows)
