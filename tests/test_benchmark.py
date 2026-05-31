from pathlib import Path

from agentprop.evaluation.runner import run_benchmark
from agentprop.workflows import (
    WORKFLOW_TEMPLATES,
    export_builtin_workflows,
    planner_coder_tester_reviewer,
)


def test_builtin_workflow_registry_includes_research_fixtures() -> None:
    assert "chain" in WORKFLOW_TEMPLATES
    assert "star" in WORKFLOW_TEMPLATES
    assert "tree" in WORKFLOW_TEMPLATES
    assert "dense_graph" in WORKFLOW_TEMPLATES
    assert "small_world_graph" in WORKFLOW_TEMPLATES
    assert "random_directed_graph" in WORKFLOW_TEMPLATES
    assert "generic_dag" in WORKFLOW_TEMPLATES
    assert "layered_pipeline" in WORKFLOW_TEMPLATES
    assert "planner_coder_tester_reviewer" in WORKFLOW_TEMPLATES
    assert "rag_pipeline" in WORKFLOW_TEMPLATES
    assert "hub_and_spoke_supervisor" in WORKFLOW_TEMPLATES


def test_run_benchmark_compares_algorithms_and_models() -> None:
    graph = planner_coder_tester_reviewer()

    rows = run_benchmark(
        graph,
        workflow_name="planner_coder_tester_reviewer",
        algorithms=["degree", "in-degree", "out-degree", "closeness", "k-core", "greedy"],
        models=["independent-cascade", "rzf"],
        budget=2,
        trials=5,
    )

    assert len(rows) == 12
    assert {row.algorithm for row in rows} == {
        "degree",
        "in-degree",
        "out-degree",
        "closeness",
        "k-core",
        "greedy",
    }
    assert all(row.coverage > 0 for row in rows)


def test_export_builtin_workflows_writes_all_templates(tmp_path: Path) -> None:
    written = export_builtin_workflows(tmp_path)

    assert len(written) == len(WORKFLOW_TEMPLATES)
    assert (tmp_path / "rag_pipeline.json").exists()
