import json
from pathlib import Path

import pytest

from agentprop.cli import main
from agentprop.core import NodeType
from agentprop.core.validation import validate_workflow_dict
from agentprop.workflows import TOPOLOGIES, scaffold_workflow


def test_pipeline_topology_chains_nodes() -> None:
    graph = scaffold_workflow(["planner", "coder", "tester", "reviewer"], topology="pipeline")
    assert graph.node_count == 4
    assert graph.edge_count == 3
    edges = {(edge.source, edge.target) for edge in graph.edges()}
    assert edges == {("planner", "coder"), ("coder", "tester"), ("tester", "reviewer")}
    assert graph.node("planner").type is NodeType.PLANNER
    assert graph.node("reviewer").type is NodeType.REVIEWER
    assert graph.node("coder").type is NodeType.EXECUTOR


def test_fan_out_topology_dispatches_from_first_node() -> None:
    graph = scaffold_workflow(["hub", "a", "b", "c"], topology="fan-out")
    assert graph.edge_count == 3
    edges = {(edge.source, edge.target) for edge in graph.edges()}
    assert edges == {("hub", "a"), ("hub", "b"), ("hub", "c")}


def test_hub_spoke_topology_is_bidirectional() -> None:
    graph = scaffold_workflow(["hub", "a", "b"], topology="hub-spoke")
    assert graph.edge_count == 4
    edges = {(edge.source, edge.target) for edge in graph.edges()}
    assert edges == {("hub", "a"), ("a", "hub"), ("hub", "b"), ("b", "hub")}


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_scaffolded_graphs_validate(topology: str) -> None:
    graph = scaffold_workflow(["planner", "coder", "tester"], topology=topology)
    validate_workflow_dict(graph.to_dict())


def test_unknown_topology_raises() -> None:
    with pytest.raises(ValueError, match="Unknown topology"):
        scaffold_workflow(["a", "b"], topology="mesh")


def test_duplicate_nodes_raise() -> None:
    with pytest.raises(ValueError, match="unique"):
        scaffold_workflow(["a", "a"], topology="pipeline")


def test_blank_nodes_raise() -> None:
    with pytest.raises(ValueError, match="non-blank"):
        scaffold_workflow(["a", "  "], topology="pipeline")


def test_fan_out_requires_two_nodes() -> None:
    with pytest.raises(ValueError, match="at least 2 nodes"):
        scaffold_workflow(["only"], topology="fan-out")


def test_cli_init_writes_valid_workflow(tmp_path: Path) -> None:
    out = tmp_path / "wf.json"
    rc = main(
        [
            "init",
            "wf",
            "--nodes",
            "planner,coder,tester",
            "--type",
            "pipeline",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    validate_workflow_dict(json.loads(out.read_text()))


def test_cli_init_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "wf.json"
    rc = main(
        ["init", "wf", "--nodes", "a,b,c", "--type", "fan-out", "--out", str(out), "--json"]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["topology"] == "fan-out"
    assert payload["node_count"] == 3
    assert payload["nodes"] == ["a", "b", "c"]


def test_cli_init_defaults_output_to_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["init", "myflow", "--nodes", "a,b"])
    assert rc == 0
    assert (tmp_path / "myflow.json").exists()
