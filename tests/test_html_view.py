"""Tests for the self-contained interactive workflow view."""

from __future__ import annotations

import json
from pathlib import Path

from agentprop.cli import main as cli_main
from agentprop.visualization import load_trace_rows, render_workflow_view, write_workflow_view
from agentprop.workflows import WORKFLOW_TEMPLATES


def _graph():  # noqa: ANN202
    return WORKFLOW_TEMPLATES["planner_coder_tester_reviewer"]()


def test_render_contains_nodes_and_no_external_assets() -> None:
    graph = _graph()
    html = render_workflow_view(graph, title="demo")
    assert "<!doctype html>" in html
    for node in graph.nodes():
        assert node.id in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in html


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "view.html"
    path = write_workflow_view(_graph(), out, title="demo")
    assert path.exists()
    assert path.read_text().startswith("<!doctype html>")


def test_trace_rows_loaded_and_embedded(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    rows = [
        {"row_type": "event", "payload": {"node_id": "coder", "tokens": 120}},
        {"row_type": "decision", "payload": {"decision": "FORCE_VERIFY"}},
    ]
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    loaded = load_trace_rows(trace)
    assert len(loaded) == 2
    html = render_workflow_view(_graph(), trace_rows=loaded)
    assert "FORCE_VERIFY" in html


def test_cli_view_command(tmp_path: Path) -> None:
    out = tmp_path / "view.html"
    code = cli_main(
        ["view", "planner_coder_tester_reviewer", "--out", str(out), "--trials", "5"]
    )
    assert code == 0
    html = out.read_text()
    assert "AgentProp workflow view" in html
    assert "verifier_placement" in html or "analysis" in html
