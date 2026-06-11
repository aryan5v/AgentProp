import json
from pathlib import Path

import agentprop.cli as cli
from agentprop.cli import main


def test_cli_optimize_emits_json(capsys) -> None:  # type: ignore[no-untyped-def]
    workflow = Path("benchmarks/workflows/planner_coder_tester_reviewer.json")

    exit_code = main(["optimize", str(workflow), "--budget", "2", "--trials", "10", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seeds"]
    assert payload["estimated_savings"] >= 0


def test_cli_analyze_accepts_builtin_workflow(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["analyze", "planner_coder_tester_reviewer", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["nodes"] == 5
    assert payload["bottlenecks"]


def test_cli_report_writes_markdown(tmp_path: Path) -> None:
    output = tmp_path / "report.md"

    exit_code = main(
        [
            "report",
            "planner_coder_tester_reviewer",
            "--budget",
            "2",
            "--trials",
            "5",
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    report = output.read_text()
    assert "AgentProp Optimization Report" in report
    assert "Pruning Risk" in report
    assert "Robustness" in report


def test_cli_report_writes_html(tmp_path: Path) -> None:
    output = tmp_path / "report.html"

    exit_code = main(
        [
            "report",
            "planner_coder_tester_reviewer",
            "--budget",
            "2",
            "--trials",
            "5",
            "--out",
            str(output),
            "--format",
            "html",
        ]
    )

    assert exit_code == 0
    html = output.read_text()
    assert "<!doctype html>" in html
    assert "Cost Comparison" in html


def test_cli_trace_and_viz_write_artifacts(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    workflow = tmp_path / "workflow.json"
    dot = tmp_path / "workflow.dot"
    trace.write_text(
        json.dumps(
            {
                "events": [
                    {"source": "planner", "target": "coder", "token_cost": 100},
                    {"source": "coder", "target": "tester", "token_cost": 80},
                ]
            }
        )
    )

    trace_exit = main(["trace", str(trace), "--out", str(workflow)])
    viz_exit = main(["viz", str(workflow), "--out", str(dot)])

    assert trace_exit == 0
    assert viz_exit == 0
    assert workflow.exists()
    assert "digraph" in dot.read_text()


def test_cli_agent_instructions_writes_markdown(tmp_path: Path) -> None:
    output = tmp_path / "instructions.md"

    exit_code = main(
        [
            "agent-instructions",
            "planner_coder_tester_reviewer",
            "--target",
            "claude-code",
            "--budget",
            "2",
            "--trials",
            "3",
            "--out",
            str(output),
        ]
    )

    assert exit_code == 0
    markdown = output.read_text()
    assert "AgentProp Brief For Claude Code" in markdown
    assert "Suggested Agent Prompt" in markdown


def test_cli_simulate_emits_json(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "simulate",
            "chain",
            "--seeds",
            "node_0",
            "--model",
            "zero-forcing",
            "--trials",
            "1",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["coverage"] == 1.0
    assert payload["activated_nodes"]


def test_cli_prune_emits_json(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "prune",
            "planner_coder_tester_reviewer",
            "--target-token-reduction",
            "0.2",
            "--model",
            "zero-forcing",
            "--trials",
            "1",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed_edges"]
    assert payload["achieved_cost_reduction"] >= 0.2


def test_cli_readiness_emits_json(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["readiness", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall_score"] >= 0.8
    assert payload["target"] == "public beta"
    assert "blockers" not in payload
    assert "items" in payload


def test_cli_version_flag(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["--version"])

    assert exit_code == 0
    version = capsys.readouterr().out.strip()
    assert version
    assert version[0].isdigit() or version.startswith("unknown")


def test_cli_workflows_list(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["workflows", "list", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    names = {row["name"] for row in payload}
    assert "planner_coder_tester_reviewer" in names


def test_cli_doctor_graph_tier(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["doctor", "--tier", "graph", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["tier"] == "graph"


def test_cli_doctor_graph_tier_marks_optional_graphviz_as_warning(
    capsys, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    def fake_which(command: str) -> str | None:
        if command == "dot":
            return None
        return f"/usr/bin/{command}"

    monkeypatch.setattr(cli.shutil, "which", fake_which)

    exit_code = main(["doctor", "--tier", "graph"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "[warn] graphviz_dot" in output
    assert "[FAIL] graphviz_dot" not in output


def test_cli_simulate_quality_cascade(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "simulate",
            "chain",
            "--seeds",
            "node_0",
            "--model",
            "quality-cascade",
            "--trials",
            "1",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "quality-cascade"


def test_cli_invalid_workflow_returns_validation_error(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    workflow = tmp_path / "bad.json"
    workflow.write_text(
        json.dumps(
            {
                "nodes": [{"id": "a"}, {"id": "a"}],
                "edges": [],
            }
        )
    )

    exit_code = main(["analyze", str(workflow)])

    assert exit_code == 2
    assert "Workflow validation failed" in capsys.readouterr().err


DOC_INDEX_COMMANDS = [
    ["optimize", "benchmarks/workflows/planner_coder_tester_reviewer.json", "--budget", "2"],
    ["simulate", "chain", "--seeds", "node_0", "--model", "zero-forcing"],
    ["simulate", "chain", "--seeds", "node_0", "--model", "quality-cascade"],
    ["prune", "planner_coder_tester_reviewer", "--target-token-reduction", "0.3"],
    ["benchmark", "planner_coder_tester_reviewer", "--budget", "2", "--trials", "5"],
    ["analyze", "planner_coder_tester_reviewer", "--json"],
    ["workflows", "list"],
    ["doctor", "--tier", "graph"],
]


def test_docs_index_common_commands_parse() -> None:
    for argv in DOC_INDEX_COMMANDS:
        cmd = list(argv)
        if cmd[0] == "benchmark":
            cmd.extend(["--trials", "1"])
        exit_code = main(cmd)
        assert exit_code == 0, f"failed for: {' '.join(argv)}"


def test_cli_ingest_trace_pipeline(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    trace = tmp_path / "trace.json"
    workflow = tmp_path / "workflow.json"
    brief = tmp_path / "brief.md"
    trace.write_text(
        json.dumps(
            {
                "events": [
                    {"source": "planner", "target": "coder", "token_cost": 120},
                    {"source": "coder", "target": "tester", "token_cost": 90},
                ]
            }
        )
    )

    exit_code = main(
        [
            "ingest-trace",
            str(trace),
            "--out-workflow",
            str(workflow),
            "--out-brief",
            str(brief),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert workflow.exists()
    assert brief.exists()
    assert payload["seeds"]


def test_cli_control_demo_writes_artifacts(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "control-demo",
            "--demo",
            "terminal",
            "--out-dir",
            str(tmp_path),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["artifacts"]["trace"]).exists()
    assert Path(payload["artifacts"]["summary"]).exists()
    assert Path(payload["artifacts"]["report"]).exists()
    assert payload["summary"]["outcome"]["passed"] is True
