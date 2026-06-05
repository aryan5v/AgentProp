import json
from pathlib import Path

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
    assert payload["alpha_ready"] is True
    assert payload["public_ready"] is False
    assert "Real routed LLM case-study results" in payload["blockers"]


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
