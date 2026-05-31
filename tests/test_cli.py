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
    assert "AgentProp Optimization Report" in output.read_text()


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
