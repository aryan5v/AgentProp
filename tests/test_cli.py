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
