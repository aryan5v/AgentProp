import os
import subprocess
import sys
from pathlib import Path

from examples.coding_agent_full_suite import run as run_full_suite_example


def test_coding_agent_full_suite_example_writes_beta_artifacts(tmp_path: Path) -> None:
    paths = run_full_suite_example(tmp_path)

    expected = {
        "routing_report",
        "routing_summary",
        "context_advice",
        "host_agent_prompt",
        "decisions",
        "control_trace",
        "control_summary",
        "control_report",
    }
    assert expected <= set(paths)
    for path in paths.values():
        assert path.exists()

    prompt = paths["host_agent_prompt"].read_text(encoding="utf-8")
    assert "ExecutionEvent" in prompt
    assert "FORCE_VERIFY" in prompt


def test_coding_agent_full_suite_cli_respects_out_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli-artifacts"
    result = subprocess.run(
        [
            sys.executable,
            "examples/coding_agent_full_suite.py",
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src:."},
        text=True,
    )

    assert "AgentProp full-suite coding-agent artifacts" in result.stdout
    assert (out_dir / "routing_report.md").exists()
    assert (out_dir / "control_session" / "trace.jsonl").exists()
