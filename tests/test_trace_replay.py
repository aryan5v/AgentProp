"""Tests for trace_replay module and trace-replay CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from agentprop.runtime.trace_replay import ReplayResult, format_replay_table, replay_trace


def _event_row(
    step, command, tokens, *, verifier=False, passed=None, final=False, action="CONTINUE"
):
    return {
        "type": "event",
        "event": {
            "step": step,
            "command": command,
            "exit_code": 0,
            "verifier_run": verifier,
            "verifier_passed": passed,
            "progress_made": True,
            "tokens_used": tokens,
            "elapsed_s": float(step),
            "error_signature": None,
            "final_answer_written": final,
            "trusted": True,
        },
        "decision": {
            "action": action,
            "reason": "ok",
            "strategy": None,
            "defer_command": None,
        },
    }


def _write_trace(
    path: Path,
    task_id: str = "t1",
    workflow: str = "planner_coder_tester_reviewer",
    baseline_tokens: int | None = None,
    include_post_finalize: bool = False,
) -> None:
    rows = [
        {
            "type": "analysis",
            "task_id": task_id,
            "baseline_tokens": baseline_tokens,
            "analysis": {"workflow": workflow, "nodes": 4, "edges": 3},
        },
        _event_row(1, "plan", 300),
        _event_row(2, "code", 500),
        _event_row(3, "verify", 200, verifier=True, passed=True, final=True, action="FINALIZE"),
    ]
    if include_post_finalize:
        rows.append(_event_row(4, "extra work after final", 700))
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_replay_basic(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    result = replay_trace(trace)
    assert isinstance(result, ReplayResult)
    assert result.task_id == "t1"
    assert result.workflow == "planner_coder_tester_reviewer"
    assert len(result.rows) == 3
    assert result.total_tokens_no_control == 300 + 500 + 200


def test_replay_uses_trace_baseline_tokens(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, baseline_tokens=1500)
    result = replay_trace(trace)
    assert result.baseline_tokens == 1500
    assert result.total_tokens_no_control == 1500
    assert result.total_tokens_with_control == 1000
    assert result.token_delta == 500
    assert result.reduction_pct == 500 / 1500 * 100


def test_replay_accepts_cli_baseline_override(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, baseline_tokens=1500)
    result = replay_trace(trace, baseline_tokens=2000)
    assert result.baseline_tokens == 2000
    assert result.total_tokens_no_control == 2000
    assert result.token_delta == 1000


def test_replay_stops_a2_cost_after_finalize(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, baseline_tokens=2200, include_post_finalize=True)
    result = replay_trace(trace)
    assert result.total_tokens_no_control == 2200
    assert result.total_tokens_with_control == 1000
    assert result.rows[-1].decision_with_control == "STOPPED"


def test_replay_no_control_shows_continue(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    result = replay_trace(trace, no_control=True)
    for row in result.rows:
        assert row.decision_no_control == "CONTINUE"


def test_format_replay_table(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)
    result = replay_trace(trace)
    table = format_replay_table(result)
    assert "Trace Replay" in table
    assert "A0 (no-control)" in table
    assert "A2 (with-control)" in table
    assert "Baseline tokens" in table
    assert "Reduction" in table


def test_replay_empty_trace(tmp_path):
    trace = tmp_path / "trace.jsonl"
    analysis_row = {
        "type": "analysis",
        "task_id": "x",
        "analysis": {"workflow": "planner_coder_tester_reviewer"},
    }
    trace.write_text(json.dumps(analysis_row) + "\n")
    result = replay_trace(trace)
    assert result.rows == []
    assert result.total_tokens_no_control == 0


def test_trace_replay_cli(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)

    from agentprop.cli import main as cli_main

    rc = cli_main(["trace-replay", str(trace)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Trace Replay" in captured.out


def test_trace_replay_cli_json(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)

    from agentprop.cli import main as cli_main

    rc = cli_main(["trace-replay", str(trace), "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert "reduction_pct" in payload
    assert "rows" in payload


def test_trace_replay_cli_baseline_tokens(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace)

    from agentprop.cli import main as cli_main

    rc = cli_main(["trace-replay", str(trace), "--baseline-tokens", "1200", "--json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["baseline_tokens"] == 1200
    assert payload["token_delta"] == 200


def test_replay_unknown_workflow_falls_back_to_recorded_decisions(tmp_path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, workflow="custom_not_available")
    result = replay_trace(trace)
    assert result.replay_warning
    assert result.rows[-1].decision_with_control == "FINALIZE"
