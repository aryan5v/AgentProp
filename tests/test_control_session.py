import json
from pathlib import Path

from agentprop.runtime import ControlSession, ControlSessionConfig, ExecutionEvent
from agentprop.runtime.demos import run_control_demo


def test_control_session_records_analysis_events_and_outcome(tmp_path: Path) -> None:
    session = ControlSession(
        ControlSessionConfig(
            workflow="planner_coder_tester_reviewer",
            task_id="unit-control",
            category="implementation",
            baseline_tokens=100,
            repeated_error_threshold=2,
        )
    )

    session.observe(
        ExecutionEvent(
            step=1,
            verifier_run=True,
            verifier_passed=False,
            error_signature="same-error",
            tokens_used=30,
        )
    )
    decision = session.observe(
        ExecutionEvent(
            step=2,
            verifier_run=True,
            verifier_passed=False,
            error_signature="same-error",
            tokens_used=20,
        )
    )
    outcome = session.record_outcome(passed=True, quality_score=1.0)
    trace_path = session.save_trace(tmp_path / "trace.jsonl")

    assert decision.action == "SWITCH_STRATEGY"
    assert outcome["token_savings"] == 0.5
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert rows[0]["type"] == "analysis"
    assert rows[-1]["type"] == "outcome"


def test_control_session_distrusts_untrusted_self_reported_pass() -> None:
    session = ControlSession.start(
        "planner_coder_tester_reviewer",
        task_id="untrusted-pass",
        category="implementation",
    )

    decision = session.observe(
        ExecutionEvent(
            step=1,
            verifier_run=True,
            verifier_passed=True,
            final_answer_written=True,
            trusted=False,
        )
    )

    assert decision.action == "FORCE_VERIFY"
    assert "self-reported pass" in decision.reason


def test_control_demo_writes_expected_artifacts(tmp_path: Path) -> None:
    result = run_control_demo("framework", tmp_path)

    assert result.artifacts["trace"].exists()
    assert result.artifacts["summary"].exists()
    assert result.artifacts["report"].exists()
    summary = json.loads(result.artifacts["summary"].read_text())
    assert summary["analysis"]["verifier_candidates"]
    assert summary["outcome"]["passed"] is True
