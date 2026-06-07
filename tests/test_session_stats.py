import json
from pathlib import Path

import pytest

from agentprop.cli import main
from agentprop.integrations import aggregate_session_stats, render_session_stats_markdown
from agentprop.integrations.session_store import SessionStore
from agentprop.runtime import ExecutionEvent


def _populate_sessions(sess_dir: Path, outcomes: list[bool]) -> None:
    store = SessionStore(sess_dir)
    for index, passed in enumerate(outcomes):
        session_id, session = store.start_session(
            workflow="planner_coder_tester_reviewer",
            task_id=f"task-{index}",
            category="coding",
            baseline_tokens=10_000,
        )
        session.observe(ExecutionEvent(step=1, command="run", tokens_used=500, progress_made=True))
        session.observe(
            ExecutionEvent(
                step=2,
                command="test",
                tokens_used=400,
                verifier_run=True,
                verifier_passed=passed,
                progress_made=True,
            )
        )
        store.finish_session(session_id, passed=passed, quality_score=0.9 if passed else 0.4)


def test_aggregate_session_stats_counts_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTPROP_GLOBAL_STATE", str(tmp_path / "learned_state.json"))
    sess_dir = tmp_path / "sessions"
    _populate_sessions(sess_dir, [True, True, False])

    report = aggregate_session_stats(sess_dir)
    assert report.session_count == 3
    assert report.sessions_with_outcome == 3
    assert report.passed_count == 2
    assert report.pass_rate == pytest.approx(2 / 3)
    assert report.category_counts == {"coding": 3}
    assert report.total_decisions == sum(report.decision_counts.values())


def test_aggregate_ignores_state_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTPROP_GLOBAL_STATE", str(tmp_path / "learned_state.json"))
    sess_dir = tmp_path / "sessions"
    _populate_sessions(sess_dir, [True])
    # bandit.json and risk_state.json live alongside records but must not be counted.
    assert (sess_dir / "bandit.json").exists()
    report = aggregate_session_stats(sess_dir)
    assert report.session_count == 1


def test_aggregate_missing_dir_returns_empty(tmp_path: Path) -> None:
    report = aggregate_session_stats(tmp_path / "does-not-exist")
    assert report.session_count == 0
    assert report.to_dict()["pass_rate"] == 0.0


def test_render_markdown_includes_decision_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTPROP_GLOBAL_STATE", str(tmp_path / "learned_state.json"))
    sess_dir = tmp_path / "sessions"
    _populate_sessions(sess_dir, [True])
    markdown = render_session_stats_markdown(aggregate_session_stats(sess_dir), root=sess_dir)
    assert "# AgentProp Session Stats" in markdown
    assert "## Decision mix" in markdown


def test_cli_sessions_stats_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AGENTPROP_GLOBAL_STATE", str(tmp_path / "learned_state.json"))
    sess_dir = tmp_path / "sessions"
    _populate_sessions(sess_dir, [True, False])
    rc = main(["sessions", "stats", "--dir", str(sess_dir), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session_count"] == 2
    assert payload["passed_count"] == 1


def test_cli_sessions_stats_writes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTPROP_GLOBAL_STATE", str(tmp_path / "learned_state.json"))
    sess_dir = tmp_path / "sessions"
    _populate_sessions(sess_dir, [True])
    out = tmp_path / "stats.md"
    rc = main(["sessions", "stats", "--dir", str(sess_dir), "--out", str(out)])
    assert rc == 0
    assert "AgentProp Session Stats" in out.read_text()


def test_global_learned_state_persisted_and_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_state = tmp_path / "learned_state.json"
    monkeypatch.setenv("AGENTPROP_GLOBAL_STATE", str(global_state))

    _populate_sessions(tmp_path / "sessions_a", [True, False])
    assert global_state.exists()
    payload = json.loads(global_state.read_text())
    assert payload["version"] == 1
    assert "coding" in payload["bandit"]["stats"]

    # A fresh store on a brand-new directory inherits the global priors.
    fresh = SessionStore(tmp_path / "sessions_b")
    assert "coding" in fresh.bandit.stats
