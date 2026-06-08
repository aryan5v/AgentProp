import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agentprop.benchmarks.cursor_terminal_agent import (
    _default_verifier_command,
    _loop_result_from_fast_path,
    _parse_args,
    _write_harbor_usage,
    main,
)
from agentprop.integrations.cursor_usage import CursorUsageAccumulator
from agentprop.runtime import ExecutionEvent, TerminalLoopResult
from agentprop.runtime.control_loop import ExecutionStateFeatures


def test_cursor_terminal_runner_parses_minimal_config(tmp_path: Path) -> None:
    config = _parse_args(
        [
            "--instruction",
            "Fix the task",
            "--workspace",
            str(tmp_path),
            "--max-steps",
            "3",
            "--model",
            "gpt-5.5",
        ]
    )

    assert config.instruction == "Fix the task"
    assert config.workspace == tmp_path
    assert config.max_steps == 3
    assert config.model == "gpt-5.5"
    assert config.use_system_python is False
    assert config.fast_path == "off"


def test_cursor_terminal_runner_parses_system_python_and_fast_path(tmp_path: Path) -> None:
    config = _parse_args(
        [
            "--instruction",
            "Fix the task",
            "--workspace",
            str(tmp_path),
            "--use-system-python",
            "--fast-path",
            "yolo-until-verifier-miss",
            "--fast-path-timeout-s",
            "42",
        ]
    )

    assert config.use_system_python is True
    assert config.fast_path == "yolo-until-verifier-miss"
    assert config.fast_path_timeout_s == 42


def test_cursor_terminal_runner_detects_local_eval(tmp_path: Path) -> None:
    (tmp_path / "eval.py").write_text("print('ok')\n", encoding="utf-8")

    assert _default_verifier_command(tmp_path).endswith(" eval.py")


def test_write_harbor_usage_writes_agent_log_file(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("AGENTPROP_HARBOR_LOGS_DIR", str(tmp_path))
    usage = CursorUsageAccumulator()
    usage.input_tokens = 42
    usage.output_tokens = 7

    _write_harbor_usage(usage, model="composer-2.5")

    payload = json.loads((tmp_path / "agentprop-cursor-usage.json").read_text(encoding="utf-8"))
    assert payload["n_input_tokens"] == 42
    assert payload["n_output_tokens"] == 7


def test_agentprop_cursor_harbor_agent_imports_without_harbor() -> None:
    from agentprop.benchmarks.harbor_agent import AgentPropCursorAgent

    assert AgentPropCursorAgent.name() == "agentprop-cursor"
    assert AgentPropCursorAgent().version() == "0.1"


def _loop_result(*, passed: bool | None) -> TerminalLoopResult:
    features = ExecutionStateFeatures(
        step_count=1,
        total_tokens=0,
        elapsed_s=0.0,
        steps_since_verifier=0,
        steps_since_progress=0,
        repeated_error_count=0,
        verifier_failed_count=0,
        last_exit_code=0,
        evaluator_passed=passed is True,
        final_answer_written=False,
    )
    return TerminalLoopResult(
        strategy="agentprop_cursor",
        decisions=(),
        proposals=(),
        events=(),
        stdout="",
        stderr="",
        passed=passed,
        features=features,
    )


def test_main_returns_zero_when_loop_completes_unpassed(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("AGENTPROP_HARBOR_SCORE_ONLY", raising=False)
    with patch(
        "agentprop.benchmarks.cursor_terminal_agent.ControlledTerminalLoop.run",
        return_value=_loop_result(passed=False),
    ):
        code = main(
            [
                "--instruction",
                "task",
                "--workspace",
                str(tmp_path),
                "--trace-dir",
                str(tmp_path / "trace"),
            ]
        )
    assert code == 0
    assert (tmp_path / "trace" / "agentprop_cursor_exit.json").exists()


def test_main_returns_zero_on_crash_when_score_only(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("AGENTPROP_HARBOR_SCORE_ONLY", "1")
    with patch(
        "agentprop.benchmarks.cursor_terminal_agent.ControlledTerminalLoop.run",
        side_effect=RuntimeError("boom"),
    ):
        code = main(
            [
                "--instruction",
                "task",
                "--workspace",
                str(tmp_path),
                "--trace-dir",
                str(tmp_path / "trace"),
            ]
        )
    assert code == 0
    crash = json.loads((tmp_path / "trace" / "agentprop_cursor_crash.json").read_text(encoding="utf-8"))
    assert crash["exception_type"] == "RuntimeError"


def test_loop_result_from_fast_path_skips_control_when_verifier_passed() -> None:
    events = (
        ExecutionEvent(
            step=0,
            command="cursor-agent --yolo",
            exit_code=0,
            progress_made=True,
        ),
        ExecutionEvent(
            step=0,
            command="pytest -q",
            exit_code=0,
            verifier_run=True,
            verifier_passed=True,
            trusted=True,
        ),
    )
    result = _loop_result_from_fast_path(events=events, stdout="yolo", stderr="")
    assert result.passed is True
    assert result.proposals == ()
    assert result.decisions[0].action == "FINALIZE"
    assert result.strategy == "cursor_yolo_fast_path"


def test_main_returns_one_on_crash_without_score_only(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("AGENTPROP_HARBOR_SCORE_ONLY", raising=False)
    with patch(
        "agentprop.benchmarks.cursor_terminal_agent.ControlledTerminalLoop.run",
        side_effect=RuntimeError("boom"),
    ):
        code = main(
            [
                "--instruction",
                "task",
                "--workspace",
                str(tmp_path),
                "--trace-dir",
                str(tmp_path / "trace"),
            ]
        )
    assert code == 1
