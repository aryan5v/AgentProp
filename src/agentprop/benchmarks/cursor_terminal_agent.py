"""Cursor + AgentProp terminal runner for Harbor / Terminal-Bench tasks."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from agentprop.integrations import CursorAgentConfig, CursorCommandProposer
from agentprop.integrations.cursor_usage import CursorUsageAccumulator
from agentprop.rl import CategoryBanditRoutingPolicy
from agentprop.runtime import (
    ControlDecision,
    ControlledTerminalLoop,
    ExecutionEvent,
    ExecutionStateTracker,
    RuntimeRewardLogger,
    StoppingController,
    StoppingControllerConfig,
    TerminalCommandProposal,
    TerminalCommandResult,
    TerminalLoopConfig,
    TerminalLoopResult,
    TerminalTurnRequest,
)
from agentprop.runtime.terminal_loop import (
    TerminalCommandExecutor,
    TerminalCommandProposer,
    TerminalStrategySwitcher,
    TerminalVerifier,
)


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Configuration for one benchmark task run."""

    instruction: str
    model: str
    workspace: Path
    max_steps: int
    verifier_command: str | None
    trace_dir: Path
    task_id: str
    category: str
    token_budget: int | None
    wall_time_budget_s: float | None
    cursor_timeout_s: float
    command_timeout_s: float
    verifier_timeout_s: float
    use_system_python: bool
    fast_path: str
    fast_path_timeout_s: float


@dataclass(slots=True)
class _RunState:
    """Mutable per-run state shared by proposer, verifier, and strategy switching."""

    switch_count: int = 0
    proposer_profile: str = "normal"
    verifier_feedback: str = ""
    fast_path_verifier_passed: bool = False


def main(argv: list[str] | None = None) -> int:
    config = _parse_args(argv)
    config.trace_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.trace_dir / "agentprop_cursor_config.json", _config_to_dict(config))

    usage = CursorUsageAccumulator()
    run_state = _RunState()
    proposer = CursorCommandProposer(
        CursorAgentConfig(
            model=config.model,
            workspace=config.workspace,
            timeout_s=config.cursor_timeout_s,
        ),
        usage=usage,
    )
    max_steps_without_verifier = 6 if config.fast_path != "off" else 3
    controller = StoppingController(
        StoppingControllerConfig(
            max_steps_without_verifier=max_steps_without_verifier,
            max_steps_without_progress=5,
            repeated_error_threshold=2,
            token_budget=config.token_budget,
            wall_time_budget_s=config.wall_time_budget_s,
            require_independent_verification=True,
        )
    )
    reward_logger = RuntimeRewardLogger(
        CategoryBanditRoutingPolicy(
            arms=("agentprop_cursor",),
            epsilon=0.0,
            default_arm="agentprop_cursor",
        )
    )
    loop = ControlledTerminalLoop(
        controller=controller,
        config=TerminalLoopConfig(
            max_steps=config.max_steps,
            task_id=config.task_id,
            category=config.category,
            explore=False,
        ),
        reward_logger=reward_logger,
    )
    exit_code = 0
    crash_info: dict[str, Any] | None = None
    result: TerminalLoopResult | None = None
    initial_events: tuple[ExecutionEvent, ...] = ()
    initial_stdout = ""
    initial_stderr = ""
    run_metadata: dict[str, object] = {
        "runner": "agentprop-cursor",
        "model": config.model,
        "workspace": str(config.workspace),
        "fast_path": config.fast_path,
        "use_system_python": config.use_system_python,
    }
    try:
        if config.fast_path == "yolo-until-verifier-miss":
            fast_result, verifier_result = _run_fast_path(config, usage, run_state)
            initial_events = (fast_result.event,)
            initial_stdout = fast_result.stdout
            initial_stderr = fast_result.stderr
            if verifier_result is not None:
                initial_events = (*initial_events, verifier_result.event)
                initial_stdout += verifier_result.stdout
                initial_stderr += verifier_result.stderr
                if verifier_result.event.verifier_passed:
                    run_state.fast_path_verifier_passed = True
        if run_state.fast_path_verifier_passed:
            result = _loop_result_from_fast_path(
                events=initial_events,
                stdout=initial_stdout,
                stderr=initial_stderr,
            )
        else:
            loop.on_strategy_switch = _on_strategy_switch(config, usage, run_state)
            result = loop.run(
                task=config.instruction,
                proposer=_proposer_with_state(proposer, run_state, run_metadata),
                executor=_executor(config),
                verifier=_verifier(config, run_state),
                strategy_switcher=_strategy_switcher_factory(run_state),
                initial_events=initial_events,
                metadata=run_metadata,
            )
            if initial_stdout or initial_stderr:
                result = _prepend_output(result, stdout=initial_stdout, stderr=initial_stderr)
        _write_result(config.trace_dir, result)
    except Exception as exc:  # noqa: BLE001 - Harbor should still grade container state
        crash_info = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "steps": len(result.events) if result is not None else len(initial_events),
        }
        _write_json(config.trace_dir / "agentprop_cursor_crash.json", crash_info)
        if not _bool_env("AGENTPROP_HARBOR_SCORE_ONLY"):
            exit_code = 1
    finally:
        _write_exit_artifact(
            config.trace_dir,
            exit_code=exit_code,
            crash_info=crash_info,
            result=result,
            config=config,
        )
        _write_harbor_usage(usage, model=config.model)
    if _bool_env("AGENTPROP_HARBOR_SCORE_ONLY"):
        return 0
    return exit_code


def _proposer_with_state(
    proposer: CursorCommandProposer,
    run_state: _RunState,
    run_metadata: dict[str, object],
) -> TerminalCommandProposer:
    def propose(request: TerminalTurnRequest) -> TerminalCommandProposal:
        metadata = dict(run_metadata)
        metadata["proposer_profile"] = run_state.proposer_profile
        if run_state.verifier_feedback:
            metadata["verifier_feedback"] = run_state.verifier_feedback
        enriched = replace(request, metadata=metadata)
        return proposer(enriched)

    return propose


def _executor(
    config: RunnerConfig,
) -> TerminalCommandExecutor:
    def execute(
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
    ) -> TerminalCommandResult:
        tokens_used = int(proposal.metadata.get("tokens_used") or 0)
        if proposal.metadata.get("proposal_failed"):
            return TerminalCommandResult(
                event=ExecutionEvent(
                    step=request.step,
                    command=proposal.command,
                    exit_code=0,
                    progress_made=False,
                    tokens_used=tokens_used,
                    error_signature="cursor_proposal_failed",
                ),
                stdout="",
                stderr="\n".join(
                    str(error) for error in proposal.metadata.get("proposal_errors", ())
                ),
                metadata={"source": "cursor-proposal-fallback"},
            )
        workspace_before = _workspace_snapshot(config.workspace)
        completed = _run_shell(
            proposal.command,
            cwd=config.workspace,
            timeout_s=config.command_timeout_s,
            env=_task_env(config),
        )
        progress_made = completed.returncode == 0 or _workspace_changed(
            config.workspace,
            workspace_before,
        )
        return TerminalCommandResult(
            event=ExecutionEvent(
                step=request.step,
                command=proposal.command,
                exit_code=completed.returncode,
                progress_made=progress_made,
                tokens_used=tokens_used,
                elapsed_s=completed.elapsed_s,
                error_signature=_error_signature(completed),
            ),
            stdout=completed.stdout,
            stderr=completed.stderr,
            metadata={"source": "cursor-proposed-command"},
        )

    return execute


def _verifier(
    config: RunnerConfig,
    run_state: _RunState,
) -> TerminalVerifier | None:
    command = config.verifier_command or _default_verifier_command(config.workspace)
    if command is None:
        return None

    def verify(
        request: TerminalTurnRequest,
        blocked_proposal: TerminalCommandProposal | None = None,
    ) -> TerminalCommandResult:
        completed = _run_shell(
            command,
            cwd=config.workspace,
            timeout_s=config.verifier_timeout_s,
            env=_task_env(config),
        )
        if completed.returncode != 0:
            feedback = (completed.stderr or completed.stdout).strip()
            if feedback:
                run_state.verifier_feedback = feedback[:8_000]
        else:
            run_state.verifier_feedback = ""
        return TerminalCommandResult(
            event=ExecutionEvent(
                step=request.step,
                command=command,
                exit_code=completed.returncode,
                verifier_run=True,
                verifier_passed=completed.returncode == 0,
                trusted=True,
                elapsed_s=completed.elapsed_s,
                error_signature=None if completed.returncode == 0 else _error_signature(completed),
            ),
            stdout=completed.stdout,
            stderr=completed.stderr,
            metadata={"blocked_command": blocked_proposal.command if blocked_proposal else None},
        )

    return verify


def _strategy_switcher_factory(run_state: _RunState) -> TerminalStrategySwitcher:
    def switch(
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
        decision: object,
    ) -> str:
        del request, proposal, decision
        run_state.switch_count += 1
        if run_state.switch_count == 1:
            run_state.proposer_profile = "recovery"
            return "cursor_recovery_prompt"
        if run_state.switch_count == 2:
            run_state.proposer_profile = "normal"
            return "cursor_yolo_repair"
        run_state.proposer_profile = "tight_verify"
        return "cursor_tight_verify"

    return switch


def _on_strategy_switch(
    config: RunnerConfig,
    usage: CursorUsageAccumulator,
    run_state: _RunState,
):
    def handler(
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
        decision: object,
    ) -> tuple[TerminalCommandResult, ...]:
        del proposal, decision
        if run_state.switch_count != 2:
            return ()
        repair_result, verifier_result = _run_yolo_repair(config, usage, request.task)
        if verifier_result is None:
            return (repair_result,)
        return (repair_result, verifier_result)

    return handler


@dataclass(frozen=True, slots=True)
class ShellResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


def _run_shell(
    command: str,
    *,
    cwd: Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> ShellResult:
    import time

    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=env,
        )
        return ShellResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            elapsed_s=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_output(exc.stdout)
        stderr = _decode_output(exc.stderr)
        stderr = (stderr + "\n" if stderr else "") + f"command timed out after {timeout_s}s"
        return ShellResult(
            returncode=124,
            stdout=stdout,
            stderr=stderr,
            elapsed_s=time.monotonic() - start,
        )


def _run_yolo_repair(
    config: RunnerConfig,
    usage: CursorUsageAccumulator,
    task: str,
) -> tuple[TerminalCommandResult, TerminalCommandResult | None]:
    repair_config = replace(config, instruction=task)
    command = _cursor_yolo_command(repair_config)
    completed = _run_shell(
        command,
        cwd=config.workspace,
        timeout_s=min(config.fast_path_timeout_s, 600.0),
        env=_task_env(config),
    )
    from agentprop.integrations.cursor_usage import decode_cursor_agent_stdout

    decode_cursor_agent_stdout(completed.stdout, usage)
    repair_result = TerminalCommandResult(
        event=ExecutionEvent(
            step=0,
            command="cursor-agent --yolo (repair)",
            exit_code=completed.returncode,
            progress_made=completed.returncode == 0,
            elapsed_s=completed.elapsed_s,
            error_signature=_error_signature(completed),
        ),
        stdout=completed.stdout,
        stderr=completed.stderr,
        metadata={"source": "cursor-yolo-repair"},
    )
    verifier = _verifier(config, _RunState())
    if verifier is None:
        return repair_result, None
    request = TerminalTurnRequest(
        task=task,
        step=0,
        strategy="cursor_yolo_repair",
        features=_empty_features(),
        transcript=(repair_result.event,),
        metadata={"repair": True},
    )
    return repair_result, verifier(request, None)


def _loop_result_from_fast_path(
    *,
    events: tuple[ExecutionEvent, ...],
    stdout: str,
    stderr: str,
) -> TerminalLoopResult:
    """Skip the per-command control loop when yolo already passed the verifier."""

    tracker = ExecutionStateTracker()
    for event in events:
        tracker.observe(event)
    features = tracker.features()
    return TerminalLoopResult(
        strategy="cursor_yolo_fast_path",
        decisions=(ControlDecision("FINALIZE", "yolo fast path verifier passed"),),
        proposals=(),
        events=events,
        stdout=stdout,
        stderr=stderr,
        passed=features.evaluator_passed,
        features=features,
    )


def _run_fast_path(
    config: RunnerConfig,
    usage: CursorUsageAccumulator,
    run_state: _RunState,
) -> tuple[TerminalCommandResult, TerminalCommandResult | None]:
    command = _cursor_yolo_command(config)
    completed = _run_shell(
        command,
        cwd=config.workspace,
        timeout_s=config.fast_path_timeout_s,
        env=_task_env(config),
    )
    from agentprop.integrations.cursor_usage import decode_cursor_agent_stdout

    decode_cursor_agent_stdout(completed.stdout, usage)
    fast_result = TerminalCommandResult(
        event=ExecutionEvent(
            step=0,
            command="cursor-agent --yolo",
            exit_code=completed.returncode,
            progress_made=completed.returncode == 0,
            elapsed_s=completed.elapsed_s,
            error_signature=_error_signature(completed),
        ),
        stdout=completed.stdout,
        stderr=completed.stderr,
        metadata={"source": "cursor-yolo-fast-path"},
    )
    verifier = _verifier(config, run_state)
    if verifier is None:
        return fast_result, None
    request = TerminalTurnRequest(
        task=config.instruction,
        step=0,
        strategy="cursor_yolo_fast_path",
        features=_empty_features(),
        transcript=(fast_result.event,),
        metadata={"fast_path": config.fast_path},
    )
    verifier_result = verifier(request, None)
    return fast_result, verifier_result


def _cursor_yolo_command(config: RunnerConfig) -> str:
    workspace = config.workspace.resolve()
    prompt = (
        f"Complete the benchmark task in this workspace: {workspace}\n"
        "You may edit files and run commands there. "
        "Stop once the task is ready for the external verifier.\n\n"
        f"{config.instruction}"
    )
    parts = [
        "cursor-agent",
        "--yolo",
        "--print",
        "--output-format",
        "stream-json",
        "--trust",
        "--model",
        config.model,
        "--",
        prompt,
    ]
    return " ".join(shlex.quote(part) for part in parts)


def _empty_features() -> ExecutionStateFeatures:
    return ExecutionStateTracker().features()


def _task_env(config: RunnerConfig) -> dict[str, str]:
    env = dict(os.environ)
    if not config.use_system_python:
        return env
    path_parts = [
        part
        for part in env.get("PATH", "").split(os.pathsep)
        if part and part != "/opt/agentprop-venv/bin"
    ]
    env["PATH"] = os.pathsep.join(path_parts)
    env["VIRTUAL_ENV"] = ""
    return env


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _prepend_output(result: TerminalLoopResult, *, stdout: str, stderr: str) -> TerminalLoopResult:
    return replace(
        result,
        stdout=stdout + result.stdout,
        stderr=stderr + result.stderr,
    )


def _default_verifier_command(workspace: Path) -> str | None:
    env_command = os.environ.get("AGENTPROP_VERIFIER_COMMAND")
    if env_command:
        return env_command
    app_eval = Path("/app/eval.py")
    if app_eval.exists():
        return f"python3 {app_eval}"
    if (workspace / "eval.py").exists():
        return "python3 eval.py"
    if (workspace / "pytest.ini").exists() or (workspace / "tests").exists():
        return "pytest -q"
    return None


def _error_signature(result: ShellResult) -> str | None:
    if result.returncode == 0:
        return None
    text = (result.stderr or result.stdout).strip().splitlines()
    first = text[-1] if text else f"exit:{result.returncode}"
    return first[:200]


def _write_result(trace_dir: Path, result: object) -> None:
    assert isinstance(result, TerminalLoopResult)
    _write_json(
        trace_dir / "agentprop_cursor_result.json",
        {
            "passed": result.passed,
            "strategy": result.strategy,
            "features": {
                "step_count": result.features.step_count,
                "total_tokens": result.features.total_tokens,
                "elapsed_s": result.features.elapsed_s,
                "repeated_error_count": result.features.repeated_error_count,
                "verifier_failed_count": result.features.verifier_failed_count,
            },
            "decisions": [
                {
                    "action": decision.action,
                    "reason": decision.reason,
                    "strategy": decision.strategy,
                    "defer_command": decision.defer_command,
                }
                for decision in result.decisions
            ],
            "events": [
                {
                    "step": event.step,
                    "command": event.command,
                    "exit_code": event.exit_code,
                    "verifier_run": event.verifier_run,
                    "verifier_passed": event.verifier_passed,
                    "progress_made": event.progress_made,
                    "elapsed_s": event.elapsed_s,
                    "error_signature": event.error_signature,
                    "trusted": event.trusted,
                }
                for event in result.events
            ],
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _workspace_snapshot(workspace: Path) -> dict[Path, float]:
    snapshot: dict[Path, float] = {}
    if not workspace.exists():
        return snapshot
    for path in workspace.rglob("*"):
        if path.is_file():
            try:
                snapshot[path.relative_to(workspace)] = path.stat().st_mtime
            except OSError:
                continue
    return snapshot


def _workspace_changed(workspace: Path, before: dict[Path, float]) -> bool:
    after = _workspace_snapshot(workspace)
    if after.keys() != before.keys():
        return True
    return any(after[path] != before[path] for path in after)


def _write_exit_artifact(
    trace_dir: Path,
    *,
    exit_code: int,
    crash_info: dict[str, Any] | None,
    result: TerminalLoopResult | None,
    config: RunnerConfig,
) -> None:
    try:
        _write_json(
            trace_dir / "agentprop_cursor_exit.json",
            {
                "exit_code": exit_code,
                "exception_type": (crash_info or {}).get("exception_type"),
                "steps": len(result.events) if result is not None else 0,
                "passed": result.passed if result is not None else None,
                "fast_path": config.fast_path,
                "use_system_python": config.use_system_python,
            },
        )
    except OSError:
        return


def _write_harbor_usage(usage: CursorUsageAccumulator, *, model: str) -> None:
    logs_dir = Path(os.environ.get("AGENTPROP_HARBOR_LOGS_DIR", "/logs/agent"))
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            logs_dir / "agentprop-cursor-usage.json",
            usage.to_harbor_payload(model=model),
        )
    except OSError:
        return


def _config_to_dict(config: RunnerConfig) -> dict[str, object]:
    return {
        "model": config.model,
        "workspace": str(config.workspace),
        "max_steps": config.max_steps,
        "verifier_command": config.verifier_command,
        "trace_dir": str(config.trace_dir),
        "task_id": config.task_id,
        "category": config.category,
        "token_budget": config.token_budget,
        "wall_time_budget_s": config.wall_time_budget_s,
        "cursor_timeout_s": config.cursor_timeout_s,
        "command_timeout_s": config.command_timeout_s,
        "verifier_timeout_s": config.verifier_timeout_s,
        "use_system_python": config.use_system_python,
        "fast_path": config.fast_path,
        "fast_path_timeout_s": config.fast_path_timeout_s,
    }


def _parse_args(argv: list[str] | None) -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Run Cursor under AgentProp terminal control.")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model", default=os.environ.get("CURSOR_MODEL", "gpt-5.5"))
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("AGENTPROP_MAX_STEPS", "64")),
    )
    parser.add_argument("--verifier-command", default=os.environ.get("AGENTPROP_VERIFIER_COMMAND"))
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(os.environ.get("AGENTPROP_TRACE_DIR", ".agentprop/cursor-terminal-bench")),
    )
    parser.add_argument(
        "--task-id",
        default=os.environ.get("AGENTPROP_TASK_ID", "terminal-bench-task"),
    )
    parser.add_argument(
        "--category",
        default=os.environ.get("AGENTPROP_TASK_CATEGORY", "terminal-bench"),
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=_optional_int_env("AGENTPROP_TOKEN_BUDGET"),
    )
    parser.add_argument(
        "--wall-time-budget-s",
        type=float,
        default=_optional_float_env("AGENTPROP_WALL_TIME_BUDGET_S"),
    )
    parser.add_argument("--cursor-timeout-s", type=float, default=180.0)
    parser.add_argument("--command-timeout-s", type=float, default=300.0)
    parser.add_argument("--verifier-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--use-system-python",
        action="store_true",
        default=_optional_bool_env("AGENTPROP_USE_SYSTEM_PYTHON"),
        help="Run task commands/verifiers with /opt/agentprop-venv removed from PATH.",
    )
    parser.add_argument(
        "--fast-path",
        choices=("off", "yolo-until-verifier-miss"),
        default=os.environ.get("AGENTPROP_FAST_PATH", "off"),
        help="Optional Cursor yolo pass before falling back to per-command plan mode.",
    )
    parser.add_argument(
        "--fast-path-timeout-s",
        type=float,
        default=float(os.environ.get("AGENTPROP_FAST_PATH_TIMEOUT_S", "900")),
    )
    args = parser.parse_args(argv)
    return RunnerConfig(
        instruction=args.instruction,
        model=args.model,
        workspace=args.workspace,
        max_steps=args.max_steps,
        verifier_command=args.verifier_command,
        trace_dir=args.trace_dir,
        task_id=args.task_id,
        category=args.category,
        token_budget=args.token_budget,
        wall_time_budget_s=args.wall_time_budget_s,
        cursor_timeout_s=args.cursor_timeout_s,
        command_timeout_s=args.command_timeout_s,
        verifier_timeout_s=args.verifier_timeout_s,
        use_system_python=args.use_system_python,
        fast_path=args.fast_path,
        fast_path_timeout_s=args.fast_path_timeout_s,
    )


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


def _optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    return float(value) if value else None


def _optional_bool_env(name: str) -> bool:
    return _bool_env(name)


def _bool_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
