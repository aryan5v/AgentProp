"""Gemini + AgentProp terminal runner for Harbor / Terminal-Bench tasks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agentprop.integrations import GeminiAgentConfig, GeminiCommandProposer
from agentprop.rl import CategoryBanditRoutingPolicy
from agentprop.runtime import (
    ControlledTerminalLoop,
    ExecutionEvent,
    RuntimeRewardLogger,
    StoppingController,
    StoppingControllerConfig,
    TerminalCommandProposal,
    TerminalCommandResult,
    TerminalLoopConfig,
    TerminalTurnRequest,
)
from agentprop.runtime.terminal_loop import TerminalCommandExecutor, TerminalVerifier


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Configuration for one Gemini-controlled benchmark task run."""

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
    gemini_timeout_s: float
    command_timeout_s: float
    verifier_timeout_s: float


def main(argv: list[str] | None = None) -> int:
    config = _parse_args(argv)
    config.trace_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.trace_dir / "agentprop_gemini_config.json", _config_to_dict(config))

    proposer = GeminiCommandProposer(
        GeminiAgentConfig(
            model=config.model,
            workspace=config.workspace,
            timeout_s=config.gemini_timeout_s,
        )
    )
    controller = StoppingController(
        StoppingControllerConfig(
            max_steps_without_verifier=3,
            max_steps_without_progress=5,
            repeated_error_threshold=2,
            token_budget=config.token_budget,
            wall_time_budget_s=config.wall_time_budget_s,
            require_independent_verification=True,
        )
    )
    reward_logger = RuntimeRewardLogger(
        CategoryBanditRoutingPolicy(
            arms=("agentprop_gemini",),
            epsilon=0.0,
            default_arm="agentprop_gemini",
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
    result = loop.run(
        task=config.instruction,
        proposer=proposer,
        executor=_executor(config),
        verifier=_verifier(config),
        strategy_switcher=_strategy_switcher,
        metadata={
            "runner": "agentprop-gemini",
            "model": config.model,
            "workspace": str(config.workspace),
        },
    )
    _write_result(config.trace_dir, result)
    if _bool_env("AGENTPROP_HARBOR_SCORE_ONLY", default=False):
        return 0
    return 0 if result.passed is True else 1


def _executor(config: RunnerConfig) -> TerminalCommandExecutor:
    def execute(
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
    ) -> TerminalCommandResult:
        if proposal.command.strip() == "agentprop:finalize":
            return TerminalCommandResult(
                event=ExecutionEvent(
                    step=request.step,
                    command="agentprop:finalize",
                    progress_made=True,
                    final_answer_written=True,
                    tokens_used=int(proposal.metadata.get("estimated_tokens") or 0),
                ),
                stdout="",
                stderr="",
                metadata={"source": "agentprop-finalize-sentinel"},
            )
        completed = _run_shell(
            proposal.command,
            cwd=config.workspace,
            timeout_s=config.command_timeout_s,
        )
        return TerminalCommandResult(
            event=ExecutionEvent(
                step=request.step,
                command=proposal.command,
                exit_code=completed.returncode,
                progress_made=completed.returncode == 0,
                tokens_used=int(proposal.metadata.get("estimated_tokens") or 0),
                elapsed_s=completed.elapsed_s,
                error_signature=_error_signature(completed),
            ),
            stdout=completed.stdout,
            stderr=completed.stderr,
            metadata={"source": "gemini-proposed-command"},
        )

    return execute


def _verifier(config: RunnerConfig) -> TerminalVerifier | None:
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
        )
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


def _strategy_switcher(
    request: TerminalTurnRequest,
    proposal: TerminalCommandProposal,
    decision: object,
) -> str:
    del request, proposal, decision
    return "gemini_repair_after_agentprop_switch"


@dataclass(frozen=True, slots=True)
class ShellResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


def _run_shell(command: str, *, cwd: Path, timeout_s: float) -> ShellResult:
    import time

    start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    return ShellResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_s=time.monotonic() - start,
    )


def _default_verifier_command(workspace: Path) -> str | None:
    env_command = os.environ.get("AGENTPROP_VERIFIER_COMMAND")
    if env_command:
        return env_command
    app_eval = Path("/app/eval.py")
    if app_eval.exists():
        return f"{shutil.which('python3') or 'python3'} {app_eval}"
    if (workspace / "eval.py").exists():
        return f"{shutil.which('python3') or 'python3'} eval.py"
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
    from agentprop.runtime.terminal_loop import TerminalLoopResult

    assert isinstance(result, TerminalLoopResult)
    _write_json(
        trace_dir / "agentprop_gemini_result.json",
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
                    "tokens_used": event.tokens_used,
                    "elapsed_s": event.elapsed_s,
                    "error_signature": event.error_signature,
                }
                for event in result.events
            ],
            "reward_row": dict(result.reward_row) if result.reward_row else None,
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_to_dict(config: RunnerConfig) -> dict[str, object]:
    return {
        "instruction": config.instruction,
        "model": config.model,
        "workspace": str(config.workspace),
        "max_steps": config.max_steps,
        "verifier_command": config.verifier_command,
        "trace_dir": str(config.trace_dir),
        "task_id": config.task_id,
        "category": config.category,
        "token_budget": config.token_budget,
        "wall_time_budget_s": config.wall_time_budget_s,
        "gemini_timeout_s": config.gemini_timeout_s,
        "command_timeout_s": config.command_timeout_s,
        "verifier_timeout_s": config.verifier_timeout_s,
    }


def _parse_args(argv: list[str] | None) -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Run Gemini under AgentProp terminal control.")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview"))
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("AGENTPROP_MAX_STEPS", "64")))
    parser.add_argument("--verifier-command", default=os.environ.get("AGENTPROP_VERIFIER_COMMAND"))
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(os.environ.get("AGENTPROP_TRACE_DIR", ".agentprop/gemini-terminal-bench")),
    )
    parser.add_argument("--task-id", default=os.environ.get("AGENTPROP_TASK_ID", "terminal-bench-task"))
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
    parser.add_argument("--gemini-timeout-s", type=float, default=180.0)
    parser.add_argument("--command-timeout-s", type=float, default=300.0)
    parser.add_argument("--verifier-timeout-s", type=float, default=600.0)
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
        gemini_timeout_s=args.gemini_timeout_s,
        command_timeout_s=args.command_timeout_s,
        verifier_timeout_s=args.verifier_timeout_s,
    )


def _optional_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw else None


def _optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    return float(raw) if raw else None


def _bool_env(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
