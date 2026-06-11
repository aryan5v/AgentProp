"""Cursor Agent integration helpers for AgentProp-controlled terminal loops."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from agentprop.integrations.cursor_usage import CursorUsageAccumulator, decode_cursor_agent_stdout
from agentprop.runtime import (
    TerminalCommandProposal,
    TerminalTurnRequest,
    execution_features_to_dict,
)


@dataclass(frozen=True, slots=True)
class CursorAgentConfig:
    """Configuration for using Cursor Agent as a command proposer."""

    binary: str = "cursor-agent"
    model: str | None = None
    workspace: str | Path | None = None
    api_key: str | None = None
    api_key_env: str = "CURSOR_API_KEY"
    timeout_s: float = 120.0
    max_process_retries: int = 1
    trust_workspace: bool = True
    output_format: Literal["text", "stream-json"] = "stream-json"
    extra_args: tuple[str, ...] = ()

    def command(self) -> list[str]:
        command = [
            self.binary,
            "--print",
            "--output-format",
            self.output_format,
            "--mode",
            "plan",
        ]
        if self.trust_workspace:
            command.append("--trust")
        if self.model:
            command.extend(["--model", self.model])
        if self.workspace is not None:
            command.extend(["--workspace", str(self.workspace)])
        if self.api_key:
            command.extend(["--api-key", self.api_key])
        command.extend(self.extra_args)
        return command


@dataclass(frozen=True, slots=True)
class CursorAgentProcessResult:
    """Process output from one Cursor Agent proposal call."""

    stdout: str
    stderr: str
    returncode: int


CursorAgentRunner = Callable[
    [Sequence[str], str, Mapping[str, str], float],
    CursorAgentProcessResult,
]


class CursorAgentError(RuntimeError):
    """Raised when Cursor Agent cannot produce a command proposal."""


@dataclass(slots=True)
class CursorCommandProposer:
    """Use Cursor Agent to propose one shell command for AgentProp to gate.

    Cursor runs in read-only plan mode. It proposes the next command as JSON;
    AgentProp's `ControlledTerminalLoop` remains responsible for execution,
    verification, strategy switching, and trace evidence.
    """

    config: CursorAgentConfig = field(default_factory=CursorAgentConfig)
    runner: CursorAgentRunner | None = None
    usage: CursorUsageAccumulator = field(default_factory=CursorUsageAccumulator)

    def __call__(self, request: TerminalTurnRequest) -> TerminalCommandProposal:
        errors: list[str] = []
        tokens_before = self.usage.total_tokens
        try:
            for output_format in ("stream-json", "text"):
                try:
                    return self._with_token_delta(
                        self._propose(
                            request,
                            replace(self.config, output_format=output_format),
                        ),
                        tokens_before=tokens_before,
                    )
                except CursorAgentError as exc:
                    errors.append(str(exc))
            try:
                return self._with_token_delta(
                    self._propose_recovery(request),
                    tokens_before=tokens_before,
                )
            except CursorAgentError as exc:
                errors.append(str(exc))
        except Exception as exc:  # noqa: BLE001 - proposer must never abort the harness
            errors.append(f"{type(exc).__name__}: {exc}")
        return self._with_token_delta(
            TerminalCommandProposal(
                command="true",
                metadata={
                    "source": "cursor-agent",
                    "mode": "plan",
                    "proposal_failed": True,
                    "proposal_errors": errors,
                },
            ),
            tokens_before=tokens_before,
        )

    def _with_token_delta(
        self,
        proposal: TerminalCommandProposal,
        *,
        tokens_before: int,
    ) -> TerminalCommandProposal:
        tokens_used = max(0, self.usage.total_tokens - tokens_before)
        if tokens_used <= 0:
            return proposal
        metadata = dict(proposal.metadata)
        metadata["tokens_used"] = tokens_used
        return TerminalCommandProposal(command=proposal.command, metadata=metadata)

    def _propose(
        self,
        request: TerminalTurnRequest,
        config: CursorAgentConfig,
    ) -> TerminalCommandProposal:
        command = config.command()
        prompt = render_cursor_command_prompt(request)
        env = _cursor_env(config)
        result = self._run_with_retries(command, prompt, env, config)
        if result.returncode != 0:
            raise CursorAgentError(
                f"Cursor Agent exited with {result.returncode}: {result.stderr.strip()}"
            )
        self.usage.note_proposal_call()
        proposal_text, _ = decode_cursor_agent_stdout(result.stdout, self.usage)
        parsed = parse_cursor_command_output(proposal_text)
        if parsed.command == proposal_text.strip() and proposal_text != result.stdout:
            parsed = parse_cursor_command_output(result.stdout)
        if not _is_plausible_shell_command(parsed.command):
            raise CursorAgentError(
                f"Cursor Agent proposed non-executable text: {parsed.command[:120]!r}"
            )
        return TerminalCommandProposal(
            command=parsed.command,
            metadata={
                "source": "cursor-agent",
                "mode": "plan",
                "output_format": config.output_format,
                "rationale": parsed.rationale,
                "raw_output": result.stdout[:2_000],
            },
        )

    def _propose_recovery(self, request: TerminalTurnRequest) -> TerminalCommandProposal:
        config = replace(self.config, output_format="text", timeout_s=self.config.timeout_s)
        prompt = (
            "Reply with exactly one JSON object and nothing else.\n"
            'Schema: {"command": "<one shell command>", "rationale": "<short reason>"}\n'
            f"Task step: {request.step}\n"
            f"Task: {request.task[:2_000]}"
        )
        result = self._run_with_retries(
            config.command(),
            prompt,
            _cursor_env(config),
            config,
        )
        if result.returncode != 0:
            raise CursorAgentError(
                f"Cursor Agent recovery exited with {result.returncode}: {result.stderr.strip()}"
            )
        self.usage.note_proposal_call()
        proposal_text, _ = decode_cursor_agent_stdout(result.stdout, self.usage)
        parsed = parse_cursor_command_output(proposal_text)
        if not _is_plausible_shell_command(parsed.command):
            raise CursorAgentError(
                f"Cursor Agent recovery proposed non-executable text: {parsed.command[:120]!r}"
            )
        return TerminalCommandProposal(
            command=parsed.command,
            metadata={
                "source": "cursor-agent",
                "mode": "plan",
                "output_format": "text",
                "recovery": True,
                "rationale": parsed.rationale,
                "raw_output": result.stdout[:2_000],
            },
        )

    def _run_with_retries(
        self,
        command: Sequence[str],
        prompt: str,
        env: Mapping[str, str],
        config: CursorAgentConfig,
    ) -> CursorAgentProcessResult:
        runner = self.runner or _run_cursor_agent
        attempts = max(1, config.max_process_retries + 1)
        last_result: CursorAgentProcessResult | None = None
        for attempt in range(1, attempts + 1):
            result = runner(command, prompt, env, config.timeout_s)
            if result.returncode == 0:
                return result
            last_result = result
            if not _is_retryable_cursor_process_failure(result):
                break
            prompt = _retry_prompt(prompt, result.stderr, attempt=attempt)
        assert last_result is not None
        return last_result


@dataclass(frozen=True, slots=True)
class ParsedCursorCommand:
    """Command extracted from Cursor Agent output."""

    command: str
    rationale: str | None = None


def render_cursor_command_prompt(request: TerminalTurnRequest) -> str:
    """Render a prompt that asks Cursor to propose, not execute, one command."""

    features = execution_features_to_dict(request.features)
    recent_events = [
        {
            "step": event.step,
            "command": event.command,
            "exit_code": event.exit_code,
            "verifier_run": event.verifier_run,
            "verifier_passed": event.verifier_passed,
            "progress_made": event.progress_made,
            "error_signature": event.error_signature,
        }
        for event in request.transcript[-8:]
    ]
    metadata = dict(request.metadata)
    verifier_feedback = metadata.pop("verifier_feedback", None)
    payload = {
        "task": request.task,
        "step": request.step,
        "strategy": request.strategy,
        "features": features,
        "recent_events": recent_events,
        "metadata": metadata,
    }
    recovery_note = ""
    profile = metadata.get("proposer_profile")
    if request.strategy.endswith("recovery_prompt") or profile == "recovery":
        recovery_note = (
            "Recovery mode: the last verifier failed. Propose one command that directly "
            "addresses the verifier failure below.\n\n"
        )
    feedback_block = ""
    if isinstance(verifier_feedback, str) and verifier_feedback.strip():
        feedback_block = (
            "Latest verifier failure output (fix this):\n"
            f"{verifier_feedback.strip()[:4_000]}\n\n"
        )
    return (
        "You are Cursor Agent acting as a command proposer inside AgentProp.\n"
        "Do not edit files, run tools, or execute shell commands yourself.\n"
        "Return exactly one JSON object with keys `command` and `rationale`.\n"
        "Do not wrap the JSON in markdown fences or add prose before or after it.\n"
        "The command must be a single shell command for the host harness to execute.\n"
        "Prefer verifier-oriented commands when progress is stale or failures repeat.\n\n"
        f"{recovery_note}"
        f"{feedback_block}"
        f"AgentProp state:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def parse_cursor_command_output(output: str) -> ParsedCursorCommand:
    """Extract `{command, rationale}` from Cursor text/JSON output."""

    for candidate in _proposal_text_candidates(output):
        for json_blob in _iter_json_object_strings(candidate):
            parsed = _try_parse_json_object(json_blob)
            if parsed is None:
                continue
            command = parsed.get("command")
            if isinstance(command, str) and _is_plausible_shell_command(command):
                rationale = parsed.get("rationale")
                return ParsedCursorCommand(
                    command=command.strip(),
                    rationale=rationale if isinstance(rationale, str) else None,
                )
        fallback = _first_command_like_line(candidate)
        if fallback and _is_plausible_shell_command(fallback):
            return ParsedCursorCommand(command=fallback, rationale="parsed from plain text output")
    raise CursorAgentError("Cursor Agent output did not contain a command")


def cursor_agent_available(binary: str = "cursor-agent") -> bool:
    """Return whether the Cursor Agent CLI is available on PATH."""

    return shutil.which(binary) is not None


def cursor_agent_env_status(
    *,
    binary: str = "cursor-agent",
    api_key_env: str = "CURSOR_API_KEY",
) -> dict[str, object]:
    """Return public-safe Cursor integration readiness."""

    return {
        "binary": binary,
        "available": cursor_agent_available(binary),
        "api_key_env": api_key_env,
        "api_key_present": bool(os.environ.get(api_key_env)),
        "auth_hint": (
            f"set {api_key_env} or run `cursor-agent login`; never commit API keys"
        ),
    }


def _proposal_text_candidates(output: str) -> list[str]:
    stripped = output.strip()
    if not stripped:
        return []
    candidates = [stripped]
    fence = _extract_markdown_json(stripped)
    if fence:
        candidates.append(fence)
    extracted = _extract_json_object(stripped)
    if extracted:
        candidates.append(extracted)
    candidates.extend(line.strip() for line in stripped.splitlines() if line.strip())
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _extract_markdown_json(value: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", value, re.DOTALL)
    return match.group(1).strip() if match else ""


_SHELL_PREFIXES = (
    "bash",
    "sh",
    "git",
    "pip",
    "python",
    "pytest",
    "make",
    "curl",
    "cd ",
    "mkdir",
    "rm ",
    "cp ",
    "mv ",
    "sed ",
    "grep",
    "rg ",
    "cat ",
    "echo ",
    "export ",
    "apt-get",
    "nginx",
    "systemctl",
    "service ",
    "true",
    ":",
)


def _is_plausible_shell_command(command: str) -> bool:
    stripped = command.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered.startswith(
        (
            "i ",
            "i'll ",
            "i will ",
            "we ",
            "we'll ",
            "we should ",
            "let's ",
            "the ",
            "this ",
            "first ",
            "next ",
            "to ",
            "run ",
            "execute ",
        )
    ):
        return False
    if lowered in {"json", "javascript", "text", "shell", "bash", "sh"}:
        return False
    if lowered.startswith("```"):
        return False
    if stripped.startswith("{") or stripped.startswith("["):
        return False
    if any(char in stripped for char in (";", "|", "&", "$", "/", "=", "\n", "'", '"')):
        return True
    if stripped.startswith(_SHELL_PREFIXES):
        return True
    if "<<" in stripped:
        return True
    root = stripped.split(maxsplit=1)[0]
    return root in {
        "ls",
        "find",
        "touch",
        "chmod",
        "chown",
        "tee",
        "wc",
        "head",
        "tail",
        "tar",
        "docker",
        "npm",
        "node",
        "cargo",
        "go",
        "rustc",
        "gcc",
        "g++",
        "clang",
        "ld",
    }


def _iter_json_object_strings(value: str) -> list[str]:
    blobs: list[str] = []
    index = 0
    while index < len(value):
        start = value.find("{", index)
        if start == -1:
            break
        depth = 0
        for offset, char in enumerate(value[start:], start=start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    blobs.append(value[start : offset + 1])
                    index = offset + 1
                    break
        else:
            break
    return blobs


def _run_cursor_agent(
    command: Sequence[str],
    prompt: str,
    env: Mapping[str, str],
    timeout_s: float,
) -> CursorAgentProcessResult:
    try:
        completed = subprocess.run(
            [*command, prompt],
            env=dict(env),
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _decode_subprocess_output(exc.stdout)
        stderr = _decode_subprocess_output(exc.stderr)
        stderr = (stderr + "\n" if stderr else "") + f"cursor-agent timed out after {timeout_s}s"
        return CursorAgentProcessResult(stdout=stdout, stderr=stderr, returncode=124)
    except OSError as exc:
        return CursorAgentProcessResult(stdout="", stderr=str(exc), returncode=127)
    return CursorAgentProcessResult(
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def _decode_subprocess_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _is_retryable_cursor_process_failure(result: CursorAgentProcessResult) -> bool:
    text = f"{result.stderr}\n{result.stdout}".lower()
    if result.returncode == 124:
        return True
    return any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "temporarily unavailable",
            "rate limit",
            "connection reset",
            "econnreset",
            "network",
            "api error",
        )
    )


def _retry_prompt(prompt: str, stderr: str, *, attempt: int) -> str:
    detail = stderr.strip().splitlines()[-1][:240] if stderr.strip() else "unknown error"
    return (
        "The previous Cursor proposal call failed before returning a usable command. "
        "Try again with one short, safe shell command only.\n"
        f"Attempt: {attempt + 1}\n"
        f"Previous error: {detail}\n\n"
        f"{prompt}"
    )


def _cursor_env(config: CursorAgentConfig) -> dict[str, str]:
    env = dict(os.environ)
    if config.api_key and config.api_key_env:
        env[config.api_key_env] = config.api_key
    return env


def _try_parse_json_object(value: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json_object(value: str) -> str:
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return value[start : end + 1]


def _first_command_like_line(value: str) -> str | None:
    for line in value.splitlines():
        candidate = line.strip().strip("`")
        if not candidate or candidate.startswith("{") or candidate.startswith("#"):
            continue
        if candidate.lower() in {"json", "javascript", "text", "shell"}:
            continue
        if candidate.startswith(("command:", "Command:")):
            candidate = candidate.split(":", 1)[1].strip()
        if candidate:
            return candidate
    return None
