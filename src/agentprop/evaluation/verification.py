"""Execution-backed verification for workflow task outputs."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

VerificationStatus = Literal["passed", "failed", "timeout", "error"]


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of running an external verification command."""

    command: str
    status: VerificationStatus
    passed: bool
    returncode: int | None
    duration_s: float
    cwd: str | None = None
    stdout: str = ""
    stderr: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command,
            "status": self.status,
            "passed": self.passed,
            "returncode": self.returncode,
            "duration_s": self.duration_s,
            "cwd": self.cwd,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "metadata": dict(self.metadata),
        }


def run_verification_command(
    command: str,
    *,
    cwd: str | Path | None = None,
    timeout_s: float = 120.0,
    env: Mapping[str, str] | None = None,
    output_limit: int = 20_000,
) -> VerificationResult:
    """Run a verification command and capture auditable pass/fail evidence."""

    if not command.strip():
        raise ValueError("verification command must not be empty")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if output_limit < 0:
        raise ValueError("output_limit must be non-negative")

    start = time.monotonic()
    resolved_cwd = str(Path(cwd).resolve()) if cwd is not None else None
    run_env = os.environ.copy()
    if env is not None:
        run_env.update(dict(env))

    try:
        completed = subprocess.run(
            command,
            cwd=resolved_cwd,
            env=run_env,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return VerificationResult(
            command=command,
            status="timeout",
            passed=False,
            returncode=None,
            duration_s=time.monotonic() - start,
            cwd=resolved_cwd,
            stdout=_coerce_output(exc.stdout, output_limit),
            stderr=_coerce_output(exc.stderr, output_limit),
            metadata={"timeout_s": timeout_s},
        )
    except OSError as exc:
        return VerificationResult(
            command=command,
            status="error",
            passed=False,
            returncode=None,
            duration_s=time.monotonic() - start,
            cwd=resolved_cwd,
            stderr=_truncate(str(exc), output_limit),
            metadata={"error_type": type(exc).__name__},
        )

    passed = completed.returncode == 0
    return VerificationResult(
        command=command,
        status="passed" if passed else "failed",
        passed=passed,
        returncode=completed.returncode,
        duration_s=time.monotonic() - start,
        cwd=resolved_cwd,
        stdout=_truncate(completed.stdout, output_limit),
        stderr=_truncate(completed.stderr, output_limit),
    )


def verification_row_fields(result: VerificationResult) -> dict[str, object]:
    """Return the row fields used by experiment result tables."""

    return {
        "verification_status": result.status,
        "verification_passed": result.passed,
        "verification_returncode": result.returncode,
        "verification_duration_s": result.duration_s,
    }


def _coerce_output(value: str | bytes | None, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return _truncate(value.decode(errors="replace"), limit)
    return _truncate(value, limit)


def _truncate(value: str, limit: int) -> str:
    if limit == 0:
        return ""
    if len(value) <= limit:
        return value
    suffix = f"\n...[truncated to {limit} characters]"
    return value[: max(0, limit - len(suffix))] + suffix
