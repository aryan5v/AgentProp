"""Hard watchdog runner for long-running external benchmark commands."""

from __future__ import annotations

import json
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WatchdogResult:
    """Final status for a subprocess guarded by wall-clock and idle timers."""

    command: tuple[str, ...]
    status: str
    exit_code: int | None
    started_at: float
    finished_at: float
    duration_s: float
    timed_out: bool
    idle_timed_out: bool
    log_path: str
    status_path: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "timed_out": self.timed_out,
            "idle_timed_out": self.idle_timed_out,
            "log_path": self.log_path,
            "status_path": self.status_path,
            "message": self.message,
        }


def run_command_with_watchdog(
    command: list[str] | tuple[str, ...],
    *,
    log_path: str | Path,
    status_path: str | Path | None = None,
    timeout_s: float,
    idle_timeout_s: float | None = None,
    poll_interval_s: float = 1.0,
    terminate_grace_s: float = 10.0,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> WatchdogResult:
    """Run a command while preserving logs and terminating hard hangs.

    The watchdog has two independent limits:
    wall-clock timeout terminates after total runtime, while idle timeout
    terminates if the process stops emitting output for too long.
    """

    if not command:
        raise ValueError("command must not be empty")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if idle_timeout_s is not None and idle_timeout_s <= 0:
        raise ValueError("idle_timeout_s must be positive")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    status_file = Path(status_path) if status_path is not None else None
    if status_file is not None:
        status_file.parent.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    process: subprocess.Popen[str] | None = None
    status = "failed-to-start"
    exit_code: int | None = None
    timed_out = False
    idle_timed_out = False
    message = ""

    with log_file.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {' '.join(command)}\n")
        log_handle.flush()
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd) if cwd is not None else None,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            message = str(exc)
        else:
            selector = selectors.DefaultSelector()
            assert process.stdout is not None
            selector.register(process.stdout, selectors.EVENT_READ)
            last_output_at = time.time()

            while process.poll() is None:
                now = time.time()
                if now - started_at >= timeout_s:
                    timed_out = True
                    status = "timeout"
                    message = f"wall-clock timeout after {timeout_s:.1f}s"
                    _terminate_process(process, terminate_grace_s)
                    break
                if idle_timeout_s is not None and now - last_output_at >= idle_timeout_s:
                    idle_timed_out = True
                    status = "idle-timeout"
                    message = f"idle timeout after {idle_timeout_s:.1f}s without output"
                    _terminate_process(process, terminate_grace_s)
                    break

                events = selector.select(timeout=min(poll_interval_s, 1.0))
                for _key, _ in events:
                    line = process.stdout.readline()
                    if line:
                        log_handle.write(line)
                        log_handle.flush()
                        last_output_at = time.time()

            remaining = process.stdout.read()
            if remaining:
                log_handle.write(remaining)
                log_handle.flush()
            exit_code = process.returncode
            if not timed_out and not idle_timed_out:
                status = "completed" if exit_code == 0 else "failed"
                message = f"process exited with code {exit_code}"

    finished_at = time.time()
    result = WatchdogResult(
        command=tuple(command),
        status=status,
        exit_code=exit_code,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=finished_at - started_at,
        timed_out=timed_out,
        idle_timed_out=idle_timed_out,
        log_path=str(log_file),
        status_path=str(status_file) if status_file is not None else None,
        message=message,
    )
    if status_file is not None:
        status_file.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    return result


def _terminate_process(process: subprocess.Popen[str], terminate_grace_s: float) -> None:
    process.terminate()
    try:
        process.wait(timeout=terminate_grace_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=terminate_grace_s)
