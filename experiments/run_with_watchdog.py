"""Run a long external command with hard wall-clock and idle watchdogs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.evaluation.watchdog import run_command_with_watchdog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a command with a hard watchdog.")
    parser.add_argument(
        "--timeout",
        type=float,
        required=True,
        help="wall-clock timeout in seconds",
    )
    parser.add_argument("--idle-timeout", type=float, default=None, help="idle timeout in seconds")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    result = run_command_with_watchdog(
        command,
        log_path=args.log,
        status_path=args.status_json,
        timeout_s=args.timeout,
        idle_timeout_s=args.idle_timeout,
        poll_interval_s=args.poll_interval,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if result.status == "completed":
        return 0
    return int(result.exit_code) if result.exit_code is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
