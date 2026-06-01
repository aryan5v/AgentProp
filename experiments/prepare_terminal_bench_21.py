"""Prepare a Terminal-Bench 2.1 + Terminus-2 launch bundle without running it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.evaluation.terminal_bench import (
    HarborWatchdogConfig,
    TerminalBenchLaunchConfig,
    write_terminal_bench_launch_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Terminal-Bench 2.1 + Terminus-2 run artifacts."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark-results/terminal-bench-2.1"),
    )
    parser.add_argument("--dataset", default="terminal-bench/terminal-bench-2-1")
    parser.add_argument("--agent", default="terminus-2")
    parser.add_argument("--model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--environment", default="modal")
    parser.add_argument("--run-name", default="agentprop-tbench-21-terminus2")
    parser.add_argument("--task-count", type=int, default=None)
    parser.add_argument(
        "--output-root",
        default="benchmark-results/terminal-bench-2.1/terminus-2-agentprop",
    )
    parser.add_argument("--timeout", type=int, default=21_600)
    parser.add_argument("--idle-timeout", type=int, default=1_800)
    parser.add_argument("--registry-root", type=Path, default=None)
    args = parser.parse_args(argv)

    config = TerminalBenchLaunchConfig(
        dataset=args.dataset,
        agent=args.agent,
        model=args.model,
        environment=args.environment,
        run_name=args.run_name,
        task_count=args.task_count,
        output_root=args.output_root,
        watchdog=HarborWatchdogConfig(
            timeout_s=args.timeout,
            idle_timeout_s=args.idle_timeout,
        ),
    )
    paths = write_terminal_bench_launch_bundle(
        args.out_dir,
        config,
        registry_root=args.registry_root,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
