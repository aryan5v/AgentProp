"""Summarize saved Harbor task artifacts into launch-ready reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.evaluation.terminal_bench import write_terminal_bench_summary_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Harbor result.json artifacts.")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="AgentProp Terminal-Bench Result Summary")
    parser.add_argument("--registry-root", type=Path, default=None)
    args = parser.parse_args(argv)

    paths = write_terminal_bench_summary_report(
        args.results_root,
        args.out_dir,
        title=args.title,
        registry_root=args.registry_root,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
