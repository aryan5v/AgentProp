#!/usr/bin/env python3
"""Run the scale/quality evidence matrix and write sanitized public artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentprop.evaluation.evidence_harness import EvidenceHarnessConfig, write_evidence_artifacts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/results/scale_quality_evidence"),
        help="Directory for REPORT.md and results.json",
    )
    parser.add_argument(
        "--tasks-per-arm",
        type=int,
        default=30,
        help="Synthetic tasks per arm (use 5 for a quick smoke artifact)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Independent repeats for CI half-width",
    )
    parser.add_argument("--seed-budget", type=int, default=3)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args(argv)

    config = EvidenceHarnessConfig(
        tasks_per_arm=args.tasks_per_arm,
        repeats=args.repeats,
        seed_budget=args.seed_budget,
        trials=args.trials,
    )
    results_path = write_evidence_artifacts(config, args.out_dir)
    print(f"Wrote {results_path} ({config.tasks_per_arm * config.repeats} runs/arm)")


if __name__ == "__main__":
    main()
