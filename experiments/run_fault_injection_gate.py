#!/usr/bin/env python3
"""Run the AGE-53 GO/NO-GO fault-injection gate and write artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.evaluation.fault_injection import run_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/fault_injection_gate"))
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--go-threshold", type=float, default=0.05)
    args = parser.parse_args()

    report = run_gate(trials=args.trials, seeds=args.seeds, go_threshold=args.go_threshold)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gate_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Fault-Injection GO/NO-GO Gate",
        "",
        "Single-fault MAP localization accuracy under noisy observations:",
        "classical undirected metric-dimension placement vs directed/noisy",
        "JS-divergence placement. Decision rule: GO if the mean directed",
        f"advantage is at least {report.go_threshold:.0%} absolute.",
        "",
        f"**Decision: {'GO' if report.go else 'NO-GO'}** — mean classical "
        f"{report.mean_classical:.1%}, mean directed {report.mean_directed:.1%}, "
        f"advantage {report.mean_advantage:+.1%}.",
        "",
        "| Family | Noise | Budget | Classical | Directed | Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in report.conditions:
        delta = c.directed_accuracy - c.classical_accuracy
        lines.append(
            f"| {c.family} | {c.noise:.2f} | {c.budget} | "
            f"{c.classical_accuracy:.1%} | {c.directed_accuracy:.1%} | {delta:+.1%} |"
        )
    lines.append("")
    (args.out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Decision: {'GO' if report.go else 'NO-GO'} "
          f"(advantage {report.mean_advantage:+.1%}, threshold {report.go_threshold:.0%})")
    print(f"Artifacts: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
