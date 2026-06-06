#!/usr/bin/env python3
"""Run the scale/quality evidence matrix and write sanitized public artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from agentprop.evaluation.evidence_harness import (
    EvidenceHarnessConfig,
    run_evidence_harness,
    summaries_to_dict,
)


def main() -> None:
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
    args = parser.parse_args()

    config = EvidenceHarnessConfig(
        tasks_per_arm=args.tasks_per_arm,
        repeats=args.repeats,
        seed_budget=args.seed_budget,
        trials=args.trials,
    )
    rows, summaries = run_evidence_harness(config)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "workflows": list(config.workflows),
            "arms": list(config.arms),
            "tasks_per_arm": config.tasks_per_arm,
            "repeats": config.repeats,
            "seed_budget": config.seed_budget,
            "trials": config.trials,
            "model": config.model,
        },
        "summaries": summaries_to_dict(summaries),
        "row_count": len(rows),
    }
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    report = _render_report(payload, config)
    (out_dir / "REPORT.md").write_text(report)
    (out_dir / "README.md").write_text(_readme(config, out_dir))
    print(f"Wrote {results_path} ({len(summaries)} summary rows)")


def _render_report(payload: dict, config: EvidenceHarnessConfig) -> str:
    lines = [
        "# Scale / Quality Evidence (synthetic matrix)",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "**Label:** directional benchmark result on built-in workflow templates.",
        "Coverage and savings are propagation-simulation metrics, not live LLM task success.",
        "",
        "## Configuration",
        "",
        f"- Workflows: {', '.join(config.workflows)}",
        f"- Arms: {', '.join(config.arms)}",
        f"- Tasks per arm: {config.tasks_per_arm}",
        f"- Repeats: {config.repeats}",
        f"- Seed budget: {config.seed_budget}",
        f"- Trials: {config.trials}",
        "",
        "## Summaries (mean coverage ± 95% CI half-width)",
        "",
        "| Workflow | Arm | Mean coverage | CI half-width | Mean savings | Runs |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summaries"]:
        lines.append(
            f"| {row['workflow']} | {row['arm']} | {row['mean_coverage']:.3f} | "
            f"{row['coverage_ci_half_width']:.3f} | {row['mean_savings']:.3f} | {row['runs']} |"
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=src:. python experiments/run_evidence_harness.py "
            f"--tasks-per-arm {config.tasks_per_arm} --repeats {config.repeats}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _readme(config: EvidenceHarnessConfig, out_dir: Path) -> str:
    return f"""# Scale / Quality Evidence Artifacts

Sanitized synthetic routing matrix comparing broadcast, greedy-family, RZF,
quality-aware, and IMM arms across expanded workflow templates.

## Command

```bash
PYTHONPATH=src:. python experiments/run_evidence_harness.py \\
  --tasks-per-arm {config.tasks_per_arm} --repeats {config.repeats} \\
  --out-dir {out_dir}
```

Paper-grade reproduction (N=30/arm, 3 repeats) uses the defaults above.
For a quick smoke check, pass `--tasks-per-arm 5 --repeats 2`.

## Files

- [REPORT.md](REPORT.md) — human-readable table
- [results.json](results.json) — machine-readable aggregates (no secrets)
"""


if __name__ == "__main__":
    main()
