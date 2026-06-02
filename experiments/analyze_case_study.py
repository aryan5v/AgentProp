"""Analyze AgentProp case-study result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicyComparison:
    """Policy summary relative to the broadcast baseline."""

    policy: str
    task_count: int
    success_rate: float
    verification_observed_count: int
    mean_quality_score: float
    mean_total_cost: float
    mean_token_cost: float
    mean_message_count: float
    mean_efficiency_score: float
    median_cost_delta_vs_broadcast: float
    median_cost_reduction_vs_broadcast: float
    success_delta_vs_broadcast: float
    quality_delta_vs_broadcast: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON/CSV-friendly row."""

        return {
            "policy": self.policy,
            "task_count": self.task_count,
            "success_rate": self.success_rate,
            "verification_observed_count": self.verification_observed_count,
            "mean_quality_score": self.mean_quality_score,
            "mean_total_cost": self.mean_total_cost,
            "mean_token_cost": self.mean_token_cost,
            "mean_message_count": self.mean_message_count,
            "mean_efficiency_score": self.mean_efficiency_score,
            "median_cost_delta_vs_broadcast": self.median_cost_delta_vs_broadcast,
            "median_cost_reduction_vs_broadcast": self.median_cost_reduction_vs_broadcast,
            "success_delta_vs_broadcast": self.success_delta_vs_broadcast,
            "quality_delta_vs_broadcast": self.quality_delta_vs_broadcast,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze AgentProp case-study results.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text())
    rows = _extract_rows(payload)
    out_dir = args.out_dir or args.results.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    comparisons = compare_policies(rows)
    comparison_rows = [comparison.to_dict() for comparison in comparisons]
    summary_payload = {
        "mode": payload.get("mode"),
        "workflow": payload.get("workflow"),
        "task_count": payload.get("task_count", len({row["task_id"] for row in rows})),
        "comparisons": comparison_rows,
        "acceptance": _acceptance_summary(comparisons),
    }

    (out_dir / "analysis.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(out_dir / "policy_comparison.csv", comparison_rows)
    (out_dir / "analysis.md").write_text(_render_markdown(summary_payload))
    (out_dir / "token_savings_by_policy.svg").write_text(
        _render_bar_svg(
            comparison_rows,
            metric="median_cost_reduction_vs_broadcast",
            title="Median Cost Reduction Vs Broadcast",
            value_format="percent",
        )
    )
    (out_dir / "quality_by_policy.svg").write_text(
        _render_bar_svg(
            comparison_rows,
            metric="mean_quality_score",
            title="Mean Quality Score By Policy",
            value_format="score",
        )
    )
    print(f"Wrote {out_dir}")
    return 0


def compare_policies(rows: list[dict[str, Any]]) -> list[PolicyComparison]:
    """Compare each policy against task-matched broadcast rows."""

    policies = sorted({str(row["policy"]) for row in rows})
    broadcast_rows = [row for row in rows if row["policy"] == "broadcast"]
    broadcast_by_task = {str(row["task_id"]): row for row in broadcast_rows}
    broadcast_success = _success_rate(broadcast_rows)
    broadcast_verification_count = _verification_observed_count(broadcast_rows)
    broadcast_quality = _mean([_quality(row) for row in broadcast_rows])

    comparisons = []
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        cost_deltas = []
        cost_reductions = []
        for row in policy_rows:
            baseline = broadcast_by_task.get(str(row["task_id"]))
            if baseline is None:
                continue
            baseline_cost = _total_cost(baseline)
            row_cost = _total_cost(row)
            cost_deltas.append(row_cost - baseline_cost)
            if baseline_cost > 0:
                cost_reductions.append((baseline_cost - row_cost) / baseline_cost)
        comparison = PolicyComparison(
            policy=policy,
            task_count=len(policy_rows),
            success_rate=_success_rate(policy_rows),
            verification_observed_count=_verification_observed_count(policy_rows),
            mean_quality_score=_mean([_quality(row) for row in policy_rows]),
            mean_total_cost=_mean([_total_cost(row) for row in policy_rows]),
            mean_token_cost=_mean([float(row.get("token_cost", 0.0)) for row in policy_rows]),
            mean_message_count=_mean(
                [float(row.get("message_count", 0.0)) for row in policy_rows]
            ),
            mean_efficiency_score=_mean(
                [float(row.get("efficiency_score", 0.0)) for row in policy_rows]
            ),
            median_cost_delta_vs_broadcast=_median(cost_deltas),
            median_cost_reduction_vs_broadcast=_median(cost_reductions),
            success_delta_vs_broadcast=_success_rate(policy_rows) - broadcast_success,
            quality_delta_vs_broadcast=_mean([_quality(row) for row in policy_rows])
            - broadcast_quality,
        )
        comparisons.append(comparison)
        if broadcast_verification_count == 0:
            comparisons[-1] = PolicyComparison(
                policy=comparison.policy,
                task_count=comparison.task_count,
                success_rate=comparison.success_rate,
                verification_observed_count=comparison.verification_observed_count,
                mean_quality_score=comparison.mean_quality_score,
                mean_total_cost=comparison.mean_total_cost,
                mean_token_cost=comparison.mean_token_cost,
                mean_message_count=comparison.mean_message_count,
                mean_efficiency_score=comparison.mean_efficiency_score,
                median_cost_delta_vs_broadcast=comparison.median_cost_delta_vs_broadcast,
                median_cost_reduction_vs_broadcast=comparison.median_cost_reduction_vs_broadcast,
                success_delta_vs_broadcast=0.0,
                quality_delta_vs_broadcast=comparison.quality_delta_vs_broadcast,
            )
    return comparisons


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("case-study results must be a JSON object")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("case-study results must contain rows")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("case-study rows must be objects")
    return rows


def _acceptance_summary(comparisons: list[PolicyComparison]) -> dict[str, dict[str, bool]]:
    summary = {}
    for comparison in comparisons:
        if comparison.policy == "broadcast":
            continue
        summary[comparison.policy] = {
            "cost_reduction_at_least_20_percent": (
                comparison.median_cost_reduction_vs_broadcast >= 0.2
            ),
            "verification_success_observed": comparison.verification_observed_count > 0,
            "success_drop_within_5_points": (
                comparison.verification_observed_count > 0
                and comparison.success_delta_vs_broadcast >= -0.05
            ),
            "quality_drop_within_0_25": comparison.quality_delta_vs_broadcast >= -0.25,
        }
    return summary


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# AgentProp Case Study Analysis",
        "",
        f"- Mode: `{payload.get('mode')}`",
        f"- Workflow: `{payload.get('workflow')}`",
        f"- Tasks: `{payload.get('task_count')}`",
        "",
        "## Policy Comparison",
        "",
        "| Policy | Verification Success | Verification Rows | Quality | Mean Cost | "
        "Median Cost Reduction | Success Delta | Quality Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["comparisons"]:
        lines.append(
            "| "
            f"{row['policy']} | "
            f"{row['success_rate']:.1%} | "
            f"{row['verification_observed_count']} | "
            f"{row['mean_quality_score']:.3f} | "
            f"{row['mean_total_cost']:.1f} | "
            f"{row['median_cost_reduction_vs_broadcast']:.1%} | "
            f"{row['success_delta_vs_broadcast']:.1%} | "
            f"{row['quality_delta_vs_broadcast']:.3f} |"
        )
    lines.extend(["", "## Acceptance Checks", ""])
    for policy, checks in payload["acceptance"].items():
        lines.append(f"### {policy}")
        for name, passed in checks.items():
            status = "pass" if passed else "fail"
            lines.append(f"- {name}: `{status}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_bar_svg(
    rows: list[dict[str, float | int | str]],
    *,
    metric: str,
    title: str,
    value_format: str,
) -> str:
    width = 920
    height = 340
    margin = 54
    chart_rows = [row for row in rows if row["policy"] != "broadcast"]
    if not chart_rows:
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n'
    values = [max(float(row[metric]), 0.0) for row in chart_rows]
    max_value = max(max(values), 0.01)
    bar_width = (width - 2 * margin) / max(len(chart_rows), 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="30" font-size="18" font-family="sans-serif">{title}</text>',
    ]
    for index, row in enumerate(chart_rows):
        policy = str(row["policy"])
        value = max(float(row[metric]), 0.0)
        x = margin + index * bar_width + 10
        bar_height = (height - 125) * (value / max_value)
        y = height - margin - bar_height
        lines.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 20:.1f}" '
                f'height="{bar_height:.1f}" fill="#146c5d"/>',
                f'<text x="{x:.1f}" y="{height - 28}" font-size="12" '
                f'font-family="sans-serif" transform="rotate(25 {x:.1f},{height - 28})">',
                _escape_xml(policy),
                "</text>",
                f'<text x="{x:.1f}" y="{y - 7:.1f}" font-size="12" font-family="sans-serif">',
                _format_svg_value(value, value_format),
                "</text>",
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _success_rate(rows: list[dict[str, Any]]) -> float:
    observed = [
        bool(row["verification_passed"])
        for row in rows
        if isinstance(row.get("verification_passed"), bool)
    ]
    return _mean([1.0 if passed else 0.0 for passed in observed])


def _verification_observed_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if isinstance(row.get("verification_passed"), bool))


def _quality(row: dict[str, Any]) -> float:
    return float(row.get("quality_score", 0.0))


def _total_cost(row: dict[str, Any]) -> float:
    return float(row.get("total_cost", row.get("total_llm_tokens", 0.0)))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def _format_svg_value(value: float, value_format: str) -> str:
    if value_format == "percent":
        return f"{value:.1%}"
    return f"{value:.3f}"


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
