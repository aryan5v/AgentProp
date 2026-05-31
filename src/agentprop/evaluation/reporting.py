"""Report rendering for AgentProp recommendations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentprop.evaluation.metrics import RecommendationReport


def report_to_dict(report: RecommendationReport) -> dict[str, Any]:
    """Serialize a recommendation report to plain data."""

    return {
        "seeds": report.seeds,
        "propagation_model": report.propagation_model,
        "coverage": report.propagation.coverage,
        "expected_propagation_time": report.propagation.expected_propagation_time,
        "full_activation_probability": report.propagation.full_activation_probability,
        "activated_nodes": sorted(report.propagation.activated_nodes),
        "coverage_by_round": report.propagation.coverage_by_round or [],
        "broadcast_cost": _cost_to_dict(report.broadcast_cost),
        "optimized_cost": _cost_to_dict(report.optimized_cost),
        "estimated_savings": report.estimated_savings,
        "bottlenecks": [
            {"node": node_id, "score": score} for node_id, score in report.bottlenecks
        ],
        "pruning_candidates": [
            {"source": source, "target": target}
            for source, target in report.pruning_candidates
        ],
        "verifier_candidates": report.verifier_candidates,
    }


def render_markdown_report(
    report: RecommendationReport,
    *,
    title: str = "AgentProp Optimization Report",
    workflow_name: str | None = None,
) -> str:
    """Render a recommendation report as Markdown."""

    workflow_line = f"\nWorkflow: `{workflow_name}`\n" if workflow_name else ""
    lines = [
        f"# {title}",
        workflow_line.strip(),
        "## Recommendation",
        f"- Seeds: `{', '.join(report.seeds)}`",
        f"- Propagation model: `{report.propagation_model}`",
        f"- Coverage: `{report.propagation.coverage:.1%}`",
        _format_optional_percent(
            "Full activation probability",
            report.propagation.full_activation_probability,
        ),
        _format_optional_number(
            "Expected propagation time",
            report.propagation.expected_propagation_time,
        ),
        "",
        "## Cost Comparison",
        "| Strategy | Token Cost | Message Cost | Total Cost | Latency | Messages |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        _cost_row("Broadcast", report.broadcast_cost),
        _cost_row("Optimized", report.optimized_cost),
        f"\nEstimated savings: **{report.estimated_savings:.1%}**",
        "",
        "## Bottleneck Nodes",
        *_ranked_node_lines(report.bottlenecks),
        "",
        "## Pruning Candidates",
        *_edge_lines(report.pruning_candidates),
        "",
        "## Verifier Candidates",
        *_plain_lines(report.verifier_candidates),
        "",
        "## Activated Nodes",
        *_plain_lines(sorted(report.propagation.activated_nodes)),
        "",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


def write_report(
    report: RecommendationReport,
    path: str | Path,
    *,
    workflow_name: str | None = None,
) -> Path:
    """Write a Markdown or JSON report based on file extension."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".json":
        output_path.write_text(json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n")
    else:
        output_path.write_text(render_markdown_report(report, workflow_name=workflow_name))
    return output_path


def _cost_to_dict(cost: Any) -> dict[str, float | int]:
    return {
        "token_cost": cost.token_cost,
        "message_cost": cost.message_cost,
        "total_cost": cost.total_cost,
        "latency": cost.latency,
        "message_count": cost.message_count,
    }


def _cost_row(label: str, cost: Any) -> str:
    return (
        f"| {label} | {cost.token_cost:.0f} | {cost.message_cost:.0f} | "
        f"{cost.total_cost:.0f} | {cost.latency:.2f} | {cost.message_count} |"
    )


def _format_optional_percent(label: str, value: float | None) -> str:
    if value is None:
        return f"- {label}: `n/a`"
    return f"- {label}: `{value:.1%}`"


def _format_optional_number(label: str, value: float | None) -> str:
    if value is None:
        return f"- {label}: `n/a`"
    return f"- {label}: `{value:.2f}`"


def _ranked_node_lines(rows: list[tuple[str, float]]) -> list[str]:
    if not rows:
        return ["No bottlenecks identified."]
    return [f"- `{node_id}`: `{score:.3f}`" for node_id, score in rows]


def _edge_lines(rows: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["No pruning candidates identified."]
    return [f"- `{source}` -> `{target}`" for source, target in rows]


def _plain_lines(values: list[str]) -> list[str]:
    if not values:
        return ["None."]
    return [f"- `{value}`" for value in values]
