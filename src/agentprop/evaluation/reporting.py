"""Report rendering for AgentProp recommendations."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Literal

from agentprop.evaluation.metrics import RecommendationReport

ConcreteReportFormat = Literal["markdown", "json", "html"]
ReportFormat = Literal["auto", "markdown", "json", "html"]


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
        "pruning_risk": _pruning_risk_to_dict(report.pruning_risk),
        "robustness": _robustness_to_dict(report.robustness),
        "verifier_candidates": report.verifier_candidates,
        "context_allocations": dict(report.context_allocations),
        "routing_risks": [_routing_risk_to_dict(risk) for risk in report.routing_risks],
        "quality_objective_score": report.quality_objective_score,
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
        "## Quality and Risk",
        (
            "- Quality-aware objective score: "
            f"`{_optional_number_text(report.quality_objective_score)}`"
        ),
        *_routing_risk_lines(report.routing_risks),
        "",
        "## Context Allocation",
        *_context_allocation_lines(report.context_allocations),
        "",
        "## Bottleneck Nodes",
        *_ranked_node_lines(report.bottlenecks),
        "",
        "## Pruning Candidates",
        *_edge_lines(report.pruning_candidates),
        "",
        "## Pruning Risk",
        *_pruning_risk_lines(report.pruning_risk),
        "",
        "## Robustness",
        *_robustness_lines(report.robustness),
        "",
        "## Verifier Candidates",
        *_plain_lines(report.verifier_candidates),
        "",
        "## Activated Nodes",
        *_plain_lines(sorted(report.propagation.activated_nodes)),
        "",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


def render_html_report(
    report: RecommendationReport,
    *,
    title: str = "AgentProp Optimization Report",
    workflow_name: str | None = None,
) -> str:
    """Render a recommendation report as standalone HTML."""

    workflow_line = ""
    if workflow_name:
        workflow_line = f'<p class="eyebrow">Workflow: <code>{_html(workflow_name)}</code></p>'

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_html(title)}</title>",
            "<style>",
            _html_report_css(),
            "</style>",
            "</head>",
            "<body>",
            "<main>",
            "<header>",
            workflow_line,
            f"<h1>{_html(title)}</h1>",
            '<section class="summary-grid" aria-label="Recommendation summary">',
            _summary_card("Seeds", ", ".join(report.seeds) or "None"),
            _summary_card("Propagation Model", report.propagation_model),
            _summary_card("Coverage", _percent_text(report.propagation.coverage)),
            _summary_card("Estimated Savings", _percent_text(report.estimated_savings)),
            _summary_card(
                "Full Activation",
                _optional_percent_text(report.propagation.full_activation_probability),
            ),
            _summary_card(
                "Expected Time",
                _optional_number_text(report.propagation.expected_propagation_time),
            ),
            "</section>",
            "</header>",
            "<section>",
            "<h2>Cost Comparison</h2>",
            '<table aria-label="Cost comparison">',
            "<thead>",
            "<tr>",
            "<th>Strategy</th><th>Token Cost</th><th>Message Cost</th>",
            "<th>Total Cost</th><th>Latency</th><th>Messages</th>",
            "</tr>",
            "</thead>",
            "<tbody>",
            _html_cost_row("Broadcast", report.broadcast_cost),
            _html_cost_row("Optimized", report.optimized_cost),
            "</tbody>",
            "</table>",
            "</section>",
            _html_ranked_section("Bottleneck Nodes", report.bottlenecks),
            _html_edge_section("Pruning Candidates", report.pruning_candidates),
            _html_routing_risk_section(report.routing_risks),
            _html_pruning_risk_section(report.pruning_risk),
            _html_robustness_section(report.robustness),
            _html_plain_section("Verifier Candidates", report.verifier_candidates),
            _html_plain_section("Activated Nodes", sorted(report.propagation.activated_nodes)),
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def write_report(
    report: RecommendationReport,
    path: str | Path,
    *,
    workflow_name: str | None = None,
    report_format: ReportFormat = "auto",
) -> Path:
    """Write a report in Markdown, JSON, or HTML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_format = _resolve_report_format(output_path, report_format)
    if resolved_format == "json":
        content = json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n"
    elif resolved_format == "html":
        content = render_html_report(report, workflow_name=workflow_name)
    else:
        content = render_markdown_report(report, workflow_name=workflow_name)
    output_path.write_text(content)
    return output_path


def _resolve_report_format(path: Path, report_format: ReportFormat) -> ConcreteReportFormat:
    if report_format != "auto":
        return report_format
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix in {".html", ".htm"}:
        return "html"
    return "markdown"


def _cost_to_dict(cost: Any) -> dict[str, float | int]:
    return {
        "token_cost": cost.token_cost,
        "message_cost": cost.message_cost,
        "total_cost": cost.total_cost,
        "latency": cost.latency,
        "message_count": cost.message_count,
    }


def _robustness_to_dict(robustness: Any | None) -> dict[str, float] | None:
    if robustness is None:
        return None
    return {
        "baseline_reachable_pairs": robustness.baseline_reachable_pairs,
        "average_node_failure_loss": robustness.average_node_failure_loss,
        "worst_node_failure_loss": robustness.worst_node_failure_loss,
        "average_edge_failure_loss": robustness.average_edge_failure_loss,
        "worst_edge_failure_loss": robustness.worst_edge_failure_loss,
    }


def _pruning_risk_to_dict(pruning_risk: Any | None) -> dict[str, float] | None:
    if pruning_risk is None:
        return None
    return {
        "target_cost_reduction": pruning_risk.target_cost_reduction,
        "achieved_cost_reduction": pruning_risk.achieved_cost_reduction,
        "coverage_delta": pruning_risk.coverage_delta,
        "coverage_loss": pruning_risk.coverage_loss,
        "target_gap": pruning_risk.target_gap,
        "risk_score": pruning_risk.risk_score,
    }


def _routing_risk_to_dict(risk: Any) -> dict[str, object]:
    if hasattr(risk, "to_dict"):
        return dict(risk.to_dict())
    return {
        "node": getattr(risk, "node_id", ""),
        "severity": getattr(risk, "severity", ""),
        "risk_score": getattr(risk, "risk_score", 0.0),
        "reason": getattr(risk, "reason", ""),
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


def _routing_risk_lines(risks: list[Any]) -> list[str]:
    if not risks:
        return ["- Routing risk: `none detected`"]
    return [
        (
            f"- `{risk.node_id}`: `{risk.severity}` risk "
            f"({risk.risk_score:.3f}) - {risk.reason}"
        )
        for risk in risks
    ]


def _context_allocation_lines(allocations: dict[str, float]) -> list[str]:
    if not allocations:
        return ["No context allocation data available."]
    return [
        f"- `{node_id}`: `{ratio:.0%}`"
        for node_id, ratio in sorted(allocations.items())
    ]


def _robustness_lines(robustness: Any | None) -> list[str]:
    if robustness is None:
        return ["No robustness summary available."]
    return [
        f"- Baseline reachable pairs: `{robustness.baseline_reachable_pairs:.0f}`",
        f"- Average node-failure loss: `{robustness.average_node_failure_loss:.1%}`",
        f"- Worst node-failure loss: `{robustness.worst_node_failure_loss:.1%}`",
        f"- Average edge-failure loss: `{robustness.average_edge_failure_loss:.1%}`",
        f"- Worst edge-failure loss: `{robustness.worst_edge_failure_loss:.1%}`",
    ]


def _pruning_risk_lines(pruning_risk: Any | None) -> list[str]:
    if pruning_risk is None:
        return ["No pruning risk summary available."]
    return [
        f"- Target cost reduction: `{pruning_risk.target_cost_reduction:.1%}`",
        f"- Achieved cost reduction: `{pruning_risk.achieved_cost_reduction:.1%}`",
        f"- Coverage delta: `{pruning_risk.coverage_delta:.1%}`",
        f"- Coverage loss: `{pruning_risk.coverage_loss:.1%}`",
        f"- Target gap: `{pruning_risk.target_gap:.1%}`",
        f"- Risk score: `{pruning_risk.risk_score:.3f}`",
    ]


def _html_report_css() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --ink: #172026;
  --muted: #64707d;
  --panel: #ffffff;
  --line: #d7dde4;
  --accent: #146c5d;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main {
  width: min(1120px, calc(100% - 32px));
  margin: 0 auto;
  padding: 40px 0 56px;
}
h1, h2 { line-height: 1.15; margin: 0; }
h1 { font-size: clamp(2rem, 5vw, 4rem); max-width: 760px; }
h2 { font-size: 1.15rem; margin-bottom: 12px; }
header { margin-bottom: 28px; }
section {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-top: 16px;
  padding: 20px;
}
.eyebrow {
  color: var(--muted);
  font-size: 0.9rem;
  margin: 0 0 12px;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  background: transparent;
  border: 0;
  padding: 0;
}
.metric {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
}
.metric span, th {
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.metric strong {
  display: block;
  font-size: 1.4rem;
  margin-top: 4px;
  overflow-wrap: anywhere;
}
table {
  width: 100%;
  border-collapse: collapse;
  overflow-wrap: anywhere;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 10px 8px;
  text-align: right;
}
th:first-child, td:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: 0; }
ul { margin: 0; padding-left: 20px; }
li + li { margin-top: 6px; }
code {
  background: #edf1f4;
  border-radius: 5px;
  padding: 0.1rem 0.28rem;
}
.empty { color: var(--muted); margin: 0; }
""".strip()


def _html(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _percent_text(value: float) -> str:
    return f"{value:.1%}"


def _optional_percent_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return _percent_text(value)


def _optional_number_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _summary_card(label: str, value: str) -> str:
    return (
        '<article class="metric">'
        f"<span>{_html(label)}</span>"
        f"<strong>{_html(value)}</strong>"
        "</article>"
    )


def _html_cost_row(label: str, cost: Any) -> str:
    return (
        "<tr>"
        f"<td>{_html(label)}</td>"
        f"<td>{cost.token_cost:.0f}</td>"
        f"<td>{cost.message_cost:.0f}</td>"
        f"<td>{cost.total_cost:.0f}</td>"
        f"<td>{cost.latency:.2f}</td>"
        f"<td>{cost.message_count}</td>"
        "</tr>"
    )


def _html_ranked_section(title: str, rows: list[tuple[str, float]]) -> str:
    if not rows:
        body = '<p class="empty">No bottlenecks identified.</p>'
    else:
        items = "".join(
            f"<li><code>{_html(node_id)}</code>: <strong>{score:.3f}</strong></li>"
            for node_id, score in rows
        )
        body = f"<ul>{items}</ul>"
    return f"<section><h2>{_html(title)}</h2>{body}</section>"


def _html_edge_section(title: str, rows: list[tuple[str, str]]) -> str:
    if not rows:
        body = '<p class="empty">No pruning candidates identified.</p>'
    else:
        items = "".join(
            f"<li><code>{_html(source)}</code> to <code>{_html(target)}</code></li>"
            for source, target in rows
        )
        body = f"<ul>{items}</ul>"
    return f"<section><h2>{_html(title)}</h2>{body}</section>"


def _html_pruning_risk_section(pruning_risk: Any | None) -> str:
    if pruning_risk is None:
        body = '<p class="empty">No pruning risk summary available.</p>'
    else:
        rows = [
            ("Target cost reduction", _percent_text(pruning_risk.target_cost_reduction)),
            ("Achieved cost reduction", _percent_text(pruning_risk.achieved_cost_reduction)),
            ("Coverage delta", _percent_text(pruning_risk.coverage_delta)),
            ("Coverage loss", _percent_text(pruning_risk.coverage_loss)),
            ("Target gap", _percent_text(pruning_risk.target_gap)),
            ("Risk score", f"{pruning_risk.risk_score:.3f}"),
        ]
        body = _html_metric_table(rows, "Pruning risk")
    return f"<section><h2>Pruning Risk</h2>{body}</section>"


def _html_routing_risk_section(risks: list[Any]) -> str:
    if not risks:
        body = '<p class="empty">No routing risks detected.</p>'
    else:
        items = "".join(
            (
                f"<li><code>{_html(risk.node_id)}</code>: "
                f"<strong>{_html(risk.severity)}</strong> "
                f"({risk.risk_score:.3f}) - {_html(risk.reason)}</li>"
            )
            for risk in risks
        )
        body = f"<ul>{items}</ul>"
    return f"<section><h2>Routing Risk</h2>{body}</section>"


def _html_robustness_section(robustness: Any | None) -> str:
    if robustness is None:
        body = '<p class="empty">No robustness summary available.</p>'
    else:
        rows = [
            ("Baseline reachable pairs", f"{robustness.baseline_reachable_pairs:.0f}"),
            ("Average node-failure loss", _percent_text(robustness.average_node_failure_loss)),
            ("Worst node-failure loss", _percent_text(robustness.worst_node_failure_loss)),
            ("Average edge-failure loss", _percent_text(robustness.average_edge_failure_loss)),
            ("Worst edge-failure loss", _percent_text(robustness.worst_edge_failure_loss)),
        ]
        body = _html_metric_table(rows, "Robustness")
    return f"<section><h2>Robustness</h2>{body}</section>"


def _html_metric_table(rows: list[tuple[str, str]], label: str) -> str:
    cells = "".join(
        f"<tr><td>{_html(metric)}</td><td>{_html(value)}</td></tr>" for metric, value in rows
    )
    return (
        f'<table aria-label="{_html(label)}">'
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{cells}</tbody>"
        "</table>"
    )


def _html_plain_section(title: str, values: list[str]) -> str:
    if not values:
        body = '<p class="empty">None.</p>'
    else:
        items = "".join(f"<li><code>{_html(value)}</code></li>" for value in values)
        body = f"<ul>{items}</ul>"
    return f"<section><h2>{_html(title)}</h2>{body}</section>"
