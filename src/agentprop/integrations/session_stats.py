"""Aggregate analytics across saved control-session records.

Sessions are persisted by :class:`SessionStore` as one JSON file per session.
On their own they are siloed; this module rolls them up into a single view —
decision mix (CONTINUE / FORCE_VERIFY / SWITCH / FINALIZE), pass rate, and mean
token savings — so users can see how control behaves across many tasks.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Files written by SessionStore that are not per-session records.
_NON_RECORD_FILES = {"bandit.json", "risk_state.json", "learned_state.json"}


@dataclass(slots=True)
class SessionStatsReport:
    """Aggregated statistics across a directory of session records."""

    session_count: int = 0
    sessions_with_outcome: int = 0
    passed_count: int = 0
    total_events: int = 0
    decision_counts: dict[str, int] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    workflow_counts: dict[str, int] = field(default_factory=dict)
    mean_token_savings: float = 0.0
    mean_quality_score: float | None = None

    @property
    def total_decisions(self) -> int:
        return sum(self.decision_counts.values())

    @property
    def pass_rate(self) -> float:
        if self.sessions_with_outcome == 0:
            return 0.0
        return self.passed_count / self.sessions_with_outcome

    def decision_rate(self, action: str) -> float:
        total = self.total_decisions
        if total == 0:
            return 0.0
        return self.decision_counts.get(action, 0) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_count": self.session_count,
            "sessions_with_outcome": self.sessions_with_outcome,
            "passed_count": self.passed_count,
            "pass_rate": self.pass_rate,
            "total_events": self.total_events,
            "total_decisions": self.total_decisions,
            "decision_counts": dict(sorted(self.decision_counts.items())),
            "force_verify_rate": self.decision_rate("FORCE_VERIFY"),
            "switch_rate": self.decision_rate("SWITCH"),
            "category_counts": dict(sorted(self.category_counts.items())),
            "workflow_counts": dict(sorted(self.workflow_counts.items())),
            "mean_token_savings": self.mean_token_savings,
            "mean_quality_score": self.mean_quality_score,
        }


def aggregate_session_stats(root: str | Path) -> SessionStatsReport:
    """Aggregate every session record under ``root`` into a single report.

    Files that are not session records (bandit/risk/learned-state snapshots) and
    unreadable or malformed files are skipped silently so a single corrupt file
    never breaks the aggregate.
    """

    root_path = Path(root)
    report = SessionStatsReport()
    if not root_path.exists():
        return report

    decisions: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    workflows: Counter[str] = Counter()
    savings: list[float] = []
    qualities: list[float] = []

    for path in sorted(root_path.glob("*.json")):
        if path.name in _NON_RECORD_FILES:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict) or "session_id" not in payload:
            continue

        report.session_count += 1
        summary = payload.get("summary") or {}
        if isinstance(summary, dict):
            report.total_events += int(summary.get("event_count", 0) or 0)
            for action, count in dict(summary.get("decision_counts", {})).items():
                decisions[str(action)] += int(count or 0)

        category = str(payload.get("category", "general"))
        categories[category] += 1
        workflow = str(payload.get("workflow_name") or summary.get("workflow") or "unknown")
        workflows[workflow] += 1

        outcome = payload.get("outcome")
        if isinstance(outcome, dict):
            report.sessions_with_outcome += 1
            if bool(outcome.get("passed")):
                report.passed_count += 1
            savings.append(float(outcome.get("token_savings") or 0.0))
            quality = outcome.get("quality_score")
            if isinstance(quality, int | float):
                qualities.append(float(quality))

    report.decision_counts = dict(decisions)
    report.category_counts = dict(categories)
    report.workflow_counts = dict(workflows)
    report.mean_token_savings = sum(savings) / len(savings) if savings else 0.0
    report.mean_quality_score = sum(qualities) / len(qualities) if qualities else None
    return report


def render_session_stats_markdown(report: SessionStatsReport, *, root: str | Path) -> str:
    """Render a :class:`SessionStatsReport` as a human-readable Markdown summary."""

    lines = [
        "# AgentProp Session Stats",
        f"- Directory: `{root}`",
        f"- Sessions: `{report.session_count}` "
        f"(`{report.sessions_with_outcome}` with recorded outcome)",
        f"- Pass rate: `{report.pass_rate:.1%}`",
        f"- Total events: `{report.total_events}`",
        f"- Mean token savings: `{report.mean_token_savings:.1%}`",
    ]
    if report.mean_quality_score is not None:
        lines.append(f"- Mean quality score: `{report.mean_quality_score:.3f}`")
    lines.append("")

    if report.decision_counts:
        lines.extend(
            [
                "## Decision mix",
                "| Action | Count | Share |",
                "| --- | ---: | ---: |",
            ]
        )
        for action in sorted(report.decision_counts):
            count = report.decision_counts[action]
            lines.append(f"| {action} | {count} | {report.decision_rate(action):.1%} |")
        lines.append("")

    if report.category_counts:
        lines.extend(["## Sessions by category", "| Category | Sessions |", "| --- | ---: |"])
        for category in sorted(report.category_counts):
            lines.append(f"| {category} | {report.category_counts[category]} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
