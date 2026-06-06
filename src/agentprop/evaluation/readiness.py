"""Implementation maturity reporting for AgentProp components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MaturityStatus = Literal["stable", "alpha", "experimental"]

_STATUS_SCORE: dict[MaturityStatus, float] = {
    "stable": 1.0,
    "alpha": 0.75,
    "experimental": 0.5,
}


@dataclass(frozen=True, slots=True)
class MaturityItem:
    """One shipped or in-progress component area."""

    id: str
    area: str
    title: str
    status: MaturityStatus
    evidence: tuple[str, ...]
    weight: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "area": self.area,
            "title": self.title,
            "status": self.status,
            "evidence": list(self.evidence),
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class MaturityReport:
    """Rollup of implemented AgentProp surface area."""

    target: str
    summary: str
    overall_score: float
    counts: dict[str, int]
    items: tuple[MaturityItem, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "summary": self.summary,
            "overall_score": self.overall_score,
            "counts": dict(self.counts),
            "items": [item.to_dict() for item in self.items],
        }


def build_v1_readiness_report() -> MaturityReport:
    """Return the current implementation maturity report."""

    items = (
        MaturityItem(
            id="core-graph",
            area="Core",
            title="Directed weighted workflow graph backbone",
            status="stable",
            evidence=(
                "AgentGraph supports node/edge metadata, JSON import/export, validation, "
                "NetworkX conversion, and visualization.",
                "Built-in workflow templates cover agent-inspired and synthetic graph families.",
            ),
        ),
        MaturityItem(
            id="classical-algorithms",
            area="Core",
            title="Training-free graph algorithms",
            status="stable",
            evidence=(
                "Seed selection includes random, degree, in/out degree, PageRank, "
                "betweenness, closeness, k-core, greedy, CELF, and cost-aware greedy.",
                "Bottleneck, bridge, articulation, reliability, and failure-sensitive "
                "diagnostics are implemented.",
            ),
        ),
        MaturityItem(
            id="propagation-models",
            area="Core",
            title="Propagation model coverage",
            status="stable",
            evidence=(
                "Independent Cascade, Linear Threshold, Bootstrap Percolation, "
                "Randomized Zero Forcing, Zero Forcing, and learned propagation exist.",
                "Trace-calibrated learned propagation can be trained from trace JSON.",
            ),
        ),
        MaturityItem(
            id="pruning-verifiers",
            area="Optimization",
            title="Pruning and verifier placement",
            status="stable",
            evidence=(
                "Low-usage, betweenness/reachability-preserving, cost-aware, and "
                "redundancy-aware pruning paths are represented in the framework.",
                "Verifier placement includes risk-aware, observability, PageRank, "
                "betweenness, error-centrality, and greedy correction coverage methods.",
            ),
        ),
        MaturityItem(
            id="metrics",
            area="Evaluation",
            title="Graph, quality, and efficiency metrics",
            status="stable",
            evidence=(
                "Reports include cost, coverage, propagation time, activation probability, "
                "savings, robustness, pruning risk, and cost-adjusted quality metrics.",
                "Quality scorers cover exact-match, human labels, rubric scoring, and "
                "injected LLM-as-judge adapters.",
            ),
        ),
        MaturityItem(
            id="cli-reports",
            area="Product",
            title="CLI and report surface",
            status="stable",
            evidence=(
                "CLI covers analyze, optimize, benchmark, report, simulate, prune, trace, "
                "viz, and agent-instructions.",
                "Reports can be emitted as Markdown, JSON, or HTML.",
            ),
        ),
        MaturityItem(
            id="ml-baselines",
            area="ML/DL/RL",
            title="Dependency-light ML baselines",
            status="stable",
            evidence=(
                "Feature extraction, greedy-labeled datasets, node scorers, pairwise "
                "ranking, propagation-time regression, and generalization checks exist.",
                "Experiment suite recipes write reproducible manifests and artifacts.",
                "Dependency-light ML scorers and tabular RL policies can be persisted "
                "as JSON checkpoints.",
            ),
        ),
        MaturityItem(
            id="torch-dl",
            area="ML/DL/RL",
            title="Optional torch deep-learning backend",
            status="alpha",
            evidence=(
                "GCN, GraphSAGE, GAT, GIN, Graph Transformer, heterogeneous GNN, and "
                "edge-conditioned GNN paths are implemented behind the dl extra.",
                "Edge-conditioned training consumes richer edge feature matrices.",
            ),
        ),
        MaturityItem(
            id="rl-routing",
            area="ML/DL/RL",
            title="Expanded-action RL routing",
            status="alpha",
            evidence=(
                "Routing environment supports seed selection plus SEND_CONTEXT, "
                "ACTIVATE_VERIFIER, SEND_MESSAGE, PRUNE_EDGE, CALL_TOOL, REQUEST_SUMMARY, "
                "and STOP actions.",
                "Q-learning, REINFORCE, and PPO-style baselines emit auditable "
                "trajectories and reward traces.",
            ),
        ),
        MaturityItem(
            id="framework-adapters",
            area="Integrations",
            title="Framework interchange adapters",
            status="alpha",
            evidence=(
                "LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, and LlamaIndex workflows "
                "have dependency-light dictionary adapters.",
                "LangGraph, CrewAI, and OpenAI Agents SDK have optional best-effort native "
                "builders when those packages are installed.",
            ),
        ),
        MaturityItem(
            id="coding-agent-integrations",
            area="Integrations",
            title="Coding-agent integration surface",
            status="alpha",
            evidence=(
                "AgentProp can emit Claude Code, Codex, and generic agent briefs.",
                "A stdio MCP server exposes analyze, optimize, report, and "
                "agent-instruction tools.",
            ),
        ),
        MaturityItem(
            id="release-packaging",
            area="Release",
            title="Public alpha packaging",
            status="stable",
            evidence=(
                "License, changelog, contributing guide, release notes, CI, docs index, "
                "and security policy are published in the public repository.",
            ),
        ),
    )
    weighted_total = sum(item.weight for item in items)
    score = sum(_STATUS_SCORE[item.status] * item.weight for item in items) / weighted_total
    counts: dict[str, int] = {status: 0 for status in _STATUS_SCORE}
    for item in items:
        counts[item.status] += 1
    summary = (
        "AgentProp ships a stable graph backbone, propagation models, CLI, and "
        "training-free baselines, with alpha integrations and optional ML/DL/RL paths."
    )
    return MaturityReport(
        target="public alpha",
        summary=summary,
        overall_score=round(score, 3),
        counts=counts,
        items=items,
    )


# Backwards-compatible aliases for existing imports and CLI.
ReadinessItem = MaturityItem
ReadinessReport = MaturityReport
ReadinessStatus = MaturityStatus


def render_v1_readiness_markdown(report: MaturityReport) -> str:
    """Render a maturity report as Markdown."""

    lines = [
        "# AgentProp Implementation Maturity",
        "",
        report.summary,
        "",
        f"- Overall score: {report.overall_score:.1%}",
        "- Status counts: "
        + ", ".join(f"{status}={count}" for status, count in report.counts.items()),
        "",
        "## Components",
        "",
    ]
    for item in report.items:
        lines.append(f"### {item.area}: {item.title}")
        lines.append("")
        lines.append(f"- Status: `{item.status}`")
        for evidence in item.evidence:
            lines.append(f"- {evidence}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
