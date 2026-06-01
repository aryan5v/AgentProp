"""V1 rollout readiness reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReadinessStatus = Literal["complete", "alpha", "blocked", "missing"]

_STATUS_SCORE: dict[ReadinessStatus, float] = {
    "complete": 1.0,
    "alpha": 0.75,
    "blocked": 0.35,
    "missing": 0.0,
}


@dataclass(frozen=True, slots=True)
class ReadinessItem:
    """One auditable rollout requirement."""

    id: str
    area: str
    title: str
    status: ReadinessStatus
    evidence: tuple[str, ...]
    remaining: tuple[str, ...] = ()
    weight: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "area": self.area,
            "title": self.title,
            "status": self.status,
            "evidence": list(self.evidence),
            "remaining": list(self.remaining),
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Rollup of AgentProp readiness for the next release gate."""

    target: str
    summary: str
    overall_score: float
    alpha_ready: bool
    public_ready: bool
    counts: dict[str, int]
    blockers: tuple[str, ...]
    items: tuple[ReadinessItem, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "summary": self.summary,
            "overall_score": self.overall_score,
            "alpha_ready": self.alpha_ready,
            "public_ready": self.public_ready,
            "counts": dict(self.counts),
            "blockers": list(self.blockers),
            "items": [item.to_dict() for item in self.items],
        }


def build_v1_readiness_report() -> ReadinessReport:
    """Return the current evidence-backed V1 rollout report.

    This is intentionally explicit instead of inferred from checkboxes. The
    rollout decision depends on the quality of evidence, not only whether a file
    or command exists.
    """

    items = (
        ReadinessItem(
            id="core-graph",
            area="Core",
            title="Directed weighted workflow graph backbone",
            status="complete",
            evidence=(
                "AgentGraph supports node/edge metadata, JSON import/export, validation, "
                "NetworkX conversion, and visualization.",
                "Built-in workflow templates cover agent-inspired and synthetic graph families.",
            ),
        ),
        ReadinessItem(
            id="classical-algorithms",
            area="Core",
            title="Training-free graph algorithms",
            status="complete",
            evidence=(
                "Seed selection includes random, degree, in/out degree, PageRank, "
                "betweenness, closeness, k-core, greedy, CELF, and cost-aware greedy.",
                "Bottleneck, bridge, articulation, reliability, and failure-sensitive "
                "diagnostics are implemented.",
            ),
        ),
        ReadinessItem(
            id="propagation-models",
            area="Core",
            title="Propagation model coverage",
            status="complete",
            evidence=(
                "Independent Cascade, Linear Threshold, Bootstrap Percolation, "
                "Randomized Zero Forcing, Zero Forcing, and learned propagation exist.",
                "Trace-calibrated learned propagation can be trained from trace JSON.",
            ),
        ),
        ReadinessItem(
            id="pruning-verifiers",
            area="Optimization",
            title="Pruning and verifier placement",
            status="complete",
            evidence=(
                "Low-usage, betweenness/reachability-preserving, cost-aware, and "
                "redundancy-aware pruning paths are represented in the framework.",
                "Verifier placement includes risk-aware, observability, PageRank, "
                "betweenness, error-centrality, and greedy correction coverage methods.",
            ),
        ),
        ReadinessItem(
            id="metrics",
            area="Evaluation",
            title="Graph, quality, and efficiency metrics",
            status="complete",
            evidence=(
                "Reports include cost, coverage, propagation time, activation probability, "
                "savings, robustness, pruning risk, and cost-adjusted quality metrics.",
                "Quality scorers cover exact-match, human labels, rubric scoring, and "
                "injected LLM-as-judge adapters.",
            ),
        ),
        ReadinessItem(
            id="cli-reports",
            area="Product",
            title="CLI and report surface",
            status="complete",
            evidence=(
                "CLI covers analyze, optimize, benchmark, report, simulate, prune, trace, "
                "viz, and agent-instructions.",
                "Reports can be emitted as Markdown, JSON, or HTML.",
            ),
        ),
        ReadinessItem(
            id="ml-baselines",
            area="ML/DL/RL",
            title="Dependency-light ML baselines",
            status="complete",
            evidence=(
                "Feature extraction, greedy-labeled datasets, node scorers, pairwise "
                "ranking, propagation-time regression, and generalization checks exist.",
                "Experiment suite recipes write reproducible manifests and artifacts.",
            ),
        ),
        ReadinessItem(
            id="torch-dl",
            area="ML/DL/RL",
            title="Optional torch deep-learning backend",
            status="alpha",
            evidence=(
                "GCN, GraphSAGE, GAT, GIN, Graph Transformer, heterogeneous GNN, and "
                "edge-conditioned GNN paths are implemented behind the dl extra.",
                "Edge-conditioned training now consumes richer edge feature matrices.",
            ),
            remaining=(
                "Run larger held-out graph sweeps and hyperparameter searches.",
                "Decide whether to use Modal GPU for bigger DL evidence.",
            ),
        ),
        ReadinessItem(
            id="rl-routing",
            area="ML/DL/RL",
            title="Expanded-action RL routing",
            status="alpha",
            evidence=(
                "Routing environment supports seed selection plus SEND_CONTEXT, "
                "ACTIVATE_VERIFIER, SEND_MESSAGE, PRUNE_EDGE, CALL_TOOL, REQUEST_SUMMARY, "
                "and STOP actions.",
                "Q-learning, REINFORCE, and PPO-style baselines can emit auditable "
                "trajectories and reward traces.",
            ),
            remaining=(
                "Prove the expanded-action policies against greedy, GNN, and broadcast on "
                "real or much larger synthetic tasks.",
            ),
        ),
        ReadinessItem(
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
            remaining=(
                "Add configured runtime builders for AutoGen and LlamaIndex.",
                "Round-trip against real framework examples.",
            ),
        ),
        ReadinessItem(
            id="coding-agent-integrations",
            area="Integrations",
            title="Coding-agent integration surface",
            status="alpha",
            evidence=(
                "AgentProp can emit Claude Code, Codex, and generic agent briefs.",
                "A stdio MCP server exposes analyze, optimize, report, and "
                "agent-instruction tools.",
            ),
            remaining=(
                "Dogfood with real Claude Code/Codex workflows and tighten the generated "
                "brief format from user traces.",
            ),
        ),
        ReadinessItem(
            id="real-llm-study",
            area="Validation",
            title="Real routed LLM case-study results",
            status="blocked",
            evidence=(
                "The 20-task protocol, task set, routed LLM harness, preflight, and "
                "analysis scripts exist.",
                "LLM execution no longer treats prompt-only annotation or keyword rubrics "
                "as validation.",
            ),
            remaining=(
                "Provide Token Router or OpenAI-compatible credentials.",
                "Run the routed multi-node arms and save cost, quality, trace, and "
                "verification artifacts.",
                "Capture verification logs from an environment that actually applies and "
                "tests generated code changes.",
            ),
            weight=1.5,
        ),
        ReadinessItem(
            id="release-packaging",
            area="Release",
            title="Private alpha packaging",
            status="complete",
            evidence=(
                "License, changelog, contributing guide, release notes, CI, docs index, "
                "and private repository decision are in place.",
            ),
        ),
        ReadinessItem(
            id="public-launch-proof",
            area="Release",
            title="Public launch proof package",
            status="blocked",
            evidence=(
                "The public release decision is documented: stay private or publish only "
                "with an explicit alpha limitation.",
            ),
            remaining=(
                "Attach the first real LLM case-study result directory.",
                "Add public-facing README plots/screenshots from the completed study.",
                "Create the GitHub release once the evidence package is ready.",
            ),
            weight=1.25,
        ),
    )
    weighted_total = sum(item.weight for item in items)
    score = sum(_STATUS_SCORE[item.status] * item.weight for item in items) / weighted_total
    counts: dict[str, int] = {status: 0 for status in _STATUS_SCORE}
    for item in items:
        counts[item.status] += 1
    blockers = tuple(
        item.title for item in items if item.status in {"blocked", "missing"}
    )
    alpha_ready = score >= 0.8 and counts["missing"] == 0
    public_ready = counts["blocked"] == 0 and counts["missing"] == 0
    summary = (
        "AgentProp is ready for a serious private alpha / v1-candidate rollout, "
        "but not for unqualified public claims until the real routed LLM case study "
        "has saved results."
    )
    return ReadinessReport(
        target="v1 rollout",
        summary=summary,
        overall_score=round(score, 3),
        alpha_ready=alpha_ready,
        public_ready=public_ready,
        counts=counts,
        blockers=blockers,
        items=items,
    )


def render_v1_readiness_markdown(report: ReadinessReport) -> str:
    """Render a V1 readiness report as Markdown."""

    lines = [
        "# AgentProp V1 Readiness",
        "",
        report.summary,
        "",
        f"- Overall score: {report.overall_score:.1%}",
        f"- Private alpha ready: {'yes' if report.alpha_ready else 'no'}",
        f"- Public ready: {'yes' if report.public_ready else 'no'}",
        "- Status counts: "
        + ", ".join(f"{status}={count}" for status, count in report.counts.items()),
        "",
        "## Blockers",
        "",
    ]
    if report.blockers:
        lines.extend(f"- {blocker}" for blocker in report.blockers)
    else:
        lines.append("- None")
    lines.extend(["", "## Evidence", ""])
    for item in report.items:
        lines.append(f"### {item.area}: {item.title}")
        lines.append("")
        lines.append(f"- Status: `{item.status}`")
        for evidence in item.evidence:
            lines.append(f"- Evidence: {evidence}")
        for remaining in item.remaining:
            lines.append(f"- Remaining: {remaining}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
