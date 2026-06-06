"""Expanded harness config for live routing evidence (phase3-evidence-expansion)."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from agentprop.evaluation.metrics import compare_routing
from agentprop.evaluation.runner import make_propagation_model, select_seeds
from agentprop.workflows import WORKFLOW_TEMPLATES

DEFAULT_EVIDENCE_ARMS = (
    "broadcast",
    "greedy",
    "rzf-centrality",
    "quality-aware-greedy",
    "degree",
)


DEFAULT_EVIDENCE_WORKFLOWS = (
    "planner_coder_tester_reviewer",
    "fan_out_parallel",
    "feedback_loop",
    "shared_memory",
    "hub_and_spoke_supervisor",
)


@dataclass(slots=True)
class EvidenceHarnessConfig:
    """Configuration for multi-workflow, multi-arm evidence runs."""

    workflows: tuple[str, ...] = DEFAULT_EVIDENCE_WORKFLOWS
    arms: tuple[str, ...] = DEFAULT_EVIDENCE_ARMS
    tasks_per_arm: int = 30
    repeats: int = 3
    seed_budget: int = 3
    trials: int = 50
    model: str = "independent-cascade"


@dataclass(slots=True)
class EvidenceRow:
    workflow: str
    arm: str
    repeat: int
    task_index: int
    coverage: float
    estimated_savings: float
    quality_objective_score: float | None


@dataclass(slots=True)
class EvidenceSummary:
    workflow: str
    arm: str
    mean_coverage: float
    coverage_ci_half_width: float
    mean_savings: float
    runs: int


def run_evidence_harness(
    config: EvidenceHarnessConfig | None = None,
) -> tuple[list[EvidenceRow], list[EvidenceSummary]]:
    """Run a lightweight synthetic evidence matrix with repeats and CIs."""

    cfg = config or EvidenceHarnessConfig()
    rows: list[EvidenceRow] = []
    summaries: list[EvidenceSummary] = []

    for workflow_name in cfg.workflows:
        if workflow_name not in WORKFLOW_TEMPLATES:
            continue
        graph = WORKFLOW_TEMPLATES[workflow_name]()
        graph.warm_analysis_cache()
        model = make_propagation_model(cfg.model)
        for arm in cfg.arms:
            coverages: list[float] = []
            savings: list[float] = []
            algorithm = "rzf-centrality" if arm == "degree" else arm
            if arm == "broadcast":
                seeds = [node.id for node in graph.nodes()]
            else:
                seeds = select_seeds(
                    graph,
                    algorithm,
                    cfg.seed_budget,
                    model,
                    cfg.trials,
                )
            for repeat in range(cfg.repeats):
                for task_index in range(cfg.tasks_per_arm):
                    propagation = model.simulate(graph, seeds, trials=cfg.trials)
                    report = compare_routing(
                        graph,
                        seeds,
                        model.name,
                        propagation,
                    )
                    rows.append(
                        EvidenceRow(
                            workflow=workflow_name,
                            arm=arm,
                            repeat=repeat,
                            task_index=task_index,
                            coverage=propagation.coverage,
                            estimated_savings=report.estimated_savings,
                            quality_objective_score=report.quality_objective_score,
                        )
                    )
                    coverages.append(propagation.coverage)
                    savings.append(report.estimated_savings)
            summaries.append(
                EvidenceSummary(
                    workflow=workflow_name,
                    arm=arm,
                    mean_coverage=mean(coverages) if coverages else 0.0,
                    coverage_ci_half_width=_ci_half_width(coverages),
                    mean_savings=mean(savings) if savings else 0.0,
                    runs=len(coverages),
                )
            )
    return rows, summaries


def summaries_to_dict(summaries: list[EvidenceSummary]) -> list[dict[str, Any]]:
    return [
        {
            "workflow": summary.workflow,
            "arm": summary.arm,
            "mean_coverage": summary.mean_coverage,
            "coverage_ci_half_width": summary.coverage_ci_half_width,
            "mean_savings": summary.mean_savings,
            "runs": summary.runs,
        }
        for summary in summaries
    ]


def _ci_half_width(values: list[float], z: float = 1.96) -> float:
    if len(values) < 2:
        return 0.0
    return z * pstdev(values) / (len(values) ** 0.5)
