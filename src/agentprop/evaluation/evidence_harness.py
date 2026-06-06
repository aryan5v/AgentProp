"""Expanded harness config for live routing evidence (phase3-evidence-expansion)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from agentprop.evaluation.metrics import compare_routing
from agentprop.evaluation.runner import make_propagation_model, select_seeds
from agentprop.propagation import IndependentCascade
from agentprop.propagation.base import PropagationModel
from agentprop.workflows import WORKFLOW_TEMPLATES

DEFAULT_EVIDENCE_ARMS = (
    "broadcast",
    "greedy",
    "rzf-centrality",
    "quality-aware-greedy",
    "imm",
    "degree",
)


DEFAULT_EVIDENCE_WORKFLOWS = (
    "planner_coder_tester_reviewer",
    "fan_out_parallel",
    "feedback_loop",
    "shared_memory",
    "dynamic_conditional",
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
    trial_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "arm": self.arm,
            "repeat": self.repeat,
            "task_index": self.task_index,
            "coverage": self.coverage,
            "estimated_savings": self.estimated_savings,
            "quality_objective_score": self.quality_objective_score,
            "trial_seed": self.trial_seed,
        }


@dataclass(slots=True)
class EvidenceSummary:
    workflow: str
    arm: str
    mean_coverage: float
    coverage_ci_half_width: float
    mean_savings: float
    savings_ci_half_width: float
    runs: int


def run_evidence_harness(
    config: EvidenceHarnessConfig | None = None,
) -> tuple[list[EvidenceRow], list[EvidenceSummary]]:
    """Run a synthetic evidence matrix with repeats and confidence intervals."""

    cfg = config or EvidenceHarnessConfig()
    rows: list[EvidenceRow] = []
    summaries: list[EvidenceSummary] = []

    for workflow_name in cfg.workflows:
        if workflow_name not in WORKFLOW_TEMPLATES:
            continue
        graph = WORKFLOW_TEMPLATES[workflow_name]()
        graph.warm_analysis_cache()
        for arm in cfg.arms:
            coverages: list[float] = []
            savings: list[float] = []
            for repeat in range(cfg.repeats):
                for task_index in range(cfg.tasks_per_arm):
                    trial_seed = _trial_seed(workflow_name, arm, repeat, task_index)
                    model = _model_for_trial(cfg.model, trial_seed)
                    if arm == "broadcast":
                        seeds = [node.id for node in graph.nodes()]
                    else:
                        seeds = select_seeds(
                            graph,
                            arm,
                            cfg.seed_budget,
                            model,
                            cfg.trials,
                        )
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
                            trial_seed=trial_seed,
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
                    savings_ci_half_width=_ci_half_width(savings),
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
            "savings_ci_half_width": summary.savings_ci_half_width,
            "runs": summary.runs,
        }
        for summary in summaries
    ]


def rows_to_dicts(rows: list[EvidenceRow]) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]


def _trial_seed(workflow: str, arm: str, repeat: int, task_index: int) -> int:
    key = f"{workflow}:{arm}:{repeat}:{task_index}".encode()
    return int(hashlib.md5(key).hexdigest(), 16) % (2**31)


def _model_for_trial(model_name: str, trial_seed: int) -> PropagationModel:
    normalized = model_name.strip().lower()
    if normalized == "independent-cascade":
        return IndependentCascade(seed=trial_seed)
    return make_propagation_model(model_name)


def _ci_half_width(values: list[float], z: float = 1.96) -> float:
    if len(values) < 2:
        return 0.0
    return float(z * pstdev(values) / (len(values) ** 0.5))


def write_evidence_artifacts(
    config: EvidenceHarnessConfig,
    out_dir: Path,
) -> Path:
    """Run the harness and write REPORT.md, results.json, and outputs.jsonl."""

    rows, summaries = run_evidence_harness(config)
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
    outputs_path = out_dir / "outputs.jsonl"
    outputs_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows_to_dicts(rows)) + "\n"
    )
    (out_dir / "REPORT.md").write_text(_render_evidence_report(payload, config))
    (out_dir / "README.md").write_text(_evidence_readme(config, out_dir))
    return results_path


def _render_evidence_report(payload: dict[str, object], config: EvidenceHarnessConfig) -> str:
    summaries = payload.get("summaries", [])
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
        "| Workflow | Arm | Mean coverage | Cov CI | Mean savings | Sav CI | Runs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if isinstance(summaries, list):
        for row in summaries:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {row['workflow']} | {row['arm']} | {row['mean_coverage']:.3f} | "
                f"{row['coverage_ci_half_width']:.3f} | {row['mean_savings']:.3f} | "
                f"{row.get('savings_ci_half_width', 0.0):.3f} | {row['runs']} |"
            )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "agentprop run-evidence "
            f"--tasks-per-arm {config.tasks_per_arm} --repeats {config.repeats}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _evidence_readme(config: EvidenceHarnessConfig, out_dir: Path) -> str:
    return f"""# Scale / Quality Evidence Artifacts

Sanitized synthetic routing matrix comparing broadcast, greedy-family, RZF,
quality-aware, IMM, and degree baselines across expanded workflow templates.

## Command

```bash
agentprop run-evidence --tasks-per-arm {config.tasks_per_arm} --repeats {config.repeats} \\
  --out-dir {out_dir}
```

Paper-grade reproduction (N=30/arm, 3 repeats) uses the defaults above.
For a quick smoke check, pass `--tasks-per-arm 5 --repeats 2`.

## Files

- [REPORT.md](REPORT.md) — human-readable table
- [results.json](results.json) — machine-readable aggregates (no secrets)
- [outputs.jsonl](outputs.jsonl) — per-task rows (coverage, savings, trial seed)
"""
