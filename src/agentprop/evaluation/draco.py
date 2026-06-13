"""DRACO benchmark loading and grading (kept out of the council/ layer).

DRACO (Perplexity/Harvard, arXiv 2602.11685) grades deep-research outputs
against task-specific weighted rubrics via per-criterion LLM-as-judge. This
module mirrors the paper's exact scoring so our numbers are comparable:

    raw      = Σ_i  1[verdict_i = MET] · w_i
    normalized = clip(raw / Σ_i max(0, w_i), 0, 1) · 100%
    pass_rate  = mean_i [ (w_i>0 ∧ MET) ∨ (w_i<0 ∧ UNMET) ] · 100%

Negative-weight criteria (415 of 3,934 in DRACO) describe *errors*: a MET
verdict means the error is present, so it subtracts. This is exactly what the
Council's claim-checking targets. Dataset: hf.co/datasets/perplexity-ai/draco.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# A judge decides MET (True) / UNMET (False) for one criterion given the query
# and the system response. Wrap an LLMJudgeScorer or any callable.
JudgeFn = Callable[[str, str, str], bool]


@dataclass(frozen=True, slots=True)
class DracoCriterion:
    """One rubric criterion with its signed integer weight and axis."""

    text: str
    weight: float
    axis: str = "factual-accuracy"

    @property
    def is_penalty(self) -> bool:
        return self.weight < 0


@dataclass(frozen=True, slots=True)
class DracoTask:
    """One DRACO task: a query plus its weighted rubric."""

    task_id: str
    query: str
    criteria: tuple[DracoCriterion, ...]
    domain: str = "general"

    @property
    def positive_weight_total(self) -> float:
        return sum(max(0.0, c.weight) for c in self.criteria)


@dataclass(frozen=True, slots=True)
class DracoScore:
    """Graded result for one response, with per-axis breakdown."""

    task_id: str
    normalized_score: float  # 0-100
    pass_rate: float  # 0-100
    raw_score: float
    criteria_count: int
    met_count: int
    per_axis_pass_rate: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "normalized_score": self.normalized_score,
            "pass_rate": self.pass_rate,
            "raw_score": self.raw_score,
            "criteria_count": self.criteria_count,
            "met_count": self.met_count,
            "per_axis_pass_rate": self.per_axis_pass_rate,
        }


def grade_response(
    task: DracoTask,
    response: str,
    judge: JudgeFn,
) -> DracoScore:
    """Grade one response against a task's rubric using DRACO's exact formulas."""

    raw = 0.0
    met_count = 0
    axis_hits: dict[str, list[bool]] = {}
    for criterion in task.criteria:
        met = bool(judge(task.query, response, criterion.text))
        if met:
            raw += criterion.weight
            met_count += 1
        # Pass-rate credit: positive criteria want MET, penalties want UNMET.
        correct = met if criterion.weight > 0 else (not met)
        axis_hits.setdefault(criterion.axis, []).append(correct)
    denom = task.positive_weight_total or 1.0
    normalized = max(0.0, min(1.0, raw / denom)) * 100.0
    all_correct = [c for hits in axis_hits.values() for c in hits]
    pass_rate = (sum(all_correct) / len(all_correct) * 100.0) if all_correct else 0.0
    per_axis = {
        axis: (sum(hits) / len(hits) * 100.0) if hits else 0.0
        for axis, hits in axis_hits.items()
    }
    return DracoScore(
        task_id=task.task_id,
        normalized_score=normalized,
        pass_rate=pass_rate,
        raw_score=raw,
        criteria_count=len(task.criteria),
        met_count=met_count,
        per_axis_pass_rate=per_axis,
    )


def load_draco_jsonl(path: str | Path) -> list[DracoTask]:
    """Load DRACO tasks from a local JSONL export.

    Each line: ``{"task_id", "query", "domain", "criteria": [{"text",
    "weight", "axis"}]}``. Use this after downloading the HF dataset so runs
    are reproducible from a pinned file rather than a live fetch.
    """

    tasks: list[DracoTask] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        tasks.append(_task_from_row(row))
    return tasks


def load_draco_hf(split: str = "test", *, name: str = "perplexity-ai/draco") -> list[DracoTask]:
    """Load DRACO via the optional ``datasets`` library (lazy import)."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "load_draco_hf requires `datasets`; install it or use load_draco_jsonl"
        ) from exc
    dataset = load_dataset(name, split=split)
    return [_task_from_row(dict(row)) for row in dataset]


def _task_from_row(row: dict[str, object]) -> DracoTask:
    raw_criteria = row.get("criteria", [])
    criteria_rows = raw_criteria if isinstance(raw_criteria, list) else []
    criteria = tuple(
        DracoCriterion(
            text=str(c["text"]),
            weight=float(c["weight"]),
            axis=str(c.get("axis", "factual-accuracy")),
        )
        for c in criteria_rows
        if isinstance(c, dict) and c.get("text") is not None
    )
    return DracoTask(
        task_id=str(row.get("task_id") or row.get("id") or "task"),
        query=str(row.get("query") or row.get("prompt") or ""),
        criteria=criteria,
        domain=str(row.get("domain", "general")),
    )
