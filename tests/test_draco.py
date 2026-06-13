"""Tests for the DRACO loader and grading formulas (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from agentprop.evaluation.draco import (
    DracoCriterion,
    DracoTask,
    grade_response,
    load_draco_jsonl,
)


def _task() -> DracoTask:
    return DracoTask(
        task_id="t1",
        query="Explain X.",
        criteria=(
            DracoCriterion("States the correct value", weight=20, axis="factual-accuracy"),
            DracoCriterion("Cites a primary source", weight=10, axis="citation-quality"),
            DracoCriterion("Contains a dangerous claim", weight=-50, axis="factual-accuracy"),
        ),
    )


def test_perfect_response_scores_100() -> None:
    # MET the two positives, UNMET the penalty.
    def judge(_q: str, _r: str, criterion: str) -> bool:
        return "dangerous" not in criterion

    score = grade_response(_task(), "good cited answer", judge)
    assert score.normalized_score == 100.0
    assert score.pass_rate == 100.0
    assert score.raw_score == 30.0  # 20 + 10


def test_penalty_met_subtracts() -> None:
    # MET everything, including the dangerous-claim penalty.
    score = grade_response(_task(), "bad answer", lambda *_: True)
    # raw = 20 + 10 - 50 = -20 -> clipped to 0.
    assert score.raw_score == -20.0
    assert score.normalized_score == 0.0
    # pass_rate: 2 positives MET (good) + 1 penalty MET (bad) -> 2/3.
    assert round(score.pass_rate, 1) == round(2 / 3 * 100, 1)


def test_normalized_uses_positive_weight_denominator() -> None:
    # Only the value criterion MET; denominator is 20+10 = 30.
    def judge(_q: str, _r: str, criterion: str) -> bool:
        return "correct value" in criterion

    score = grade_response(_task(), "partial", judge)
    assert round(score.normalized_score, 1) == round(20 / 30 * 100, 1)


def test_per_axis_breakdown() -> None:
    score = grade_response(_task(), "x", lambda *_: True)
    assert "factual-accuracy" in score.per_axis_pass_rate
    assert "citation-quality" in score.per_axis_pass_rate
    # factual-accuracy: value MET (good) + dangerous MET (bad) -> 1/2 = 50%.
    assert score.per_axis_pass_rate["factual-accuracy"] == 50.0


def test_empty_rubric_is_zero() -> None:
    task = DracoTask("t", "q", criteria=())
    score = grade_response(task, "anything", lambda *_: True)
    assert score.normalized_score == 0.0
    assert score.pass_rate == 0.0


def test_load_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = [
        {
            "task_id": "a",
            "query": "Q",
            "domain": "law",
            "criteria": [{"text": "c1", "weight": 5, "axis": "factual-accuracy"}],
        }
    ]
    path = tmp_path / "draco.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    tasks = load_draco_jsonl(path)
    assert len(tasks) == 1
    assert tasks[0].domain == "law"
    assert tasks[0].criteria[0].weight == 5.0
    assert tasks[0].positive_weight_total == 5.0
