"""Task-quality scoring interfaces for real workflow evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class QualityScore:
    """A normalized task-quality score with provenance."""

    score: float
    method: str
    passed: bool
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class QualityScorer(Protocol):
    """Protocol for task-quality scorers."""

    name: str

    def score(
        self,
        *,
        expected: str | None,
        actual: str,
        context: str | None = None,
    ) -> QualityScore:
        """Score one task output."""


@dataclass(slots=True)
class ExactMatchScorer:
    """Score exact string matches for deterministic tasks."""

    normalize_whitespace: bool = True
    name: str = "exact-match"

    def score(
        self,
        *,
        expected: str | None,
        actual: str,
        context: str | None = None,
    ) -> QualityScore:
        """Return 1.0 when actual output exactly matches expected output."""

        if expected is None:
            raise ValueError("ExactMatchScorer requires an expected value")
        expected_value = _normalize(expected) if self.normalize_whitespace else expected
        actual_value = _normalize(actual) if self.normalize_whitespace else actual
        passed = expected_value == actual_value
        return QualityScore(
            score=1.0 if passed else 0.0,
            method=self.name,
            passed=passed,
            rationale="exact match" if passed else "actual output did not match expected output",
        )


@dataclass(slots=True)
class HumanLabelScorer:
    """Normalize human quality labels to 0..1."""

    max_label: float = 5.0
    pass_threshold: float = 0.8
    name: str = "human-label"

    def from_label(self, label: float, *, rationale: str = "") -> QualityScore:
        """Build a quality score from a human label."""

        if not 0 <= label <= self.max_label:
            raise ValueError("label must be between 0 and max_label")
        normalized = label / self.max_label
        return QualityScore(
            score=normalized,
            method=self.name,
            passed=normalized >= self.pass_threshold,
            rationale=rationale,
            metadata={"raw_label": label, "max_label": self.max_label},
        )


@dataclass(slots=True)
class RubricScorer:
    """Score outputs with weighted rubric checks."""

    criteria: dict[str, float]
    pass_threshold: float = 0.75
    name: str = "rubric"

    def from_criteria(self, satisfied: dict[str, bool], *, rationale: str = "") -> QualityScore:
        """Score a rubric from boolean criterion results."""

        missing = set(self.criteria) - set(satisfied)
        if missing:
            raise ValueError(f"missing rubric criteria: {sorted(missing)}")
        total_weight = sum(self.criteria.values()) or 1.0
        earned = sum(weight for criterion, weight in self.criteria.items() if satisfied[criterion])
        score = earned / total_weight
        return QualityScore(
            score=score,
            method=self.name,
            passed=score >= self.pass_threshold,
            rationale=rationale,
            metadata={"criteria": satisfied, "weights": self.criteria},
        )


@dataclass(slots=True)
class LLMJudgeScorer:
    """Adapter for an external LLM-as-judge function."""

    judge: Callable[[str | None, str, str | None], QualityScore]
    name: str = "llm-judge"

    def score(
        self,
        *,
        expected: str | None,
        actual: str,
        context: str | None = None,
    ) -> QualityScore:
        """Call an injected judge function without owning API credentials."""

        result = self.judge(expected, actual, context)
        return QualityScore(
            score=result.score,
            method=self.name,
            passed=result.passed,
            rationale=result.rationale,
            metadata=result.metadata,
        )


def aggregate_quality_scores(scores: list[QualityScore]) -> QualityScore:
    """Aggregate multiple quality scores into one summary."""

    if not scores:
        raise ValueError("scores must not be empty")
    mean_score = sum(score.score for score in scores) / len(scores)
    pass_rate = sum(1 for score in scores if score.passed) / len(scores)
    return QualityScore(
        score=mean_score,
        method="aggregate",
        passed=pass_rate >= 0.95,
        rationale=f"mean score={mean_score:.3f}, pass rate={pass_rate:.3f}",
        metadata={"count": len(scores), "pass_rate": pass_rate},
    )


def _normalize(value: str) -> str:
    return " ".join(value.strip().split())
