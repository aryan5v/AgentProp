"""Failure taxonomy for benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FailureCategory = Literal[
    "passed",
    "solution_miss",
    "incomplete_output",
    "domain_constraint_miss",
    "async_lifecycle_miss",
    "build_or_mode_mismatch",
    "timeout_or_overexploration",
    "harness_infra_failure",
    "unknown_exception",
]


@dataclass(frozen=True, slots=True)
class FailureClassification:
    """A coarse, research-safe label for one benchmark task outcome."""

    category: FailureCategory
    retry_recommended: bool
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "retry_recommended": self.retry_recommended,
            "rationale": self.rationale,
        }


def classify_benchmark_failure(
    task_name: str,
    *,
    passed: bool | None,
    exception_name: str | None = None,
) -> FailureClassification:
    """Classify obvious benchmark failures without reading private run logs."""

    if passed is True:
        return FailureClassification("passed", False, "Verifier reward indicates success.")

    task_key = task_name.rsplit("/", 1)[-1].lower()
    exception_key = (exception_name or "").lower()
    signature = f"{task_key} {exception_key}"

    if _contains_any(
        signature,
        (
            "sessionnotcreated",
            "selenium",
            "chrome instance exited",
            "tmux",
            "container cwd",
            "/app missing",
        ),
    ):
        return FailureClassification(
            "harness_infra_failure",
            True,
            "Failure signature points to browser/session/container setup rather than task logic.",
        )

    if "timeout" in signature:
        return FailureClassification(
            "timeout_or_overexploration",
            True,
            "Timeouts are retryable and should train timeout-risk separately from correctness.",
        )

    if "chess-best-move" in task_key:
        return FailureClassification(
            "incomplete_output",
            False,
            "Known direct-answer failure: the task may require all valid answers, not one answer.",
        )

    if "dna-" in task_key or "primer" in task_key:
        return FailureClassification(
            "domain_constraint_miss",
            False,
            "Biological-design tasks require explicit assembly and primer-constraint checks.",
        )

    if "cancel-async-tasks" in task_key or "async" in task_key:
        return FailureClassification(
            "async_lifecycle_miss",
            False,
            "Concurrency tasks need cleanup/cancellation invariants, not only happy-path tests.",
        )

    if "custom-memory-heap-crash" in task_key or "release" in signature:
        return FailureClassification(
            "build_or_mode_mismatch",
            False,
            "Build-sensitive tasks must verify the same mode/configuration used by the harness.",
        )

    if exception_name:
        return FailureClassification(
            "unknown_exception",
            True,
            "Exception was reported, but it does not match a known task or infra signature.",
        )

    return FailureClassification(
        "solution_miss",
        False,
        "Verifier failed without an obvious infra or timeout signature.",
    )


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)
