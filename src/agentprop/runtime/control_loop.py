"""Control-loop primitives for real benchmark/runtime execution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agentprop.rl import CategoryBanditRoutingPolicy

ControlAction = Literal["CONTINUE", "FORCE_VERIFY", "FINALIZE", "SWITCH_STRATEGY"]


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """One observed turn from a real terminal/tool/agent loop."""

    step: int
    command: str | None = None
    exit_code: int | None = None
    verifier_run: bool = False
    verifier_passed: bool | None = None
    progress_made: bool = False
    tokens_used: int = 0
    elapsed_s: float = 0.0
    error_signature: str | None = None
    final_answer_written: bool = False
    trusted: bool = True
    """Whether a verifier result is from an independent/trusted check.

    Defaults to True for backward compatibility. A harness should set this to
    False for the agent's own self-reported evaluation (e.g. a locally-run
    ``eval.py``), so the controller does not finalize on an unconfirmed pass."""


@dataclass(frozen=True, slots=True)
class ExecutionStateFeatures:
    """Compact features the controller can act on deterministically."""

    step_count: int
    total_tokens: int
    elapsed_s: float
    steps_since_verifier: int
    steps_since_progress: int
    repeated_error_count: int
    verifier_failed_count: int
    last_exit_code: int | None
    evaluator_passed: bool
    final_answer_written: bool
    last_error_signature: str | None = None
    token_budget_fraction: float | None = None
    wall_time_budget_fraction: float | None = None
    unconfirmed_pass: bool = False
    """A pass was claimed by an untrusted verifier but not independently confirmed."""


@dataclass(frozen=True, slots=True)
class ControlDecision:
    """Structured controller decision applied by the outer runtime loop."""

    action: ControlAction
    reason: str
    strategy: str | None = None
    defer_command: bool = True
    """For a FORCE_VERIFY action, whether to skip the agent's proposed command.

    True when the agent believes it is finished (an unconfirmed pass or a written
    final answer): the proposed command is moot, so we verify instead. False for a
    proactive/staleness check: the proposed command is still useful work, so the
    loop should execute it and verify alongside rather than discard it."""


@dataclass(frozen=True, slots=True)
class StoppingControllerConfig:
    """Thresholds for budget-aware verification and stopping."""

    max_steps_without_verifier: int = 4
    max_steps_without_progress: int = 6
    repeated_error_threshold: int = 2
    token_budget: int | None = None
    wall_time_budget_s: float | None = None
    require_independent_verification: bool = True
    """When True, a self-reported (untrusted) pass triggers FORCE_VERIFY rather
    than finalizing — closing the false-local-pass failure mode."""


@dataclass(slots=True)
class ExecutionStateTracker:
    """Accumulate real execution events and expose controller features.

    Now incremental: observe() maintains running totals, last_* step markers,
    and trailing repeat counts so that features() is O(1) per call instead of
    O(#events) rescans/sums. The events list is still kept (for transcripts and
    replay), but the expensive per-step rescans in _steps_since / _trailing / sums
    are eliminated.
    """

    events: list[ExecutionEvent] = field(default_factory=list)
    # Incremental aggregates (maintained in observe)
    total_tokens: int = 0
    elapsed_s: float = 0.0
    current_step: int = 0
    last_verifier_step: int = 0
    last_progress_step: int = 0
    verifier_failed_count: int = 0
    has_trusted_pass: bool = False
    has_claimed_pass: bool = False
    has_final_answer: bool = False
    last_exit_code: int | None = None
    last_error_signature: str | None = None
    repeated_error_count: int = 0
    _trailing_sig: str | None = None  # for incremental repeated_error tracking

    def observe(self, event: ExecutionEvent) -> ExecutionStateFeatures:
        """Record one event (incremental) and return updated features."""

        self.events.append(event)
        self.total_tokens += event.tokens_used or 0
        self.elapsed_s += event.elapsed_s or 0.0
        self.current_step = max(self.current_step, event.step)

        if event.verifier_run:
            self.last_verifier_step = event.step
            if event.verifier_passed is False:
                self.verifier_failed_count += 1
        if event.progress_made:
            self.last_progress_step = event.step
        # Mirror "latest = events[-1]" semantics exactly for these fields
        self.last_exit_code = event.exit_code
        self.last_error_signature = event.error_signature

        if event.final_answer_written:
            self.has_final_answer = True
        if event.verifier_passed is True:
            self.has_claimed_pass = True
            if event.trusted:
                self.has_trusted_pass = True

        # Incremental trailing repeated errors (over events, matching old _trailing_repeated_errors)
        new_sig = event.error_signature
        if new_sig is None:
            self.repeated_error_count = 0
            self._trailing_sig = None
        else:
            if new_sig == self._trailing_sig:
                self.repeated_error_count += 1
            else:
                self.repeated_error_count = 1
            self._trailing_sig = new_sig

        return self.features()

    def features(
        self,
        *,
        token_budget: int | None = None,
        wall_time_budget_s: float | None = None,
    ) -> ExecutionStateFeatures:
        """Extract controller features from observed events (now O(1))."""

        if not self.events:
            return ExecutionStateFeatures(
                step_count=0,
                total_tokens=0,
                elapsed_s=0.0,
                steps_since_verifier=0,
                steps_since_progress=0,
                repeated_error_count=0,
                verifier_failed_count=0,
                last_exit_code=None,
                evaluator_passed=False,
                final_answer_written=False,
                token_budget_fraction=_budget_fraction(0, token_budget),
                wall_time_budget_fraction=_budget_fraction(0.0, wall_time_budget_s),
                unconfirmed_pass=False,
            )

        # All heavy work (sums, any-scans, step rescans) is maintained incrementally in observe().
        steps_since_verifier = max(0, self.current_step - self.last_verifier_step)
        steps_since_progress = max(0, self.current_step - self.last_progress_step)

        return ExecutionStateFeatures(
            step_count=len(self.events),
            total_tokens=self.total_tokens,
            elapsed_s=self.elapsed_s,
            steps_since_verifier=steps_since_verifier,
            steps_since_progress=steps_since_progress,
            repeated_error_count=self.repeated_error_count,
            verifier_failed_count=self.verifier_failed_count,
            last_exit_code=self.last_exit_code,
            evaluator_passed=self.has_trusted_pass,
            final_answer_written=self.has_final_answer,
            last_error_signature=self.last_error_signature,
            token_budget_fraction=_budget_fraction(self.total_tokens, token_budget),
            wall_time_budget_fraction=_budget_fraction(self.elapsed_s, wall_time_budget_s),
            unconfirmed_pass=self.has_claimed_pass and not self.has_trusted_pass,
        )


@dataclass(slots=True)
class StoppingController:
    """Deterministic policy for continuing, verifying, finalizing, or switching."""

    config: StoppingControllerConfig = field(default_factory=StoppingControllerConfig)

    def decide(self, features: ExecutionStateFeatures) -> ControlDecision:
        """Choose the next mechanical action for a real execution loop."""

        if features.evaluator_passed:
            return ControlDecision("FINALIZE", "independent verifier passed")
        if self.config.require_independent_verification and features.unconfirmed_pass:
            return ControlDecision(
                "FORCE_VERIFY", "confirm self-reported pass with an independent check"
            )
        if features.final_answer_written:
            if self.config.require_independent_verification and not features.evaluator_passed:
                return ControlDecision(
                    "FORCE_VERIFY", "verify final answer before finalizing"
                )
            return ControlDecision("FINALIZE", "final answer already written")
        if (
            self.config.token_budget is not None
            and features.total_tokens >= self.config.token_budget
        ):
            return ControlDecision("FINALIZE", "token budget reached")
        if (
            self.config.wall_time_budget_s is not None
            and features.elapsed_s >= self.config.wall_time_budget_s
        ):
            return ControlDecision("FINALIZE", "wall-clock budget reached")
        if features.repeated_error_count >= self.config.repeated_error_threshold:
            return ControlDecision("SWITCH_STRATEGY", "same error repeated")
        # Proactive checks below run *alongside* the agent's command (defer_command
        # =False) so a routine verification never discards useful in-flight work.
        if features.steps_since_verifier >= self.config.max_steps_without_verifier:
            return ControlDecision(
                "FORCE_VERIFY", "verification is stale", defer_command=False
            )
        if features.steps_since_progress >= self.config.max_steps_without_progress:
            return ControlDecision(
                "FORCE_VERIFY", "progress is stale", defer_command=False
            )
        return ControlDecision("CONTINUE", "within execution budget")


@dataclass(slots=True)
class RuntimeRewardLogger:
    """Log real outcomes and update category-conditioned routing bandits."""

    bandit: CategoryBanditRoutingPolicy
    jsonl_path: Path | None = None
    rows: list[dict[str, object]] = field(default_factory=list)

    def record(
        self,
        *,
        task_id: str,
        category: str,
        strategy: str,
        passed: bool,
        token_savings: float,
        quality_score: float | None = None,
        features: ExecutionStateFeatures | None = None,
        action: str | None = None,
        outcome: Mapping[str, object] | None = None,
        regression_risk: float = 0.0,
    ) -> dict[str, object]:
        """Record one real task outcome and update the bandit arm (now with regression risk)."""

        self.bandit.update(
            category,
            strategy,
            passed=passed,
            token_savings=token_savings,
            quality_score=quality_score,
            regression_risk=regression_risk,
        )
        row: dict[str, object] = {
            "task_id": task_id,
            "category": category,
            "strategy": strategy,
            "action": action or "COMPLETE",
            "passed": passed,
            "token_savings": token_savings,
            "quality_score": quality_score,
            "regression_risk": regression_risk,
            "outcome": {
                "passed": passed,
                "token_savings": token_savings,
                "quality_score": quality_score,
                "regression_risk": regression_risk,
                **dict(outcome or {}),
            },
            "bandit_values": self.bandit.values(category),
        }
        if features is not None:
            row["state"] = execution_features_to_dict(features)
            row["features"] = row["state"]
        self.rows.append(row)
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row


def execution_features_to_dict(features: ExecutionStateFeatures) -> dict[str, object]:
    """Serialize execution-state features for training logs and reports."""

    return {
        "step_count": features.step_count,
        "total_tokens": features.total_tokens,
        "elapsed_s": features.elapsed_s,
        "steps_since_verifier": features.steps_since_verifier,
        "steps_since_progress": features.steps_since_progress,
        "repeated_error_count": features.repeated_error_count,
        "verifier_failed_count": features.verifier_failed_count,
        "last_exit_code": features.last_exit_code,
        "evaluator_passed": features.evaluator_passed,
        "final_answer_written": features.final_answer_written,
        "last_error_signature": features.last_error_signature,
        "token_budget_fraction": features.token_budget_fraction,
        "wall_time_budget_fraction": features.wall_time_budget_fraction,
        "unconfirmed_pass": features.unconfirmed_pass,
    }


def _budget_fraction(used: float, budget: float | None) -> float | None:
    if budget is None or budget <= 0:
        return None
    return min(max(used / budget, 0.0), 1.0)


def _steps_since(events: list[ExecutionEvent], predicate: object) -> int:
    test = predicate  # keep mypy happy with callability narrowed below
    if not callable(test):
        raise TypeError("predicate must be callable")
    # Count distinct steps, not events: a single step may record several events
    # (for example an executed command verified alongside it), and a step counts as
    # matched if any of its events satisfy the predicate.
    matched_by_step: dict[int, bool] = {}
    for event in events:
        matched_by_step[event.step] = matched_by_step.get(event.step, False) or bool(
            test(event)
        )
    count = 0
    for step in sorted(matched_by_step, reverse=True):
        if matched_by_step[step]:
            return count
        count += 1
    return count


def _trailing_repeated_errors(events: list[ExecutionEvent]) -> int:
    signature = events[-1].error_signature
    if signature is None:
        return 0
    count = 0
    for event in reversed(events):
        if event.error_signature != signature:
            break
        count += 1
    return count
