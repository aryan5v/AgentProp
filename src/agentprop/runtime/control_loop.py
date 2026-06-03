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
    """Accumulate real execution events and expose controller features."""

    events: list[ExecutionEvent] = field(default_factory=list)

    def observe(self, event: ExecutionEvent) -> ExecutionStateFeatures:
        """Record one event and return updated features."""

        self.events.append(event)
        return self.features()

    def features(
        self,
        *,
        token_budget: int | None = None,
        wall_time_budget_s: float | None = None,
    ) -> ExecutionStateFeatures:
        """Extract controller features from observed events."""

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

        latest = self.events[-1]
        total_tokens = sum(event.tokens_used for event in self.events)
        elapsed_s = sum(event.elapsed_s for event in self.events)
        trusted_pass = any(
            event.verifier_passed is True and event.trusted for event in self.events
        )
        claimed_pass = any(event.verifier_passed is True for event in self.events)
        return ExecutionStateFeatures(
            step_count=len(self.events),
            total_tokens=total_tokens,
            elapsed_s=elapsed_s,
            steps_since_verifier=_steps_since(self.events, lambda event: event.verifier_run),
            steps_since_progress=_steps_since(self.events, lambda event: event.progress_made),
            repeated_error_count=_trailing_repeated_errors(self.events),
            verifier_failed_count=sum(
                1 for event in self.events if event.verifier_run and event.verifier_passed is False
            ),
            last_exit_code=latest.exit_code,
            evaluator_passed=trusted_pass,
            final_answer_written=any(event.final_answer_written for event in self.events),
            last_error_signature=latest.error_signature,
            token_budget_fraction=_budget_fraction(total_tokens, token_budget),
            wall_time_budget_fraction=_budget_fraction(elapsed_s, wall_time_budget_s),
            unconfirmed_pass=claimed_pass and not trusted_pass,
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
    ) -> dict[str, object]:
        """Record one real task outcome and update the bandit arm."""

        self.bandit.update(
            category,
            strategy,
            passed=passed,
            token_savings=token_savings,
            quality_score=quality_score,
        )
        row: dict[str, object] = {
            "task_id": task_id,
            "category": category,
            "strategy": strategy,
            "action": action or "COMPLETE",
            "passed": passed,
            "token_savings": token_savings,
            "quality_score": quality_score,
            "outcome": {
                "passed": passed,
                "token_savings": token_savings,
                "quality_score": quality_score,
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
    count = 0
    for event in reversed(events):
        if test(event):
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
