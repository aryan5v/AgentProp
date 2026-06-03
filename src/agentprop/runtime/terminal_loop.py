"""Per-command terminal control loop for real agent execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

from agentprop.rl import CategoryBanditRoutingPolicy
from agentprop.runtime.control_loop import (
    ControlDecision,
    ExecutionEvent,
    ExecutionStateFeatures,
    ExecutionStateTracker,
    RuntimeRewardLogger,
    StoppingController,
)


class TerminalCommandProposer(Protocol):
    """Propose the next terminal command without executing it."""

    def __call__(self, request: TerminalTurnRequest) -> TerminalCommandProposal:
        """Return a candidate command for the current task state."""


class TerminalCommandExecutor(Protocol):
    """Execute a command that AgentProp allowed through the control gate."""

    def __call__(
        self,
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
    ) -> TerminalCommandResult:
        """Execute the proposed command and return the observed event."""


class TerminalVerifier(Protocol):
    """Run a verifier when AgentProp decides verification is required."""

    def __call__(
        self,
        request: TerminalTurnRequest,
        blocked_proposal: TerminalCommandProposal | None = None,
    ) -> TerminalCommandResult:
        """Run verification instead of the pending command."""


class TerminalStrategySwitcher(Protocol):
    """Choose a new strategy after AgentProp requests a strategy switch."""

    def __call__(
        self,
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
        decision: ControlDecision,
    ) -> str:
        """Return the next strategy name."""


@dataclass(frozen=True, slots=True)
class TerminalTurnRequest:
    """State visible before one terminal command is proposed or executed."""

    task: str
    step: int
    strategy: str
    features: ExecutionStateFeatures
    transcript: tuple[ExecutionEvent, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TerminalCommandProposal:
    """A terminal command proposed by an agent but not yet executed."""

    command: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TerminalCommandResult:
    """Observed result from an executed terminal command or verifier."""

    event: ExecutionEvent
    stdout: str = ""
    stderr: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TerminalLoopConfig:
    """Execution settings for a controlled terminal loop."""

    max_steps: int = 64
    default_strategy: str = "agentprop_controller"
    fallback_strategy: str = "baseline"
    task_id: str | None = None
    category: str | None = None
    baseline_tokens: int | None = None
    quality_score: float | None = None
    explore: bool = True
    """Whether bandit strategy selection may explore. Set False when scoring
    held-out tasks so a graded run exploits the learned policy instead of risking
    a random exploration pick."""


@dataclass(frozen=True, slots=True)
class TerminalLoopResult:
    """Complete trace from a command-gated terminal loop."""

    strategy: str
    decisions: tuple[ControlDecision, ...]
    proposals: tuple[TerminalCommandProposal, ...]
    events: tuple[ExecutionEvent, ...]
    stdout: str
    stderr: str
    passed: bool | None
    features: ExecutionStateFeatures
    reward_row: Mapping[str, object] | None = None


@dataclass(slots=True)
class ControlledTerminalLoop:
    """Gate every proposed terminal command through AgentProp control."""

    controller: StoppingController = field(default_factory=StoppingController)
    config: TerminalLoopConfig = field(default_factory=TerminalLoopConfig)
    bandit: CategoryBanditRoutingPolicy | None = None
    reward_logger: RuntimeRewardLogger | None = None

    def run(
        self,
        *,
        task: str,
        proposer: TerminalCommandProposer,
        executor: TerminalCommandExecutor,
        verifier: TerminalVerifier | None = None,
        strategy_switcher: TerminalStrategySwitcher | None = None,
        initial_events: tuple[ExecutionEvent, ...] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> TerminalLoopResult:
        """Run a terminal agent while AgentProp controls every command boundary."""

        tracker = _tracker_from_events(initial_events)
        decisions: list[ControlDecision] = []
        proposals: list[TerminalCommandProposal] = []
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        strategy = self._initial_strategy()
        run_metadata = dict(metadata or {})

        for step in range(1, self.config.max_steps + 1):
            request = TerminalTurnRequest(
                task=task,
                step=step,
                strategy=strategy,
                features=self._features(tracker),
                transcript=tuple(tracker.events),
                metadata=run_metadata,
            )
            proposal = proposer(request)
            proposals.append(proposal)
            decision = self.controller.decide(request.features)
            decisions.append(decision)

            if decision.action == "FINALIZE":
                break
            if decision.action == "FORCE_VERIFY":
                if verifier is None:
                    # A deferred verify confirms a claimed completion; with no
                    # verifier available the claim can never be cleared, so stop
                    # rather than executing commands until the budget is exhausted.
                    # A proactive (non-deferred) check just falls through to run the
                    # proposed command this step.
                    if decision.defer_command:
                        break
                else:
                    # A proactive (non-deferred) check keeps the agent's proposed
                    # work: run it first, then verify alongside instead of discarding.
                    if not decision.defer_command:
                        work = executor(request, proposal)
                        self._observe_result(tracker, work, stdout_parts, stderr_parts)
                        # Refresh the request so the verifier sees the command it ran.
                        request = replace(
                            request,
                            features=self._features(tracker),
                            transcript=tuple(tracker.events),
                        )
                    result = verifier(request, proposal)
                    # Guarantee the forced verification counts as a verifier run so
                    # the staleness counter resets and we do not trigger a verify-
                    # every-step storm if the harness omits verifier_run on its result.
                    result = _as_verifier_run(result)
                    self._observe_result(tracker, result, stdout_parts, stderr_parts)
                    continue
            if decision.action == "SWITCH_STRATEGY":
                strategy = self._switch_strategy(
                    request=request,
                    proposal=proposal,
                    decision=decision,
                    strategy_switcher=strategy_switcher,
                )
                tracker.observe(
                    ExecutionEvent(
                        step=step,
                        command="agentprop:switch_strategy",
                        progress_made=True,
                    )
                )
                continue

            result = executor(request, proposal)
            self._observe_result(tracker, result, stdout_parts, stderr_parts)

        features = self._features(tracker)
        passed = self._resolve_passed(features, tracker.events)
        reward_row = self._record_reward(
            strategy=strategy,
            passed=passed,
            features=features,
            action=decisions[-1].action if decisions else None,
        )
        return TerminalLoopResult(
            strategy=strategy,
            decisions=tuple(decisions),
            proposals=tuple(proposals),
            events=tuple(tracker.events),
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            passed=passed,
            features=features,
            reward_row=reward_row,
        )

    def _initial_strategy(self) -> str:
        if self.bandit is not None and self.config.category is not None:
            return self.bandit.choose(self.config.category, explore=self.config.explore)
        return self.config.default_strategy

    def _switch_strategy(
        self,
        *,
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
        decision: ControlDecision,
        strategy_switcher: TerminalStrategySwitcher | None,
    ) -> str:
        if strategy_switcher is not None:
            return strategy_switcher(request, proposal, decision)
        return decision.strategy or self.config.fallback_strategy

    def _observe_result(
        self,
        tracker: ExecutionStateTracker,
        result: TerminalCommandResult | None,
        stdout_parts: list[str],
        stderr_parts: list[str],
    ) -> None:
        if result is None:
            raise RuntimeError("terminal executor returned None")
        tracker.observe(result.event)
        stdout_parts.append(result.stdout)
        stderr_parts.append(result.stderr)

    def _record_reward(
        self,
        *,
        strategy: str,
        passed: bool | None,
        features: ExecutionStateFeatures,
        action: str | None,
    ) -> Mapping[str, object] | None:
        if (
            self.reward_logger is None
            or self.config.task_id is None
            or self.config.category is None
            or passed is None
        ):
            return None
        return self.reward_logger.record(
            task_id=self.config.task_id,
            category=self.config.category,
            strategy=strategy,
            passed=passed,
            token_savings=_token_savings(
                baseline_tokens=self.config.baseline_tokens,
                observed_tokens=features.total_tokens,
            ),
            quality_score=self.config.quality_score,
            features=features,
            action=action,
            outcome={
                "total_tokens": features.total_tokens,
                "elapsed_s": features.elapsed_s,
            },
        )

    def _features(self, tracker: ExecutionStateTracker) -> ExecutionStateFeatures:
        return tracker.features(
            token_budget=self.controller.config.token_budget,
            wall_time_budget_s=self.controller.config.wall_time_budget_s,
        )

    def _resolve_passed(
        self,
        features: ExecutionStateFeatures,
        events: list[ExecutionEvent],
    ) -> bool | None:
        """Decide the final outcome without trusting an unconfirmed pass.

        A trusted/independent verifier pass finalizes as a pass. When
        ``require_independent_verification`` is on, the fallback ignores
        self-reported (untrusted) results so a false-local-pass is never
        recorded as a success; otherwise it uses the last verifier result.
        """

        if features.evaluator_passed:
            return True
        trusted_only = self.controller.config.require_independent_verification
        return _last_verifier_result(events, trusted_only=trusted_only)


def _tracker_from_events(events: tuple[ExecutionEvent, ...]) -> ExecutionStateTracker:
    tracker = ExecutionStateTracker()
    for event in events:
        tracker.observe(event)
    return tracker


def _as_verifier_run(result: TerminalCommandResult) -> TerminalCommandResult:
    """Ensure a forced-verification result is recorded as a verifier run.

    Without this, a harness that returns a verifier result with ``verifier_run``
    left False never resets ``steps_since_verifier``, so the controller forces a
    verification on every subsequent step — the dominant cost regression observed
    for the gated arm.
    """

    if result.event.verifier_run:
        return result
    return TerminalCommandResult(
        event=replace(result.event, verifier_run=True),
        stdout=result.stdout,
        stderr=result.stderr,
        metadata=result.metadata,
    )


def _last_verifier_result(
    events: list[ExecutionEvent], *, trusted_only: bool = False
) -> bool | None:
    for event in reversed(events):
        if event.verifier_passed is None:
            continue
        if trusted_only and not event.trusted:
            continue
        return event.verifier_passed
    return None


def _token_savings(*, baseline_tokens: int | None, observed_tokens: int) -> float:
    if baseline_tokens is None or baseline_tokens <= 0:
        return 0.0
    return (baseline_tokens - observed_tokens) / baseline_tokens
