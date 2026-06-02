"""Provider-neutral AgentProp control loop for real agent execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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


class AgentTurnExecutor(Protocol):
    """Execute one agent turn under the current AgentProp strategy."""

    def __call__(self, request: AgentTurnRequest) -> AgentTurnResult:
        """Run one agent turn and return the observed event."""


class AgentLoopVerifier(Protocol):
    """Run an external verifier when AgentProp requests a check."""

    def __call__(self, request: AgentTurnRequest) -> AgentTurnResult:
        """Run a verifier turn and return the observed event."""


class AgentStrategySwitcher(Protocol):
    """Choose a new strategy after the controller asks to switch."""

    def __call__(self, request: AgentTurnRequest, decision: ControlDecision) -> str:
        """Return the next strategy name."""


@dataclass(frozen=True, slots=True)
class AgentTurnRequest:
    """Concrete state visible to one controlled agent or verifier turn."""

    task: str
    step: int
    strategy: str
    features: ExecutionStateFeatures
    transcript: tuple[ExecutionEvent, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    """Observed result from one controlled agent or verifier turn."""

    event: ExecutionEvent
    output: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentLoopConfig:
    """Execution settings for a controlled agent loop."""

    max_steps: int = 64
    default_strategy: str = "agentprop_controller"
    fallback_strategy: str = "baseline"
    task_id: str | None = None
    category: str | None = None
    baseline_tokens: int | None = None
    quality_score: float | None = None


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    """Complete trace from a controlled agent loop."""

    strategy: str
    decisions: tuple[ControlDecision, ...]
    events: tuple[ExecutionEvent, ...]
    final_output: str
    passed: bool | None
    features: ExecutionStateFeatures
    reward_row: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class AgentLoopDecision:
    """One controller decision over the current observed loop state."""

    strategy: str
    decision: ControlDecision
    request: AgentTurnRequest


@dataclass(slots=True)
class ControlledAgentLoop:
    """Wrap a real agent loop with AgentProp stopping, verification, and rewards."""

    controller: StoppingController = field(default_factory=StoppingController)
    config: AgentLoopConfig = field(default_factory=AgentLoopConfig)
    bandit: CategoryBanditRoutingPolicy | None = None
    reward_logger: RuntimeRewardLogger | None = None

    def decide(
        self,
        *,
        task: str,
        initial_events: tuple[ExecutionEvent, ...] = (),
        strategy: str | None = None,
        step: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> AgentLoopDecision:
        """Inspect observed events and return the next control decision."""

        tracker = _tracker_from_events(initial_events)
        active_strategy = strategy or self._initial_strategy()
        features = self._features(tracker)
        request = AgentTurnRequest(
            task=task,
            step=step or (len(initial_events) + 1),
            strategy=active_strategy,
            features=features,
            transcript=tuple(tracker.events),
            metadata=dict(metadata or {}),
        )
        return AgentLoopDecision(
            strategy=active_strategy,
            decision=self.controller.decide(request.features),
            request=request,
        )

    def run(
        self,
        *,
        task: str,
        turn_executor: AgentTurnExecutor,
        verifier: AgentLoopVerifier | None = None,
        strategy_switcher: AgentStrategySwitcher | None = None,
        initial_events: tuple[ExecutionEvent, ...] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> AgentLoopResult:
        """Run agent turns while AgentProp controls verify/switch/finalize decisions."""

        tracker = _tracker_from_events(initial_events)
        decisions: list[ControlDecision] = []
        final_output = ""
        strategy = self._initial_strategy()
        run_metadata = dict(metadata or {})

        for step in range(1, self.config.max_steps + 1):
            features = self._features(tracker)
            decision = self.controller.decide(features)
            decisions.append(decision)
            request = AgentTurnRequest(
                task=task,
                step=step,
                strategy=strategy,
                features=features,
                transcript=tuple(tracker.events),
                metadata=run_metadata,
            )

            if decision.action == "FINALIZE":
                break
            if decision.action == "FORCE_VERIFY" and verifier is not None:
                result = verifier(request)
                self._observe_result(tracker, result)
                final_output = result.output or final_output
                continue
            if decision.action == "SWITCH_STRATEGY":
                strategy = self._switch_strategy(
                    request=request,
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

            result = turn_executor(request)
            self._observe_result(tracker, result)
            final_output = result.output or final_output

        features = self._features(tracker)
        passed = True if features.evaluator_passed else _last_verifier_result(tracker.events)
        reward_row = self._record_reward(
            strategy=strategy,
            passed=passed,
            features=features,
            action=decisions[-1].action if decisions else None,
        )
        return AgentLoopResult(
            strategy=strategy,
            decisions=tuple(decisions),
            events=tuple(tracker.events),
            final_output=final_output,
            passed=passed,
            features=features,
            reward_row=reward_row,
        )

    def _initial_strategy(self) -> str:
        if self.bandit is not None and self.config.category is not None:
            return self.bandit.choose(self.config.category)
        return self.config.default_strategy

    def _switch_strategy(
        self,
        *,
        request: AgentTurnRequest,
        decision: ControlDecision,
        strategy_switcher: AgentStrategySwitcher | None,
    ) -> str:
        if strategy_switcher is not None:
            return strategy_switcher(request, decision)
        return decision.strategy or self.config.fallback_strategy

    def _observe_result(
        self,
        tracker: ExecutionStateTracker,
        result: AgentTurnResult | None,
    ) -> None:
        if result is None:
            raise RuntimeError("agent loop executor returned None")
        tracker.observe(result.event)

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


def _last_verifier_result(events: list[ExecutionEvent]) -> bool | None:
    for event in reversed(events):
        if event.verifier_passed is not None:
            return event.verifier_passed
    return None


def _tracker_from_events(events: tuple[ExecutionEvent, ...]) -> ExecutionStateTracker:
    tracker = ExecutionStateTracker()
    for event in events:
        tracker.observe(event)
    return tracker


def _token_savings(*, baseline_tokens: int | None, observed_tokens: int) -> float:
    if baseline_tokens is None or baseline_tokens <= 0:
        return 0.0
    return (baseline_tokens - observed_tokens) / baseline_tokens
