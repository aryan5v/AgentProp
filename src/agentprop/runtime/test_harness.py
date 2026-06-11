"""Deterministic replay harness for controller decision traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from agentprop.runtime.control_loop import (
    ControlDecision,
    ExecutionEvent,
    StoppingController,
    StoppingControllerConfig,
)
from agentprop.runtime.session import _decision_to_dict, _event_to_dict


@dataclass(frozen=True, slots=True)
class TestHarness:
    """Replay ``ExecutionEvent`` rows through ``StoppingController``."""

    __test__ = False

    events: tuple[ExecutionEvent, ...]
    decisions: tuple[ControlDecision, ...]

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        config: StoppingControllerConfig | None = None,
    ) -> TestHarness:
        """Load events from a trace JSONL file and replay them deterministically."""

        events = tuple(_event_from_row(row) for row in _read_jsonl(Path(path)))
        return cls.replay(events, config=config)

    @classmethod
    def from_fixture(
        cls,
        name: str,
        *,
        config: StoppingControllerConfig | None = None,
    ) -> TestHarness:
        """Load a bundled reference trace by name."""

        if not name.endswith(".jsonl"):
            name = f"{name}.jsonl"
        fixture = resources.files("agentprop.fixtures.traces").joinpath(name)
        with resources.as_file(fixture) as path:
            return cls.from_jsonl(path, config=config)

    @classmethod
    def replay(
        cls,
        events: tuple[ExecutionEvent, ...] | list[ExecutionEvent],
        *,
        config: StoppingControllerConfig | None = None,
    ) -> TestHarness:
        """Replay in-memory events through a fresh controller."""

        controller = StoppingController(config or StoppingControllerConfig())
        from agentprop.runtime.control_loop import ExecutionStateTracker

        tracker = ExecutionStateTracker()
        decisions: list[ControlDecision] = []
        for event in events:
            decisions.append(controller.decide(tracker.observe(event)))
        return cls(events=tuple(events), decisions=tuple(decisions))

    def decision_at_step(self, step: int) -> ControlDecision:
        """Return the decision after the event with ``event.step == step``."""

        for event, decision in zip(self.events, self.decisions, strict=True):
            if event.step == step:
                return decision
        raise IndexError(f"No decision recorded at step {step}")

    def decision_actions(self) -> tuple[str, ...]:
        """Return decision actions for ergonomic assertions."""

        return tuple(decision.action for decision in self.decisions)

    def finalized_on_confirmed_pass(self) -> bool:
        """Return true when a trusted verifier pass leads to FINALIZE."""

        for event, decision in zip(self.events, self.decisions, strict=True):
            if event.verifier_passed is True and event.trusted and decision.action == "FINALIZE":
                return True
        return False

    def verify_count(self) -> int:
        """Count FORCE_VERIFY decisions."""

        return sum(1 for decision in self.decisions if decision.action == "FORCE_VERIFY")

    def to_jsonl(self) -> str:
        """Serialize replayed events and decisions."""

        rows = []
        for event, decision in zip(self.events, self.decisions, strict=True):
            rows.append(
                {
                    "type": "event",
                    "event": _event_to_dict(event),
                    "decision": _decision_to_dict(decision),
                }
            )
        return "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _event_from_row(row: dict[str, Any]) -> ExecutionEvent:
    payload = row.get("event", row)
    if not isinstance(payload, dict):
        raise ValueError("Trace row must contain an event object")
    return ExecutionEvent(
        step=int(payload.get("step", 0)),
        command=payload.get("command"),
        exit_code=payload.get("exit_code"),
        verifier_run=bool(payload.get("verifier_run", False)),
        verifier_passed=payload.get("verifier_passed"),
        progress_made=bool(payload.get("progress_made", False)),
        tokens_used=int(payload.get("tokens_used") or 0),
        elapsed_s=float(payload.get("elapsed_s") or 0.0),
        error_signature=payload.get("error_signature"),
        final_answer_written=bool(payload.get("final_answer_written", False)),
        trusted=bool(payload.get("trusted", True)),
    )
