"""Durable JSONL storage for control-session execution state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentprop.runtime.control_loop import (
    ControlDecision,
    ExecutionEvent,
    ExecutionStateFeatures,
    ExecutionStateTracker,
    StoppingController,
    StoppingControllerConfig,
    execution_features_to_dict,
)
from agentprop.runtime.observability import Scrubber, scrub_event
from agentprop.runtime.session import _decision_to_dict, _event_to_dict


@dataclass(frozen=True, slots=True)
class RunState:
    """Persisted controller state reconstructed at a run boundary."""

    run_id: str
    events: tuple[ExecutionEvent, ...]
    features: ExecutionStateFeatures
    decisions: tuple[ControlDecision, ...] = ()
    bandit_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the snapshot."""

        return {
            "run_id": self.run_id,
            "events": [_event_to_dict(event) for event in self.events],
            "features": execution_features_to_dict(self.features),
            "decisions": [_decision_to_dict(decision) for decision in self.decisions],
            "bandit_state": self.bandit_state,
        }


class JSONLEventStore:
    """Append-only event store with one JSONL file per run."""

    def __init__(
        self,
        root: str | Path,
        *,
        scrubber: Scrubber | None = None,
    ) -> None:
        self.root = Path(root)
        self.scrubber = scrubber

    def append(
        self,
        run_id: str,
        event: ExecutionEvent,
        *,
        decision: ControlDecision | None = None,
        features: ExecutionStateFeatures | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Append one event row after applying redaction."""

        self.root.mkdir(parents=True, exist_ok=True)
        output = self.path_for(run_id)
        safe_event = scrub_event(event, self.scrubber)
        row: dict[str, Any] = {
            "type": "event",
            "run_id": run_id,
            "event": _event_to_dict(safe_event),
        }
        if decision is not None:
            row["decision"] = _decision_to_dict(decision)
        if features is not None:
            row["features"] = execution_features_to_dict(features)
        if metadata:
            row["metadata"] = metadata
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return output

    def load_events(self, run_id: str) -> tuple[ExecutionEvent, ...]:
        """Load all events for a run."""

        path = self.path_for(run_id)
        if not path.exists():
            return ()
        events: list[ExecutionEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row.get("event")
            if isinstance(payload, dict):
                events.append(_event_from_dict(payload))
        return tuple(events)

    def snapshot(self, run_id: str) -> RunState:
        """Reconstruct a ``RunState`` from persisted events."""

        events = self.load_events(run_id)
        tracker = ExecutionStateTracker()
        for event in events:
            tracker.observe(event)
        return RunState(run_id=run_id, events=events, features=tracker.features())

    def path_for(self, run_id: str) -> Path:
        """Return the JSONL path for ``run_id``."""

        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_id)
        return self.root / f"{safe}.jsonl"


class DurableController:
    """Stopping controller that persists and resumes execution history."""

    def __init__(
        self,
        store: JSONLEventStore,
        *,
        run_id: str,
        config: StoppingControllerConfig | None = None,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.controller = StoppingController(config or StoppingControllerConfig())
        self.tracker = ExecutionStateTracker(list(store.load_events(run_id)))

    @classmethod
    def resume(
        cls,
        run_id: str,
        *,
        store: JSONLEventStore,
        config: StoppingControllerConfig | None = None,
    ) -> DurableController:
        """Resume a controller from the append-only log."""

        return cls(store, run_id=run_id, config=config)

    def observe(
        self,
        event: ExecutionEvent,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ControlDecision:
        """Record one event, persist state, and return the next decision."""

        features = self.tracker.observe(event)
        decision = self.controller.decide(features)
        self.store.append(
            self.run_id,
            event,
            decision=decision,
            features=features,
            metadata=metadata,
        )
        return decision

    def snapshot(self) -> RunState:
        """Return an in-memory run snapshot."""

        return RunState(
            run_id=self.run_id,
            events=tuple(self.tracker.events),
            features=self.tracker.features(),
        )


Controller = DurableController


def _event_from_dict(payload: dict[str, Any]) -> ExecutionEvent:
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
