"""LangGraph-first runtime wrapper for AgentProp."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentprop.core import AgentGraph
from agentprop.integrations.framework_adapters import graph_from_langgraph_object
from agentprop.runtime.control_loop import ControlDecision, ExecutionEvent
from agentprop.runtime.session import ControlSession


@dataclass(frozen=True, slots=True)
class ControlledRunResult:
    """Result returned by ``wrap(...).run(...)``."""

    result: Any
    decision_trace: tuple[dict[str, Any], ...]
    cost_actual: float
    verifier_log: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class ControlledLangGraph:
    """Dependency-light wrapper around a LangGraph workflow object."""

    workflow: Any
    graph: AgentGraph
    budget: Mapping[str, float] = field(default_factory=dict)
    trace_path: Path | None = None
    session_factory: Callable[[str], ControlSession] | None = None

    def run(self, input_data: Any = None, **kwargs: Any) -> ControlledRunResult:
        """Run the wrapped workflow and emit an AgentProp decision trace."""

        run_id = str(kwargs.pop("run_id", "langgraph-run"))
        session = self._session(run_id)
        result = _invoke_workflow(self.workflow, input_data, kwargs)
        event = _event_from_result(result)
        decision = session.observe(event)
        self._write_trace_if_configured(session)
        return _controlled_result(result, session.trace_rows, decision)

    async def arun(self, input_data: Any = None, **kwargs: Any) -> ControlledRunResult:
        """Async variant for LangGraph ``ainvoke`` workflows."""

        run_id = str(kwargs.pop("run_id", "langgraph-run"))
        session = self._session(run_id)
        result = await _ainvoke_workflow(self.workflow, input_data, kwargs)
        event = _event_from_result(result)
        decision = session.observe(event)
        self._write_trace_if_configured(session)
        return _controlled_result(result, session.trace_rows, decision)

    def invoke(self, input_data: Any = None, **kwargs: Any) -> Any:
        """LangGraph-compatible invoke surface returning the raw workflow result."""

        return self.run(input_data, **kwargs).result

    async def ainvoke(self, input_data: Any = None, **kwargs: Any) -> Any:
        """LangGraph-compatible async invoke surface returning the raw result."""

        return (await self.arun(input_data, **kwargs)).result

    def _session(self, run_id: str) -> ControlSession:
        if self.session_factory is not None:
            return self.session_factory(run_id)
        return ControlSession.start(
            self.graph,
            task_id=run_id,
            token_budget=int(self.budget["tokens"]) if "tokens" in self.budget else None,
            baseline_tokens=int(self.budget["tokens"]) if "tokens" in self.budget else None,
            trace_path=self.trace_path,
        )

    def _write_trace_if_configured(self, session: ControlSession) -> None:
        if self.trace_path is not None:
            session.save_trace(self.trace_path)


def wrap(
    workflow: Any,
    *,
    budget: Mapping[str, float] | None = None,
    trace_path: str | Path | None = None,
) -> ControlledLangGraph:
    """Wrap a LangGraph workflow with AgentProp control and trace capture."""

    graph = workflow if isinstance(workflow, AgentGraph) else graph_from_langgraph_object(workflow)
    return ControlledLangGraph(
        workflow=workflow,
        graph=graph,
        budget=dict(budget or {}),
        trace_path=Path(trace_path) if trace_path is not None else None,
    )


def _invoke_workflow(workflow: Any, input_data: Any, kwargs: dict[str, Any]) -> Any:
    if hasattr(workflow, "invoke"):
        return workflow.invoke(input_data, **kwargs)
    if callable(workflow):
        if input_data is None:
            return workflow(**kwargs)
        return workflow(input_data, **kwargs)
    return workflow


async def _ainvoke_workflow(workflow: Any, input_data: Any, kwargs: dict[str, Any]) -> Any:
    if hasattr(workflow, "ainvoke"):
        return await workflow.ainvoke(input_data, **kwargs)
    result = _invoke_workflow(workflow, input_data, kwargs)
    if inspect.isawaitable(result):
        awaited: Awaitable[Any] = result
        return await awaited
    return result


def _event_from_result(result: Any) -> ExecutionEvent:
    tokens = 0
    if isinstance(result, Mapping):
        raw_tokens = result.get("tokens") or result.get("tokens_used") or result.get("cost_tokens")
        if raw_tokens is not None:
            tokens = int(raw_tokens)
    return ExecutionEvent(
        step=1,
        command="langgraph.invoke",
        exit_code=0,
        progress_made=True,
        tokens_used=tokens,
        final_answer_written=True,
        verifier_run=False,
    )


def _controlled_result(
    result: Any,
    trace_rows: list[dict[str, Any]],
    decision: ControlDecision,
) -> ControlledRunResult:
    verifier_log = tuple(
        row
        for row in trace_rows
        if isinstance(row.get("event"), dict) and row["event"].get("verifier_run")
    )
    tokens = 0.0
    for row in trace_rows:
        event = row.get("event")
        if isinstance(event, dict):
            tokens += float(event.get("tokens_used") or 0)
    trace = tuple(json.loads(json.dumps(row, sort_keys=True)) for row in trace_rows)
    if not verifier_log and decision.action == "FORCE_VERIFY":
        verifier_log = ({"action": decision.action, "reason": decision.reason},)
    return ControlledRunResult(
        result=result,
        decision_trace=trace,
        cost_actual=tokens,
        verifier_log=verifier_log,
    )
