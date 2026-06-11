"""User-facing control sessions for analysis-backed runtime execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentprop.algorithms import bottleneck_nodes, low_weight_edges, risk_aware_verifier_placement
from agentprop.core import AgentGraph
from agentprop.core.dynamic_graph import DynamicGraphSession
from agentprop.ml.risk_predictors import TimeoutRiskPredictor
from agentprop.rl import CategoryBanditRoutingPolicy
from agentprop.rl.graph_features import reward_record_graph_features
from agentprop.runtime.control_loop import (
    ControlDecision,
    ExecutionEvent,
    ExecutionStateFeatures,
    ExecutionStateTracker,
    RuntimeRewardLogger,
    StoppingController,
    StoppingControllerConfig,
    execution_features_to_dict,
)
from agentprop.workflows import WORKFLOW_TEMPLATES


@dataclass(frozen=True, slots=True)
class ControlSessionConfig:
    """Configuration for a user-facing AgentProp control session."""

    workflow: str | AgentGraph
    task_id: str
    category: str = "general"
    token_budget: int | None = None
    wall_time_budget_s: float | None = None
    baseline_tokens: int | None = None
    trace_path: str | Path | None = None
    max_steps_without_verifier: int = 4
    max_steps_without_progress: int = 6
    repeated_error_threshold: int = 2
    require_independent_verification: bool = True
    seed_budget: int = 2


@dataclass(frozen=True, slots=True)
class ControlAnalysis:
    """Small graph-analysis snapshot that travels with a control session."""

    workflow_name: str
    node_count: int
    edge_count: int
    bottlenecks: tuple[tuple[str, float], ...]
    pruning_candidates: tuple[tuple[str, str], ...]
    verifier_candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize the analysis snapshot."""

        return {
            "workflow": self.workflow_name,
            "nodes": self.node_count,
            "edges": self.edge_count,
            "bottlenecks": [
                {"node": node_id, "score": score} for node_id, score in self.bottlenecks
            ],
            "pruning_candidates": [
                {"source": source, "target": target}
                for source, target in self.pruning_candidates
            ],
            "verifier_candidates": list(self.verifier_candidates),
        }


@dataclass(slots=True)
class ControlSession:
    """Analysis-backed facade for wrapping real agent execution.

    A session records observed execution events, exposes the controller's next
    decision, and writes a portable trace/report that can be shared with users.
    """

    config: ControlSessionConfig
    graph: AgentGraph = field(init=False)
    workflow_name: str = field(init=False)
    analysis: ControlAnalysis = field(init=False)
    controller: StoppingController = field(init=False)
    tracker: ExecutionStateTracker = field(default_factory=ExecutionStateTracker)
    decisions: list[ControlDecision] = field(default_factory=list)
    trace_rows: list[dict[str, object]] = field(default_factory=list)
    outcome: dict[str, object] | None = None
    _reward_logger: RuntimeRewardLogger = field(init=False, repr=False)
    _dynamic: DynamicGraphSession | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        workflow_name, graph = _load_session_workflow(self.config.workflow)
        self.workflow_name = workflow_name
        self.graph = graph
        self.analysis = _analyze_graph(
            graph,
            workflow_name=workflow_name,
            seed_budget=self.config.seed_budget,
        )
        self.controller = StoppingController(
            StoppingControllerConfig(
                max_steps_without_verifier=self.config.max_steps_without_verifier,
                max_steps_without_progress=self.config.max_steps_without_progress,
                repeated_error_threshold=self.config.repeated_error_threshold,
                token_budget=self.config.token_budget,
                wall_time_budget_s=self.config.wall_time_budget_s,
                require_independent_verification=self.config.require_independent_verification,
            )
        )
        self._reward_logger = RuntimeRewardLogger(
            CategoryBanditRoutingPolicy(
                arms=("agentprop_controller", "baseline"),
                epsilon=0.0,
                default_arm="agentprop_controller",
            ),
            graph_context=reward_record_graph_features(
                graph,
                verifiers=tuple(self.analysis.verifier_candidates),
            ),
        )
        self._record(
            "analysis",
            {
                "task_id": self.config.task_id,
                "category": self.config.category,
                "baseline_tokens": self.config.baseline_tokens,
                "token_budget": self.config.token_budget,
                "wall_time_budget_s": self.config.wall_time_budget_s,
                "analysis": self.analysis.to_dict(),
            },
        )

    @classmethod
    def start(
        cls,
        workflow: str | AgentGraph,
        *,
        task_id: str,
        category: str = "general",
        token_budget: int | None = None,
        wall_time_budget_s: float | None = None,
        baseline_tokens: int | None = None,
        trace_path: str | Path | None = None,
    ) -> ControlSession:
        """Create a session with the common public options."""

        return cls(
            ControlSessionConfig(
                workflow=workflow,
                task_id=task_id,
                category=category,
                token_budget=token_budget,
                wall_time_budget_s=wall_time_budget_s,
                baseline_tokens=baseline_tokens,
                trace_path=trace_path,
            )
        )

    @property
    def dynamic(self) -> DynamicGraphSession | None:
        """Runtime graph mutation session, when enabled."""

        return self._dynamic

    def enable_dynamic_graph(self) -> DynamicGraphSession:
        """Start tracking runtime add/remove/conditional edge mutations."""

        self._dynamic = DynamicGraphSession(base_graph=self.graph)
        return self._dynamic

    def effective_graph(self, context: Mapping[str, Any] | None = None) -> AgentGraph:
        """Return the working graph, optionally filtering conditional edges."""

        if self._dynamic is not None:
            return self._dynamic.active_graph(context)
        if context:
            return self.graph.filter_active_edges(context)
        return self.graph

    def mutate_add_node(self, node_id: str, **metadata: Any) -> None:
        if self._dynamic is None:
            self.enable_dynamic_graph()
        assert self._dynamic is not None
        self._dynamic.add_node(node_id, **metadata)
        self._sync_graph_from_dynamic()

    def mutate_remove_node(self, node_id: str) -> None:
        if self._dynamic is None:
            self.enable_dynamic_graph()
        assert self._dynamic is not None
        self._dynamic.remove_node(node_id)
        self._sync_graph_from_dynamic()

    def mutate_add_edge(self, source: str, target: str, **metadata: Any) -> None:
        if self._dynamic is None:
            self.enable_dynamic_graph()
        assert self._dynamic is not None
        self._dynamic.add_edge(source, target, **metadata)
        self._sync_graph_from_dynamic()

    def mutate_add_conditional_edge(
        self,
        source: str,
        target: str,
        *,
        condition_key: str,
        condition_value: object,
        **metadata: Any,
    ) -> None:
        if self._dynamic is None:
            self.enable_dynamic_graph()
        assert self._dynamic is not None
        self._dynamic.add_conditional_edge(
            source,
            target,
            condition_key=condition_key,
            condition_value=condition_value,
            **metadata,
        )
        self._sync_graph_from_dynamic()

    def mutate_remove_edge(self, source: str, target: str) -> None:
        if self._dynamic is None:
            self.enable_dynamic_graph()
        assert self._dynamic is not None
        self._dynamic.remove_edge(source, target)
        self._sync_graph_from_dynamic()

    def observe(self, event: ExecutionEvent) -> ControlDecision:
        """Record one execution event and return the next control decision."""

        features = self.tracker.observe(event)  # incremental: O(1) features, no full history rescan
        decision = self.controller.decide(features)
        self.decisions.append(decision)
        self._record(
            "event",
            {
                "event": _event_to_dict(event),
                "features": execution_features_to_dict(features),
                "decision": _decision_to_dict(decision),
            },
        )
        return decision

    def decide(self) -> ControlDecision:
        """Return the current control decision without adding a new event."""

        features = self._features()
        decision = self.controller.decide(features)
        self.decisions.append(decision)
        self._record(
            "decision",
            {
                "features": execution_features_to_dict(features),
                "decision": _decision_to_dict(decision),
            },
        )
        return decision

    def record_outcome(
        self,
        *,
        passed: bool,
        strategy: str = "agentprop_controller",
        quality_score: float | None = None,
        metadata: dict[str, object] | None = None,
        regression_risk: float = 0.0,
        timeout_risk: float | None = None,
    ) -> dict[str, object]:
        """Record the final outcome and update the session's reward row.

        regression_risk (simple 0-1 signal) can be supplied by the caller after
        consulting an ExpectedSuccessProfile built from trace_loader empirical
        rows. It flows into the bandit update so the policy learns to avoid
        strategies that historically caused regressions.
        """

        features = self._features()
        if timeout_risk is None:
            timeout_risk = TimeoutRiskPredictor().predict(
                self.graph,
                activated_nodes={node.id for node in self.graph.nodes()},
                wall_time_budget_s=self.config.wall_time_budget_s,
                observed_elapsed_s=features.elapsed_s,
            )
        quality_loss = None if quality_score is None else max(0.0, 1.0 - quality_score)
        reward_row = self._reward_logger.record(
            task_id=self.config.task_id,
            category=self.config.category,
            strategy=strategy,
            passed=passed,
            token_savings=_token_savings(
                baseline_tokens=self.config.baseline_tokens,
                observed_tokens=features.total_tokens,
            ),
            quality_score=quality_score,
            features=features,
            action=self.decisions[-1].action if self.decisions else None,
            outcome=metadata,
            regression_risk=regression_risk,
            timeout_risk=timeout_risk,
            quality_loss=quality_loss,
        )
        self.outcome = reward_row
        self._record("outcome", reward_row)
        return reward_row

    def save_trace(self, path: str | Path | None = None) -> Path:
        """Write the session trace as JSONL and return the path."""

        output = Path(path or self.config.trace_path or "agentprop-control-trace.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in self.trace_rows) + "\n",
            encoding="utf-8",
        )
        return output

    def summary(self) -> dict[str, object]:
        """Return a compact summary for CLI/MCP users."""

        features = self._features()
        decision_counts: dict[str, int] = {}
        for decision in self.decisions:
            decision_counts[decision.action] = decision_counts.get(decision.action, 0) + 1
        return {
            "task_id": self.config.task_id,
            "category": self.config.category,
            "workflow": self.workflow_name,
            "analysis": self.analysis.to_dict(),
            "event_count": len(self.tracker.events),
            "decision_counts": decision_counts,
            "latest_decision": (
                _decision_to_dict(self.decisions[-1]) if self.decisions else None
            ),
            "features": execution_features_to_dict(features),
            "outcome": self.outcome,
        }

    def render_report(self) -> str:
        """Render a concise Markdown report for the wrapped execution."""

        summary = self.summary()
        latest = summary["latest_decision"]
        latest_action = latest["action"] if isinstance(latest, dict) else "none"
        features = summary["features"]
        if not isinstance(features, dict):
            features = {}
        lines = [
            "# AgentProp Control Session Report",
            "",
            f"- Task: `{self.config.task_id}`",
            f"- Category: `{self.config.category}`",
            f"- Workflow: `{self.workflow_name}`",
            f"- Events observed: `{summary['event_count']}`",
            f"- Latest decision: `{latest_action}`",
            "",
            "## Analysis Snapshot",
            f"- Nodes: `{self.analysis.node_count}`",
            f"- Edges: `{self.analysis.edge_count}`",
            f"- Verifier candidates: `{_join_or_none(self.analysis.verifier_candidates)}`",
            "- Bottlenecks: "
            f"`{_join_or_none(node for node, _ in self.analysis.bottlenecks[:3])}`",
            "- Pruning candidates: "
            f"`{_join_or_none(f'{s}->{t}' for s, t in self.analysis.pruning_candidates[:3])}`",
            "",
            "## Runtime Features",
            f"- Tokens used: `{features.get('total_tokens', 0)}`",
            f"- Elapsed seconds: `{features.get('elapsed_s', 0.0)}`",
            f"- Steps since verifier: `{features.get('steps_since_verifier', 0)}`",
            f"- Repeated errors: `{features.get('repeated_error_count', 0)}`",
            f"- Unconfirmed pass: `{features.get('unconfirmed_pass', False)}`",
            "",
            "## Decisions",
            *_decision_lines(self.decisions),
            "",
        ]
        if self.outcome is not None:
            lines.extend(
                [
                    "## Outcome",
                    f"- Passed: `{self.outcome.get('passed')}`",
                    f"- Token savings: `{self.outcome.get('token_savings')}`",
                    "",
                ]
            )
        return "\n".join(lines)

    def write_artifacts(self, out_dir: str | Path) -> dict[str, Path]:
        """Write trace, summary, and Markdown report into a directory."""

        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        trace_path = self.save_trace(output / "trace.jsonl")
        summary_path = output / "summary.json"
        report_path = output / "report.md"
        summary_path.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(self.render_report(), encoding="utf-8")
        return {"trace": trace_path, "summary": summary_path, "report": report_path}

    def _features(self) -> ExecutionStateFeatures:
        return self.tracker.features(
            token_budget=self.config.token_budget,
            wall_time_budget_s=self.config.wall_time_budget_s,
        )

    def _sync_graph_from_dynamic(self) -> None:
        assert self._dynamic is not None
        self.graph = self._dynamic.graph
        self.graph.warm_analysis_cache()
        self.analysis = _analyze_graph(
            self.graph,
            workflow_name=self.workflow_name,
            seed_budget=self.config.seed_budget,
        )
        self._record(
            "graph_mutation",
            {
                "version": self._dynamic.version,
                "mutations": self._dynamic.mutations_to_dict(),
            },
        )

    def _record(self, row_type: str, payload: dict[str, object]) -> None:
        row = {"type": row_type, **payload}
        self.trace_rows.append(row)


class AsyncControlSession:
    """Async wrapper around :class:`ControlSession` for use in async agent loops.

    All public methods mirror the sync API but are awaitable.  The underlying
    state machine remains synchronous; each call is dispatched via
    :func:`asyncio.to_thread` so the event loop is never blocked and no nested
    ``asyncio.run()`` calls are needed.

    Usage::

        session = await AsyncControlSession.start(
            "planner_coder_tester_reviewer",
            task_id="my-task",
        )
        decision = await session.observe(event)
    """

    def __init__(self, inner: ControlSession) -> None:
        self._inner = inner

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    async def start(
        cls,
        workflow: str | AgentGraph,
        *,
        task_id: str,
        category: str = "general",
        token_budget: int | None = None,
        wall_time_budget_s: float | None = None,
        baseline_tokens: int | None = None,
        trace_path: str | Path | None = None,
    ) -> AsyncControlSession:
        """Async constructor — mirrors :meth:`ControlSession.start`."""
        inner = await asyncio.to_thread(
            ControlSession.start,
            workflow,
            task_id=task_id,
            category=category,
            token_budget=token_budget,
            wall_time_budget_s=wall_time_budget_s,
            baseline_tokens=baseline_tokens,
            trace_path=trace_path,
        )
        return cls(inner)

    # ------------------------------------------------------------------
    # Core async interface
    # ------------------------------------------------------------------

    async def observe(self, event: ExecutionEvent) -> ControlDecision:
        """Async version of :meth:`ControlSession.observe`."""
        return await asyncio.to_thread(self._inner.observe, event)

    async def decide(self) -> ControlDecision:
        """Async version of :meth:`ControlSession.decide`."""
        return await asyncio.to_thread(self._inner.decide)

    async def record_outcome(
        self,
        *,
        passed: bool,
        strategy: str = "agentprop_controller",
        quality_score: float | None = None,
        metadata: dict[str, object] | None = None,
        regression_risk: float = 0.0,
        timeout_risk: float | None = None,
    ) -> dict[str, object]:
        """Async version of :meth:`ControlSession.record_outcome`."""
        return await asyncio.to_thread(
            self._inner.record_outcome,
            passed=passed,
            strategy=strategy,
            quality_score=quality_score,
            metadata=metadata,
            regression_risk=regression_risk,
            timeout_risk=timeout_risk,
        )

    async def write_artifacts(self, out_dir: str | Path) -> dict[str, Path]:
        """Async version of :meth:`ControlSession.write_artifacts`."""
        return await asyncio.to_thread(self._inner.write_artifacts, out_dir)

    async def save_trace(self, path: str | Path | None = None) -> Path:
        """Async version of :meth:`ControlSession.save_trace`."""
        return await asyncio.to_thread(self._inner.save_trace, path)

    # ------------------------------------------------------------------
    # Pass-through properties / sync helpers
    # ------------------------------------------------------------------

    @property
    def config(self) -> ControlSessionConfig:
        return self._inner.config

    @property
    def analysis(self) -> ControlAnalysis:
        return self._inner.analysis

    @property
    def decisions(self) -> list[ControlDecision]:
        return self._inner.decisions

    def summary(self) -> dict[str, object]:
        return self._inner.summary()

    def render_report(self) -> str:
        return self._inner.render_report()

    def enable_dynamic_graph(self) -> Any:
        return self._inner.enable_dynamic_graph()

    @property
    def dynamic(self) -> Any:
        return self._inner.dynamic

    def effective_graph(self, context: Mapping[str, Any] | None = None) -> AgentGraph:
        return self._inner.effective_graph(context)

    async def mutate_add_node(self, node_id: str, **metadata: Any) -> None:
        await asyncio.to_thread(self._inner.mutate_add_node, node_id, **metadata)

    async def mutate_remove_node(self, node_id: str) -> None:
        await asyncio.to_thread(self._inner.mutate_remove_node, node_id)

    async def mutate_add_edge(self, source: str, target: str, **metadata: Any) -> None:
        await asyncio.to_thread(self._inner.mutate_add_edge, source, target, **metadata)

    async def mutate_add_conditional_edge(
        self,
        source: str,
        target: str,
        *,
        condition_key: str,
        condition_value: object,
        **metadata: Any,
    ) -> None:
        await asyncio.to_thread(
            self._inner.mutate_add_conditional_edge,
            source,
            target,
            condition_key=condition_key,
            condition_value=condition_value,
            **metadata,
        )

    async def mutate_remove_edge(self, source: str, target: str) -> None:
        await asyncio.to_thread(self._inner.mutate_remove_edge, source, target)


def _load_session_workflow(workflow: str | AgentGraph) -> tuple[str, AgentGraph]:
    if isinstance(workflow, AgentGraph):
        return "custom", workflow
    if workflow in WORKFLOW_TEMPLATES:
        return workflow, WORKFLOW_TEMPLATES[workflow]()
    path = Path(workflow)
    return path.stem, AgentGraph.from_json(path)


def _analyze_graph(
    graph: AgentGraph,
    *,
    workflow_name: str,
    seed_budget: int,
) -> ControlAnalysis:
    return ControlAnalysis(
        workflow_name=workflow_name,
        node_count=graph.node_count,
        edge_count=graph.edge_count,
        bottlenecks=tuple(bottleneck_nodes(graph)),
        pruning_candidates=tuple(low_weight_edges(graph)),
        verifier_candidates=tuple(
            risk_aware_verifier_placement(graph, min(seed_budget, graph.node_count))
        ),
    )


def _event_to_dict(event: ExecutionEvent) -> dict[str, object]:
    return {
        "step": event.step,
        "command": event.command,
        "exit_code": event.exit_code,
        "verifier_run": event.verifier_run,
        "verifier_passed": event.verifier_passed,
        "progress_made": event.progress_made,
        "tokens_used": event.tokens_used,
        "elapsed_s": event.elapsed_s,
        "error_signature": event.error_signature,
        "final_answer_written": event.final_answer_written,
        "trusted": event.trusted,
    }


def _decision_to_dict(decision: ControlDecision) -> dict[str, object]:
    return {
        "action": decision.action,
        "reason": decision.reason,
        "strategy": decision.strategy,
        "defer_command": decision.defer_command,
    }


def _token_savings(*, baseline_tokens: int | None, observed_tokens: int) -> float:
    if baseline_tokens is None or baseline_tokens <= 0:
        return 0.0
    return (baseline_tokens - observed_tokens) / baseline_tokens


def _join_or_none(values: Iterable[object] | str) -> str:
    if isinstance(values, str):
        return values
    items = [str(value) for value in values]
    return ", ".join(items) if items else "none"


def _decision_lines(decisions: list[ControlDecision]) -> list[str]:
    if not decisions:
        return ["- No decisions recorded."]
    return [
        f"- `{index}` `{decision.action}`: {decision.reason}"
        for index, decision in enumerate(decisions, start=1)
    ]
