"""The Council orchestrator: decompose, assign, verify, synthesize — supervised.

One ``Council.run(task)`` call:
1. decomposes the task into a sub-task DAG (planner);
2. if the decomposition is confident, **assigns** each sub-task to the cheapest
   capable model and runs them in parallel; otherwise falls back to a
   **Fusion-style ensemble** (every pool model answers the whole task);
3. **claim-checks** each sub-answer and quarantines unsupported ones;
4. **quality-weighted synthesizes** the survivors into a final answer;
5. supervises the whole run under a ``ControlSession`` (token/time budget,
   schema-v2 reward logging) and emits an ``ExecutionEvent`` per stage.

It is benchmark-agnostic: the same object runs research, coding, or QA — only
the pool, retrieval tool, and (downstream) scorer change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentprop.council.assignment import Assigner
from agentprop.council.claim_check import CheckedSubAnswer, ClaimChecker
from agentprop.council.model_pool import ModelPool, ModelResponse
from agentprop.council.planner import LLMPlanner, Plan, SubTask
from agentprop.council.retrieval import NullRetrieval, RetrievalTool
from agentprop.council.synthesizer import Synthesizer
from agentprop.propagation.quality_cascade import QualityCascade
from agentprop.runtime.control_loop import ExecutionEvent
from agentprop.runtime.session import ControlSession

_SUBTASK_SYSTEM = """You are a research specialist answering ONE focused \
sub-question. Be precise, cite sources for every factual claim, and clearly \
state uncertainty. Do not answer beyond the sub-question."""


@dataclass(frozen=True, slots=True)
class CouncilResult:
    """Final answer plus full cost/latency/provenance accounting."""

    task: str
    answer: str
    mode: str
    """``assign`` (decompose-and-assign) or ``ensemble`` (Fusion fallback)."""
    subtask_count: int
    quarantined_count: int
    total_cost_usd: float
    wall_latency_s: float
    total_tokens: int
    citations: tuple[str, ...]
    plan_confidence: float
    trace: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "answer": self.answer,
            "mode": self.mode,
            "subtask_count": self.subtask_count,
            "quarantined_count": self.quarantined_count,
            "total_cost_usd": self.total_cost_usd,
            "wall_latency_s": self.wall_latency_s,
            "total_tokens": self.total_tokens,
            "citations": list(self.citations),
            "plan_confidence": self.plan_confidence,
            "trace": self.trace,
        }


@dataclass(slots=True)
class Council:
    """Supervise a model pool as one graph for a multi-step task."""

    pool: ModelPool
    planner: LLMPlanner
    synthesizer: Synthesizer
    assigner: Assigner = field(default_factory=Assigner)
    claim_checker: ClaimChecker = field(default_factory=ClaimChecker)
    retrieval: RetrievalTool = field(default_factory=NullRetrieval)
    confidence_threshold: float = 0.4
    ensemble_models: tuple[str, ...] | None = None
    """Models used in ensemble fallback; defaults to the whole pool."""
    token_budget: int | None = None
    subtask_system: str = _SUBTASK_SYSTEM

    def run(self, task: str, *, task_id: str = "council-task") -> CouncilResult:
        plan = self.planner.decompose(self.pool, task)
        if plan.confidence < self.confidence_threshold:
            return self._run_ensemble(task, task_id, plan)
        return self._run_assigned(task, task_id, plan)

    # -- decompose-and-assign (the headline path) --------------------------

    def _run_assigned(self, task: str, task_id: str, plan: Plan) -> CouncilResult:
        graph = plan.graph()
        session = ControlSession.start(
            graph, task_id=task_id, category="council", token_budget=self.token_budget
        )
        quality = QualityCascade().simulate(graph, ["input"]).node_qualities

        assignments = self.assigner.assign(plan, self.pool)
        by_id = {s.id: s for s in plan.subtasks}
        calls = []
        for assignment in assignments:
            sub = by_id[assignment.subtask_id]
            retrieval = self.retrieval.for_subtask(sub.question)
            calls.append((assignment, sub, retrieval.extra_body))

        responses = self.pool.map_assignments(
            [
                (a.model, self.subtask_system, sub.question)
                for a, sub, _ in calls
            ],
            extra_body=None,
        )
        # Re-run only the search-augmented sub-tasks with their plugin body when
        # retrieval differs per sub-task. (extra_body is per-call; map_assignments
        # shares one body, so apply retrieval individually when enabled.)
        resolved: list[tuple[SubTask, ModelResponse]] = []
        for (assignment, sub, extra_body), resp in zip(calls, responses, strict=True):
            if extra_body:
                resp = self.pool.call(
                    assignment.model,
                    system_prompt=self.subtask_system,
                    user_prompt=sub.question,
                    extra_body=extra_body,
                )
            resolved.append((sub, resp))

        checked: list[CheckedSubAnswer] = []
        for step, (sub, resp) in enumerate(resolved, start=1):
            result = self.claim_checker.check(sub, resp)
            checked.append(result)
            session.observe(
                ExecutionEvent(
                    step=step,
                    command=f"subtask:{sub.id}:{resp.model}",
                    tokens_used=resp.usage.total_tokens,
                    elapsed_s=resp.latency_s,
                    progress_made=resp.ok and not result.quarantined,
                    error_signature=None if resp.ok else "model_error",
                    verifier_run=True,
                    verifier_passed=not result.quarantined,
                    trusted=True,
                )
            )

        synth = self.synthesizer.synthesize(
            self.pool, task, checked, quality=quality
        )
        session.observe(
            ExecutionEvent(
                step=len(resolved) + 1,
                command=f"synthesize:{synth.model}",
                tokens_used=0,
                elapsed_s=synth.latency_s,
                progress_made=True,
                final_answer_written=True,
            )
        )
        return self._finalize(
            task,
            "assign",
            plan,
            resolved,
            checked,
            synth,
            session,
            planner_cost=self.planner.last_cost_usd,
        )

    # -- ensemble fallback (Fusion parity) ---------------------------------

    def _run_ensemble(self, task: str, task_id: str, plan: Plan) -> CouncilResult:
        models = self.ensemble_models or tuple(s.name for s in self.pool.specs)
        graph = plan.graph()
        session = ControlSession.start(
            graph, task_id=task_id, category="council-ensemble",
            token_budget=self.token_budget,
        )
        retrieval = self.retrieval.for_subtask(task)
        responses = self.pool.fan_out(
            list(models),
            system_prompt=self.subtask_system,
            user_prompt=task,
            extra_body=retrieval.extra_body or None,
        )
        whole = SubTask(id="whole", question=task, needs_search=True)
        checked: list[CheckedSubAnswer] = []
        for step, (model, resp) in enumerate(responses.items(), start=1):
            result = CheckedSubAnswer(
                subtask_id=model,
                model=model,
                text=resp.text,
                risk=self.claim_checker.risk_fn(whole, resp),
                quarantined=False,
                citations=resp.citations,
            )
            checked.append(result)
            session.observe(
                ExecutionEvent(
                    step=step,
                    command=f"ensemble:{model}",
                    tokens_used=resp.usage.total_tokens,
                    elapsed_s=resp.latency_s,
                    progress_made=resp.ok,
                )
            )
        synth = self.synthesizer.synthesize(self.pool, task, checked)
        session.observe(
            ExecutionEvent(
                step=len(responses) + 1,
                command=f"synthesize:{synth.model}",
                elapsed_s=synth.latency_s,
                progress_made=True,
                final_answer_written=True,
            )
        )
        resolved = [(whole, responses[m]) for m in responses]
        return self._finalize(
            task, "ensemble", plan, resolved, checked, synth, session,
            planner_cost=self.planner.last_cost_usd,
        )

    # -- shared finalize ----------------------------------------------------

    def _finalize(
        self,
        task: str,
        mode: str,
        plan: Plan,
        resolved: list[tuple[SubTask, ModelResponse]],
        checked: list[CheckedSubAnswer],
        synth: object,
        session: ControlSession,
        *,
        planner_cost: float,
    ) -> CouncilResult:
        subtask_cost = sum(r.cost_usd for _, r in resolved)
        subtask_tokens = sum(r.usage.total_tokens for _, r in resolved)
        synth_cost = getattr(synth, "cost_usd", 0.0)
        wall = max((r.latency_s for _, r in resolved), default=0.0) + getattr(
            synth, "latency_s", 0.0
        )
        quarantined = sum(1 for c in checked if c.quarantined)
        session.record_outcome(
            passed=True,
            quality_score=1.0 - (quarantined / max(len(checked), 1)),
            metadata={"mode": mode, "subtasks": len(checked)},
        )
        return CouncilResult(
            task=task,
            answer=getattr(synth, "text", ""),
            mode=mode,
            subtask_count=len(checked),
            quarantined_count=quarantined,
            total_cost_usd=planner_cost + subtask_cost + synth_cost,
            wall_latency_s=wall,
            total_tokens=subtask_tokens,
            citations=getattr(synth, "citations", ()),
            plan_confidence=plan.confidence,
            trace=list(session.trace_rows),
        )
