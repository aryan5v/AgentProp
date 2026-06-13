"""Assign each sub-task to the cheapest capable model (optimized task-sharing).

This is the core efficiency lever over Fusion: instead of every model answering
the whole task, each sub-task goes to one model chosen by capability + cost +
(optionally) a learned policy. Capability is a hard constraint from
``ModelSpec``; among the candidates that qualify, a
``ContextualThompsonSamplingPolicy`` over graph-position features picks the arm
when trained, otherwise we fall back to cheapest-capable.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentprop.council.model_pool import ModelPool
from agentprop.council.planner import Plan, SubTask
from agentprop.rl.contextual_thompson import ContextualThompsonSamplingPolicy

_DIFFICULTY_SCORE = {"easy": 0.0, "medium": 0.5, "hard": 1.0}


@dataclass(frozen=True, slots=True)
class Assignment:
    """One sub-task routed to one model."""

    subtask_id: str
    model: str
    reason: str


def subtask_features(subtask: SubTask, depth: int) -> dict[str, float]:
    """Context vector for the routing policy (matches reward-log schema v2 spirit)."""

    return {
        "depth": float(depth),
        "difficulty": _DIFFICULTY_SCORE.get(subtask.difficulty, 0.5),
        "needs_search": 1.0 if subtask.needs_search else 0.0,
        "fan_in": float(len(subtask.depends_on)),
    }


@dataclass(slots=True)
class Assigner:
    """Route plan sub-tasks to pool models under capability + cost + policy."""

    policy: ContextualThompsonSamplingPolicy | None = None

    def assign(self, plan: Plan, pool: ModelPool) -> list[Assignment]:
        depth_of = _depths(plan)
        assignments: list[Assignment] = []
        for sub in plan.subtasks:
            candidates = pool.candidates(
                min_tier=sub.min_tier, required_tags=sub.required_tags
            )
            if not candidates:
                # Relax tags before tier: better to retrieve-less than to fail.
                candidates = pool.candidates(min_tier=sub.min_tier)
            if not candidates:
                candidates = pool.candidates()
            if not candidates:
                raise ValueError("model pool has no usable models for assignment")
            candidate_names = [spec.name for spec in candidates]
            if self.policy is not None and set(candidate_names) <= set(self.policy.arms):
                features = subtask_features(sub, depth_of[sub.id])
                chosen = self.policy.choose(features)
                if chosen in candidate_names:
                    assignments.append(
                        Assignment(sub.id, chosen, "contextual-thompson")
                    )
                    continue
            # Cold start / unconstrained policy: cheapest capable model.
            assignments.append(
                Assignment(sub.id, candidate_names[0], "cheapest-capable")
            )
        return assignments


def _depths(plan: Plan) -> dict[str, int]:
    depth: dict[str, int] = {}
    by_id = {s.id: s for s in plan.subtasks}

    def resolve(sid: str, seen: frozenset[str]) -> int:
        if sid in depth:
            return depth[sid]
        sub = by_id.get(sid)
        if sub is None or not sub.depends_on or sid in seen:
            depth[sid] = 0
            return 0
        d = 1 + max(resolve(p, seen | {sid}) for p in sub.depends_on if p in by_id)
        depth[sid] = d
        return d

    for sub in plan.subtasks:
        resolve(sub.id, frozenset())
    return depth
