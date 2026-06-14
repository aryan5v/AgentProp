"""Decompose a task into a sub-task DAG (the Council's plan).

The plan is the graph: sub-tasks are nodes, dependencies are edges. Each
sub-task declares the capability it needs (minimum tier, tags like ``search``)
so the assignment policy can route it to the cheapest *capable* model rather
than running every model on the whole task. The planner emits a confidence; a
low-confidence decomposition triggers the Council's ensemble fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentprop.core import AgentGraph
from agentprop.core.types import NodeType

_PLANNER_SYSTEM = """You are a research planner. Decompose the user's task into \
a minimal DAG of concrete sub-questions. Each sub-question must be independently \
answerable and, together, fully cover the task. Return STRICT JSON:
{
  "subtasks": [
    {"id": "s1", "question": "...", "depends_on": [], "needs_search": true,
     "difficulty": "easy|medium|hard"}
  ],
  "synthesis_instruction": "how to combine the sub-answers into a final report",
  "confidence": 0.0-1.0
}
Use as few sub-questions as the task genuinely needs. Mark depends_on only for \
true dependencies. confidence reflects how well this decomposition covers the task."""

_DIFFICULTY_TIER = {"easy": 1, "medium": 2, "hard": 3}


@dataclass(frozen=True, slots=True)
class SubTask:
    """One node of the plan DAG."""

    id: str
    question: str
    depends_on: tuple[str, ...] = ()
    needs_search: bool = False
    difficulty: str = "medium"

    @property
    def min_tier(self) -> int:
        return _DIFFICULTY_TIER.get(self.difficulty, 2)

    @property
    def required_tags(self) -> tuple[str, ...]:
        return ("search",) if self.needs_search else ()


@dataclass(frozen=True, slots=True)
class Plan:
    """A decomposition: sub-task DAG + how to synthesize + confidence."""

    task: str
    subtasks: tuple[SubTask, ...]
    synthesis_instruction: str = ""
    confidence: float = 1.0

    def graph(self) -> AgentGraph:
        """Materialize the plan as an AgentGraph: input -> subtasks -> output."""

        graph = AgentGraph()
        graph.add_node("input", type=NodeType.MEMORY, role="task")
        for sub in self.subtasks:
            graph.add_node(
                sub.id,
                type=NodeType.AGENT,
                role="subtask",
                name=sub.question[:60],
                difficulty=sub.difficulty,
                needs_search=sub.needs_search,
            )
            if sub.depends_on:
                for parent in sub.depends_on:
                    graph.add_edge(parent, sub.id, weight=1.0)
            else:
                graph.add_edge("input", sub.id, weight=1.0)
        graph.add_node("synthesizer", type=NodeType.AGENT, role="synthesizer")
        for sub in self.subtasks:
            graph.add_edge(sub.id, "synthesizer", weight=1.0)
        graph.add_node("output", type=NodeType.OUTPUT, role="output")
        graph.add_edge("synthesizer", "output", weight=1.0)
        return graph


def parse_plan(task: str, raw: str) -> Plan:
    """Parse a planner model's JSON response into a Plan (tolerant of fences)."""

    payload = _loads_lenient(raw)
    subtasks: list[SubTask] = []
    for index, item in enumerate(payload.get("subtasks", []) or []):
        if not isinstance(item, dict) or not item.get("question"):
            continue
        subtasks.append(
            SubTask(
                id=str(item.get("id") or f"s{index + 1}"),
                question=str(item["question"]),
                depends_on=tuple(str(d) for d in item.get("depends_on", []) or []),
                needs_search=bool(item.get("needs_search", False)),
                difficulty=str(item.get("difficulty", "medium")),
            )
        )
    if not subtasks:
        # Degenerate: treat the whole task as one sub-question.
        subtasks = [SubTask(id="s1", question=task, needs_search=True)]
        confidence = 0.0
    else:
        try:
            confidence = float(payload.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    return Plan(
        task=task,
        subtasks=tuple(subtasks),
        synthesis_instruction=str(payload.get("synthesis_instruction", "")),
        confidence=confidence,
    )


@dataclass(slots=True)
class LLMPlanner:
    """Decompose a task by calling a planner model in the pool."""

    model: str
    system_prompt: str = _PLANNER_SYSTEM
    temperature: float = 0.2
    last_cost_usd: float = field(default=0.0, init=False)

    def decompose(self, pool: object, task: str) -> Plan:
        """Call ``pool.call(model, ...)`` and parse the plan."""

        response = pool.call(  # type: ignore[attr-defined]
            self.model,
            system_prompt=self.system_prompt,
            user_prompt=task,
            temperature=self.temperature,
        )
        self.last_cost_usd = getattr(response, "cost_usd", 0.0)
        return parse_plan(task, getattr(response, "text", ""))


def _loads_lenient(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
