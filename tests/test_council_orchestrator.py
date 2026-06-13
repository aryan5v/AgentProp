"""Integration tests for the Council orchestrator (no network)."""

from __future__ import annotations

import json

from agentprop.council import (
    Assigner,
    ClaimChecker,
    Council,
    LLMPlanner,
    ModelPool,
    ModelSpec,
    OpenRouterWebSearch,
    SubTask,
    parse_plan,
)
from agentprop.council.assignment import subtask_features
from agentprop.evaluation.llm_execution import LLMExecutionResult, LLMUsage

PLAN_JSON = {
    "subtasks": [
        {"id": "s1", "question": "What is X?", "depends_on": [], "needs_search": True,
         "difficulty": "easy"},
        {"id": "s2", "question": "Compare X and Y", "depends_on": ["s1"],
         "needs_search": True, "difficulty": "hard"},
    ],
    "synthesis_instruction": "Combine into a report.",
    "confidence": 0.8,
}


class _RoleClient:
    """Returns plan JSON for the planner model, prose for everything else."""

    def __init__(self, *, model: str, **_: object) -> None:
        self.model = model

    def chat(self, *, system_prompt: str, user_prompt: str, **_: object) -> LLMExecutionResult:
        if "planner" in system_prompt.lower():
            text = json.dumps(PLAN_JSON)
            cites: tuple[str, ...] = ()
        elif "synthesizer" in system_prompt.lower():
            text = "FINAL SYNTHESIZED ANSWER"
            cites = ("https://syn.example",)
        else:
            text = f"sub-answer from {self.model}"
            cites = ("https://evidence.example",)
        return LLMExecutionResult(
            model=self.model,
            prompt=user_prompt,
            response=text,
            usage=LLMUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300),
            latency_s=1.0,
            citations=cites,
        )


def _pool(monkeypatch) -> ModelPool:
    pool = ModelPool(
        specs=(
            ModelSpec("cheap", 0.1, 0.2, capability_tier=1, tags=("search",)),
            ModelSpec("frontier", 3.0, 6.0, capability_tier=3, tags=("search",)),
        ),
        api_key="k",
    )
    monkeypatch.setattr(
        "agentprop.council.model_pool.OpenAICompatibleChatClient", _RoleClient
    )
    return pool


def _council(pool: ModelPool, **overrides: object) -> Council:
    kwargs: dict = {
        "pool": pool,
        "planner": LLMPlanner(model="cheap"),
        "synthesizer": __import__(
            "agentprop.council.synthesizer", fromlist=["Synthesizer"]
        ).Synthesizer(model="frontier"),
        "retrieval": OpenRouterWebSearch(),
    }
    kwargs.update(overrides)
    return Council(**kwargs)  # type: ignore[arg-type]


def test_parse_plan_roundtrip() -> None:
    plan = parse_plan("task", json.dumps(PLAN_JSON))
    assert len(plan.subtasks) == 2
    assert plan.subtasks[1].depends_on == ("s1",)
    assert plan.subtasks[1].min_tier == 3  # hard
    assert plan.confidence == 0.8


def test_parse_plan_tolerates_fences_and_junk() -> None:
    fenced = "```json\n" + json.dumps(PLAN_JSON) + "\n```"
    assert len(parse_plan("t", fenced).subtasks) == 2
    empty = parse_plan("the task", "not json at all")
    assert empty.confidence == 0.0  # degenerate single-subtask fallback
    assert len(empty.subtasks) == 1


def test_plan_graph_shape() -> None:
    plan = parse_plan("task", json.dumps(PLAN_JSON))
    graph = plan.graph()
    node_ids = {n.id for n in graph.nodes()}
    assert {"input", "s1", "s2", "synthesizer", "output"} <= node_ids


def test_assigner_routes_hard_to_capable_cheapest() -> None:
    plan = parse_plan("task", json.dumps(PLAN_JSON))
    pool = ModelPool(
        specs=(
            ModelSpec("cheap", 0.1, 0.2, capability_tier=1, tags=("search",)),
            ModelSpec("frontier", 3.0, 6.0, capability_tier=3, tags=("search",)),
        ),
        api_key="k",
    )
    assignments = {a.subtask_id: a.model for a in Assigner().assign(plan, pool)}
    assert assignments["s1"] == "cheap"      # easy → cheapest capable
    assert assignments["s2"] == "frontier"   # hard (tier 3) → only frontier qualifies


def test_council_assigned_path(monkeypatch) -> None:
    council = _council(_pool(monkeypatch))
    result = council.run("Research X vs Y", task_id="t1")
    assert result.mode == "assign"
    assert result.answer == "FINAL SYNTHESIZED ANSWER"
    assert result.subtask_count == 2
    assert result.total_cost_usd > 0
    assert result.total_tokens > 0
    assert "https://syn.example" in result.citations
    assert result.trace  # ControlSession recorded events


def test_council_ensemble_fallback_on_low_confidence(monkeypatch) -> None:
    low = dict(PLAN_JSON, confidence=0.1)

    class _LowConf(_RoleClient):
        def chat(self, *, system_prompt: str, user_prompt: str, **_: object):
            if "planner" in system_prompt.lower():
                return LLMExecutionResult(
                    model=self.model, prompt=user_prompt, response=json.dumps(low),
                    usage=LLMUsage(10, 10, 20), latency_s=0.1,
                )
            return super().chat(system_prompt=system_prompt, user_prompt=user_prompt)

    pool = ModelPool(
        specs=(ModelSpec("cheap", 0.1, 0.2, tags=("search",)),
               ModelSpec("frontier", 3.0, 6.0, capability_tier=3, tags=("search",))),
        api_key="k",
    )
    monkeypatch.setattr("agentprop.council.model_pool.OpenAICompatibleChatClient", _LowConf)
    result = _council(pool).run("ambiguous task")
    assert result.mode == "ensemble"
    assert result.subtask_count == 2  # one per pool model


def test_claim_checker_quarantines_uncited_search_subtask() -> None:
    from agentprop.council.model_pool import ModelResponse

    sub = SubTask(id="s", question="q", needs_search=True)
    no_cite = ModelResponse("m", "claim with no source", LLMUsage(), 0.0, 0.1, citations=())
    checked = ClaimChecker().check(sub, no_cite)
    assert checked.quarantined
    cited = ModelResponse(
        "m", "claim", LLMUsage(), 0.0, 0.1, citations=("https://a", "https://b")
    )
    assert not ClaimChecker().check(sub, cited).quarantined


def test_subtask_features_shape() -> None:
    sub = SubTask("s", "q", depends_on=("a",), needs_search=True, difficulty="hard")
    f = subtask_features(sub, depth=2)
    assert f["depth"] == 2.0
    assert f["difficulty"] == 1.0
    assert f["needs_search"] == 1.0
    assert f["fan_in"] == 1.0
