"""Tests for hierarchical context and fact-level verifiers."""

from __future__ import annotations

from agentprop.core import AgentGraph, NodeType
from agentprop.integrations.context_advisor import (
    ContextExpansionAdvisor,
    langgraph_checkpoint_advice,
)
from agentprop.runtime.critical_facts import CriticalFact, CriticalFactStore
from agentprop.runtime.hierarchical_context import (
    build_hierarchical_bundle,
    place_fact_level_verifiers,
)


def test_build_hierarchical_bundle_preserves_facts() -> None:
    bundle = build_hierarchical_bundle(
        shared_context="You must verify the API signature before coding.",
        task="implement handler",
        node_id="coder",
        ratio=1.0,
    )
    rendered = bundle.render()
    assert "API" in rendered or "verify" in rendered.lower()
    assert bundle.task_context == "implement handler"


def test_fact_level_verifier_placement() -> None:
    graph = AgentGraph.from_dict(
        {
            "nodes": [
                {"id": "planner"},
                {"id": "coder"},
                {"id": "verifier", "type": NodeType.VERIFIER.value},
            ],
            "edges": [
                {"source": "planner", "target": "coder"},
                {"source": "coder", "target": "verifier"},
            ],
        }
    )
    facts = [
        CriticalFact(text="must use pytest", source_span="task", score=0.9),
        CriticalFact(text="never skip lint", source_span="task", score=0.8),
    ]
    placements = place_fact_level_verifiers(graph, facts, budget=2)
    assert len(placements) == 2
    assert placements[0].verifier_node_id == "verifier"


def test_context_advisor_expands_for_verifier() -> None:
    graph = AgentGraph.from_dict(
        {
            "nodes": [
                {"id": "coder"},
                {"id": "reviewer", "type": NodeType.VERIFIER.value},
            ],
            "edges": [{"source": "coder", "target": "reviewer"}],
        }
    )
    store = CriticalFactStore()
    store.register_node_facts(
        "coder",
        [CriticalFact(text="required schema", source_span="trace", score=0.95)],
    )
    advisor = ContextExpansionAdvisor(graph=graph, fact_store=store)
    advice = advisor.should_expand("reviewer", current_ratio=0.2)
    assert advice.expand is True
    payload = langgraph_checkpoint_advice(advisor, node_id="reviewer", state={"task": "t"})
    assert payload["agentprop_expand_context"] is True
