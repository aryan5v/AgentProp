"""Tests for structured critical-fact context routing."""

from __future__ import annotations

from agentprop.core import NodeType
from agentprop.core.models import AgentNode
from agentprop.runtime.critical_facts import build_context_slice, extract_critical_facts


def test_extract_critical_facts_preserves_conventions() -> None:
    context = (
        "Background info. "
        "You must follow the conventions doc format. "
        "Use `parse_json` for all inputs."
    )
    facts = extract_critical_facts(context, task="parse_json")
    assert any("conventions" in fact.text.lower() for fact in facts)
    assert any("`parse_json`" in fact.text for fact in facts)


def test_build_context_slice_keeps_must_have_facts() -> None:
    context = (
        "Filler sentence one. Filler sentence two. "
        "You must never skip verifier checks. "
        "More filler text that can be dropped safely."
    )
    node = AgentNode(id="coder", type=NodeType.EXECUTOR)
    sliced = build_context_slice(context, task="coding", ratio=0.35, node=node)
    assert "verifier" in sliced.lower()
