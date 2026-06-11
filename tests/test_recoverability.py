"""Tests for the forward-simulation cascade-risk advisor."""

from __future__ import annotations

import pytest

from agentprop.core import AgentGraph
from agentprop.runtime import (
    CascadeRiskAdvisor,
    ControlDecision,
    ExecutionStateFeatures,
    estimate_cascade_risk,
)


def _chain_graph(probability: float) -> AgentGraph:
    graph = AgentGraph()
    for node_id in ("a", "b", "c", "d"):
        graph.add_node(node_id)
    for source, target in (("a", "b"), ("b", "c"), ("c", "d")):
        graph.add_edge(source, target, activation_probability=probability)
    return graph


def _features(**overrides: object) -> ExecutionStateFeatures:
    defaults: dict[str, object] = {
        "step_count": 5,
        "total_tokens": 1000,
        "elapsed_s": 10.0,
        "steps_since_verifier": 3,
        "steps_since_progress": 1,
        "repeated_error_count": 1,
        "verifier_failed_count": 0,
        "last_exit_code": 1,
        "evaluator_passed": False,
        "final_answer_written": False,
    }
    defaults.update(overrides)
    return ExecutionStateFeatures(**defaults)  # type: ignore[arg-type]


def test_estimate_orders_by_position() -> None:
    graph = _chain_graph(0.9)
    head = estimate_cascade_risk(graph, "a", trials=200, seed=1)
    tail = estimate_cascade_risk(graph, "d", trials=200, seed=1)
    assert head.impact.mean > tail.impact.mean
    assert head.downstream_nodes == 3
    assert tail.downstream_nodes == 0


def test_estimate_unknown_node_raises() -> None:
    with pytest.raises(ValueError):
        estimate_cascade_risk(_chain_graph(0.5), "zzz")


def test_advisor_escalates_high_impact_node() -> None:
    advisor = CascadeRiskAdvisor(_chain_graph(0.95), impact_threshold=0.5, trials=200)
    decision = advisor.advise(
        ControlDecision("CONTINUE", "within budget"),
        _features(),
        node_id="a",
    )
    assert decision.action == "FORCE_VERIFY"
    assert "cascade" in decision.reason


def test_advisor_leaves_low_impact_alone() -> None:
    advisor = CascadeRiskAdvisor(_chain_graph(0.95), impact_threshold=0.5, trials=200)
    decision = advisor.advise(
        ControlDecision("CONTINUE", "within budget"),
        _features(),
        node_id="d",
    )
    assert decision.action == "CONTINUE"


def test_advisor_never_downgrades_and_respects_clean_state() -> None:
    advisor = CascadeRiskAdvisor(_chain_graph(0.95), impact_threshold=0.1, trials=50)
    finalize = ControlDecision("FINALIZE", "done")
    assert advisor.advise(finalize, _features(), node_id="a") is finalize
    clean = _features(repeated_error_count=0, last_exit_code=0)
    decision = advisor.advise(ControlDecision("CONTINUE", "ok"), clean, node_id="a")
    assert decision.action == "CONTINUE"


def test_advisor_waits_after_fresh_verification() -> None:
    advisor = CascadeRiskAdvisor(_chain_graph(0.95), impact_threshold=0.1, trials=50)
    fresh = _features(steps_since_verifier=0)
    decision = advisor.advise(ControlDecision("CONTINUE", "ok"), fresh, node_id="a")
    assert decision.action == "CONTINUE"
