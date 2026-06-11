"""Tests for graph-position features in bandit reward records (issue #61)."""

from __future__ import annotations

from agentprop.rl import (
    REWARD_RECORD_SCHEMA_VERSION,
    node_position_features,
    reward_record_graph_features,
    workflow_embedding,
)
from agentprop.runtime import ControlSession, ExecutionEvent
from agentprop.workflows import WORKFLOW_TEMPLATES


def _graph():  # noqa: ANN202
    return WORKFLOW_TEMPLATES["planner_coder_tester_reviewer"]()


def test_workflow_embedding_is_flat_and_stable() -> None:
    graph = _graph()
    embedding = workflow_embedding(graph)
    assert embedding["node_count"] == float(graph.node_count)
    assert embedding["edge_count"] == float(graph.edge_count)
    assert embedding["max_depth"] >= 1.0
    assert all(isinstance(v, float) for v in embedding.values())
    assert embedding == workflow_embedding(graph)


def test_node_position_features_fields() -> None:
    graph = _graph()
    node_id = graph.nodes()[0].id
    verifiers = tuple(n.id for n in graph.nodes()[:2])
    features = node_position_features(graph, node_id, verifiers=verifiers)
    assert features["node_id"] == node_id
    assert isinstance(features["depth"], float)
    assert 0.0 <= float(features["quality_score"] or 0.0) <= 1.0
    assert 0.0 <= float(features["resolving_coverage_active"] or 0.0) <= 1.0
    contribution = float(features["resolving_coverage_contribution"] or 0.0)
    assert contribution >= 0.0


def test_reward_record_payload_schema() -> None:
    graph = _graph()
    payload = reward_record_graph_features(graph, verifiers=(graph.nodes()[0].id,))
    assert payload["schema_version"] == REWARD_RECORD_SCHEMA_VERSION
    assert "workflow_embedding" in payload
    assert "node" not in payload
    with_node = reward_record_graph_features(graph, node_id=graph.nodes()[0].id)
    assert "node" in with_node


def test_control_session_outcome_row_carries_graph_features() -> None:
    session = ControlSession.start(
        "planner_coder_tester_reviewer",
        task_id="t1",
        category="coding",
    )
    session.observe(ExecutionEvent(step=1, progress_made=True, tokens_used=100))
    row = session.record_outcome(passed=True)
    graph_features = row["graph_features"]
    assert isinstance(graph_features, dict)
    assert graph_features["schema_version"] == REWARD_RECORD_SCHEMA_VERSION
    assert "workflow_embedding" in graph_features
