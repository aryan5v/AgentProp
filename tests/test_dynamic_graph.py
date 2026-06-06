"""Tests for runtime graph mutations and conditional edges."""

from __future__ import annotations

from agentprop.core import AgentGraph
from agentprop.core.dynamic_graph import DynamicGraphSession, edge_is_active


def test_remove_node_and_edge_invalidate_structure() -> None:
    graph = AgentGraph.from_dict(
        {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
            ],
        }
    )
    graph.remove_edge("a", "b")
    assert not graph.has_edge("a", "b")
    graph.remove_node("c")
    assert not graph.has_node("c")
    assert graph.node_count == 2


def test_conditional_edge_filtering() -> None:
    graph = AgentGraph.from_dict(
        {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [
                {"source": "a", "target": "b"},
            ],
        }
    )
    graph.add_conditional_edge("a", "c", condition_key="mode", condition_value="debug")
    active = graph.filter_active_edges({"mode": "debug"})
    assert active.has_edge("a", "c")
    inactive = graph.filter_active_edges({"mode": "prod"})
    assert not inactive.has_edge("a", "c")
    assert inactive.has_edge("a", "b")


def test_dynamic_graph_session_tracks_mutations() -> None:
    base = AgentGraph.from_dict(
        {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}],
        }
    )
    session = DynamicGraphSession(base_graph=base)
    session.add_node("c")
    session.remove_edge("a", "b")
    assert session.version == 2
    assert len(session.mutations) == 2
    assert session.graph.has_node("c")
    assert not session.graph.has_edge("a", "b")


def test_dynamic_session_conditional_edge() -> None:
    base = AgentGraph.from_dict(
        {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [{"source": "a", "target": "b"}],
        }
    )
    session = DynamicGraphSession(base_graph=base)
    session.add_conditional_edge("a", "c", condition_key="flag", condition_value=True)
    active = session.active_graph({"flag": True})
    assert active.has_edge("a", "c")
    inactive = session.active_graph({"flag": False})
    assert not inactive.has_edge("a", "c")


def test_edge_is_active_without_condition() -> None:
    graph = AgentGraph.from_dict(
        {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}],
        }
    )
    edge = graph.edge("a", "b")
    assert edge_is_active(edge, {})
