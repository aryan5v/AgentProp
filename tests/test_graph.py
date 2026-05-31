from pathlib import Path

import pytest

from agentprop import AgentGraph, NodeType


def test_agent_graph_adds_nodes_and_edges() -> None:
    graph = AgentGraph()
    graph.add_agent("planner", token_cost=1000)
    graph.add_verifier("reviewer", reliability=0.95)
    graph.add_edge("planner", "reviewer", message_cost=250, weight=0.8)

    assert graph.node_count == 2
    assert graph.edge_count == 1
    assert graph.node("planner").type is NodeType.AGENT
    assert graph.edge("planner", "reviewer").message_cost == 250


def test_agent_graph_rejects_edges_with_missing_nodes() -> None:
    graph = AgentGraph()
    graph.add_agent("planner")

    with pytest.raises(ValueError, match="Unknown target node"):
        graph.add_edge("planner", "missing")


def test_agent_graph_round_trips_json(tmp_path: Path) -> None:
    path = tmp_path / "workflow.json"
    graph = AgentGraph()
    graph.add_agent("planner", tool_access=["search"])
    graph.add_tool("search")
    graph.add_edge("planner", "search", activation_probability=0.7)

    graph.to_json(path)
    loaded = AgentGraph.from_json(path)

    assert loaded.node("planner").tool_access == ("search",)
    assert loaded.edge("planner", "search").activation_probability == 0.7
    assert loaded.to_networkx().has_edge("planner", "search")
