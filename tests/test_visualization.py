from agentprop.visualization import graph_to_dot
from agentprop.workflows import planner_coder_tester_reviewer


def test_graph_to_dot_contains_nodes_and_edges() -> None:
    dot = graph_to_dot(planner_coder_tester_reviewer(), name="Demo")

    assert "digraph Demo" in dot
    assert '"planner" -> "coder"' in dot
    assert "PLANNER" in dot
