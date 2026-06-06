from agentprop.integrations import graph_from_framework_dict, to_framework_dict
from agentprop.workflows import planner_coder_tester_reviewer


def test_langgraph_dict_round_trip() -> None:
    graph = planner_coder_tester_reviewer()
    spec = to_framework_dict(graph, "langgraph")
    round_tripped = graph_from_framework_dict(spec, "langgraph")

    assert round_tripped.node_count == graph.node_count
    assert round_tripped.edge_count == graph.edge_count
    assert {node.id for node in round_tripped.nodes()} == {node.id for node in graph.nodes()}
