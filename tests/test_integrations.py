import json
from pathlib import Path

from agentprop.integrations import (
    graph_from_autogen_dict,
    graph_from_crewai_dict,
    graph_from_framework_dict,
    graph_from_langgraph_dict,
    graph_from_llamaindex_dict,
    graph_from_openai_agents_dict,
    graph_from_trace,
    to_autogen_dict,
    to_crewai_dict,
    to_framework_dict,
    to_langgraph_dict,
    to_llamaindex_dict,
    to_openai_agents_dict,
)
from agentprop.workflows import planner_coder_tester_reviewer


def test_graph_from_trace_aggregates_messages(tmp_path: Path) -> None:
    trace = {
        "events": [
            {
                "source": "planner",
                "target": "coder",
                "source_type": "PLANNER",
                "target_type": "EXECUTOR",
                "token_cost": 500,
                "latency": 0.7,
                "success": True,
            },
            {
                "source": "coder",
                "target": "tester",
                "target_type": "VERIFIER",
                "token_cost": 300,
                "latency": 0.5,
                "success": False,
            },
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))

    result = graph_from_trace(path)

    assert result.message_count == 2
    assert result.total_token_cost == 800
    assert result.graph.node("tester").error_rate == 1.0
    assert result.graph.edge("planner", "coder").message_cost == 500


def test_langgraph_adapter_exports_and_imports_node_edge_spec() -> None:
    graph = planner_coder_tester_reviewer()

    payload = to_langgraph_dict(graph)
    imported = graph_from_langgraph_dict(payload)

    assert payload["framework"] == "langgraph"
    assert "planner" in payload["entrypoints"]
    assert imported.node_count == graph.node_count
    assert imported.edge_count == graph.edge_count


def test_autogen_adapter_exports_agents_and_transitions() -> None:
    graph = planner_coder_tester_reviewer()

    payload = to_autogen_dict(graph)
    imported = graph_from_autogen_dict(payload)

    assert payload["framework"] == "autogen"
    assert any(agent["id"] == "planner" for agent in payload["agents"])
    assert imported.edge("planner", "coder").source == "planner"


def test_crewai_adapter_exports_agents_and_tasks() -> None:
    graph = planner_coder_tester_reviewer()

    payload = to_crewai_dict(graph)
    imported = graph_from_crewai_dict(payload)

    assert payload["framework"] == "crewai"
    assert payload["tasks"]
    assert payload["dependencies"]
    assert imported.node("coder").id == "coder"
    assert imported.node_count == graph.node_count
    assert imported.edge_count == graph.edge_count


def test_openai_agents_adapter_exports_handoffs() -> None:
    graph = planner_coder_tester_reviewer()

    payload = to_openai_agents_dict(graph)
    imported = graph_from_openai_agents_dict(payload)

    planner = next(agent for agent in payload["agents"] if agent["name"] == "planner")
    assert "coder" in planner["handoffs"]
    assert imported.edge("planner", "coder").target == "coder"


def test_llamaindex_adapter_exports_steps_and_events() -> None:
    graph = planner_coder_tester_reviewer()

    payload = to_llamaindex_dict(graph)
    imported = graph_from_llamaindex_dict(payload)

    assert payload["framework"] == "llamaindex"
    assert payload["steps"]
    assert imported.node_count == graph.node_count


def test_generic_framework_adapter_dispatches_aliases() -> None:
    graph = planner_coder_tester_reviewer()

    payload = to_framework_dict(graph, "openai_agents_sdk")
    imported = graph_from_framework_dict(payload, "openai")

    assert payload["framework"] == "openai-agents"
    assert imported.edge_count >= graph.edge_count
