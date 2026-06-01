import json
import sys
from pathlib import Path
from types import ModuleType

from agentprop.integrations import (
    NativeFrameworkUnavailable,
    graph_from_autogen_dict,
    graph_from_crewai_dict,
    graph_from_framework_dict,
    graph_from_langgraph_dict,
    graph_from_llamaindex_dict,
    graph_from_openai_agents_dict,
    graph_from_trace,
    native_framework_status,
    to_autogen_dict,
    to_crewai_dict,
    to_framework_dict,
    to_langgraph_dict,
    to_llamaindex_dict,
    to_native_framework,
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


def test_native_framework_status_reports_optional_packages() -> None:
    statuses = native_framework_status(["langgraph", "autogen"])

    by_name = {status.framework: status for status in statuses}
    assert by_name["langgraph"].native_adapter is True
    assert by_name["autogen"].native_adapter is False
    assert by_name["langgraph"].import_names == ("langgraph.graph",)


def test_native_langgraph_builder_uses_installed_state_graph(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    graph_module = ModuleType("langgraph.graph")

    class FakeStateGraph:
        def __init__(self, state_type):
            self.state_type = state_type
            self.nodes = {}
            self.edges = []
            self.entrypoints = []
            self.finish_points = []

        def add_node(self, name, handler):
            self.nodes[name] = handler

        def add_edge(self, source, target):
            self.edges.append((source, target))

        def set_entry_point(self, name):
            self.entrypoints.append(name)

        def set_finish_point(self, name):
            self.finish_points.append(name)

    graph_module.StateGraph = FakeStateGraph
    monkeypatch.setitem(sys.modules, "langgraph", ModuleType("langgraph"))
    monkeypatch.setitem(sys.modules, "langgraph.graph", graph_module)

    native = to_native_framework(planner_coder_tester_reviewer(), "langgraph")

    assert native.state_type is dict
    assert "planner" in native.nodes
    assert ("planner", "coder") in native.edges
    assert "planner" in native.entrypoints
    assert "final" in native.finish_points


def test_native_crewai_builder_uses_installed_classes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    crewai_module = ModuleType("crewai")

    class FakeAgent:
        def __init__(self, role, goal, backstory, allow_delegation):
            self.role = role
            self.goal = goal
            self.backstory = backstory
            self.allow_delegation = allow_delegation

    class FakeTask:
        def __init__(self, description, expected_output, agent):
            self.description = description
            self.expected_output = expected_output
            self.agent = agent

    class FakeCrew:
        def __init__(self, agents, tasks):
            self.agents = agents
            self.tasks = tasks

    crewai_module.Agent = FakeAgent
    crewai_module.Task = FakeTask
    crewai_module.Crew = FakeCrew
    monkeypatch.setitem(sys.modules, "crewai", crewai_module)

    native = to_native_framework(planner_coder_tester_reviewer(), "crewai")

    assert len(native.agents) == 4
    assert native.tasks
    assert any(task.agent.role == "coder" for task in native.tasks)


def test_native_openai_agents_builder_uses_installed_agent_class(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    agents_module = ModuleType("agents")

    class FakeAgent:
        def __init__(self, name, instructions, handoff_description):
            self.name = name
            self.instructions = instructions
            self.handoff_description = handoff_description

    agents_module.Agent = FakeAgent
    monkeypatch.setitem(sys.modules, "agents", agents_module)

    native = to_native_framework(planner_coder_tester_reviewer(), "openai")

    assert [agent.name for agent in native] == ["planner", "coder", "tester", "reviewer"]
    assert native[0].handoff_description == "AgentProp node planner"


def test_native_runtime_builder_explains_unsupported_framework() -> None:
    try:
        to_native_framework(planner_coder_tester_reviewer(), "autogen")
    except NativeFrameworkUnavailable as exc:
        assert "model clients" in str(exc)
    else:
        raise AssertionError("expected NativeFrameworkUnavailable")
