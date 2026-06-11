"""Dependency-light adapters for common agent workflow frameworks."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from inspect import Parameter, signature
from typing import Any

from agentprop.core import AgentGraph, NodeType

SUPPORTED_FRAMEWORKS = {
    "langgraph",
    "autogen",
    "crewai",
    "openai-agents",
    "llamaindex",
}


@dataclass(frozen=True, slots=True)
class NativeFrameworkStatus:
    """Availability status for an optional native framework adapter."""

    framework: str
    import_names: tuple[str, ...]
    available: bool
    native_adapter: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "framework": self.framework,
            "import_names": list(self.import_names),
            "available": self.available,
            "native_adapter": self.native_adapter,
            "notes": list(self.notes),
        }


class NativeFrameworkUnavailable(RuntimeError):
    """Raised when a requested native framework adapter cannot be built."""


def to_framework_dict(graph: AgentGraph, framework: str) -> dict[str, Any]:
    """Export an AgentGraph as a framework-oriented dictionary."""

    normalized = _normalize_framework(framework)
    if normalized == "langgraph":
        return to_langgraph_dict(graph)
    if normalized == "autogen":
        return to_autogen_dict(graph)
    if normalized == "crewai":
        return to_crewai_dict(graph)
    if normalized == "openai-agents":
        return to_openai_agents_dict(graph)
    if normalized == "llamaindex":
        return to_llamaindex_dict(graph)
    raise ValueError(f"Unsupported framework: {framework}")


def native_framework_status(frameworks: Iterable[str] | None = None) -> list[NativeFrameworkStatus]:
    """Report optional native adapter availability without requiring heavy dependencies."""

    targets = [
        _normalize_framework(framework)
        for framework in frameworks or sorted(SUPPORTED_FRAMEWORKS)
    ]
    return [_native_status(framework) for framework in targets]


def to_native_framework(graph: AgentGraph, framework: str) -> Any:
    """Build a best-effort native workflow object for an installed framework.

    The base package keeps these dependencies optional. If the framework package
    is not installed, or its runtime API needs user-supplied execution objects,
    this function raises NativeFrameworkUnavailable with a concrete next step.
    """

    normalized = _normalize_framework(framework)
    if normalized == "langgraph":
        return _to_native_langgraph(graph)
    if normalized == "crewai":
        return _to_native_crewai(graph)
    if normalized == "openai-agents":
        return _to_native_openai_agents(graph)
    if normalized == "autogen":
        raise NativeFrameworkUnavailable(
            "AutoGen native execution requires model clients and agent runtime wiring; "
            "use to_autogen_dict() for interchange until a configured runtime builder is added."
        )
    if normalized == "llamaindex":
        raise NativeFrameworkUnavailable(
            "LlamaIndex Workflow native classes require user-defined step functions; "
            "use to_llamaindex_dict() for interchange until a configured runtime builder is added."
        )
    raise ValueError(f"Unsupported framework: {framework}")


def graph_from_framework_dict(data: Mapping[str, Any], framework: str) -> AgentGraph:
    """Import a supported framework-oriented dictionary as an AgentGraph."""

    normalized = _normalize_framework(framework)
    if normalized == "langgraph":
        return graph_from_langgraph_dict(data)
    if normalized == "autogen":
        return graph_from_autogen_dict(data)
    if normalized == "crewai":
        return graph_from_crewai_dict(data)
    if normalized == "openai-agents":
        return graph_from_openai_agents_dict(data)
    if normalized == "llamaindex":
        return graph_from_llamaindex_dict(data)
    raise ValueError(f"Unsupported framework: {framework}")


def to_langgraph_dict(graph: AgentGraph) -> dict[str, Any]:
    """Export a LangGraph-style node/edge spec."""

    return {
        "framework": "langgraph",
        "nodes": [_node_payload(node_id, graph) for node_id in _node_ids(graph)],
        "edges": [_edge_payload(edge) for edge in graph.edges()],
        "entrypoints": _entrypoints(graph),
        "finish_points": _output_nodes(graph),
    }


def graph_from_langgraph_dict(data: Mapping[str, Any]) -> AgentGraph:
    """Import a LangGraph-style dictionary with nodes and edges."""

    return _graph_from_node_edge_payloads(
        _required_list(data, "nodes"),
        _required_list(data, "edges"),
    )


def graph_from_langgraph_object(workflow: Any) -> AgentGraph:
    """Best-effort import of a LangGraph ``StateGraph`` or compiled graph object.

    The adapter intentionally uses duck typing so AgentProp can analyze real
    LangGraph objects while keeping ``langgraph`` an optional dependency.
    """

    graph_obj = workflow.get_graph() if callable(getattr(workflow, "get_graph", None)) else workflow
    nodes_obj = getattr(graph_obj, "nodes", None)
    edges_obj = getattr(graph_obj, "edges", None)

    nodes = _langgraph_nodes(nodes_obj)
    edges = _langgraph_edges(edges_obj)
    if not nodes:
        raise NativeFrameworkUnavailable(
            "Could not inspect LangGraph nodes; pass an AgentGraph or a compiled "
            "LangGraph object exposing get_graph().nodes."
        )
    return _graph_from_node_edge_payloads(nodes, edges)


def to_autogen_dict(graph: AgentGraph) -> dict[str, Any]:
    """Export an AutoGen-style agent transition spec."""

    return {
        "framework": "autogen",
        "agents": [
            _agent_payload(node_id, graph)
            for node_id in _node_ids(graph)
            if graph.node(node_id).type != NodeType.OUTPUT
        ],
        "transitions": [_edge_payload(edge) for edge in graph.edges()],
        "termination_agents": _output_nodes(graph),
    }


def graph_from_autogen_dict(data: Mapping[str, Any]) -> AgentGraph:
    """Import an AutoGen-style dictionary with agents and transitions."""

    agents = _required_list(data, "agents")
    transitions = _list_field(data, "transitions")
    nodes = [
        {
            "id": _id_from_payload(agent),
            "type": agent.get("type", NodeType.AGENT.value),
            "role": agent.get("system_message", agent.get("role")),
            "tools": agent.get("tools", []),
        }
        for agent in agents
        if isinstance(agent, Mapping)
    ]
    for output in _list_field(data, "termination_agents"):
        nodes.append({"id": str(output), "type": NodeType.OUTPUT.value})
    return _graph_from_node_edge_payloads(nodes, transitions)


def to_crewai_dict(graph: AgentGraph) -> dict[str, Any]:
    """Export a CrewAI-style agents/tasks spec."""

    agents = [
        _agent_payload(node_id, graph)
        for node_id in _node_ids(graph)
        if graph.node(node_id).type != NodeType.OUTPUT
    ]
    tasks = [
        {
            "id": edge.target,
            "agent": edge.target,
            "context": [edge.source],
            "metadata": _edge_payload(edge),
        }
        for edge in graph.edges()
        if graph.node(edge.target).type != NodeType.OUTPUT
    ]
    return {
        "framework": "crewai",
        "agents": agents,
        "tasks": tasks,
        "dependencies": [_edge_payload(edge) for edge in graph.edges()],
        "outputs": _output_nodes(graph),
    }


def graph_from_crewai_dict(data: Mapping[str, Any]) -> AgentGraph:
    """Import a CrewAI-style dictionary with agents and tasks."""

    agents = _required_list(data, "agents")
    tasks = _list_field(data, "tasks")
    nodes = [
        {
            "id": _id_from_payload(agent),
            "type": agent.get("type", NodeType.AGENT.value),
            "role": agent.get("goal", agent.get("role")),
            "tools": agent.get("tools", []),
        }
        for agent in agents
        if isinstance(agent, Mapping)
    ]
    for output in _list_field(data, "outputs"):
        nodes.append({"id": str(output), "type": NodeType.OUTPUT.value})
    dependencies = _list_field(data, "dependencies")
    edges = dependencies if dependencies else _crewai_task_edges(tasks)
    return _graph_from_node_edge_payloads(nodes, edges)


def to_openai_agents_dict(graph: AgentGraph) -> dict[str, Any]:
    """Export an OpenAI Agents SDK-style agents and handoffs spec."""

    agents = []
    for node_id in _node_ids(graph):
        node = graph.node(node_id)
        if node.type == NodeType.OUTPUT:
            continue
        agents.append(
            {
                "name": node.id,
                "instructions": node.role or node.name or node.id,
                "tools": list(node.tool_access),
                "handoffs": graph.successors(node.id),
                "metadata": _node_payload(node.id, graph),
            }
        )
    return {"framework": "openai-agents", "agents": agents}


def graph_from_openai_agents_dict(data: Mapping[str, Any]) -> AgentGraph:
    """Import an OpenAI Agents SDK-style dictionary."""

    agents = _required_list(data, "agents")
    nodes = []
    edges = []
    for agent in agents:
        if not isinstance(agent, Mapping):
            continue
        node_id = _id_from_payload(agent)
        nodes.append(
            {
                "id": node_id,
                "type": agent.get("type", NodeType.AGENT.value),
                "role": agent.get("instructions", agent.get("role")),
                "tools": agent.get("tools", []),
            }
        )
        for target in _list_field(agent, "handoffs"):
            edges.append({"source": node_id, "target": str(target)})
    for edge in edges:
        if not any(str(node["id"]) == edge["target"] for node in nodes):
            nodes.append({"id": edge["target"], "type": NodeType.OUTPUT.value})
    return _graph_from_node_edge_payloads(nodes, edges)


def to_llamaindex_dict(graph: AgentGraph) -> dict[str, Any]:
    """Export a LlamaIndex workflow-style steps/events spec."""

    return {
        "framework": "llamaindex",
        "steps": [_node_payload(node_id, graph) for node_id in _node_ids(graph)],
        "events": [_edge_payload(edge) for edge in graph.edges()],
    }


def graph_from_llamaindex_dict(data: Mapping[str, Any]) -> AgentGraph:
    """Import a LlamaIndex workflow-style dictionary."""

    return _graph_from_node_edge_payloads(
        _required_list(data, "steps"),
        _list_field(data, "events"),
    )


def _to_native_langgraph(graph: AgentGraph) -> Any:
    module = _import_first("langgraph.graph")
    state_graph_cls = getattr(module, "StateGraph", None)
    if state_graph_cls is None:
        raise NativeFrameworkUnavailable("Installed langgraph package does not expose StateGraph.")
    builder = state_graph_cls(dict)
    for node_id in _node_ids(graph):
        builder.add_node(node_id, _passthrough_node(node_id))
    for edge in graph.edges():
        builder.add_edge(edge.source, edge.target)
    for entrypoint in _entrypoints(graph):
        _call_if_present(builder, "set_entry_point", entrypoint)
    for output_node in _output_nodes(graph):
        _call_if_present(builder, "set_finish_point", output_node)
    return builder


def _to_native_crewai(graph: AgentGraph) -> Any:
    module = _import_first("crewai")
    agent_cls = getattr(module, "Agent", None)
    task_cls = getattr(module, "Task", None)
    crew_cls = getattr(module, "Crew", None)
    if agent_cls is None or task_cls is None or crew_cls is None:
        raise NativeFrameworkUnavailable(
            "Installed crewai package must expose Agent, Task, and Crew."
        )

    agents_by_id = {}
    for node_id in _node_ids(graph):
        node = graph.node(node_id)
        if node.type == NodeType.OUTPUT:
            continue
        agents_by_id[node_id] = _call_with_supported_kwargs(
            agent_cls,
            role=node.name or node.id,
            goal=node.role or f"Handle the {node.id} workflow step.",
            backstory=f"AgentProp node {node.id}",
            allow_delegation=True,
        )

    tasks = []
    for edge in graph.edges():
        if edge.target not in agents_by_id:
            continue
        tasks.append(
            _call_with_supported_kwargs(
                task_cls,
                description=f"Process context from {edge.source} for {edge.target}.",
                expected_output=f"Output for {edge.target}.",
                agent=agents_by_id[edge.target],
            )
        )
    return _call_with_supported_kwargs(
        crew_cls,
        agents=list(agents_by_id.values()),
        tasks=tasks,
    )


def _to_native_openai_agents(graph: AgentGraph) -> list[Any]:
    module = _import_first("agents", "openai_agents")
    agent_cls = getattr(module, "Agent", None)
    if agent_cls is None:
        raise NativeFrameworkUnavailable("Installed OpenAI Agents package does not expose Agent.")
    agents = []
    for node_id in _node_ids(graph):
        node = graph.node(node_id)
        if node.type == NodeType.OUTPUT:
            continue
        agents.append(
            _call_with_supported_kwargs(
                agent_cls,
                name=node.name or node.id,
                instructions=node.role or f"Handle the {node.id} workflow step.",
                handoff_description=f"AgentProp node {node.id}",
            )
        )
    return agents


def _graph_from_node_edge_payloads(
    nodes: Iterable[Mapping[str, Any]],
    edges: Iterable[Mapping[str, Any]],
) -> AgentGraph:
    graph = AgentGraph()
    for node in nodes:
        node_id = _id_from_payload(node)
        node_type = _node_type(node.get("type"))
        graph.add_node(
            node_id,
            type=node_type,
            name=_optional_str(node.get("name")),
            role=_optional_str(node.get("role")),
            token_cost=_float_value(node.get("token_cost")),
            latency=_float_value(node.get("latency")),
            reliability=_float_value(node.get("reliability"), default=1.0),
            error_rate=_float_value(node.get("error_rate")),
            tool_access=_tools_from_payload(node),
            framework_metadata={
                key: value
                for key, value in dict(node).items()
                if key
                not in {
                    "id",
                    "name",
                    "role",
                    "type",
                    "token_cost",
                    "latency",
                    "reliability",
                    "error_rate",
                    "tools",
                }
            },
        )

    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if not source or not target:
            continue
        if source not in graph.to_networkx():
            graph.add_node(source)
        if target not in graph.to_networkx():
            graph.add_node(target)
        graph.add_edge(
            source,
            target,
            message_cost=_float_value(edge.get("message_cost")),
            latency=_float_value(edge.get("latency")),
            relevance=_float_value(edge.get("relevance"), default=1.0),
            reliability=_float_value(edge.get("reliability"), default=1.0),
            activation_probability=_float_value(
                edge.get("activation_probability"),
                default=1.0,
            ),
            weight=_float_value(edge.get("weight"), default=1.0),
        )
    return graph


def _crewai_task_edges(tasks: list[Any]) -> list[dict[str, str]]:
    edges = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        target = str(task.get("agent", task.get("id", "")))
        for source in _list_field(task, "context"):
            edges.append({"source": str(source), "target": target})
    return edges


def _node_payload(node_id: str, graph: AgentGraph) -> dict[str, Any]:
    node = graph.node(node_id)
    return {
        "id": node.id,
        "name": node.name or node.id,
        "role": node.role or node.type.value.lower(),
        "type": node.type.value,
        "token_cost": node.token_cost,
        "latency": node.latency,
        "reliability": node.reliability,
        "error_rate": node.error_rate,
        "tools": list(node.tool_access),
    }


def _agent_payload(node_id: str, graph: AgentGraph) -> dict[str, Any]:
    payload = _node_payload(node_id, graph)
    payload["system_message"] = payload["role"]
    payload["goal"] = payload["role"]
    return payload


def _edge_payload(edge: Any) -> dict[str, Any]:
    return {
        "source": edge.source,
        "target": edge.target,
        "message_cost": edge.message_cost,
        "latency": edge.latency,
        "relevance": edge.relevance,
        "reliability": edge.reliability,
        "activation_probability": edge.activation_probability,
        "weight": edge.weight,
    }


def _entrypoints(graph: AgentGraph) -> list[str]:
    nx_graph = graph.to_networkx()
    return sorted(str(node_id) for node_id in nx_graph.nodes if nx_graph.in_degree(node_id) == 0)


def _output_nodes(graph: AgentGraph) -> list[str]:
    return sorted(node.id for node in graph.nodes() if node.type == NodeType.OUTPUT)


def _node_ids(graph: AgentGraph) -> list[str]:
    return [node.id for node in graph.nodes()]


def _normalize_framework(framework: str) -> str:
    normalized = framework.lower().replace("_", "-")
    aliases = {
        "openai": "openai-agents",
        "openai-agents-sdk": "openai-agents",
        "llama-index": "llamaindex",
        "llamaindex-workflows": "llamaindex",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_FRAMEWORKS:
        raise ValueError(f"Unsupported framework: {framework}")
    return normalized


def _native_status(framework: str) -> NativeFrameworkStatus:
    import_names = _native_import_names(framework)
    available = any(_can_import(name) for name in import_names)
    native_adapter = framework in {"langgraph", "crewai", "openai-agents"}
    notes: tuple[str, ...] = ()
    if not native_adapter:
        notes = ("Native runtime builder needs user-supplied runtime functions or model clients.",)
    elif not available:
        notes = ("Optional framework package is not installed.",)
    return NativeFrameworkStatus(
        framework=framework,
        import_names=import_names,
        available=available,
        native_adapter=native_adapter,
        notes=notes,
    )


def _native_import_names(framework: str) -> tuple[str, ...]:
    if framework == "langgraph":
        return ("langgraph.graph",)
    if framework == "autogen":
        return ("autogen_agentchat", "autogen")
    if framework == "crewai":
        return ("crewai",)
    if framework == "openai-agents":
        return ("agents", "openai_agents")
    if framework == "llamaindex":
        return ("llama_index.core.workflow", "llama_index")
    raise ValueError(f"Unsupported framework: {framework}")


def _import_first(*names: str) -> Any:
    for name in names:
        try:
            return import_module(name)
        except ImportError:
            continue
    joined = ", ".join(names)
    raise NativeFrameworkUnavailable(f"Install one of these optional packages first: {joined}")


def _can_import(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _call_if_present(obj: Any, method_name: str, *args: Any) -> None:
    method = getattr(obj, method_name, None)
    if callable(method):
        method(*args)


def _call_with_supported_kwargs(factory: Any, **kwargs: Any) -> Any:
    try:
        params = signature(factory).parameters
    except (TypeError, ValueError):
        return factory(**kwargs)
    accepts_kwargs = any(param.kind == Parameter.VAR_KEYWORD for param in params.values())
    supported = (
        kwargs
        if accepts_kwargs
        else {key: value for key, value in kwargs.items() if key in params}
    )
    return factory(**supported)


def _passthrough_node(node_id: str) -> Any:
    def run(state: Mapping[str, Any]) -> dict[str, Any]:
        next_state = dict(state)
        next_state["agentprop_last_node"] = node_id
        return next_state

    return run


def _id_from_payload(payload: Mapping[str, Any]) -> str:
    value = payload.get("id", payload.get("name"))
    if not isinstance(value, str) or not value:
        raise ValueError("framework node payload requires a non-empty id or name")
    return value


def _required_list(data: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"framework payload requires a {key} list")
    return [item for item in value if isinstance(item, Mapping)]


def _list_field(data: Mapping[str, Any], key: str) -> list[Any]:
    value = data.get(key, [])
    return value if isinstance(value, list) else []


def _langgraph_nodes(nodes_obj: Any) -> list[dict[str, Any]]:
    if callable(nodes_obj):
        nodes_obj = nodes_obj()
    if isinstance(nodes_obj, Mapping):
        return [
            {"id": str(node_id), "type": NodeType.AGENT.value}
            for node_id in nodes_obj
            if str(node_id) not in {"__start__", "__end__", "START", "END"}
        ]
    if isinstance(nodes_obj, Iterable) and not isinstance(nodes_obj, str | bytes):
        nodes: list[dict[str, Any]] = []
        for item in nodes_obj:
            node_id = getattr(item, "id", getattr(item, "name", item))
            node_id_str = str(node_id)
            if node_id_str not in {"__start__", "__end__", "START", "END"}:
                nodes.append({"id": node_id_str, "type": NodeType.AGENT.value})
        return nodes
    return []


def _langgraph_edges(edges_obj: Any) -> list[dict[str, Any]]:
    if callable(edges_obj):
        edges_obj = edges_obj()
    if edges_obj is None:
        return []
    edges: list[dict[str, Any]] = []
    for item in edges_obj:
        if isinstance(item, Mapping):
            source = item.get("source")
            target = item.get("target")
        elif isinstance(item, tuple) and len(item) >= 2:
            source, target = item[0], item[1]
        else:
            source = getattr(item, "source", None)
            target = getattr(item, "target", None)
        if source is None or target is None:
            continue
        source_str = str(source)
        target_str = str(target)
        if source_str in {"__start__", "START"} or target_str in {"__end__", "END"}:
            continue
        edges.append({"source": source_str, "target": target_str})
    return edges


def _node_type(value: Any) -> NodeType:
    if value is None:
        return NodeType.AGENT
    try:
        return NodeType(str(value).upper())
    except ValueError:
        return NodeType.AGENT


def _tools_from_payload(payload: Mapping[str, Any]) -> list[str]:
    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        return []
    return [str(tool) for tool in tools]


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _float_value(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | str):
        return float(value)
    return default
