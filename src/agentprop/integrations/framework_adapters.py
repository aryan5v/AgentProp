"""Dependency-light adapters for common agent workflow frameworks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agentprop.core import AgentGraph, NodeType

SUPPORTED_FRAMEWORKS = {
    "langgraph",
    "autogen",
    "crewai",
    "openai-agents",
    "llamaindex",
}


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
