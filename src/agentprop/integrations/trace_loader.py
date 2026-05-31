"""Load workflow graphs from message/tool/verifier trace logs."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentprop.core import AgentGraph, NodeType


@dataclass(slots=True)
class TraceLoadResult:
    """Graph plus trace-derived aggregate statistics."""

    graph: AgentGraph
    message_count: int
    total_token_cost: float
    total_latency: float


def graph_from_trace(path: str | Path) -> TraceLoadResult:
    """Load an AgentGraph from a trace JSON file."""

    data = json.loads(Path(path).read_text())
    return graph_from_trace_dict(data)


def graph_from_trace_dict(data: dict[str, Any]) -> TraceLoadResult:
    """Build a graph from trace dictionary data."""

    events = data.get("events", data.get("messages", []))
    if not isinstance(events, list):
        raise ValueError("trace must contain an events or messages list")

    node_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    edge_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    node_types: dict[str, NodeType] = {}

    for event in events:
        if not isinstance(event, dict):
            raise ValueError("trace event must be an object")
        source = _required_str(event, "source")
        target = _required_str(event, "target")
        source_type = _node_type(event.get("source_type"))
        target_type = _node_type(event.get("target_type"))
        token_cost = float(event.get("token_cost", event.get("tokens", 0.0)))
        latency = float(event.get("latency", 0.0))
        success = bool(event.get("success", True))

        node_types.setdefault(source, source_type)
        node_types.setdefault(target, target_type)
        node_stats[source]["out_messages"] += 1
        node_stats[target]["in_messages"] += 1
        node_stats[target]["token_cost"] += token_cost
        node_stats[target]["latency"] += latency
        if not success:
            node_stats[target]["errors"] += 1

        edge_key = (source, target)
        edge_stats[edge_key]["message_count"] += 1
        edge_stats[edge_key]["message_cost"] += token_cost
        edge_stats[edge_key]["latency"] += latency
        if success:
            edge_stats[edge_key]["successes"] += 1

    graph = AgentGraph()
    for node_id, stats in sorted(node_stats.items()):
        total_messages = stats["in_messages"] + stats["out_messages"]
        error_rate = stats["errors"] / max(stats["in_messages"], 1.0)
        graph.add_node(
            node_id,
            type=node_types.get(node_id, NodeType.AGENT),
            token_cost=stats["token_cost"],
            latency=stats["latency"] / max(stats["in_messages"], 1.0),
            reliability=1.0 - error_rate,
            error_rate=error_rate,
            trace_message_count=total_messages,
        )

    for (source, target), stats in sorted(edge_stats.items()):
        message_count = stats["message_count"]
        graph.add_edge(
            source,
            target,
            message_cost=stats["message_cost"],
            latency=stats["latency"] / max(message_count, 1.0),
            reliability=stats["successes"] / max(message_count, 1.0),
            activation_probability=min(1.0, message_count / max(len(events), 1)),
            weight=message_count,
            trace_message_count=message_count,
        )

    return TraceLoadResult(
        graph=graph,
        message_count=len(events),
        total_token_cost=sum(stats["message_cost"] for stats in edge_stats.values()),
        total_latency=sum(stats["latency"] for stats in edge_stats.values()),
    )


def _required_str(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"trace event missing non-empty {key}")
    return value


def _node_type(value: Any) -> NodeType:
    if value is None:
        return NodeType.AGENT
    return NodeType(str(value))
