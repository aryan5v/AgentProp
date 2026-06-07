"""Scaffold starter workflow graphs from a node list and a topology.

This powers ``agentprop init``: instead of hand-authoring a workflow JSON file,
users name a few nodes and pick a topology, and AgentProp emits a valid starter
graph they can edit. The output passes :func:`validate_workflow_dict` and can be
loaded directly by :meth:`AgentGraph.from_json`.
"""

from __future__ import annotations

from agentprop.core import AgentGraph, NodeType

#: Supported starter topologies for ``scaffold_workflow``.
TOPOLOGIES: tuple[str, ...] = ("pipeline", "fan-out", "hub-spoke")

# Default per-node metrics. These are intentionally generic placeholders that
# users are expected to tune; they exist so the starter graph is immediately
# runnable through analysis/optimize without divide-by-zero or empty-cost edges.
_DEFAULT_TOKEN_COST = 1000.0
_PLANNER_TOKEN_COST = 1200.0
_DEFAULT_LATENCY = 1.2
_DEFAULT_RELIABILITY = 0.85
_DEFAULT_ERROR_RATE = 0.10
_DEFAULT_MESSAGE_COST = 400.0
_DEFAULT_EDGE_WEIGHT = 0.8


def scaffold_workflow(nodes: list[str], *, topology: str = "pipeline") -> AgentGraph:
    """Build a starter :class:`AgentGraph` from node ids and a topology.

    Args:
        nodes: Ordered, non-empty list of unique node ids. The first node is
            treated as the planner/dispatcher; the last (for pipelines) as the
            reviewer.
        topology: One of :data:`TOPOLOGIES`:

            * ``"pipeline"`` — a linear chain ``n0 -> n1 -> ... -> n_k``.
            * ``"fan-out"`` — ``n0`` dispatches to every other node.
            * ``"hub-spoke"`` — ``n0`` is a hub with two-way edges to each spoke.

    Returns:
        A populated, validated :class:`AgentGraph`.

    Raises:
        ValueError: If ``nodes`` is empty, contains duplicates or blank ids, the
            topology is unknown, or the topology needs more nodes than supplied.
    """

    if topology not in TOPOLOGIES:
        raise ValueError(
            f"Unknown topology: {topology!r}. Choose one of {', '.join(TOPOLOGIES)}."
        )
    cleaned = [node.strip() for node in nodes]
    if not cleaned or any(not node for node in cleaned):
        raise ValueError("nodes must be a non-empty list of non-blank ids.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("nodes must be unique; found duplicate ids.")
    if topology in {"fan-out", "hub-spoke"} and len(cleaned) < 2:
        raise ValueError(f"topology {topology!r} requires at least 2 nodes.")

    graph = AgentGraph()
    for index, node_id in enumerate(cleaned):
        graph.add_node(
            node_id,
            type=_node_type_for(index, len(cleaned), topology),
            token_cost=_PLANNER_TOKEN_COST if index == 0 else _DEFAULT_TOKEN_COST,
            latency=_DEFAULT_LATENCY,
            reliability=_DEFAULT_RELIABILITY,
            error_rate=_DEFAULT_ERROR_RATE,
        )

    if topology == "pipeline":
        _wire_pipeline(graph, cleaned)
    elif topology == "fan-out":
        _wire_fan_out(graph, cleaned)
    else:  # hub-spoke
        _wire_hub_spoke(graph, cleaned)

    return graph


def _node_type_for(index: int, count: int, topology: str) -> NodeType:
    if index == 0:
        return NodeType.PLANNER
    if topology == "pipeline" and index == count - 1:
        return NodeType.REVIEWER
    return NodeType.EXECUTOR


def _wire_pipeline(graph: AgentGraph, nodes: list[str]) -> None:
    for source, target in zip(nodes, nodes[1:], strict=False):
        graph.add_edge(
            source,
            target,
            message_cost=_DEFAULT_MESSAGE_COST,
            latency=0.4,
            weight=_DEFAULT_EDGE_WEIGHT,
        )


def _wire_fan_out(graph: AgentGraph, nodes: list[str]) -> None:
    hub = nodes[0]
    for target in nodes[1:]:
        graph.add_edge(
            hub,
            target,
            message_cost=_DEFAULT_MESSAGE_COST,
            latency=0.4,
            weight=_DEFAULT_EDGE_WEIGHT,
        )


def _wire_hub_spoke(graph: AgentGraph, nodes: list[str]) -> None:
    hub = nodes[0]
    for spoke in nodes[1:]:
        graph.add_edge(
            hub,
            spoke,
            message_cost=_DEFAULT_MESSAGE_COST,
            latency=0.3,
            weight=_DEFAULT_EDGE_WEIGHT,
        )
        graph.add_edge(
            spoke,
            hub,
            message_cost=_DEFAULT_MESSAGE_COST * 0.75,
            latency=0.3,
            weight=_DEFAULT_EDGE_WEIGHT * 0.85,
        )
