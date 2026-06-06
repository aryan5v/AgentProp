"""Runtime graph mutations and conditional edge resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agentprop.core.graph import AgentGraph
from agentprop.core.models import AgentEdge, AgentNode


@dataclass(frozen=True, slots=True)
class GraphMutation:
    """One structural change applied to a dynamic workflow graph."""

    action: str
    node_id: str | None = None
    source: str | None = None
    target: str | None = None
    version: int = 0


@dataclass(slots=True)
class DynamicGraphSession:
    """Track runtime add/remove mutations and resolve conditional edges."""

    base_graph: AgentGraph
    version: int = 0
    mutations: list[GraphMutation] = field(default_factory=list)
    _working: AgentGraph = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._working = AgentGraph.from_dict(self.base_graph.to_dict())

    @property
    def graph(self) -> AgentGraph:
        return self._working

    def add_node(self, node_id: str, **metadata: Any) -> AgentNode:
        node = self._working.add_node(node_id, **metadata)
        self._record("add_node", node_id=node_id)
        return node

    def remove_node(self, node_id: str) -> None:
        self._working.remove_node(node_id)
        self._record("remove_node", node_id=node_id)

    def add_edge(self, source: str, target: str, **metadata: Any) -> AgentEdge:
        edge = self._working.add_edge(source, target, **metadata)
        self._record("add_edge", source=source, target=target)
        return edge

    def add_conditional_edge(
        self,
        source: str,
        target: str,
        *,
        condition_key: str,
        condition_value: object,
        **metadata: Any,
    ) -> AgentEdge:
        edge = self._working.add_conditional_edge(
            source,
            target,
            condition_key=condition_key,
            condition_value=condition_value,
            **metadata,
        )
        self._record("add_edge", source=source, target=target)
        return edge

    def remove_edge(self, source: str, target: str) -> None:
        self._working.remove_edge(source, target)
        self._record("remove_edge", source=source, target=target)

    def active_graph(self, context: Mapping[str, Any] | None = None) -> AgentGraph:
        """Return a snapshot with only edges whose conditions match *context*."""

        return self._working.filter_active_edges(context or {})

    def mutations_to_dict(self) -> list[dict[str, object]]:
        return [
            {
                "action": mutation.action,
                "node_id": mutation.node_id,
                "source": mutation.source,
                "target": mutation.target,
                "version": mutation.version,
            }
            for mutation in self.mutations
        ]

    def _record(self, action: str, **payload: str | None) -> None:
        self.version += 1
        self.mutations.append(
            GraphMutation(
                action=action,
                node_id=payload.get("node_id"),
                source=payload.get("source"),
                target=payload.get("target"),
                version=self.version,
            )
        )


def edge_is_active(edge: AgentEdge, context: Mapping[str, Any]) -> bool:
    """Return whether a conditional edge is active under *context*."""

    condition_key = edge.metadata.get("condition_key")
    if condition_key is None:
        return True
    expected = edge.metadata.get("condition_value")
    actual = context.get(str(condition_key))
    if expected is None:
        return bool(actual)
    return actual == expected
