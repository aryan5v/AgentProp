"""Core directed graph abstraction for AgentProp."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from agentprop.core.models import AgentEdge, AgentNode
from agentprop.core.types import NodeType
from agentprop.core.validation import validate_workflow_dict


@dataclass(slots=True)
class GraphAnalysisCache:
    """Per-graph memo for derived analysis results (distances, centralities, etc).

    Populated lazily by algorithms to avoid repeated expensive graph computations
    such as all-pairs shortest paths for metric dimension / resolving sets.
    Callers should treat fields as read-only; use AgentGraph helpers to populate.
    """

    undirected_node_ids: list[str] | None = None
    undirected_distances: dict[str, dict[str, int]] | None = None
    # Future phases will add: betweenness, core_numbers, ancestor closures, etc.


class AgentGraph:
    """A directed weighted graph of agents, tools, memories, and verifiers."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._analysis_cache: GraphAnalysisCache = GraphAnalysisCache()

    @property
    def node_count(self) -> int:
        return int(self._graph.number_of_nodes())

    @property
    def edge_count(self) -> int:
        return int(self._graph.number_of_edges())

    def add_node(
        self,
        node_id: str,
        *,
        type: NodeType = NodeType.AGENT,
        name: str | None = None,
        role: str | None = None,
        token_cost: float = 0.0,
        latency: float = 0.0,
        reliability: float = 1.0,
        error_rate: float = 0.0,
        context_capacity: int | None = None,
        tool_access: tuple[str, ...] | list[str] = (),
        importance_score: float | None = None,
        **metadata: Any,
    ) -> AgentNode:
        """Add or replace a node and return its typed metadata."""

        if not node_id:
            raise ValueError("node_id must be non-empty")

        node = AgentNode(
            id=node_id,
            type=type,
            name=name,
            role=role,
            token_cost=token_cost,
            latency=latency,
            reliability=reliability,
            error_rate=error_rate,
            context_capacity=context_capacity,
            tool_access=tuple(tool_access),
            importance_score=importance_score,
            metadata=metadata,
        )
        self._graph.add_node(node_id, **node.to_dict())
        self._clear_analysis_cache()
        return node

    def add_agent(self, node_id: str, **metadata: Any) -> AgentNode:
        """Convenience wrapper for adding an agent node."""

        return self.add_node(node_id, type=NodeType.AGENT, **metadata)

    def add_verifier(self, node_id: str, **metadata: Any) -> AgentNode:
        """Convenience wrapper for adding a verifier node."""

        return self.add_node(node_id, type=NodeType.VERIFIER, **metadata)

    def add_tool(self, node_id: str, **metadata: Any) -> AgentNode:
        """Convenience wrapper for adding a tool node."""

        return self.add_node(node_id, type=NodeType.TOOL, **metadata)

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        cost: float | None = None,
        message_cost: float = 0.0,
        latency: float = 0.0,
        relevance: float = 1.0,
        reliability: float = 1.0,
        activation_probability: float = 0.85,
        dependency_strength: float = 1.0,
        weight: float = 1.0,
        **metadata: Any,
    ) -> AgentEdge:
        """Add a directed edge and return its typed metadata."""

        if source not in self._graph:
            raise ValueError(f"Unknown source node: {source}")
        if target not in self._graph:
            raise ValueError(f"Unknown target node: {target}")

        if cost is not None:
            message_cost = cost

        edge = AgentEdge(
            source=source,
            target=target,
            message_cost=message_cost,
            latency=latency,
            relevance=relevance,
            reliability=reliability,
            activation_probability=activation_probability,
            dependency_strength=dependency_strength,
            weight=weight,
            metadata=metadata,
        )
        self._graph.add_edge(source, target, **edge.to_dict())
        self._clear_analysis_cache()
        return edge

    def node(self, node_id: str) -> AgentNode:
        """Return typed node metadata."""

        if node_id not in self._graph:
            raise KeyError(node_id)
        return AgentNode.from_dict(dict(self._graph.nodes[node_id]))

    def edge(self, source: str, target: str) -> AgentEdge:
        """Return typed edge metadata."""

        if not self._graph.has_edge(source, target):
            raise KeyError((source, target))
        return AgentEdge.from_dict(dict(self._graph.edges[source, target]))

    def nodes(self) -> list[AgentNode]:
        """Return all nodes as typed metadata objects."""

        return [AgentNode.from_dict(dict(data)) for _, data in self._graph.nodes(data=True)]

    def edges(self) -> list[AgentEdge]:
        """Return all edges as typed metadata objects."""

        return [AgentEdge.from_dict(dict(data)) for _, _, data in self._graph.edges(data=True)]

    def to_networkx(self) -> nx.DiGraph:
        """Return a copy of the underlying NetworkX directed graph."""

        return self._graph.copy()

    @classmethod
    def from_networkx(cls, nx_graph: nx.DiGraph) -> AgentGraph:
        """Build an AgentGraph from a NetworkX directed graph."""

        graph = cls()
        for node_id, data in nx_graph.nodes(data=True):
            node_data = dict(data)
            node_data.setdefault("id", str(node_id))
            node = AgentNode.from_dict(node_data)
            graph._graph.add_node(node.id, **node.to_dict())

        for source, target, data in nx_graph.edges(data=True):
            edge_data = dict(data)
            edge_data.setdefault("source", str(source))
            edge_data.setdefault("target", str(target))
            edge = AgentEdge.from_dict(edge_data)
            graph._graph.add_edge(edge.source, edge.target, **edge.to_dict())

        return graph

    def successors(self, node_id: str) -> list[str]:
        """Return outgoing neighbor ids."""

        return list(self._graph.successors(node_id))

    def predecessors(self, node_id: str) -> list[str]:
        """Return incoming neighbor ids."""

        return list(self._graph.predecessors(node_id))

    def _clear_analysis_cache(self) -> None:
        """Invalidate cached analysis results (e.g. after structural mutation)."""
        self._analysis_cache.undirected_node_ids = None
        self._analysis_cache.undirected_distances = None

    def _ensure_undirected_distances(self) -> tuple[list[str], dict[str, dict[str, int]]]:
        """Return (node_ids, distances) using memoized all-pairs shortest paths on the
        undirected version of the graph. Used by metric dimension / resolving set
        algorithms to avoid O(n * (n+m)) recomputation on every call.
        """
        cache = self._analysis_cache
        if cache.undirected_distances is None or cache.undirected_node_ids is None:
            nx_ug = self._graph.to_undirected()
            node_ids = sorted(str(n) for n in nx_ug.nodes())
            distances = dict(nx.all_pairs_shortest_path_length(nx_ug))
            cache.undirected_node_ids = node_ids
            cache.undirected_distances = distances
        return cache.undirected_node_ids, cache.undirected_distances

    def get_undirected_distances(self) -> tuple[list[str], dict[str, dict[str, int]]]:
        """Return cached (node_ids, distances) for undirected shortest paths.

        This powers metric-dimension verifier placement and resolving coverage
        without repeated all-pairs computations. The cache is invalidated on
        structural changes (add_node/add_edge).
        """
        return self._ensure_undirected_distances()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to JSON-compatible data."""

        return {
            "nodes": [node.to_dict() for node in self.nodes()],
            "edges": [edge.to_dict() for edge in self.edges()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentGraph:
        """Build a graph from JSON-compatible data."""

        validate_workflow_dict(data)
        graph = cls()
        for raw_node in data.get("nodes", []):
            node = AgentNode.from_dict(raw_node)
            graph._graph.add_node(node.id, **node.to_dict())

        for raw_edge in data.get("edges", []):
            edge = AgentEdge.from_dict(raw_edge)
            if edge.source not in graph._graph:
                raise ValueError(f"Unknown source node in edge: {edge.source}")
            if edge.target not in graph._graph:
                raise ValueError(f"Unknown target node in edge: {edge.target}")
            graph._graph.add_edge(edge.source, edge.target, **edge.to_dict())

        return graph

    def to_json(self, path: str | Path) -> None:
        """Write the graph as formatted JSON."""

        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> AgentGraph:
        """Load a graph from JSON."""

        return cls.from_dict(json.loads(Path(path).read_text()))
