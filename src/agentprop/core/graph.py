"""Core directed graph abstraction for AgentProp."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from agentprop.core.models import AgentEdge, AgentNode
from agentprop.core.types import NodeType

if TYPE_CHECKING:
    pass
from agentprop.core.validation import validate_workflow_dict


@dataclass(slots=True)
class GraphAnalysisCache:
    """Per-graph memo for derived analysis results (distances, centralities, closures, etc).

    Populated lazily by algorithms to avoid repeated expensive graph computations
    (all-pairs shortest paths, betweenness, core numbers, ancestor/descendant closures,
    etc.). Callers should treat fields as read-only; use the AgentGraph accessors.
    The cache is invalidated on any structural mutation (add_node / add_edge).
    """

    # Distances (for metric dimension / resolving sets)
    undirected_node_ids: list[str] | None = None
    undirected_distances: dict[str, dict[str, int]] | None = None

    # Centralities (weighted where applicable)
    betweenness: dict[str, float] | None = None
    edge_betweenness: dict[tuple[str, str], float] | None = None
    closeness: dict[str, float] | None = None   # downstream closeness (reverse graph)
    core_numbers: dict[str, int] | None = None  # undirected k-core

    # Reachability closures (frequently used by bottlenecks, observability, pruning risk)
    ancestor_closures: dict[str, set[str]] | None = None
    descendant_closures: dict[str, set[str]] | None = None

    # Stable fingerprint for the current graph structure (used to decide when to trust cache
    # across sessions or for future persistent caching). Invalidated on mutation.
    fingerprint: str | None = None

    # Integer-indexed adjacency for fast propagation (phase1-fast-propagation).
    propagation_index: Any | None = None


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

    def has_node(self, node_id: str) -> bool:
        """Return True when ``node_id`` exists (no graph copy)."""

        return bool(node_id in self._graph)

    def has_edge(self, source: str, target: str) -> bool:
        """Return True when a directed edge exists (no graph copy)."""

        return bool(self._graph.has_edge(source, target))

    def node_ids(self) -> list[str]:
        """Return all node ids without materializing AgentNode objects."""

        return [str(node_id) for node_id in self._graph.nodes()]

    def is_dag(self) -> bool:
        """Return True when the graph is a directed acyclic graph."""

        return bool(nx.is_directed_acyclic_graph(self._graph))

    def topological_order(self) -> list[str]:
        """Return a topological order for DAG workflows."""

        if not self.is_dag():
            raise ValueError("graph is not a DAG")
        return [str(node_id) for node_id in nx.topological_sort(self._graph)]

    def out_degree(self, node_id: str) -> int:
        """Outgoing edge count for a node."""

        return int(self._graph.out_degree(node_id))

    def in_degree(self, node_id: str) -> int:
        """Incoming edge count for a node."""

        return int(self._graph.in_degree(node_id))

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
        c = self._analysis_cache
        c.undirected_node_ids = None
        c.undirected_distances = None
        c.betweenness = None
        c.edge_betweenness = None
        c.closeness = None
        c.core_numbers = None
        c.ancestor_closures = None
        c.descendant_closures = None
        c.fingerprint = None
        c.propagation_index = None

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

    # --- Phase 1 centrality / closure cache accessors (phase1-centrality-cache) ---

    def _compute_fingerprint(self) -> str:
        """Stable lightweight fingerprint of the current graph structure.

        Used to guard cached values and (in the future) for cross-session or
        on-disk cache keys. Changes on any node/edge add/remove or (for now)
        on any mutation that calls _clear_analysis_cache.
        """
        nodes = tuple(sorted(str(n) for n in self._graph.nodes()))
        # Include weight when present for weighted centralities to be safe
        edges = tuple(
            sorted(
                (str(u), str(v), float(d.get("weight", 1.0)))
                for u, v, d in self._graph.edges(data=True)
            )
        )
        import hashlib
        fingerprint_data = f"{nodes}:{edges}".encode()
        h = hashlib.md5(fingerprint_data).hexdigest()
        return f"n{len(nodes)}e{len(edges)}:{h}"

    def _ensure_centrality_cache(self) -> None:
        """Populate the expensive centrality fields if missing."""
        c = self._analysis_cache
        if c.betweenness is None or c.closeness is None or c.core_numbers is None:
            nx_graph = self._graph
            c.betweenness = (
                nx.betweenness_centrality(nx_graph, weight="weight") if self.node_count else {}
            )
            # downstream closeness (standard for "how central as a source of info")
            c.closeness = (
                nx.closeness_centrality(nx_graph.reverse(copy=True)) if self.node_count else {}
            )
            c.core_numbers = nx.core_number(nx_graph.to_undirected()) if self.node_count else {}
            c.fingerprint = self._compute_fingerprint()

        if c.edge_betweenness is None:
            nx_graph = self._graph
            c.edge_betweenness = (
                nx.edge_betweenness_centrality(nx_graph, weight="weight") if self.edge_count else {}
            )

    def _ensure_reachability_closures(self) -> None:
        """Populate ancestor and descendant closures for all nodes (lazily)."""
        c = self._analysis_cache
        if c.ancestor_closures is None or c.descendant_closures is None:
            nx_graph = self._graph
            c.ancestor_closures = {}
            c.descendant_closures = {}
            for nid in nx_graph.nodes():
                s = str(nid)
                c.ancestor_closures[s] = {str(x) for x in nx.ancestors(nx_graph, nid)}
                c.descendant_closures[s] = {str(x) for x in nx.descendants(nx_graph, nid)}
            c.fingerprint = self._compute_fingerprint()

    def get_betweenness_centrality(self) -> dict[str, float]:
        """Cached weighted betweenness centrality."""
        self._ensure_centrality_cache()
        return self._analysis_cache.betweenness or {}

    def get_edge_betweenness_centrality(self) -> dict[tuple[str, str], float]:
        """Cached weighted edge betweenness centrality."""
        self._ensure_centrality_cache()
        return self._analysis_cache.edge_betweenness or {}

    def get_closeness_centrality(self) -> dict[str, float]:
        """Cached closeness on the reverse graph (downstream influence)."""
        self._ensure_centrality_cache()
        return self._analysis_cache.closeness or {}

    def get_core_numbers(self) -> dict[str, int]:
        """Cached undirected k-core numbers."""
        self._ensure_centrality_cache()
        return self._analysis_cache.core_numbers or {}

    def get_ancestors(self, node_id: str) -> set[str]:
        """Cached set of (proper) ancestors of node_id."""
        self._ensure_reachability_closures()
        return (self._analysis_cache.ancestor_closures or {}).get(str(node_id), set())

    def get_descendants(self, node_id: str) -> set[str]:
        """Cached set of (proper) descendants of node_id."""
        self._ensure_reachability_closures()
        return (self._analysis_cache.descendant_closures or {}).get(str(node_id), set())

    def get_reachable_pair_count(self) -> float:
        """Total number of reachable (source, descendant) pairs. Cached via closures."""
        self._ensure_reachability_closures()
        closures = self._analysis_cache.descendant_closures or {}
        return float(sum(len(s) for s in closures.values()))

    def get_propagation_index(self) -> Any:
        """Return cached integer-indexed adjacency for propagation kernels."""

        from agentprop.core.propagation_index import build_propagation_index

        cache = self._analysis_cache
        if cache.propagation_index is None:
            cache.propagation_index = build_propagation_index(self)
            cache.fingerprint = self._compute_fingerprint()
        return cache.propagation_index

    def analysis_fingerprint(self) -> str | None:
        """Return the current graph fingerprint if the analysis cache is warm."""

        return self._analysis_cache.fingerprint

    def warm_analysis_cache(self) -> str:
        """Precompute distances, centralities, and propagation index; return fingerprint."""

        self.get_undirected_distances()
        self._ensure_centrality_cache()
        self._ensure_reachability_closures()
        self.get_propagation_index()
        return self._analysis_cache.fingerprint or self._compute_fingerprint()

    def export_analysis_cache(self) -> GraphAnalysisCache:
        """Return a shallow snapshot of the warm analysis cache."""

        self.warm_analysis_cache()
        return self._analysis_cache

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
