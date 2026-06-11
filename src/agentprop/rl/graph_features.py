"""Graph-position features for bandit reward records (RL flywheel, phase 1).

The contextual-bandit literature solves LLM routing at the query level; the
edge AgentProp brings is the *workflow-graph context* around each decision.
This module computes that context so every controlled run logs it from day
one, giving later phases (contextual bandit, policy transfer) training data.
"""

from __future__ import annotations

from collections import deque
from statistics import fmean

from agentprop.algorithms.verifier_placement import resolving_coverage
from agentprop.core import AgentGraph
from agentprop.ml.features import extract_edge_features, extract_graph_features
from agentprop.propagation.quality_cascade import QualityCascade

REWARD_RECORD_SCHEMA_VERSION = 2


def workflow_embedding(graph: AgentGraph) -> dict[str, float]:
    """Mean/max pooled node and edge features plus a node-type histogram.

    The embedding is a flat name->value mapping so it can be logged as JSON
    and later consumed as a fixed-length vector (sorted by key).
    """

    embedding: dict[str, float] = {
        "node_count": float(graph.node_count),
        "edge_count": float(graph.edge_count),
        "max_depth": float(max(_node_depths(graph).values(), default=0)),
    }
    node_features = extract_graph_features(graph)
    for index, name in enumerate(node_features.feature_names):
        column = [row[index] for row in node_features.node_features.values()]
        embedding[f"node_{name}_mean"] = fmean(column) if column else 0.0
        embedding[f"node_{name}_max"] = max(column, default=0.0)
    edge_features = extract_edge_features(graph)
    for index, name in enumerate(edge_features.feature_names):
        column = [row[index] for row in edge_features.edge_features.values()]
        embedding[f"edge_{name}_mean"] = fmean(column) if column else 0.0
        embedding[f"edge_{name}_max"] = max(column, default=0.0)
    total = max(graph.node_count, 1)
    type_counts: dict[str, int] = {}
    for node in graph.nodes():
        key = str(node.type.value if hasattr(node.type, "value") else node.type)
        type_counts[key] = type_counts.get(key, 0) + 1
    for type_name, count in sorted(type_counts.items()):
        embedding[f"type_{type_name}_fraction"] = count / total
    return embedding


def node_position_features(
    graph: AgentGraph,
    node_id: str,
    *,
    verifiers: tuple[str, ...] | list[str] = (),
) -> dict[str, float | str | None]:
    """Position of one node: type, DAG depth, quality score, verifier coverage."""

    node = next((n for n in graph.nodes() if n.id == node_id), None)
    if node is None:
        raise ValueError(f"unknown node: {node_id}")
    depths = _node_depths(graph)
    sources = [n.id for n in graph.nodes() if depths.get(n.id, 0) == 0]
    qualities = (
        QualityCascade().simulate(graph, sources).node_qualities if sources else {}
    )
    active = [v for v in verifiers if v != node_id]
    coverage_with = resolving_coverage(graph, list(verifiers)) if verifiers else 0.0
    coverage_without = resolving_coverage(graph, active) if active else 0.0
    return {
        "node_id": node_id,
        "node_type": str(node.type.value if hasattr(node.type, "value") else node.type),
        "depth": float(depths.get(node_id, 0)),
        "quality_score": float(qualities.get(node_id, 0.0)),
        "resolving_coverage_active": coverage_with,
        "resolving_coverage_contribution": max(0.0, coverage_with - coverage_without),
    }


def reward_record_graph_features(
    graph: AgentGraph,
    *,
    node_id: str | None = None,
    verifiers: tuple[str, ...] | list[str] = (),
) -> dict[str, object]:
    """Full graph-context payload for one bandit reward record."""

    payload: dict[str, object] = {
        "schema_version": REWARD_RECORD_SCHEMA_VERSION,
        "workflow_embedding": workflow_embedding(graph),
        "resolving_coverage_active": (
            resolving_coverage(graph, list(verifiers)) if verifiers else 0.0
        ),
        "active_verifiers": list(verifiers),
    }
    if node_id is not None:
        payload["node"] = node_position_features(graph, node_id, verifiers=verifiers)
    return payload


def _node_depths(graph: AgentGraph) -> dict[str, int]:
    """BFS depth from in-degree-zero sources; cycle-safe."""

    nx_graph = graph.to_networkx()
    sources = [n for n in nx_graph.nodes if nx_graph.in_degree(n) == 0]
    if not sources:
        sources = list(nx_graph.nodes)[:1]
    depths: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((s, 0) for s in sources)
    while queue:
        node, depth = queue.popleft()
        if node in depths and depths[node] <= depth:
            continue
        depths[node] = depth
        for successor in nx_graph.successors(node):
            if successor not in depths:
                queue.append((successor, depth + 1))
    for node in nx_graph.nodes:
        depths.setdefault(node, 0)
    return depths
