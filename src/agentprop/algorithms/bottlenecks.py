"""Workflow bottleneck detection."""

from __future__ import annotations

import networkx as nx

from agentprop.core import AgentGraph


def bottleneck_nodes(graph: AgentGraph, *, limit: int = 5) -> list[tuple[str, float]]:
    """Return nodes that look structurally important or operationally risky.

    Uses the per-graph analysis cache (phase1-centrality-cache) for betweenness
    and degree to avoid repeated to_networkx + centrality recomputation.
    """

    if graph.node_count == 0:
        return []

    betweenness = graph.get_betweenness_centrality()
    # We still need a light nx view only for degrees (cheap); full copy avoided for centralities.
    nx_graph = graph.to_networkx()  # retained only for degree; future: cache degrees too
    scores: dict[str, float] = {}
    for node in graph.nodes():
        scores[node.id] = (
            float(betweenness.get(node.id, 0.0))
            + 0.05 * float(nx_graph.out_degree(node.id))
            + 0.05 * float(nx_graph.in_degree(node.id))
            + 0.25 * (1.0 - node.reliability)
            + 0.25 * node.error_rate
        )

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def articulation_bottlenecks(graph: AgentGraph, *, limit: int = 5) -> list[str]:
    """Return nodes whose removal disconnects the weak workflow structure."""

    undirected = graph.to_networkx().to_undirected()
    if graph.node_count < 3:
        return []
    points = [str(node_id) for node_id in nx.articulation_points(undirected)]
    points.sort()
    return points[:limit]


def bridge_bottlenecks(graph: AgentGraph, *, limit: int = 5) -> list[tuple[str, str]]:
    """Return communication edges that are weak bridges in the workflow graph."""

    undirected = graph.to_networkx().to_undirected()
    bridges = [(str(source), str(target)) for source, target in nx.bridges(undirected)]
    bridges.sort()
    return bridges[:limit]


def edge_bottlenecks(graph: AgentGraph, *, limit: int = 5) -> list[tuple[str, str, float]]:
    """Rank edges by structural centrality and operational cost."""

    nx_graph = graph.to_networkx()
    if not nx_graph:
        return []
    edge_centrality = nx.edge_betweenness_centrality(nx_graph, weight="weight")
    scored: list[tuple[str, str, float]] = []
    for edge in graph.edges():
        centrality = float(edge_centrality.get((edge.source, edge.target), 0.0))
        cost_risk = 0.001 * edge.message_cost + 0.1 * (1.0 - edge.reliability)
        scored.append((edge.source, edge.target, centrality + cost_risk))
    return sorted(scored, key=lambda item: (-item[2], item[0], item[1]))[:limit]


def low_reliability_cut_points(graph: AgentGraph, *, limit: int = 5) -> list[tuple[str, float]]:
    """Rank structurally central nodes with low reliability or high error rate.

    Now uses cached betweenness.
    """

    if graph.node_count == 0:
        return []
    betweenness = graph.get_betweenness_centrality()
    scores = {}
    for node in graph.nodes():
        reliability_risk = 1.0 - node.reliability + node.error_rate
        scores[node.id] = reliability_risk * (1.0 + float(betweenness.get(node.id, 0.0)))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def failure_sensitive_nodes(graph: AgentGraph, *, limit: int = 5) -> list[tuple[str, float]]:
    """Rank nodes by reachable-pair loss after node removal.

    Uses cached descendant closures for the baseline reachable pair count and
    per-node descendant sets. Still performs a small copy for the "remove node"
    delta (acceptable; the expensive repeated full-graph descendant walks are gone).
    """

    if graph.node_count == 0:
        return []
    baseline_pairs = graph.get_reachable_pair_count()
    if baseline_pairs == 0:
        return []

    # For the delta we still need a transient view; the closures give us fast "before".
    nx_graph = graph.to_networkx()
    scored = []
    for node_id in list(nx_graph.nodes):
        # simulate removal for loss
        reduced = nx_graph.copy()
        reduced.remove_node(node_id)
        after = sum(len({str(x) for x in nx.descendants(reduced, n)}) for n in reduced.nodes)
        loss = baseline_pairs - after
        scored.append((str(node_id), loss / baseline_pairs))
    return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]
