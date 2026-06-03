"""Quality cascade propagation model for continuous context-quality routing."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from agentprop.core import AgentGraph, NodeType
from agentprop.propagation.base import PropagationResult


@dataclass(slots=True)
class QualityCascadeResult(PropagationResult):
    """PropagationResult extended with per-node quality scores."""

    node_qualities: dict[str, float] = field(default_factory=dict)
    mean_output_quality: float = 0.0


class QualityCascade:
    """Propagate continuous quality scores rather than binary activation.

    Each seed node starts with quality 1.0. Non-seed nodes receive quality
    degraded by edge relevance and reliability. This models how compressed
    context reduces downstream output quality in LLM workflows — unlike
    Independent Cascade which treats activation as binary.

    Aggregation modes:
    - "max": node takes the best-quality incoming signal (default).
      Models an agent that can selectively use its best input.
    - "mean": node averages incoming quality weighted by edge weight.
      Models an agent that integrates all inputs equally.
    """

    name = "quality-cascade"

    def __init__(
        self,
        *,
        quality_floor: float = 0.1,
        aggregation: str = "max",
    ) -> None:
        if quality_floor < 0.0 or quality_floor > 1.0:
            raise ValueError("quality_floor must be between 0 and 1")
        if aggregation not in {"max", "mean"}:
            raise ValueError("aggregation must be 'max' or 'mean'")
        self.quality_floor = quality_floor
        self.aggregation = aggregation

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 1,
    ) -> QualityCascadeResult:
        """Propagate quality from seeds through the workflow graph."""

        seed_set = set(seeds)
        valid_ids = {n.id for n in graph.nodes()}
        qualities: dict[str, float] = {s: 1.0 for s in seed_set if s in valid_ids}
        nx_graph = graph.to_networkx()

        if nx.is_directed_acyclic_graph(nx_graph):
            self._propagate_dag(graph, nx_graph, qualities, seed_set)
        else:
            self._propagate_cyclic(graph, nx_graph, qualities, seed_set)

        activation_rounds: dict[str, int] = {
            node_id: 0
            for node_id in qualities
            if qualities[node_id] >= self.quality_floor
        }
        node_count = max(graph.node_count, 1)
        activated = {nid for nid, q in qualities.items() if q >= self.quality_floor}
        coverage = len(activated) / node_count
        propagation_time = max(activation_rounds.values(), default=0)

        output_qualities = [
            qualities[node.id]
            for node in graph.nodes()
            if node.type == NodeType.OUTPUT and node.id in qualities
        ]
        if not output_qualities:
            terminal_ids = {
                str(n) for n in nx_graph.nodes()
                if nx_graph.out_degree(n) == 0 and str(n) in qualities
            }
            output_qualities = [qualities[nid] for nid in terminal_ids]
        mean_output_quality = sum(output_qualities) / max(len(output_qualities), 1)

        return QualityCascadeResult(
            activated_nodes=activated,
            propagation_time=propagation_time,
            coverage=coverage,
            activation_rounds=activation_rounds,
            trials=trials,
            full_activation_probability=1.0 if len(activated) == graph.node_count else 0.0,
            expected_propagation_time=float(propagation_time),
            coverage_by_round=None,
            node_qualities=qualities,
            mean_output_quality=mean_output_quality,
        )

    def _propagate_dag(
        self,
        graph: AgentGraph,
        nx_graph: nx.DiGraph,
        qualities: dict[str, float],
        seed_set: set[str],
    ) -> None:
        for node_id in nx.topological_sort(nx_graph):
            node_id = str(node_id)
            if node_id in seed_set:
                continue
            quality = self._incoming_quality(graph, node_id, qualities)
            if quality >= self.quality_floor:
                qualities[node_id] = quality

    def _propagate_cyclic(
        self,
        graph: AgentGraph,
        nx_graph: nx.DiGraph,
        qualities: dict[str, float],
        seed_set: set[str],
    ) -> None:
        max_rounds = graph.node_count
        for _ in range(max_rounds):
            updated = False
            for node_id in nx_graph.nodes():
                node_id = str(node_id)
                if node_id in seed_set:
                    continue
                quality = self._incoming_quality(graph, node_id, qualities)
                if quality >= self.quality_floor and qualities.get(node_id, 0.0) < quality:
                    qualities[node_id] = quality
                    updated = True
            if not updated:
                break

    def _incoming_quality(
        self,
        graph: AgentGraph,
        node_id: str,
        qualities: dict[str, float],
    ) -> float:
        predecessors = graph.predecessors(node_id)
        active_preds = [
            p for p in predecessors if p in qualities and qualities[p] >= self.quality_floor
        ]
        if not active_preds:
            return 0.0

        if self.aggregation == "max":
            return max(
                qualities[p] * graph.edge(p, node_id).relevance * graph.edge(p, node_id).reliability
                for p in active_preds
            )

        total_weight = sum(max(graph.edge(p, node_id).weight, 0.0) for p in active_preds)
        if total_weight == 0.0:
            return max(
                qualities[p] * graph.edge(p, node_id).relevance * graph.edge(p, node_id).reliability
                for p in active_preds
            )
        return sum(
            qualities[p]
            * graph.edge(p, node_id).relevance
            * graph.edge(p, node_id).reliability
            * max(graph.edge(p, node_id).weight, 0.0)
            / total_weight
            for p in active_preds
        )
