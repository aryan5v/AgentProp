"""User-facing workflow analysis reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentprop.algorithms import bottleneck_nodes, low_weight_edges
from agentprop.algorithms.seed_selection import auto_seed_algorithm
from agentprop.algorithms.verifier_placement import (
    fault_tolerant_resolving_coverage,
    metric_dimension_verifier_placement,
    resolving_coverage,
)
from agentprop.core import AgentGraph
from agentprop.evaluation.runner import make_propagation_model, select_seeds
from agentprop.integrations.framework_adapters import graph_from_langgraph_object
from agentprop.workflows import WORKFLOW_TEMPLATES


@dataclass(frozen=True, slots=True)
class AnalyzeReport:
    """Read-only graph analysis for developer onboarding and CI reports."""

    workflow: str
    nodes: int
    edges: int
    verifier_placement: tuple[str, ...]
    resolving_coverage: float
    fault_tolerant_coverage: float
    broadcast_cost: float
    constrained_cost: float
    constrained_savings: float
    bottlenecks: tuple[tuple[str, float], ...]
    recommended_seed_budget: int
    recommended_seeds: tuple[str, ...]
    pruning_candidates: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to JSON-compatible data."""

        return {
            "workflow": self.workflow,
            "nodes": self.nodes,
            "edges": self.edges,
            "verifier_placement": list(self.verifier_placement),
            "resolving_coverage": self.resolving_coverage,
            "fault_tolerant_coverage": self.fault_tolerant_coverage,
            "broadcast_cost": self.broadcast_cost,
            "constrained_cost": self.constrained_cost,
            "constrained_savings": self.constrained_savings,
            "bottlenecks": [
                {"node": node_id, "score": score} for node_id, score in self.bottlenecks
            ],
            "recommended_seed_budget": self.recommended_seed_budget,
            "recommended_seeds": list(self.recommended_seeds),
            "pruning_candidates": [
                {"source": source, "target": target}
                for source, target in self.pruning_candidates
            ],
        }

    def to_json(self) -> str:
        """Render the report as deterministic JSON."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        """Render the report as Markdown."""

        lines = [
            "# AgentProp Workflow Analysis",
            "",
            f"- Workflow: `{self.workflow}`",
            f"- Nodes: `{self.nodes}`",
            f"- Edges: `{self.edges}`",
            f"- Recommended seed budget: `{self.recommended_seed_budget}`",
            f"- Recommended seeds: `{_join(self.recommended_seeds)}`",
            "",
            "## Verifier Placement",
            f"- Recommended verifiers: `{_join(self.verifier_placement)}`",
            f"- Resolving coverage: `{self.resolving_coverage:.1%}`",
            f"- Single-dropout coverage: `{self.fault_tolerant_coverage:.1%}`",
            "",
            "## Cost Estimate",
            f"- Broadcast baseline: `{self.broadcast_cost:.0f}`",
            f"- Constrained routing: `{self.constrained_cost:.0f}`",
            f"- Constrained savings: `{self.constrained_savings:.1%}`",
            "",
            "## Bottlenecks",
            *_ranked_lines(self.bottlenecks),
            "",
            "## Pruning Candidates",
            *_edge_lines(self.pruning_candidates),
            "",
        ]
        return "\n".join(lines)


def analyze(
    workflow: str | Path | AgentGraph | object,
    *,
    workflow_name: str | None = None,
    seed_budget: int | None = None,
    verifier_budget: int | None = None,
    trials: int = 50,
) -> AnalyzeReport:
    """Analyze a workflow JSON path, built-in workflow, AgentGraph, or LangGraph object."""

    name, graph = _coerce_workflow(workflow, workflow_name=workflow_name)
    budget = seed_budget or _recommended_seed_budget(graph)
    verifier_k = verifier_budget or min(max(1, budget), max(graph.node_count, 1))
    verifier_placement = tuple(
        metric_dimension_verifier_placement(graph, verifier_k, fault_tolerant=True)
    )
    if not verifier_placement and graph.node_count:
        verifier_placement = tuple(node.id for node in graph.nodes()[:verifier_k])

    model = make_propagation_model("independent-cascade")
    algorithm = auto_seed_algorithm(graph, requested="auto")
    recommended_seeds = tuple(select_seeds(graph, algorithm, budget, model, trials))
    broadcast_cost = _broadcast_cost(graph)
    constrained_cost = _constrained_cost(graph, set(recommended_seeds), set(verifier_placement))
    constrained_savings = 0.0
    if broadcast_cost > 0:
        constrained_savings = max(0.0, (broadcast_cost - constrained_cost) / broadcast_cost)

    return AnalyzeReport(
        workflow=name,
        nodes=graph.node_count,
        edges=graph.edge_count,
        verifier_placement=verifier_placement,
        resolving_coverage=resolving_coverage(graph, list(verifier_placement)),
        fault_tolerant_coverage=fault_tolerant_resolving_coverage(
            graph, list(verifier_placement)
        ),
        broadcast_cost=broadcast_cost,
        constrained_cost=constrained_cost,
        constrained_savings=constrained_savings,
        bottlenecks=tuple(bottleneck_nodes(graph)),
        recommended_seed_budget=budget,
        recommended_seeds=recommended_seeds,
        pruning_candidates=tuple(low_weight_edges(graph)),
    )


def _coerce_workflow(
    workflow: str | Path | AgentGraph | object,
    *,
    workflow_name: str | None,
) -> tuple[str, AgentGraph]:
    if isinstance(workflow, AgentGraph):
        return workflow_name or "custom", workflow
    if isinstance(workflow, Path):
        return workflow.stem, AgentGraph.from_json(workflow)
    if isinstance(workflow, str):
        if workflow in WORKFLOW_TEMPLATES:
            return workflow, WORKFLOW_TEMPLATES[workflow]()
        path = Path(workflow)
        if path.exists():
            return path.stem, AgentGraph.from_json(path)
        raise ValueError(f"Unknown workflow or path: {workflow}")
    return workflow_name or workflow.__class__.__name__, graph_from_langgraph_object(workflow)


def _recommended_seed_budget(graph: AgentGraph) -> int:
    if graph.node_count <= 1:
        return graph.node_count
    if graph.node_count <= 8:
        return 2
    if graph.node_count <= 25:
        return 3
    return max(4, min(8, int(round(graph.node_count**0.5))))


def _broadcast_cost(graph: AgentGraph) -> float:
    node_cost = sum(node.token_cost for node in graph.nodes())
    edge_cost = sum(edge.message_cost for edge in graph.edges())
    return float(node_cost + edge_cost)


def _constrained_cost(
    graph: AgentGraph,
    seeds: set[str],
    verifiers: set[str],
) -> float:
    total = 0.0
    for node in graph.nodes():
        ratio = 1.0 if node.id in seeds or node.id in verifiers else 0.35
        total += node.token_cost * ratio
    total += sum(edge.message_cost * 0.6 for edge in graph.edges())
    return float(total)


def _join(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"


def _ranked_lines(rows: tuple[tuple[str, float], ...]) -> list[str]:
    if not rows:
        return ["- None"]
    return [f"- `{node_id}`: `{score:.3f}`" for node_id, score in rows]


def _edge_lines(rows: tuple[tuple[str, str], ...]) -> list[str]:
    if not rows:
        return ["- None"]
    return [f"- `{source}` -> `{target}`" for source, target in rows]
