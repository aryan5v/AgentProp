"""Reusable agent workflow templates."""

from __future__ import annotations

import random
from typing import Any

from agentprop.core import AgentGraph, NodeType


def planner_coder_tester_reviewer() -> AgentGraph:
    """Planner-coder-tester-reviewer workflow used as the first demo."""

    graph = AgentGraph()
    graph.add_node(
        "planner",
        type=NodeType.PLANNER,
        token_cost=1200,
        latency=1.4,
        reliability=0.86,
        error_rate=0.10,
        role="decomposes the task and routes work",
        importance_score=0.55,
    )
    graph.add_node(
        "coder",
        type=NodeType.EXECUTOR,
        token_cost=1700,
        latency=2.1,
        reliability=0.80,
        error_rate=0.18,
        role="implements the solution",
        importance_score=1.0,
    )
    graph.add_node(
        "tester",
        type=NodeType.VERIFIER,
        token_cost=950,
        latency=1.6,
        reliability=0.88,
        error_rate=0.08,
        role="tests the implementation",
        importance_score=0.85,
    )
    graph.add_node(
        "reviewer",
        type=NodeType.REVIEWER,
        token_cost=1100,
        latency=1.8,
        reliability=0.90,
        error_rate=0.07,
        role="reviews correctness and maintainability",
        importance_score=0.65,
    )
    graph.add_node(
        "final",
        type=NodeType.OUTPUT,
        token_cost=350,
        latency=0.4,
        reliability=0.95,
        importance_score=0.0,
    )

    graph.add_edge("planner", "coder", message_cost=650, latency=0.7, weight=0.9)
    graph.add_edge("coder", "tester", message_cost=450, latency=0.5, weight=0.8)
    graph.add_edge("tester", "reviewer", message_cost=350, latency=0.5, weight=0.75)
    graph.add_edge("reviewer", "final", message_cost=250, latency=0.3, weight=0.9)
    graph.add_edge("planner", "reviewer", message_cost=400, latency=0.4, weight=0.45, relevance=0.7)
    graph.add_edge("tester", "coder", message_cost=300, latency=0.4, weight=0.55, relevance=0.8)
    return graph


def research_writer_verifier() -> AgentGraph:
    """Research-writing workflow with a verifier stage."""

    graph = AgentGraph()
    graph.add_node("planner", type=NodeType.PLANNER, token_cost=900, latency=1.0, reliability=0.86)
    graph.add_agent("researcher_a", token_cost=1300, latency=1.9, reliability=0.82, error_rate=0.14)
    graph.add_agent("researcher_b", token_cost=1300, latency=1.9, reliability=0.82, error_rate=0.14)
    graph.add_agent("writer", token_cost=1500, latency=2.0, reliability=0.84, error_rate=0.12)
    graph.add_verifier("verifier", token_cost=900, latency=1.4, reliability=0.91, error_rate=0.06)
    graph.add_node("final", type=NodeType.OUTPUT, token_cost=300, latency=0.3, reliability=0.95)

    graph.add_edge("planner", "researcher_a", message_cost=400, latency=0.4, weight=0.8)
    graph.add_edge("planner", "researcher_b", message_cost=400, latency=0.4, weight=0.8)
    graph.add_edge("researcher_a", "writer", message_cost=500, latency=0.5, weight=0.85)
    graph.add_edge("researcher_b", "writer", message_cost=500, latency=0.5, weight=0.85)
    graph.add_edge("writer", "verifier", message_cost=450, latency=0.4, weight=0.9)
    graph.add_edge("verifier", "final", message_cost=250, latency=0.2, weight=0.95)
    return graph


def debate_judge() -> AgentGraph:
    """Three-agent debate workflow."""

    graph = AgentGraph()
    graph.add_node("prompt", type=NodeType.DOCUMENT, token_cost=900, latency=0.2, reliability=0.95)
    for agent in ("agent_a", "agent_b", "agent_c"):
        graph.add_agent(agent, token_cost=1200, latency=1.8, reliability=0.80, error_rate=0.16)
        graph.add_edge("prompt", agent, message_cost=500, latency=0.4, weight=0.8)
    graph.add_verifier("judge", token_cost=1000, latency=1.5, reliability=0.88, error_rate=0.08)
    graph.add_node("final", type=NodeType.OUTPUT, token_cost=250, latency=0.2, reliability=0.95)
    for agent in ("agent_a", "agent_b", "agent_c"):
        graph.add_edge(agent, "judge", message_cost=450, latency=0.4, weight=0.75)
    graph.add_edge("judge", "final", message_cost=250, latency=0.2, weight=0.95)
    return graph


def rag_pipeline() -> AgentGraph:
    """Retrieval-augmented generation workflow."""

    graph = AgentGraph()
    graph.add_node("query", type=NodeType.DOCUMENT, token_cost=500, latency=0.1, reliability=0.95)
    graph.add_tool("retriever", token_cost=700, latency=1.0, reliability=0.85, error_rate=0.10)
    graph.add_agent("summarizer", token_cost=1000, latency=1.4, reliability=0.83, error_rate=0.12)
    graph.add_agent("reasoner", token_cost=1400, latency=1.9, reliability=0.82, error_rate=0.15)
    graph.add_verifier("verifier", token_cost=850, latency=1.1, reliability=0.91, error_rate=0.06)
    graph.add_node("final", type=NodeType.OUTPUT, token_cost=250, latency=0.2, reliability=0.95)

    graph.add_edge("query", "retriever", message_cost=250, latency=0.2, weight=0.9)
    graph.add_edge("retriever", "summarizer", message_cost=650, latency=0.5, weight=0.85)
    graph.add_edge("summarizer", "reasoner", message_cost=500, latency=0.5, weight=0.8)
    graph.add_edge("reasoner", "verifier", message_cost=450, latency=0.4, weight=0.9)
    graph.add_edge("verifier", "final", message_cost=250, latency=0.2, weight=0.95)
    graph.add_edge("retriever", "reasoner", message_cost=400, latency=0.3, weight=0.55)
    return graph


def tool_use_pipeline() -> AgentGraph:
    """Workflow with tool calls feeding analyst and tester agents."""

    graph = AgentGraph()
    graph.add_node("planner", type=NodeType.PLANNER, token_cost=900, latency=1.0, reliability=0.86)
    graph.add_tool("search_tool", token_cost=500, latency=1.2, reliability=0.88, error_rate=0.08)
    graph.add_tool("code_tool", token_cost=650, latency=1.4, reliability=0.84, error_rate=0.11)
    graph.add_agent("analyst", token_cost=1200, latency=1.6, reliability=0.84, error_rate=0.12)
    graph.add_verifier("tester", token_cost=950, latency=1.5, reliability=0.90, error_rate=0.07)
    graph.add_node("final", type=NodeType.OUTPUT, token_cost=250, latency=0.2, reliability=0.95)

    graph.add_edge("planner", "search_tool", message_cost=300, latency=0.3, weight=0.75)
    graph.add_edge("planner", "code_tool", message_cost=300, latency=0.3, weight=0.75)
    graph.add_edge("search_tool", "analyst", message_cost=550, latency=0.4, weight=0.85)
    graph.add_edge("code_tool", "tester", message_cost=500, latency=0.4, weight=0.85)
    graph.add_edge("analyst", "final", message_cost=350, latency=0.3, weight=0.8)
    graph.add_edge("tester", "final", message_cost=350, latency=0.3, weight=0.8)
    graph.add_edge("tester", "code_tool", message_cost=250, latency=0.3, weight=0.5)
    return graph


def hub_and_spoke_supervisor() -> AgentGraph:
    """Supervisor routes context to specialist agents and aggregates outputs."""

    graph = AgentGraph()
    graph.add_node(
        "supervisor",
        type=NodeType.PLANNER,
        token_cost=1300,
        latency=1.5,
        reliability=0.87,
    )
    for agent in ("researcher", "coder", "critic", "summarizer"):
        graph.add_agent(agent, token_cost=1050, latency=1.4, reliability=0.82, error_rate=0.13)
        graph.add_edge("supervisor", agent, message_cost=450, latency=0.3, weight=0.8)
        graph.add_edge(agent, "supervisor", message_cost=350, latency=0.3, weight=0.7)
    graph.add_verifier("verifier", token_cost=900, latency=1.1, reliability=0.91, error_rate=0.06)
    graph.add_node("final", type=NodeType.OUTPUT, token_cost=250, latency=0.2, reliability=0.95)
    graph.add_edge("supervisor", "verifier", message_cost=400, latency=0.3, weight=0.8)
    graph.add_edge("verifier", "final", message_cost=250, latency=0.2, weight=0.95)
    return graph


def chain_workflow(length: int = 6) -> AgentGraph:
    """Synthetic chain workflow for propagation and cut-point tests."""

    graph = AgentGraph()
    for index in range(length):
        node_id = f"node_{index}"
        node_type = NodeType.OUTPUT if index == length - 1 else NodeType.AGENT
        graph.add_node(node_id, type=node_type, **_synthetic_node_metrics(index))
        if index > 0:
            graph.add_edge(
                f"node_{index - 1}",
                node_id,
                message_cost=220 + 25 * index,
                latency=0.2,
                weight=0.85,
            )
    return graph


def star_workflow(spokes: int = 5) -> AgentGraph:
    """Synthetic star workflow with a central supervisor."""

    graph = AgentGraph()
    graph.add_node("hub", type=NodeType.PLANNER, **_synthetic_node_metrics(0))
    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(spokes + 1))
    for index in range(spokes):
        spoke = f"spoke_{index}"
        graph.add_agent(spoke, **_synthetic_node_metrics(index + 1))
        graph.add_edge("hub", spoke, message_cost=260, latency=0.2, weight=0.8)
        graph.add_edge(spoke, "final", message_cost=180, latency=0.2, weight=0.75)
    return graph


def tree_workflow(branching: int = 2, depth: int = 3) -> AgentGraph:
    """Synthetic rooted tree workflow."""

    graph = AgentGraph()
    graph.add_node("root", type=NodeType.PLANNER, **_synthetic_node_metrics(0))
    current_layer = ["root"]
    counter = 1
    for _ in range(depth):
        next_layer = []
        for parent in current_layer:
            for branch in range(branching):
                node_id = f"{parent}_{branch}"
                graph.add_agent(node_id, **_synthetic_node_metrics(counter))
                graph.add_edge(parent, node_id, message_cost=240, latency=0.2, weight=0.8)
                next_layer.append(node_id)
                counter += 1
        current_layer = next_layer
    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(counter))
    for leaf in current_layer:
        graph.add_edge(leaf, "final", message_cost=160, latency=0.2, weight=0.75)
    return graph


def dense_workflow(size: int = 6) -> AgentGraph:
    """Synthetic dense directed workflow."""

    graph = AgentGraph()
    node_ids = [f"node_{index}" for index in range(size)]
    for index, node_id in enumerate(node_ids):
        graph.add_agent(node_id, **_synthetic_node_metrics(index))
    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(size))
    for source_index, source in enumerate(node_ids):
        for target_index, target in enumerate(node_ids):
            if source_index < target_index:
                graph.add_edge(source, target, message_cost=180, latency=0.15, weight=0.65)
        graph.add_edge(source, "final", message_cost=120, latency=0.1, weight=0.55)
    return graph


def small_world_workflow(size: int = 8, neighborhood: int = 2) -> AgentGraph:
    """Deterministic small-world-style workflow with local and shortcut edges."""

    graph = AgentGraph()
    node_ids = [f"node_{index}" for index in range(size)]
    for index, node_id in enumerate(node_ids):
        graph.add_agent(node_id, **_synthetic_node_metrics(index))
    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(size))

    for index, source in enumerate(node_ids):
        for offset in range(1, neighborhood + 1):
            target = node_ids[(index + offset) % size]
            graph.add_edge(source, target, message_cost=180, latency=0.2, weight=0.7)
        shortcut = node_ids[(index * 3 + 1) % size]
        if shortcut != source:
            graph.add_edge(source, shortcut, message_cost=260, latency=0.25, weight=0.45)
        if index % 2 == 0:
            graph.add_edge(source, "final", message_cost=130, latency=0.1, weight=0.6)
    return graph


def random_directed_workflow(
    size: int = 8,
    edge_probability: float = 0.25,
    seed: int = 0,
) -> AgentGraph:
    """Deterministic random directed graph workflow."""

    rng = random.Random(seed)
    graph = AgentGraph()
    node_ids = [f"node_{index}" for index in range(size)]
    for index, node_id in enumerate(node_ids):
        graph.add_agent(node_id, **_synthetic_node_metrics(index))
    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(size))

    for source in node_ids:
        for target in node_ids:
            if source != target and rng.random() < edge_probability:
                graph.add_edge(source, target, message_cost=180, latency=0.2, weight=0.5)
    for node_id in node_ids[-3:]:
        graph.add_edge(node_id, "final", message_cost=140, latency=0.1, weight=0.65)
    return graph


def generic_dag_workflow(layers: int = 3, width: int = 3) -> AgentGraph:
    """Synthetic layered DAG workflow."""

    graph = AgentGraph()
    previous_layer: list[str] = []
    counter = 0
    for layer in range(layers):
        current_layer = []
        for position in range(width):
            node_id = f"layer_{layer}_node_{position}"
            node_type = NodeType.PLANNER if layer == 0 and position == 0 else NodeType.AGENT
            graph.add_node(node_id, type=node_type, **_synthetic_node_metrics(counter))
            current_layer.append(node_id)
            counter += 1
        for source in previous_layer:
            for target in current_layer:
                graph.add_edge(source, target, message_cost=210, latency=0.2, weight=0.72)
        previous_layer = current_layer

    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(counter))
    for node_id in previous_layer:
        graph.add_edge(node_id, "final", message_cost=130, latency=0.1, weight=0.8)
    return graph


def fan_out_parallel_workflow(branches: int = 4) -> AgentGraph:
    """Planner fans out to parallel workers that merge at a verifier."""

    graph = AgentGraph()
    graph.add_node("planner", type=NodeType.PLANNER, **_synthetic_node_metrics(0))
    workers = [f"worker_{index}" for index in range(branches)]
    for index, worker in enumerate(workers):
        graph.add_agent(worker, **_synthetic_node_metrics(index + 1))
        graph.add_edge("planner", worker, message_cost=320, latency=0.25, weight=0.82)
    graph.add_verifier("verifier", **_synthetic_node_metrics(branches + 1))
    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(branches + 2))
    for worker in workers:
        graph.add_edge(worker, "verifier", message_cost=240, latency=0.2, weight=0.75)
    graph.add_edge("verifier", "final", message_cost=180, latency=0.15, weight=0.9)
    return graph


def feedback_loop_workflow() -> AgentGraph:
    """Coder ↔ tester feedback loop with planner seed and reviewer output."""

    graph = AgentGraph()
    graph.add_node("planner", type=NodeType.PLANNER, **_synthetic_node_metrics(0))
    graph.add_agent("coder", **_synthetic_node_metrics(1))
    graph.add_verifier("tester", **_synthetic_node_metrics(2))
    graph.add_node("reviewer", type=NodeType.REVIEWER, **_synthetic_node_metrics(3))
    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(4))
    graph.add_edge("planner", "coder", message_cost=420, latency=0.3, weight=0.85)
    graph.add_edge("coder", "tester", message_cost=360, latency=0.25, weight=0.8)
    graph.add_edge("tester", "coder", message_cost=280, latency=0.25, weight=0.7, relevance=0.85)
    graph.add_edge("tester", "reviewer", message_cost=300, latency=0.2, weight=0.75)
    graph.add_edge("reviewer", "final", message_cost=200, latency=0.15, weight=0.9)
    graph.add_edge("planner", "reviewer", message_cost=250, latency=0.2, weight=0.55)
    return graph


def dynamic_conditional_workflow() -> AgentGraph:
    """Planner routes to fast or thorough paths via conditional edges."""

    graph = AgentGraph()
    graph.add_node("planner", type=NodeType.PLANNER, **_synthetic_node_metrics(0))
    graph.add_agent("fast_worker", **_synthetic_node_metrics(1))
    graph.add_agent("thorough_worker", **_synthetic_node_metrics(2))
    graph.add_verifier("verifier", **_synthetic_node_metrics(3))
    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(4))
    graph.add_edge("planner", "fast_worker", message_cost=280, latency=0.2, weight=0.8)
    graph.add_conditional_edge(
        "planner",
        "thorough_worker",
        condition_key="route",
        condition_value="thorough",
        message_cost=360,
        latency=0.3,
        weight=0.85,
    )
    graph.add_edge("fast_worker", "verifier", message_cost=240, latency=0.2, weight=0.75)
    graph.add_edge("thorough_worker", "verifier", message_cost=300, latency=0.25, weight=0.82)
    graph.add_edge("verifier", "final", message_cost=180, latency=0.15, weight=0.9)
    return graph


def shared_memory_workflow() -> AgentGraph:
    """Agents read/write a shared memory node (document hub)."""

    graph = AgentGraph()
    graph.add_node(
        "shared_memory",
        type=NodeType.MEMORY,
        token_cost=500,
        latency=0.1,
        reliability=0.95,
        importance_score=0.9,
    )
    for agent in ("planner", "researcher", "writer", "verifier"):
        role_index = {"planner": 0, "researcher": 1, "writer": 2, "verifier": 3}[agent]
        metrics = _synthetic_node_metrics(role_index)
        node_type = NodeType.VERIFIER if agent == "verifier" else NodeType.AGENT
        if agent == "planner":
            node_type = NodeType.PLANNER
        graph.add_node(agent, type=node_type, **metrics)
        graph.add_edge(agent, "shared_memory", message_cost=220, latency=0.15, weight=0.7)
        graph.add_edge("shared_memory", agent, message_cost=260, latency=0.15, weight=0.75)
    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(4))
    graph.add_edge("writer", "verifier", message_cost=300, latency=0.2, weight=0.8)
    graph.add_edge("verifier", "final", message_cost=180, latency=0.15, weight=0.9)
    graph.add_edge("planner", "researcher", message_cost=350, latency=0.25, weight=0.8)
    graph.add_edge("researcher", "writer", message_cost=320, latency=0.25, weight=0.8)
    return graph


def layered_pipeline_workflow() -> AgentGraph:
    """Synthetic layered workflow with planner, workers, verifiers, and output."""

    graph = AgentGraph()
    graph.add_node("planner", type=NodeType.PLANNER, **_synthetic_node_metrics(0))
    for worker in ("worker_a", "worker_b", "worker_c"):
        graph.add_agent(worker, **_synthetic_node_metrics(1))
        graph.add_edge("planner", worker, message_cost=260, latency=0.25, weight=0.82)
    for verifier in ("verifier_a", "verifier_b"):
        graph.add_verifier(verifier, **_synthetic_node_metrics(2))
        for worker in ("worker_a", "worker_b", "worker_c"):
            graph.add_edge(worker, verifier, message_cost=190, latency=0.2, weight=0.7)
    graph.add_node("final", type=NodeType.OUTPUT, **_synthetic_node_metrics(3))
    graph.add_edge("verifier_a", "final", message_cost=140, latency=0.1, weight=0.8)
    graph.add_edge("verifier_b", "final", message_cost=140, latency=0.1, weight=0.8)
    return graph


def inject_quality_decay(
    graph: AgentGraph,
    *,
    seed: int = 0,
    relevance_range: tuple[float, float] = (0.6, 1.0),
    reliability_range: tuple[float, float] = (0.75, 0.97),
) -> AgentGraph:
    """Return a copy of ``graph`` with heterogeneous edge relevance/reliability.

    Most built-in templates set ``relevance = reliability = 1.0`` on nearly
    every edge, which makes the quality cascade degenerate (output quality stays
    at 1.0 regardless of routing). This transform assigns each edge a
    deterministic, reproducible relevance and reliability sampled from the given
    ranges, keyed by ``(seed, source, target)`` so the same workflow always maps
    to the same decayed graph. The original graph is left unmodified.
    """

    decayed = AgentGraph.from_dict(graph.to_dict())
    for edge in graph.edges():
        rng = random.Random(f"{seed}:{edge.source}->{edge.target}")
        relevance = round(rng.uniform(*relevance_range), 3)
        reliability = round(rng.uniform(*reliability_range), 3)
        decayed.add_edge(
            edge.source,
            edge.target,
            message_cost=edge.message_cost,
            latency=edge.latency,
            relevance=relevance,
            reliability=reliability,
            activation_probability=edge.activation_probability,
            dependency_strength=edge.dependency_strength,
            weight=edge.weight,
            **edge.metadata,
        )
    return decayed


def _synthetic_node_metrics(index: int) -> dict[str, Any]:
    return {
        "token_cost": 650.0 + 75.0 * (index % 5),
        "latency": 0.6 + 0.15 * (index % 4),
        "reliability": 0.9 - 0.02 * (index % 4),
        "error_rate": 0.04 + 0.02 * (index % 4),
    }


WORKFLOW_TEMPLATES = {
    "chain": chain_workflow,
    "dynamic_conditional": dynamic_conditional_workflow,
    "fan_out_parallel": fan_out_parallel_workflow,
    "feedback_loop": feedback_loop_workflow,
    "shared_memory": shared_memory_workflow,
    "planner_coder_tester_reviewer": planner_coder_tester_reviewer,
    "research_writer_verifier": research_writer_verifier,
    "debate_judge": debate_judge,
    "dense_graph": dense_workflow,
    "generic_dag": generic_dag_workflow,
    "rag_pipeline": rag_pipeline,
    "random_directed_graph": random_directed_workflow,
    "tool_use_pipeline": tool_use_pipeline,
    "hub_and_spoke_supervisor": hub_and_spoke_supervisor,
    "layered_pipeline": layered_pipeline_workflow,
    "small_world_graph": small_world_workflow,
    "star": star_workflow,
    "tree": tree_workflow,
}

WORKFLOW_DESCRIPTIONS: dict[str, str] = {
    "chain": "Linear propagation path for cut-point and bridge tests",
    "dynamic_conditional": "Runtime-conditional fast vs thorough routing branches",
    "fan_out_parallel": "Planner fan-out to parallel workers merging at verifier",
    "feedback_loop": "Coder-tester correction loop with reviewer gate",
    "shared_memory": "Hub-and-spoke shared memory with planner/researcher/writer",
    "planner_coder_tester_reviewer": "Software-agent loop with correction feedback",
    "research_writer_verifier": "Parallel research branches merging at writer",
    "debate_judge": "Many-to-one debate aggregation at judge node",
    "dense_graph": "Redundant paths and centrality tie stress case",
    "generic_dag": "Layered directed acyclic propagation graph",
    "rag_pipeline": "Retrieval-augmented generation with verifier",
    "random_directed_graph": "Deterministic random directed stress graph",
    "tool_use_pipeline": "Tool nodes with planner and executor handoffs",
    "hub_and_spoke_supervisor": "Supervisor hub routing to worker spokes",
    "layered_pipeline": "Planner, worker, verifier, and output layers",
    "small_world_graph": "Local neighborhoods with shortcut edges",
    "star": "Hub-dominant supervisor-style bottleneck",
    "tree": "Branching propagation and articulation structure",
}
