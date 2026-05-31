"""Reusable agent workflow templates."""

from __future__ import annotations

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
    )
    graph.add_node(
        "coder",
        type=NodeType.EXECUTOR,
        token_cost=1700,
        latency=2.1,
        reliability=0.80,
        error_rate=0.18,
        role="implements the solution",
    )
    graph.add_node(
        "tester",
        type=NodeType.VERIFIER,
        token_cost=950,
        latency=1.6,
        reliability=0.88,
        error_rate=0.08,
        role="tests the implementation",
    )
    graph.add_node(
        "reviewer",
        type=NodeType.REVIEWER,
        token_cost=1100,
        latency=1.8,
        reliability=0.90,
        error_rate=0.07,
        role="reviews correctness and maintainability",
    )
    graph.add_node("final", type=NodeType.OUTPUT, token_cost=350, latency=0.4, reliability=0.95)

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
