"""Tests for ControlSession dynamic graph mutations."""

from __future__ import annotations

from agentprop.core.types import NodeType
from agentprop.runtime.controller import AgentPropRuntimeController, RuntimeNodeResult
from agentprop.runtime.session import ControlSession


def test_control_session_dynamic_mutations_refresh_analysis() -> None:
    session = ControlSession.start("chain", task_id="t1")
    initial_nodes = session.analysis.node_count
    session.mutate_add_node("extra_agent", type=NodeType.AGENT)
    assert session.analysis.node_count == initial_nodes + 1
    assert session.dynamic is not None
    assert session.dynamic.version >= 1


def test_runtime_controller_respects_routing_context() -> None:
    from agentprop.workflows import WORKFLOW_TEMPLATES

    graph = WORKFLOW_TEMPLATES["dynamic_conditional"]()
    controller = AgentPropRuntimeController(graph)
    executed: list[str] = []

    def executor(request):
        executed.append(request.node.id)
        return RuntimeNodeResult(node_id=request.node.id, output=request.node.id)

    controller.run(
        task="route",
        shared_context="context",
        executor=executor,
        metadata={"routing_context": {"route": "thorough"}},
    )
    assert "thorough_worker" in executed
    assert "fast_worker" in executed


def test_effective_graph_filters_conditional_edges() -> None:
    session = ControlSession.start("dynamic_conditional", task_id="t2")
    fast = session.effective_graph({"route": "fast"})
    thorough = session.effective_graph({"route": "thorough"})
    assert fast.has_edge("planner", "fast_worker")
    assert not fast.has_edge("planner", "thorough_worker")
    assert thorough.has_edge("planner", "thorough_worker")
