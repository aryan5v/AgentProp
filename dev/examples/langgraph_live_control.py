"""Async LangGraph agent with AgentProp AsyncControlSession.

Shows the "wrap don't replace" pattern: a minimal LangGraph StateGraph
emits ExecutionEvents at each node, and AsyncControlSession gates whether
execution continues.

Run:
    python dev/examples/langgraph_live_control.py

No API key required — nodes use deterministic mock responses. To use a real
LLM, set OPENAI_API_KEY and swap in a LangChain chat model.
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
from typing import Any

# ------------------------------------------------------------------
# LangGraph import guard
# ------------------------------------------------------------------
try:
    from langgraph.graph import END, StateGraph
except ImportError:
    print(
        "LangGraph is not installed. Install it with:\n"
        "  pip install langgraph\n"
        "This example requires langgraph>=0.1.0.",
        file=sys.stderr,
    )
    sys.exit(1)

from agentprop.runtime import AsyncControlSession
from agentprop.runtime.control_loop import ExecutionEvent

# ------------------------------------------------------------------
# Shared state type
# ------------------------------------------------------------------
State = dict[str, Any]

# ------------------------------------------------------------------
# Mock node implementations (no API key needed)
# ------------------------------------------------------------------

_MOCK_SEED = int(os.getenv("MOCK_SEED", "42"))
_rng = random.Random(_MOCK_SEED)


def _planner_node(state: State) -> State:
    tokens = _rng.randint(200, 400)
    return {
        **state,
        "plan": "1. Analyse inputs\n2. Implement solution\n3. Verify output",
        "step": state.get("step", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "last_node": "planner",
        "last_tokens": tokens,
    }


def _coder_node(state: State) -> State:
    tokens = _rng.randint(400, 800)
    exit_code = 0 if _rng.random() > 0.2 else 1
    return {
        **state,
        "code": "def solve():\n    return 42",
        "exit_code": exit_code,
        "step": state.get("step", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "last_node": "coder",
        "last_tokens": tokens,
        "last_exit_code": exit_code,
    }


def _tester_node(state: State) -> State:
    tokens = _rng.randint(150, 300)
    passed = _rng.random() > 0.3
    return {
        **state,
        "test_passed": passed,
        "step": state.get("step", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "last_node": "tester",
        "last_tokens": tokens,
        "verifier_passed": passed,
    }


def _reviewer_node(state: State) -> State:
    tokens = _rng.randint(100, 200)
    return {
        **state,
        "review": "Looks good.",
        "step": state.get("step", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + tokens,
        "last_node": "reviewer",
        "last_tokens": tokens,
        "final_answer_written": True,
    }


# ------------------------------------------------------------------
# Build the LangGraph StateGraph
# ------------------------------------------------------------------

def build_graph() -> Any:
    g: StateGraph = StateGraph(State)
    g.add_node("planner", _planner_node)
    g.add_node("coder", _coder_node)
    g.add_node("tester", _tester_node)
    g.add_node("reviewer", _reviewer_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "coder")
    g.add_edge("coder", "tester")
    g.add_conditional_edges(
        "tester",
        lambda s: "reviewer" if s.get("test_passed") else END,
    )
    g.add_edge("reviewer", END)
    return g.compile()


# ------------------------------------------------------------------
# Async driver with AgentProp control
# ------------------------------------------------------------------

def _state_to_event(state: State) -> ExecutionEvent:
    node = state.get("last_node", "unknown")
    is_verifier = node == "tester"
    return ExecutionEvent(
        step=int(state.get("step", 0)),
        command=node,
        exit_code=state.get("last_exit_code"),
        verifier_run=is_verifier,
        verifier_passed=state.get("verifier_passed") if is_verifier else None,
        progress_made=True,
        tokens_used=int(state.get("last_tokens", 0)),
        elapsed_s=0.1 * int(state.get("step", 1)),
        final_answer_written=bool(state.get("final_answer_written", False)),
        trusted=is_verifier,
    )


async def run_with_control(task_id: str = "langgraph-demo") -> None:
    print(f"=== AgentProp + LangGraph live control demo (task={task_id}) ===\n")

    session = await AsyncControlSession.start(
        "planner_coder_tester_reviewer",
        task_id=task_id,
        category="coding",
        token_budget=3000,
    )
    print(f"Workflow: {session.analysis.workflow_name}")
    print(f"Verifier candidates: {', '.join(session.analysis.verifier_candidates) or 'none'}\n")

    graph = build_graph()
    state: State = {"step": 0, "tokens_used": 0}
    step_count = 0

    # Stream through the graph node by node
    async for chunk in graph.astream(state, stream_mode="updates"):
        for node_name, node_state in chunk.items():
            state = {**state, **node_state}
            event = _state_to_event(state)
            decision = await session.observe(event)

            tokens = event.tokens_used
            print(
                f"  [{node_name:10s}] tokens={tokens:4d}  "
                f"decision={decision.action:<18s}  reason={decision.reason}"
            )
            step_count += 1

            if decision.action in ("FINALIZE", "FORCE_VERIFY"):
                print(f"\n  --> Control decision: {decision.action}. Stopping early.")
                break
        else:
            continue
        break

    await session.record_outcome(
        passed=bool(state.get("test_passed", False)),
        strategy="agentprop_controller",
    )

    summary = session.summary()
    counts = summary.get("decision_counts", {})
    total_tokens = state.get("tokens_used", 0)
    print("\n--- Summary ---")
    print(f"Steps: {step_count}  |  Total tokens: {total_tokens}")
    print(f"Decision counts: {counts}")
    print(
        f"Workflow analysis: {session.analysis.node_count} nodes, "
        f"{session.analysis.edge_count} edges"
    )


if __name__ == "__main__":
    asyncio.run(run_with_control())
