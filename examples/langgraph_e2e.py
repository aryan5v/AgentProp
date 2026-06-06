"""End-to-end LangGraph round-trip plus ControlSession analysis."""

from __future__ import annotations

import json
from pathlib import Path

from agentprop import ControlSession, ExecutionEvent
from agentprop.integrations import graph_from_framework_dict, to_framework_dict, to_native_framework
from agentprop.integrations.framework_adapters import NativeFrameworkUnavailable
from agentprop.workflows import planner_coder_tester_reviewer


def main() -> None:
    graph = planner_coder_tester_reviewer()

    spec = to_framework_dict(graph, "langgraph")
    round_tripped = graph_from_framework_dict(spec, "langgraph")
    assert round_tripped.node_count == graph.node_count
    assert round_tripped.edge_count == graph.edge_count
    print(f"LangGraph dict round-trip ok: {round_tripped.node_count} nodes")

    try:
        builder = to_native_framework(graph, "langgraph")
        print(f"Native LangGraph builder: {type(builder).__name__}")
    except NativeFrameworkUnavailable as error:
        print(f"Native builder skipped: {error}")

    out_dir = Path("reports/langgraph-e2e")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "langgraph_spec.json").write_text(json.dumps(spec, indent=2))

    session = ControlSession.start("planner_coder_tester_reviewer", task_id="langgraph-e2e")
    session.observe(
        ExecutionEvent(step=1, command="langgraph.invoke", tokens_used=1_500, verifier_passed=True)
    )
    session.record_outcome(passed=True, quality_score=1.0)
    paths = session.write_artifacts(out_dir)
    print("ControlSession artifacts:", ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
