import json
from pathlib import Path

from agentprop.integrations import graph_from_trace


def test_graph_from_trace_aggregates_messages(tmp_path: Path) -> None:
    trace = {
        "events": [
            {
                "source": "planner",
                "target": "coder",
                "source_type": "PLANNER",
                "target_type": "EXECUTOR",
                "token_cost": 500,
                "latency": 0.7,
                "success": True,
            },
            {
                "source": "coder",
                "target": "tester",
                "target_type": "VERIFIER",
                "token_cost": 300,
                "latency": 0.5,
                "success": False,
            },
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))

    result = graph_from_trace(path)

    assert result.message_count == 2
    assert result.total_token_cost == 800
    assert result.graph.node("tester").error_rate == 1.0
    assert result.graph.edge("planner", "coder").message_cost == 500
