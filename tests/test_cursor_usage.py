import json
from pathlib import Path

from agentprop.benchmarks.harbor_agent import AgentPropCursorAgent
from agentprop.integrations.cursor_usage import CursorUsageAccumulator, decode_cursor_agent_stdout


def test_cursor_usage_total_tokens_sums_components() -> None:
    usage = CursorUsageAccumulator(
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=3,
        cache_write_tokens=2,
    )
    assert usage.total_tokens == 20


def test_cursor_usage_accumulator_parses_stream_json() -> None:
    usage = CursorUsageAccumulator()
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": '{"command": "pytest -q", "rationale": "verify"}',
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "usage": {
                        "inputTokens": 1000,
                        "outputTokens": 200,
                        "cacheReadTokens": 50,
                        "totalCost": 0.12,
                    },
                }
            ),
        ]
    )

    text, saw_stream = decode_cursor_agent_stdout(stdout, usage)

    assert saw_stream is True
    assert "pytest -q" in text
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 200
    assert usage.cache_read_tokens == 50
    assert usage.cost_usd == 0.12
    payload = usage.to_harbor_payload(model="composer-2.5")
    assert payload["n_input_tokens"] == 1050
    assert payload["n_output_tokens"] == 200
    assert payload["cost_usd"] == 0.12
    assert payload["cost_source"] == "cursor_cli"


def test_cursor_usage_estimates_cost_when_not_reported() -> None:
    usage = CursorUsageAccumulator()
    usage.input_tokens = 1_000_000
    usage.output_tokens = 1_000_000
    payload = usage.to_harbor_payload()
    assert payload["cost_source"] == "estimated"
    assert payload["cost_usd"] == 3.0


def test_harbor_agent_populate_context_reads_usage_file(tmp_path: Path) -> None:
    usage_path = tmp_path / "agentprop-cursor-usage.json"
    usage_path.write_text(
        json.dumps(
            {
                "n_input_tokens": 1200,
                "n_cache_tokens": 200,
                "n_output_tokens": 300,
                "cost_usd": 0.45,
            }
        ),
        encoding="utf-8",
    )

    class _Context:
        n_input_tokens: int | None = None
        n_cache_tokens: int | None = None
        n_output_tokens: int | None = None
        cost_usd: float | None = None

    context = _Context()
    agent = AgentPropCursorAgent.__new__(AgentPropCursorAgent)
    agent.logs_dir = tmp_path
    agent.populate_context_post_run(context)  # type: ignore[arg-type]

    assert context.n_input_tokens == 1200
    assert context.n_cache_tokens == 200
    assert context.n_output_tokens == 300
    assert context.cost_usd == 0.45
