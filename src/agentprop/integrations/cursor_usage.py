"""Aggregate Cursor CLI stream-json usage for Harbor telemetry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

_COMPOSER_RATES_PER_MILLION = {
    "input": 0.5,
    "output": 2.5,
    "cache_read": 0.2,
    "cache_write": 0.5,
}


@dataclass(slots=True)
class CursorUsageAccumulator:
    """Token and cost totals across multiple cursor-agent subprocess calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    cost_reported: bool = False
    proposal_calls: int = 0

    def ingest_event(self, event: dict[str, Any]) -> None:
        if event.get("type") != "result":
            return
        usage = event.get("usage")
        if not isinstance(usage, dict):
            return
        self.input_tokens += int(usage.get("inputTokens") or 0)
        self.output_tokens += int(usage.get("outputTokens") or 0)
        self.cache_read_tokens += int(usage.get("cacheReadTokens") or 0)
        self.cache_write_tokens += int(usage.get("cacheWriteTokens") or 0)
        reported = usage.get("totalCost")
        if reported is None:
            reported = usage.get("cost")
        if isinstance(reported, int | float):
            self.cost_usd = (self.cost_usd or 0.0) + float(reported)
            self.cost_reported = True

    def note_proposal_call(self) -> None:
        self.proposal_calls += 1

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def finalize_cost(self, *, model: str | None = None) -> None:
        if self.cost_reported and self.cost_usd is not None:
            return
        _ = model
        rates = _COMPOSER_RATES_PER_MILLION
        self.cost_usd = (
            self.input_tokens * rates["input"] / 1_000_000
            + self.output_tokens * rates["output"] / 1_000_000
            + self.cache_read_tokens * rates["cache_read"] / 1_000_000
            + self.cache_write_tokens * rates["cache_write"] / 1_000_000
        )

    def to_harbor_payload(self, *, model: str | None = None) -> dict[str, int | float | str | None]:
        self.finalize_cost(model=model)
        return {
            "n_input_tokens": self.input_tokens + self.cache_read_tokens + self.cache_write_tokens,
            "n_cache_tokens": self.cache_read_tokens,
            "n_output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "proposal_calls": self.proposal_calls,
            "cost_source": "cursor_cli" if self.cost_reported else "estimated",
        }


def decode_cursor_agent_stdout(
    stdout: str,
    usage: CursorUsageAccumulator | None = None,
) -> tuple[str, bool]:
    """Normalize Cursor stdout to proposal text; return (text, saw_stream_json)."""

    text_parts: list[str] = []
    saw_stream = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or "type" not in event:
            continue
        saw_stream = True
        if usage is not None:
            usage.ingest_event(event)
        event_type = event.get("type")
        if event_type == "assistant":
            message = event.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "text":
                            continue
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            text_parts.append(text.strip())
        elif event_type == "result":
            result_text = event.get("result")
            if isinstance(result_text, str) and result_text.strip():
                text_parts.append(result_text.strip())
    if saw_stream:
        return "\n".join(text_parts), True
    return stdout, False
