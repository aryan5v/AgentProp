"""Replay a saved ControlSession trace to compare A0 (no-control) vs A2 (with-control)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentprop.runtime.control_loop import ControlDecision, ExecutionEvent
from agentprop.runtime.session import ControlSession, ControlSessionConfig


@dataclass(slots=True)
class ReplayRow:
    step: int
    command: str | None
    tokens_used: int
    decision_with_control: str
    decision_no_control: str
    token_delta: int


@dataclass(slots=True)
class ReplayResult:
    task_id: str
    workflow: str
    rows: list[ReplayRow]
    total_tokens_with_control: int
    total_tokens_no_control: int
    token_delta: int
    reduction_pct: float


def replay_trace(
    trace_path: Path,
    *,
    no_control: bool = False,
) -> ReplayResult:
    """Read a trace.jsonl and replay events, returning a side-by-side comparison.

    When ``no_control`` is True, the A0 column shows what would have happened
    if every decision were CONTINUE (no forced verification, no stopping).
    The A2 column always shows the original trace decisions.
    """
    rows_raw = _load_trace(trace_path)

    task_id = "unknown"
    workflow = "unknown"
    for row in rows_raw:
        if row.get("type") == "analysis":
            task_id = str(row.get("task_id", "unknown"))
            analysis = row.get("analysis") or {}
            if isinstance(analysis, dict):
                workflow = str(analysis.get("workflow", "unknown"))
            break

    event_rows = [r for r in rows_raw if r.get("type") == "event"]
    if not event_rows:
        return ReplayResult(
            task_id=task_id,
            workflow=workflow,
            rows=[],
            total_tokens_with_control=0,
            total_tokens_no_control=0,
            token_delta=0,
            reduction_pct=0.0,
        )

    # Build a replay session for the A2 (with-control) path
    config = ControlSessionConfig(workflow=workflow, task_id=task_id)
    session_a2 = ControlSession(config)

    replay_rows: list[ReplayRow] = []
    total_a2 = 0
    total_a0 = 0

    for raw in event_rows:
        event_dict = raw.get("event") or {}
        if not isinstance(event_dict, dict):
            continue
        event = _dict_to_event(event_dict)
        tokens = int(event.tokens_used or 0)
        total_a0 += tokens

        decision_a2: ControlDecision = session_a2.observe(event)
        decision_a2_str = decision_a2.action

        if no_control:
            decision_a0_str = "CONTINUE"
        else:
            decision_a0_str = str((raw.get("decision") or {}).get("action", "CONTINUE"))

        total_a2 += tokens

        replay_rows.append(
            ReplayRow(
                step=int(event.step),
                command=event.command,
                tokens_used=tokens,
                decision_with_control=decision_a2_str,
                decision_no_control=decision_a0_str,
                token_delta=0,
            )
        )

    token_delta = total_a0 - total_a2
    reduction_pct = (token_delta / total_a0 * 100) if total_a0 > 0 else 0.0

    return ReplayResult(
        task_id=task_id,
        workflow=workflow,
        rows=replay_rows,
        total_tokens_with_control=total_a2,
        total_tokens_no_control=total_a0,
        token_delta=token_delta,
        reduction_pct=reduction_pct,
    )


def format_replay_table(result: ReplayResult) -> str:
    """Render a ReplayResult as a human-readable Markdown table."""
    lines = [
        f"# Trace Replay: `{result.task_id}`",
        f"- Workflow: `{result.workflow}`",
        f"- Steps: `{len(result.rows)}`",
        "",
        "| Step | Command | Tokens | A0 (no-control) | A2 (with-control) |",
        "| ---: | --- | ---: | --- | --- |",
    ]
    for row in result.rows:
        cmd = (row.command or "")[:40].replace("|", "\\|")
        lines.append(
            f"| {row.step} | `{cmd}` | {row.tokens_used} "
            f"| {row.decision_no_control} | {row.decision_with_control} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "| | Tokens |",
            "| --- | ---: |",
            f"| A0 (no-control) | {result.total_tokens_no_control} |",
            f"| A2 (with-control) | {result.total_tokens_with_control} |",
            f"| Delta | {result.token_delta:+d} |",
            f"| Reduction | {result.reduction_pct:.1f}% |",
        ]
    )
    return "\n".join(lines)


def _load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _dict_to_event(d: dict[str, Any]) -> ExecutionEvent:
    return ExecutionEvent(
        step=int(d.get("step", 0)),
        command=d.get("command"),
        exit_code=d.get("exit_code"),
        verifier_run=bool(d.get("verifier_run", False)),
        verifier_passed=d.get("verifier_passed"),
        progress_made=bool(d.get("progress_made", True)),
        tokens_used=int(d.get("tokens_used") or 0),
        elapsed_s=float(d.get("elapsed_s") or 0.0),
        error_signature=d.get("error_signature"),
        final_answer_written=bool(d.get("final_answer_written", False)),
        trusted=bool(d.get("trusted", True)),
    )
