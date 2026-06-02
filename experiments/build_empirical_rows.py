"""Build empirical ML/DL/RL training rows from trace and outcome artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentprop.integrations import empirical_rows_from_trace_dicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert workflow traces plus task outcomes into empirical training rows."
    )
    parser.add_argument(
        "--trace",
        type=Path,
        action="append",
        required=True,
        help="Trace JSON/JSONL file. May be passed more than once.",
    )
    parser.add_argument(
        "--outcome-results",
        type=Path,
        default=None,
        help="Optional results JSON/JSONL with pass/fail, quality, cost, or latency fields.",
    )
    parser.add_argument(
        "--default-summary-ratio",
        type=float,
        default=0.35,
        help="Context ratio assigned to compressed/summary-only trace events.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/empirical_rows.json"),
    )
    args = parser.parse_args(argv)

    traces = _load_records(args.trace, record_kind="trace")
    outcomes = (
        _load_records([args.outcome_results], record_kind="outcome")
        if args.outcome_results is not None
        else []
    )
    result = empirical_rows_from_trace_dicts(
        traces,
        outcome_rows=outcomes,
        default_summary_ratio=args.default_summary_ratio,
    )

    payload = {
        "rows": result.rows,
        "row_count": len(result.rows),
        "trace_count": len(traces),
        "outcome_count": len(outcomes),
        "skipped_trace_count": result.skipped_trace_count,
        "default_summary_ratio": args.default_summary_ratio,
        "trace_sources": [str(path) for path in args.trace],
        "outcome_source": str(args.outcome_results) if args.outcome_results else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {args.out} "
        f"({len(result.rows)} rows, {result.skipped_trace_count} skipped traces)"
    )
    return 0


def _load_records(paths: list[Path | None], *, record_kind: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path is None:
            continue
        loaded = _load_path(path)
        records.extend(_records_from_payload(loaded, record_kind=record_kind))
    return records


def _load_path(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _records_from_payload(payload: Any, *, record_kind: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError(f"{record_kind} artifact must be a JSON object, array, or JSONL file")

    if record_kind == "trace" and _looks_like_trace(payload):
        return [dict(payload)]

    for key in ("rows", "traces", "tasks", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]

    if record_kind == "outcome":
        return [dict(payload)]
    raise ValueError("trace artifact must contain events/messages or a rows/traces list")


def _looks_like_trace(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("events"), list) or isinstance(payload.get("messages"), list)


if __name__ == "__main__":
    raise SystemExit(main())
