"""Load workflow graphs from message/tool/verifier trace logs."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentprop.core import AgentGraph, NodeType


@dataclass(slots=True)
class TraceLoadResult:
    """Graph plus trace-derived aggregate statistics."""

    graph: AgentGraph
    message_count: int
    total_token_cost: float
    total_latency: float


@dataclass(frozen=True, slots=True)
class TraceGraphCalibrationResult:
    """A base graph calibrated with trace-derived parameters."""

    graph: AgentGraph
    calibrated_node_count: int
    calibrated_edge_count: int
    message_count: int


@dataclass(frozen=True, slots=True)
class TraceEmpiricalRowsResult:
    """Empirical training rows derived from workflow execution traces."""

    rows: list[dict[str, Any]]
    skipped_trace_count: int


def graph_from_trace(path: str | Path) -> TraceLoadResult:
    """Load an AgentGraph from a trace JSON file."""

    data = json.loads(Path(path).read_text())
    return graph_from_trace_dict(data)


def calibrate_graph_from_trace(
    graph: AgentGraph,
    path: str | Path,
    *,
    smoothing: float = 0.5,
) -> TraceGraphCalibrationResult:
    """Return a calibrated copy of a graph using trace JSON from disk."""

    data = json.loads(Path(path).read_text())
    return calibrate_graph_from_trace_dict(graph, data, smoothing=smoothing)


def calibrate_graph_from_trace_dict(
    graph: AgentGraph,
    data: dict[str, Any],
    *,
    smoothing: float = 0.5,
) -> TraceGraphCalibrationResult:
    """Return a calibrated copy using trace events.

    Edge activation probability estimates how often a target handoff was
    attempted when the source appeared in the trace. Edge reliability estimates
    whether attempted handoffs succeeded. This avoids the default
    ``activation_probability=1.0`` cascade when trace evidence exists.
    """

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    events = data.get("events", data.get("messages", []))
    if not isinstance(events, list):
        raise ValueError("trace must contain an events or messages list")

    source_counts: dict[str, float] = defaultdict(float)
    node_incoming: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    edge_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for event in events:
        if not isinstance(event, dict):
            raise ValueError("trace event must be an object")
        source = _required_str(event, "source")
        target = _required_str(event, "target")
        token_cost = _float_field(event, "token_cost", fallback_key="tokens")
        latency = _float_field(event, "latency")
        success = bool(event.get("success", True))

        source_counts[source] += 1.0
        node_incoming[target]["count"] += 1.0
        node_incoming[target]["token_cost"] += token_cost
        node_incoming[target]["latency"] += latency
        if success:
            node_incoming[target]["successes"] += 1.0

        edge_key = (source, target)
        edge_stats[edge_key]["attempts"] += 1.0
        edge_stats[edge_key]["token_cost"] += token_cost
        edge_stats[edge_key]["latency"] += latency
        if success:
            edge_stats[edge_key]["successes"] += 1.0

    payload = graph.to_dict()
    calibrated_nodes = _calibrate_node_payloads(payload["nodes"], node_incoming, smoothing)
    calibrated_edges = _calibrate_edge_payloads(
        payload["edges"],
        graph,
        source_counts,
        edge_stats,
        smoothing,
    )
    return TraceGraphCalibrationResult(
        graph=AgentGraph.from_dict(payload),
        calibrated_node_count=calibrated_nodes,
        calibrated_edge_count=calibrated_edges,
        message_count=len(events),
    )


def empirical_row_from_trace_dict(
    data: dict[str, Any],
    *,
    outcome_row: dict[str, Any] | None = None,
    default_summary_ratio: float = 0.35,
) -> dict[str, Any] | None:
    """Build one empirical ML/RL training row from a trace and outcome fields.

    The resulting row is accepted by empirical seed, verifier, torch GNN, and
    RL reward-calibration paths. Traces without pass/fail or quality signal are
    skipped because they cannot provide a task-success training target.
    """

    outcome = outcome_row or {}
    merged = {**data, **outcome}
    outcome_score = _trace_outcome(merged)
    if outcome_score is None:
        return None

    events = data.get("events", data.get("messages", []))
    if not isinstance(events, list):
        raise ValueError("trace must contain an events or messages list")

    selected_seeds = _trace_selected_seeds(data, events)
    context_allocations = _trace_context_allocations(
        data,
        events,
        default_summary_ratio=default_summary_ratio,
    )
    row: dict[str, Any] = {
        "task_id": str(merged.get("task_id") or merged.get("task_name") or "unknown-task"),
        "policy": str(merged.get("policy") or "unknown-policy"),
        "selected_seeds": selected_seeds,
        "context_allocations": context_allocations,
        "verification_passed": _optional_bool(merged.get("verification_passed")),
        "quality_score": _optional_float(merged.get("quality_score")),
        "quality_passed": _optional_bool(merged.get("quality_passed")),
        "token_cost": _optional_float(
            merged.get("token_cost"),
            merged.get("total_llm_tokens"),
            merged.get("total_tokens"),
        ),
        "message_cost": _optional_float(merged.get("message_cost")),
        "latency": _optional_float(merged.get("latency"), merged.get("latency_s")),
        "cost_adjusted_success": _optional_float(merged.get("cost_adjusted_success")),
    }
    category = merged.get("category")
    if isinstance(category, str):
        row["category"] = category
    activated_verifiers = _trace_activated_verifiers(merged)
    if activated_verifiers:
        row["activated_verifiers"] = activated_verifiers
    pruned_edges = _trace_pruned_edges(merged)
    if pruned_edges:
        row["pruned_edges"] = pruned_edges
    return {key: value for key, value in row.items() if value is not None}


def empirical_rows_from_trace_dicts(
    traces: list[dict[str, Any]],
    *,
    outcome_rows: list[dict[str, Any]] | None = None,
    default_summary_ratio: float = 0.35,
) -> TraceEmpiricalRowsResult:
    """Build empirical training rows from traces and optional result rows."""

    outcomes = _outcome_lookup(outcome_rows or [])
    rows = []
    skipped = 0
    for trace in traces:
        outcome = _matching_outcome(trace, outcomes)
        row = empirical_row_from_trace_dict(
            trace,
            outcome_row=outcome,
            default_summary_ratio=default_summary_ratio,
        )
        if row is None:
            skipped += 1
        else:
            rows.append(row)
    return TraceEmpiricalRowsResult(rows=rows, skipped_trace_count=skipped)


def graph_from_trace_dict(data: dict[str, Any]) -> TraceLoadResult:
    """Build a graph from trace dictionary data."""

    events = data.get("events", data.get("messages", []))
    if not isinstance(events, list):
        raise ValueError("trace must contain an events or messages list")

    node_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    edge_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    node_types: dict[str, NodeType] = {}

    for event in events:
        if not isinstance(event, dict):
            raise ValueError("trace event must be an object")
        source = _required_str(event, "source")
        target = _required_str(event, "target")
        source_type = _node_type(event.get("source_type"))
        target_type = _node_type(event.get("target_type"))
        token_cost = _float_field(event, "token_cost", fallback_key="tokens")
        latency = _float_field(event, "latency")
        success = bool(event.get("success", True))

        node_types.setdefault(source, source_type)
        node_types.setdefault(target, target_type)
        node_stats[source]["out_messages"] += 1
        node_stats[target]["in_messages"] += 1
        node_stats[target]["token_cost"] += token_cost
        node_stats[target]["latency"] += latency
        if not success:
            node_stats[target]["errors"] += 1

        edge_key = (source, target)
        edge_stats[edge_key]["message_count"] += 1
        edge_stats[edge_key]["message_cost"] += token_cost
        edge_stats[edge_key]["latency"] += latency
        if success:
            edge_stats[edge_key]["successes"] += 1

    graph = AgentGraph()
    for node_id, stats in sorted(node_stats.items()):
        total_messages = stats["in_messages"] + stats["out_messages"]
        error_rate = stats["errors"] / max(stats["in_messages"], 1.0)
        graph.add_node(
            node_id,
            type=node_types.get(node_id, NodeType.AGENT),
            token_cost=stats["token_cost"],
            latency=stats["latency"] / max(stats["in_messages"], 1.0),
            reliability=1.0 - error_rate,
            error_rate=error_rate,
            trace_message_count=total_messages,
        )

    for (source, target), stats in sorted(edge_stats.items()):
        message_count = stats["message_count"]
        graph.add_edge(
            source,
            target,
            message_cost=stats["message_cost"],
            latency=stats["latency"] / max(message_count, 1.0),
            reliability=stats["successes"] / max(message_count, 1.0),
            activation_probability=min(1.0, message_count / max(len(events), 1)),
            weight=message_count,
            trace_message_count=message_count,
            trace_success_count=stats["successes"],
        )

    return TraceLoadResult(
        graph=graph,
        message_count=len(events),
        total_token_cost=sum(stats["message_cost"] for stats in edge_stats.values()),
        total_latency=sum(stats["latency"] for stats in edge_stats.values()),
    )


def _required_str(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"trace event missing non-empty {key}")
    return value


def _node_type(value: Any) -> NodeType:
    if value is None:
        return NodeType.AGENT
    return NodeType(str(value))


def _float_field(
    event: dict[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
) -> float:
    value = event.get(key)
    if value is None and fallback_key is not None:
        value = event.get(fallback_key)
    if value is None:
        return 0.0
    if not isinstance(value, int | float | str):
        raise ValueError(f"trace event field {key} must be numeric")
    return float(value)


def _calibrate_node_payloads(
    nodes: list[dict[str, Any]],
    node_incoming: dict[str, dict[str, float]],
    smoothing: float,
) -> int:
    calibrated = 0
    for node in nodes:
        node_id = str(node.get("id", ""))
        stats = node_incoming.get(node_id)
        if not stats:
            continue
        count = stats["count"]
        successes = stats["successes"]
        reliability = _smoothed_binary_rate(successes, count, smoothing)
        node["reliability"] = reliability
        node["error_rate"] = max(0.0, min(1.0, 1.0 - reliability))
        node["token_cost"] = stats["token_cost"] / max(count, 1.0)
        node["latency"] = stats["latency"] / max(count, 1.0)
        metadata = _metadata(node)
        metadata["trace_calibrated"] = True
        metadata["trace_in_message_count"] = count
        metadata["trace_success_count"] = successes
        calibrated += 1
    return calibrated


def _calibrate_edge_payloads(
    edges: list[dict[str, Any]],
    graph: AgentGraph,
    source_counts: dict[str, float],
    edge_stats: dict[tuple[str, str], dict[str, float]],
    smoothing: float,
) -> int:
    calibrated = 0
    out_degrees = {node.id: len(graph.successors(node.id)) for node in graph.nodes()}
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        source_count = source_counts.get(source)
        if source_count is None:
            continue

        stats = edge_stats.get((source, target), {})
        attempts = stats.get("attempts", 0.0)
        denominator = source_count + smoothing * max(out_degrees.get(source, 1), 1)
        activation = (attempts + smoothing) / max(denominator, 1.0)
        edge["activation_probability"] = max(0.0, min(1.0, activation))

        metadata = _metadata(edge)
        metadata["trace_calibrated"] = True
        metadata["trace_source_message_count"] = source_count
        metadata["trace_message_count"] = attempts

        if attempts > 0.0:
            successes = stats["successes"]
            edge["reliability"] = _smoothed_binary_rate(successes, attempts, smoothing)
            edge["message_cost"] = stats["token_cost"] / attempts
            edge["latency"] = stats["latency"] / attempts
            edge["weight"] = attempts
            metadata["trace_success_count"] = successes
            calibrated += 1
    return calibrated


def _smoothed_binary_rate(successes: float, attempts: float, smoothing: float) -> float:
    if attempts <= 0:
        return 0.0
    if smoothing == 0:
        return max(0.0, min(1.0, successes / attempts))
    return max(0.0, min(1.0, (successes + smoothing) / (attempts + 2.0 * smoothing)))


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    replacement: dict[str, Any] = {}
    payload["metadata"] = replacement
    return replacement


def _trace_selected_seeds(data: dict[str, Any], events: list[Any]) -> list[str]:
    selected = _string_list(data.get("selected_seeds"))
    if selected:
        return selected
    seeds = []
    for event in events:
        if not isinstance(event, dict):
            continue
        source = event.get("source")
        target = event.get("target")
        if source == "task" and isinstance(target, str) and target != "final":
            seeds.append(target)
    return _unique_strings(seeds)


def _trace_context_allocations(
    data: dict[str, Any],
    events: list[Any],
    *,
    default_summary_ratio: float,
) -> dict[str, float]:
    explicit = _string_float_dict(data.get("context_allocations") or data.get("context_ratios"))
    if explicit:
        return explicit
    allocations: dict[str, float] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        target = event.get("target")
        if not isinstance(target, str) or target == "final":
            continue
        ratio = _optional_float(event.get("context_ratio"))
        if ratio is None:
            full_context = event.get("full_context")
            if isinstance(full_context, bool):
                ratio = 1.0 if full_context else default_summary_ratio
            elif event.get("source") == "task":
                ratio = 1.0
        if ratio is not None:
            allocations[target] = max(0.0, min(1.0, ratio))
    return allocations


def _trace_activated_verifiers(data: dict[str, Any]) -> list[str]:
    for key in ("activated_verifiers", "verifier_nodes", "verifier_placements"):
        values = _string_list(data.get(key))
        if values:
            return values
    return []


def _trace_pruned_edges(data: dict[str, Any]) -> list[tuple[str, str]]:
    raw_edges = data.get("pruned_edges")
    if not isinstance(raw_edges, list | tuple):
        return []
    edges = []
    for item in raw_edges:
        if isinstance(item, list | tuple) and len(item) == 2:
            source, target = item
            if isinstance(source, str) and isinstance(target, str):
                edges.append((source, target))
        elif isinstance(item, dict):
            source = item.get("source")
            target = item.get("target")
            if isinstance(source, str) and isinstance(target, str):
                edges.append((source, target))
    return edges


def _trace_outcome(row: dict[str, Any]) -> float | None:
    verification = _optional_bool(row.get("verification_passed"))
    if verification is not None:
        return 1.0 if verification else 0.0
    quality_passed = _optional_bool(row.get("quality_passed"))
    if quality_passed is not None:
        return 1.0 if quality_passed else 0.0
    quality_score = _optional_float(row.get("quality_score"))
    if quality_score is not None:
        return max(0.0, min(1.0, quality_score))
    passed = _optional_bool(row.get("passed"))
    if passed is not None:
        return 1.0 if passed else 0.0
    return None


def _outcome_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str | None], dict[str, Any]]:
    lookup = {}
    for row in rows:
        task_id = row.get("task_id") or row.get("task_name")
        if not isinstance(task_id, str):
            continue
        policy = row.get("policy")
        lookup[(task_id, policy if isinstance(policy, str) else None)] = row
    return lookup


def _matching_outcome(
    trace: dict[str, Any],
    outcomes: dict[tuple[str, str | None], dict[str, Any]],
) -> dict[str, Any] | None:
    task_id = trace.get("task_id") or trace.get("task_name")
    if not isinstance(task_id, str):
        return None
    policy = trace.get("policy")
    if isinstance(policy, str) and (task_id, policy) in outcomes:
        return outcomes[(task_id, policy)]
    return outcomes.get((task_id, None))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [item for item in value if isinstance(item, str)]
    return []


def _unique_strings(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _string_float_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, raw_value in value.items():
        if isinstance(key, str):
            numeric = _optional_float(raw_value)
            if numeric is not None:
                result[key] = max(0.0, min(1.0, numeric))
    return result


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_float(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None
