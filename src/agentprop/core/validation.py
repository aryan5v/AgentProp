"""Validation for AgentProp workflow JSON documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentprop.core.types import NodeType


@dataclass(slots=True)
class ValidationIssue:
    """A validation issue with a JSON-ish path."""

    path: str
    message: str

    def format(self) -> str:
        """Return a readable issue string."""

        return f"{self.path}: {self.message}"


class WorkflowValidationError(ValueError):
    """Raised when a workflow JSON document is invalid."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        super().__init__("\n".join(issue.format() for issue in issues))


def validate_workflow_dict(data: dict[str, Any]) -> None:
    """Validate a workflow dictionary and raise on invalid input."""

    issues: list[ValidationIssue] = []
    _validate_top_level(data, issues)
    if issues:
        raise WorkflowValidationError(issues)

    node_ids = _validate_nodes(data["nodes"], issues)
    _validate_edges(data["edges"], node_ids, issues)
    if issues:
        raise WorkflowValidationError(issues)


def _validate_top_level(data: dict[str, Any], issues: list[ValidationIssue]) -> None:
    if not isinstance(data, dict):
        issues.append(ValidationIssue("$", "workflow must be a JSON object"))
        return
    if "nodes" not in data:
        issues.append(ValidationIssue("$.nodes", "missing required nodes list"))
    elif not isinstance(data["nodes"], list):
        issues.append(ValidationIssue("$.nodes", "must be a list"))
    if "edges" not in data:
        issues.append(ValidationIssue("$.edges", "missing required edges list"))
    elif not isinstance(data["edges"], list):
        issues.append(ValidationIssue("$.edges", "must be a list"))


def _validate_nodes(raw_nodes: list[Any], issues: list[ValidationIssue]) -> set[str]:
    node_ids: set[str] = set()
    for index, raw_node in enumerate(raw_nodes):
        path = f"$.nodes[{index}]"
        if not isinstance(raw_node, dict):
            issues.append(ValidationIssue(path, "node must be an object"))
            continue

        node_id = raw_node.get("id")
        if not isinstance(node_id, str) or not node_id:
            issues.append(ValidationIssue(f"{path}.id", "must be a non-empty string"))
            continue
        if node_id in node_ids:
            issues.append(ValidationIssue(f"{path}.id", f"duplicate node id: {node_id}"))
        node_ids.add(node_id)

        node_type = raw_node.get("type", NodeType.AGENT.value)
        if node_type not in {node_type.value for node_type in NodeType}:
            issues.append(ValidationIssue(f"{path}.type", f"unknown node type: {node_type}"))

        _validate_probability(raw_node, "reliability", path, issues)
        _validate_probability(raw_node, "error_rate", path, issues)
        _validate_non_negative(raw_node, "token_cost", path, issues)
        _validate_non_negative(raw_node, "latency", path, issues)

    return node_ids


def _validate_edges(
    raw_edges: list[Any],
    node_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    for index, raw_edge in enumerate(raw_edges):
        path = f"$.edges[{index}]"
        if not isinstance(raw_edge, dict):
            issues.append(ValidationIssue(path, "edge must be an object"))
            continue

        source = raw_edge.get("source")
        target = raw_edge.get("target")
        if source not in node_ids:
            issues.append(ValidationIssue(f"{path}.source", f"unknown source node: {source}"))
        if target not in node_ids:
            issues.append(ValidationIssue(f"{path}.target", f"unknown target node: {target}"))
        if source == target:
            issues.append(ValidationIssue(path, "self-loops are not supported in v1"))

        _validate_non_negative(raw_edge, "message_cost", path, issues)
        _validate_non_negative(raw_edge, "latency", path, issues)
        _validate_non_negative(raw_edge, "weight", path, issues)
        _validate_probability(raw_edge, "relevance", path, issues)
        _validate_probability(raw_edge, "reliability", path, issues)
        _validate_probability(raw_edge, "activation_probability", path, issues)
        _validate_probability(raw_edge, "dependency_strength", path, issues)


def _validate_probability(
    raw_object: dict[str, Any],
    field: str,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    if field not in raw_object or raw_object[field] is None:
        return
    value = raw_object[field]
    if not isinstance(value, int | float) or not 0 <= value <= 1:
        issues.append(ValidationIssue(f"{path}.{field}", "must be a number between 0 and 1"))


def _validate_non_negative(
    raw_object: dict[str, Any],
    field: str,
    path: str,
    issues: list[ValidationIssue],
) -> None:
    if field not in raw_object or raw_object[field] is None:
        return
    value = raw_object[field]
    if not isinstance(value, int | float) or value < 0:
        issues.append(ValidationIssue(f"{path}.{field}", "must be a non-negative number"))
