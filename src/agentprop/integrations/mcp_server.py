"""Small JSON-RPC/MCP-style server for AgentProp tools."""

from __future__ import annotations

import json
import sys
from typing import Any, cast

from agentprop.algorithms import bottleneck_nodes, low_weight_edges, risk_aware_verifier_placement
from agentprop.cli import _build_recommendation_report, _load_workflow
from agentprop.core import AgentGraph
from agentprop.evaluation.metrics import RecommendationReport
from agentprop.evaluation.reporting import render_markdown_report, report_to_dict
from agentprop.integrations.agent_instructions import (
    CodingAgentTarget,
    render_coding_agent_instructions,
)

SERVER_INFO = {"name": "agentprop", "version": "0.1.0a2"}


def main() -> int:
    """Run the stdio JSON-RPC server."""

    for line in sys.stdin:
        if not line.strip():
            continue
        response = handle_json_rpc(json.loads(line))
        if response is not None:
            print(json.dumps(response, sort_keys=True), flush=True)
    return 0


def handle_json_rpc(message: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message."""

    request_id = message.get("id")
    method = str(message.get("method", ""))
    try:
        if method == "initialize":
            result: Any = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            }
        elif method == "tools/list":
            result = {"tools": _tool_specs()}
        elif method == "tools/call":
            params = message.get("params", {})
            if not isinstance(params, dict):
                raise ValueError("tools/call params must be an object")
            result = _call_tool(str(params.get("name", "")), params.get("arguments", {}))
        else:
            raise ValueError(f"unsupported method: {method}")
    except Exception as exc:
        if request_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": str(exc)},
        }
    if request_id is None:
        return None
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _tool_specs() -> list[dict[str, Any]]:
    workflow_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workflow": {"type": "string"},
        },
        "required": ["workflow"],
    }
    optimize_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "workflow": {"type": "string"},
            "budget": {"type": "integer", "default": 2},
            "algorithm": {"type": "string", "default": "greedy"},
            "model": {"type": "string", "default": "independent-cascade"},
            "trials": {"type": "integer", "default": 100},
        },
        "required": ["workflow"],
    }
    return [
        {
            "name": "agentprop_analyze",
            "description": "Analyze graph bottlenecks, pruning candidates, and verifier placement.",
            "inputSchema": workflow_schema,
        },
        {
            "name": "agentprop_optimize",
            "description": "Recommend context seed agents and return cost/coverage metrics.",
            "inputSchema": optimize_schema,
        },
        {
            "name": "agentprop_agent_instructions",
            "description": "Generate a Claude Code/Codex-ready AgentProp workflow brief.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    **optimize_schema["properties"],
                    "target": {
                        "type": "string",
                        "enum": ["claude-code", "codex", "generic"],
                        "default": "generic",
                    },
                },
                "required": ["workflow"],
            },
        },
        {
            "name": "agentprop_report",
            "description": "Generate a Markdown optimization report.",
            "inputSchema": optimize_schema,
        },
    ]


def _call_tool(name: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    if name == "agentprop_analyze":
        return _text_result(json.dumps(_analyze(arguments), indent=2, sort_keys=True))
    if name == "agentprop_optimize":
        return _text_result(json.dumps(_optimize(arguments), indent=2, sort_keys=True))
    if name == "agentprop_agent_instructions":
        return _text_result(_instructions(arguments))
    if name == "agentprop_report":
        return _text_result(_report(arguments))
    raise ValueError(f"unknown tool: {name}")


def _analyze(arguments: dict[str, Any]) -> dict[str, Any]:
    workflow_name, graph = _load_workflow(_required_workflow(arguments))
    return {
        "workflow": workflow_name,
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "bottlenecks": bottleneck_nodes(graph),
        "pruning_candidates": low_weight_edges(graph),
        "verifier_candidates": risk_aware_verifier_placement(graph, min(3, graph.node_count)),
    }


def _optimize(arguments: dict[str, Any]) -> dict[str, Any]:
    _, graph = _load_workflow(_required_workflow(arguments))
    report = _recommendation(arguments, graph)
    return report_to_dict(report)


def _instructions(arguments: dict[str, Any]) -> str:
    workflow_name, graph = _load_workflow(_required_workflow(arguments))
    report = _recommendation(arguments, graph)
    target = str(arguments.get("target", "generic"))
    if target not in {"claude-code", "codex", "generic"}:
        raise ValueError("target must be claude-code, codex, or generic")
    return render_coding_agent_instructions(
        report,
        workflow_name=workflow_name,
        target=cast(CodingAgentTarget, target),
    )


def _report(arguments: dict[str, Any]) -> str:
    workflow_name, graph = _load_workflow(_required_workflow(arguments))
    return render_markdown_report(_recommendation(arguments, graph), workflow_name=workflow_name)


def _recommendation(arguments: dict[str, Any], graph: AgentGraph) -> RecommendationReport:
    return cast(
        RecommendationReport,
        _build_recommendation_report(
            graph,
            algorithm=str(arguments.get("algorithm", "greedy")),
            model_name=str(arguments.get("model", "independent-cascade")),
            budget=int(arguments.get("budget", 2)),
            trials=int(arguments.get("trials", 100)),
        ),
    )


def _required_workflow(arguments: dict[str, Any]) -> str:
    workflow = arguments.get("workflow")
    if not isinstance(workflow, str) or not workflow:
        raise ValueError("workflow is required")
    return workflow


def _text_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


if __name__ == "__main__":
    raise SystemExit(main())
