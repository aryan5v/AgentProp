"""Small JSON-RPC/MCP-style server for AgentProp tools."""

from __future__ import annotations

import json
import sys
from typing import Any, cast

from agentprop.algorithms import bottleneck_nodes, low_weight_edges, risk_aware_verifier_placement
from agentprop.cli import _build_recommendation_report, _load_workflow
from agentprop.core import AgentGraph
from agentprop.evaluation.metrics import RecommendationReport, build_what_if_k_curve
from agentprop.evaluation.reporting import render_markdown_report, report_to_dict
from agentprop.evaluation.runner import make_propagation_model
from agentprop.integrations.agent_instructions import (
    CodingAgentTarget,
    render_coding_agent_instructions,
)
from agentprop.integrations.session_store import SessionStore, warm_shared_analysis_cache
from agentprop.runtime import ControlSession, ExecutionEvent

SERVER_INFO = {"name": "agentprop", "version": "0.1.0a3"}
_SESSION_STORE: SessionStore | None = None


def _get_session_store() -> SessionStore:
    global _SESSION_STORE
    if _SESSION_STORE is None:
        _SESSION_STORE = SessionStore()
    return _SESSION_STORE


def main() -> int:
    """Run the FastMCP server when installed, otherwise the JSON-RPC fallback."""

    try:
        app = create_fastmcp_app()
    except RuntimeError:
        return _run_json_rpc_server()
    app.run()
    return 0


def _run_json_rpc_server() -> int:
    """Run the dependency-free stdio JSON-RPC server."""

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
        {
            "name": "agentprop_control_start",
            "description": "Start an analysis-backed runtime control session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow": {"type": "string"},
                    "task_id": {"type": "string"},
                    "category": {"type": "string", "default": "general"},
                    "token_budget": {"type": "integer"},
                    "wall_time_budget_s": {"type": "number"},
                    "baseline_tokens": {"type": "integer"},
                },
                "required": ["workflow", "task_id"],
            },
        },
        {
            "name": "agentprop_control_observe",
            "description": "Record one execution event and return the control decision.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "step": {"type": "integer"},
                    "command": {"type": "string"},
                    "exit_code": {"type": "integer"},
                    "verifier_run": {"type": "boolean", "default": False},
                    "verifier_passed": {"type": "boolean"},
                    "progress_made": {"type": "boolean", "default": False},
                    "tokens_used": {"type": "integer", "default": 0},
                    "elapsed_s": {"type": "number", "default": 0.0},
                    "error_signature": {"type": "string"},
                    "final_answer_written": {"type": "boolean", "default": False},
                    "trusted": {"type": "boolean", "default": True},
                },
                "required": ["session_id", "step"],
            },
        },
        {
            "name": "agentprop_control_decide",
            "description": "Return the current control decision for an active session.",
            "inputSchema": {
                "type": "object",
                "properties": {"session_id": {"type": "string"}},
                "required": ["session_id"],
            },
        },
        {
            "name": "agentprop_what_if_k",
            "description": "Return coverage/cost uncertainty curve for seed budgets k=1..max_k.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "workflow": {"type": "string"},
                    "max_k": {"type": "integer", "default": 3},
                    "model": {"type": "string", "default": "independent-cascade"},
                    "trials": {"type": "integer", "default": 50},
                },
                "required": ["workflow"],
            },
        },
        {
            "name": "agentprop_control_finish",
            "description": "Record the final outcome and close a control session.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "passed": {"type": "boolean"},
                    "strategy": {"type": "string", "default": "agentprop_controller"},
                    "quality_score": {"type": "number"},
                },
                "required": ["session_id", "passed"],
            },
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
    if name == "agentprop_control_start":
        return _json_text_result(_control_start(arguments))
    if name == "agentprop_control_observe":
        return _json_text_result(_control_observe(arguments))
    if name == "agentprop_control_decide":
        return _json_text_result(_control_decide(arguments))
    if name == "agentprop_control_finish":
        return _json_text_result(_control_finish(arguments))
    if name == "agentprop_what_if_k":
        return _json_text_result(_what_if_k(arguments))
    raise ValueError(f"unknown tool: {name}")


def create_fastmcp_app() -> Any:
    """Create the FastMCP app exposing AgentProp analysis and live-control tools."""

    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install agentprop[mcp] to run the FastMCP server.") from exc

    mcp = FastMCP("AgentProp")

    @mcp.tool  # type: ignore[untyped-decorator]
    def agentprop_analyze(workflow: str) -> dict[str, Any]:
        """Analyze graph bottlenecks, pruning candidates, and verifier placement."""

        return _analyze({"workflow": workflow})

    @mcp.tool  # type: ignore[untyped-decorator]
    def agentprop_optimize(
        workflow: str,
        budget: int = 2,
        algorithm: str = "auto",
        model: str = "independent-cascade",
        trials: int = 100,
    ) -> dict[str, Any]:
        """Recommend context seed agents and return cost/coverage metrics."""

        return _optimize(
            {
                "workflow": workflow,
                "budget": budget,
                "algorithm": algorithm,
                "model": model,
                "trials": trials,
            }
        )

    @mcp.tool  # type: ignore[untyped-decorator]
    def agentprop_agent_instructions(
        workflow: str,
        target: str = "generic",
        budget: int = 2,
        algorithm: str = "greedy",
        model: str = "independent-cascade",
        trials: int = 100,
    ) -> str:
        """Generate a Claude Code/Codex-ready AgentProp workflow brief."""

        return _instructions(
            {
                "workflow": workflow,
                "target": target,
                "budget": budget,
                "algorithm": algorithm,
                "model": model,
                "trials": trials,
            }
        )

    @mcp.tool  # type: ignore[untyped-decorator]
    def agentprop_report(
        workflow: str,
        budget: int = 2,
        algorithm: str = "greedy",
        model: str = "independent-cascade",
        trials: int = 100,
    ) -> str:
        """Generate a Markdown optimization report."""

        return _report(
            {
                "workflow": workflow,
                "budget": budget,
                "algorithm": algorithm,
                "model": model,
                "trials": trials,
            }
        )

    @mcp.tool  # type: ignore[untyped-decorator]
    def agentprop_control_start(
        workflow: str,
        task_id: str,
        category: str = "general",
        token_budget: int | None = None,
        wall_time_budget_s: float | None = None,
        baseline_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Start an analysis-backed runtime control session."""

        return _control_start(
            {
                "workflow": workflow,
                "task_id": task_id,
                "category": category,
                "token_budget": token_budget,
                "wall_time_budget_s": wall_time_budget_s,
                "baseline_tokens": baseline_tokens,
            }
        )

    @mcp.tool  # type: ignore[untyped-decorator]
    def agentprop_control_observe(
        session_id: str,
        step: int,
        command: str | None = None,
        exit_code: int | None = None,
        verifier_run: bool = False,
        verifier_passed: bool | None = None,
        progress_made: bool = False,
        tokens_used: int = 0,
        elapsed_s: float = 0.0,
        error_signature: str | None = None,
        final_answer_written: bool = False,
        trusted: bool = True,
    ) -> dict[str, Any]:
        """Record one execution event and return the control decision."""

        return _control_observe(
            {
                "session_id": session_id,
                "step": step,
                "command": command,
                "exit_code": exit_code,
                "verifier_run": verifier_run,
                "verifier_passed": verifier_passed,
                "progress_made": progress_made,
                "tokens_used": tokens_used,
                "elapsed_s": elapsed_s,
                "error_signature": error_signature,
                "final_answer_written": final_answer_written,
                "trusted": trusted,
            }
        )

    @mcp.tool  # type: ignore[untyped-decorator]
    def agentprop_control_decide(session_id: str) -> dict[str, Any]:
        """Return the current control decision for an active session."""

        return _control_decide({"session_id": session_id})

    @mcp.tool  # type: ignore[untyped-decorator]
    def agentprop_control_finish(
        session_id: str,
        passed: bool,
        strategy: str = "agentprop_controller",
        quality_score: float | None = None,
    ) -> dict[str, Any]:
        """Record the final outcome and close a control session."""

        return _control_finish(
            {
                "session_id": session_id,
                "passed": passed,
                "strategy": strategy,
                "quality_score": quality_score,
            }
        )

    return mcp


def _analyze(arguments: dict[str, Any]) -> dict[str, Any]:
    workflow_name, graph = _load_workflow(_required_workflow(arguments))
    warm_shared_analysis_cache(graph)
    return {
        "workflow": workflow_name,
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "bottlenecks": bottleneck_nodes(graph),
        "pruning_candidates": low_weight_edges(graph),
        "verifier_candidates": risk_aware_verifier_placement(graph, min(3, graph.node_count)),
    }


def _control_start(arguments: dict[str, Any]) -> dict[str, Any]:
    workflow = _required_workflow(arguments)
    task_id = _required_string(arguments, "task_id")
    session_id, session = _get_session_store().start_session(
        workflow=workflow,
        task_id=task_id,
        category=str(arguments.get("category") or "general"),
        token_budget=_optional_int(arguments, "token_budget"),
        wall_time_budget_s=_optional_float(arguments, "wall_time_budget_s"),
        baseline_tokens=_optional_int(arguments, "baseline_tokens"),
    )
    return {"session_id": session_id, "summary": session.summary()}


def _control_observe(arguments: dict[str, Any]) -> dict[str, Any]:
    session = _session(arguments)
    decision = session.observe(
        ExecutionEvent(
            step=int(arguments.get("step", 0)),
            command=_optional_string(arguments, "command"),
            exit_code=_optional_int(arguments, "exit_code"),
            verifier_run=bool(arguments.get("verifier_run", False)),
            verifier_passed=_optional_bool(arguments, "verifier_passed"),
            progress_made=bool(arguments.get("progress_made", False)),
            tokens_used=int(arguments.get("tokens_used", 0)),
            elapsed_s=float(arguments.get("elapsed_s", 0.0)),
            error_signature=_optional_string(arguments, "error_signature"),
            final_answer_written=bool(arguments.get("final_answer_written", False)),
            trusted=bool(arguments.get("trusted", True)),
        )
    )
    return {
        "decision": {
            "action": decision.action,
            "reason": decision.reason,
            "strategy": decision.strategy,
            "defer_command": decision.defer_command,
        },
        "summary": session.summary(),
    }


def _control_decide(arguments: dict[str, Any]) -> dict[str, Any]:
    session = _session(arguments)
    decision = session.decide()
    return {
        "decision": {
            "action": decision.action,
            "reason": decision.reason,
            "strategy": decision.strategy,
            "defer_command": decision.defer_command,
        },
        "summary": session.summary(),
    }


def _control_finish(arguments: dict[str, Any]) -> dict[str, Any]:
    session_id = _required_string(arguments, "session_id")
    outcome = _get_session_store().finish_session(
        session_id,
        passed=bool(arguments.get("passed", False)),
        strategy=str(arguments.get("strategy") or "agentprop_controller"),
        quality_score=_optional_float(arguments, "quality_score"),
        regression_risk=float(arguments.get("regression_risk", 0.0) or 0.0),
    )
    session = _get_session_store().get_session(session_id)
    return {"session_id": session_id, "outcome": outcome, "summary": session.summary()}


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
            algorithm=str(arguments.get("algorithm", "auto")),
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


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} is required")
    return value


def _optional_string(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_int(arguments: dict[str, Any], key: str) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    return int(value)


def _optional_float(arguments: dict[str, Any], key: str) -> float | None:
    value = arguments.get(key)
    if value is None:
        return None
    return float(value)


def _optional_bool(arguments: dict[str, Any], key: str) -> bool | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _session(arguments: dict[str, Any]) -> ControlSession:
    session_id = _required_string(arguments, "session_id")
    return _get_session_store().get_session(session_id)


def _what_if_k(arguments: dict[str, Any]) -> dict[str, Any]:
    _, graph = _load_workflow(_required_workflow(arguments))
    warm_shared_analysis_cache(graph)
    model = make_propagation_model(str(arguments.get("model", "independent-cascade")))
    max_k = int(arguments.get("max_k", arguments.get("budget", 3)))
    trials = int(arguments.get("trials", 50))
    entries = build_what_if_k_curve(
        graph,
        model=model,
        candidate_seeds=[node.id for node in graph.nodes()],
        max_k=min(max_k, graph.node_count),
        trials=trials,
    )
    return {
        "workflow_nodes": graph.node_count,
        "trials": trials,
        "curve": [
            {
                "k": entry.k,
                "seeds": entry.seeds,
                "coverage": entry.coverage,
                "coverage_uncertainty": entry.coverage_std,
                "estimated_savings": entry.estimated_savings,
                "quality_objective_score": entry.quality_objective_score,
            }
            for entry in entries
        ],
    }


def _text_result(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _json_text_result(payload: dict[str, Any]) -> dict[str, Any]:
    return _text_result(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
