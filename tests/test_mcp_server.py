import json

from agentprop.integrations.mcp_server import handle_json_rpc


def test_mcp_server_lists_agentprop_tools() -> None:
    response = handle_json_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response is not None
    names = {tool["name"] for tool in response["result"]["tools"]}
    assert "agentprop_analyze" in names
    assert "agentprop_agent_instructions" in names
    assert "agentprop_control_start" in names


def test_mcp_server_calls_agent_instructions_tool() -> None:
    response = handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "agentprop_agent_instructions",
                "arguments": {
                    "workflow": "planner_coder_tester_reviewer",
                    "target": "codex",
                    "budget": 2,
                    "trials": 3,
                },
            },
        }
    )

    assert response is not None
    text = response["result"]["content"][0]["text"]
    assert "AgentProp Brief For Codex" in text
    assert "Suggested Agent Prompt" in text


def test_mcp_server_returns_errors_for_bad_tool_call() -> None:
    response = handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "missing_tool", "arguments": {}},
        }
    )

    assert response is not None
    assert response["error"]["code"] == -32000


def test_mcp_control_tools_manage_session_lifecycle() -> None:
    start = handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "agentprop_control_start",
                "arguments": {
                    "workflow": "planner_coder_tester_reviewer",
                    "task_id": "mcp-control",
                    "category": "implementation",
                    "baseline_tokens": 100,
                },
            },
        }
    )

    assert start is not None
    start_payload = json.loads(start["result"]["content"][0]["text"])
    session_id = start_payload["session_id"]

    observed = handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "agentprop_control_observe",
                "arguments": {
                    "session_id": session_id,
                    "step": 1,
                    "verifier_run": True,
                    "verifier_passed": True,
                    "final_answer_written": True,
                    "trusted": False,
                    "tokens_used": 40,
                },
            },
        }
    )

    assert observed is not None
    observed_payload = json.loads(observed["result"]["content"][0]["text"])
    assert observed_payload["decision"]["action"] == "FORCE_VERIFY"

    finish = handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "agentprop_control_finish",
                "arguments": {"session_id": session_id, "passed": True},
            },
        }
    )

    assert finish is not None
    finish_payload = json.loads(finish["result"]["content"][0]["text"])
    assert finish_payload["outcome"]["passed"] is True
