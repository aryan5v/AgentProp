from agentprop.integrations.mcp_server import handle_json_rpc


def test_mcp_server_lists_agentprop_tools() -> None:
    response = handle_json_rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

    assert response is not None
    names = {tool["name"] for tool in response["result"]["tools"]}
    assert "agentprop_analyze" in names
    assert "agentprop_agent_instructions" in names


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
