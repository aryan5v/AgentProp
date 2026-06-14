import json

import pytest

from agentprop.integrations import (
    CursorAgentConfig,
    CursorAgentError,
    CursorAgentProcessResult,
    CursorCommandProposer,
    parse_cursor_command_output,
    render_cursor_command_prompt,
)
from agentprop.runtime import (
    ExecutionEvent,
    ExecutionStateTracker,
    TerminalTurnRequest,
)


def test_parse_cursor_command_output_accepts_json() -> None:
    parsed = parse_cursor_command_output(
        '{"command": "pytest -q", "rationale": "verify before finalizing"}'
    )

    assert parsed.command == "pytest -q"
    assert parsed.rationale == "verify before finalizing"


def test_parse_cursor_command_output_extracts_embedded_json() -> None:
    parsed = parse_cursor_command_output(
        'Here is the next command:\n{"command": "rg failing_test", "rationale": "inspect"}'
    )

    assert parsed.command == "rg failing_test"


def test_parse_cursor_command_output_rejects_empty_output() -> None:
    with pytest.raises(CursorAgentError):
        parse_cursor_command_output("")


def test_cursor_command_proposer_uses_plan_mode_and_returns_metadata() -> None:
    calls: list[tuple[list[str], str]] = []

    def runner(command, prompt, env, timeout_s):  # type: ignore[no-untyped-def]
        calls.append((list(command), prompt))
        assert env
        assert timeout_s == 3.0
        return CursorAgentProcessResult(
            stdout='{"command": "pytest tests/test_api.py", "rationale": "target verifier"}',
            stderr="",
            returncode=0,
        )

    request = _request()
    proposer = CursorCommandProposer(
        CursorAgentConfig(
            binary="cursor-agent",
            model="gpt-5.5",
            workspace="/tmp/project",
            timeout_s=3.0,
        ),
        runner=runner,
    )

    proposal = proposer(request)

    assert proposal.command == "pytest tests/test_api.py"
    assert proposal.metadata["source"] == "cursor-agent"
    command, prompt = calls[0]
    assert "--mode" in command
    assert "plan" in command
    assert "--output-format" in command
    assert "stream-json" in command
    assert "--model" in command
    assert "gpt-5.5" in command
    assert "Do not edit files" in prompt
    assert "recent_events" in prompt


def test_cursor_command_proposer_parses_stream_json_output() -> None:
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": '{"command": "ls -la", "rationale": "inspect"}',
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "usage": {"inputTokens": 10, "outputTokens": 5},
                }
            ),
        ]
    )

    def runner(command, prompt, env, timeout_s):  # type: ignore[no-untyped-def]
        return CursorAgentProcessResult(stdout=stdout, stderr="", returncode=0)

    proposer = CursorCommandProposer(CursorAgentConfig(), runner=runner)
    proposal = proposer(_request())

    assert proposal.command == "ls -la"
    assert proposer.usage.input_tokens == 10
    assert proposer.usage.output_tokens == 5


def test_parse_cursor_command_output_accepts_markdown_fence() -> None:
    parsed = parse_cursor_command_output(
        '```json\n{"command": "make test", "rationale": "build"}\n```'
    )

    assert parsed.command == "make test"


def test_parse_cursor_command_output_rejects_fence_marker_and_prose() -> None:
    with pytest.raises(CursorAgentError):
        parse_cursor_command_output("```json")

    with pytest.raises(CursorAgentError):
        parse_cursor_command_output("Proposing one comprehensive setup command for step 1")


def test_cursor_command_proposer_returns_noop_on_unexpected_exception() -> None:
    def runner(command, prompt, env, timeout_s):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    proposer = CursorCommandProposer(CursorAgentConfig(), runner=runner)
    proposal = proposer(_request())

    assert proposal.command == "true"
    assert proposal.metadata["proposal_failed"] is True
    assert any("RuntimeError" in error for error in proposal.metadata["proposal_errors"])


def test_cursor_command_proposer_returns_noop_when_parse_fails() -> None:
    def runner(command, prompt, env, timeout_s):  # type: ignore[no-untyped-def]
        return CursorAgentProcessResult(
            stdout="```json\nnot valid\n```",
            stderr="",
            returncode=0,
        )

    proposer = CursorCommandProposer(CursorAgentConfig(), runner=runner)
    proposal = proposer(_request())

    assert proposal.command == "true"
    assert proposal.metadata["proposal_failed"] is True


def test_cursor_command_proposer_returns_noop_on_failed_process() -> None:
    def runner(command, prompt, env, timeout_s):  # type: ignore[no-untyped-def]
        return CursorAgentProcessResult(stdout="", stderr="auth failed", returncode=1)

    proposer = CursorCommandProposer(CursorAgentConfig(), runner=runner)
    proposal = proposer(_request())

    assert proposal.command == "true"
    assert proposal.metadata["proposal_failed"] is True
    assert any("auth failed" in error for error in proposal.metadata["proposal_errors"])


def test_cursor_command_proposer_retries_transient_process_failure() -> None:
    calls = 0

    def runner(command, prompt, env, timeout_s):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 1:
            return CursorAgentProcessResult(
                stdout="",
                stderr="cursor-agent timed out after 3s",
                returncode=124,
            )
        return CursorAgentProcessResult(
            stdout='{"command": "pytest -q", "rationale": "retry succeeded"}',
            stderr="",
            returncode=0,
        )

    proposer = CursorCommandProposer(
        CursorAgentConfig(max_process_retries=1),
        runner=runner,
    )
    proposal = proposer(_request())

    assert proposal.command == "pytest -q"
    assert calls == 2


def test_render_cursor_command_prompt_includes_agentprop_state() -> None:
    prompt = render_cursor_command_prompt(_request())

    assert "AgentProp state" in prompt
    assert '"task": "fix parser"' in prompt
    assert '"error_signature": "AssertionError"' in prompt


def _request() -> TerminalTurnRequest:
    tracker = ExecutionStateTracker()
    tracker.observe(
        ExecutionEvent(
            step=1,
            command="pytest -q",
            exit_code=1,
            error_signature="AssertionError",
            tokens_used=100,
        )
    )
    return TerminalTurnRequest(
        task="fix parser",
        step=2,
        strategy="agentprop_controller",
        features=tracker.features(),
        transcript=tuple(tracker.events),
        metadata={"category": "implementation"},
    )
