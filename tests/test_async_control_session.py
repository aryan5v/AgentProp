"""Tests for AsyncControlSession — async wrapper around ControlSession.

These tests drive the async API via ``asyncio.run`` so they do not require
the optional ``pytest-asyncio`` plugin.
"""

from __future__ import annotations

import asyncio

from agentprop.runtime import AsyncControlSession, ControlDecision, ExecutionEvent
from agentprop.runtime.session import ControlAnalysis


def _make_event(
    step: int = 1, *, verifier: bool = False, passed: bool | None = None
) -> ExecutionEvent:
    return ExecutionEvent(
        step=step,
        command=f"step_{step}",
        tokens_used=100,
        verifier_run=verifier,
        verifier_passed=passed,
        progress_made=True,
    )


def test_async_start_returns_session():
    async def scenario() -> AsyncControlSession:
        return await AsyncControlSession.start(
            "planner_coder_tester_reviewer",
            task_id="test-async",
        )

    session = asyncio.run(scenario())
    assert isinstance(session, AsyncControlSession)
    assert isinstance(session.analysis, ControlAnalysis)
    assert session.analysis.node_count > 0


def test_observe_returns_control_decision():
    async def scenario() -> ControlDecision:
        session = await AsyncControlSession.start(
            "planner_coder_tester_reviewer",
            task_id="test-observe",
        )
        return await session.observe(_make_event(1))

    decision = asyncio.run(scenario())
    assert isinstance(decision, ControlDecision)
    assert decision.action in ("CONTINUE", "FORCE_VERIFY", "SWITCH_STRATEGY", "FINALIZE")


def test_multiple_observations_accumulate():
    async def scenario() -> int:
        session = await AsyncControlSession.start(
            "planner_coder_tester_reviewer",
            task_id="test-multi",
        )
        for i in range(3):
            await session.observe(_make_event(i + 1))
        return len(session.decisions)

    assert asyncio.run(scenario()) == 3


def test_record_outcome():
    async def scenario() -> dict:
        session = await AsyncControlSession.start(
            "planner_coder_tester_reviewer",
            task_id="test-outcome",
        )
        await session.observe(_make_event(1))
        return await session.record_outcome(passed=True)

    outcome = asyncio.run(scenario())
    assert isinstance(outcome, dict)
    assert outcome.get("passed") is True


def test_write_artifacts(tmp_path):
    async def scenario() -> dict:
        session = await AsyncControlSession.start(
            "planner_coder_tester_reviewer",
            task_id="test-artifacts",
        )
        await session.observe(_make_event(1, verifier=True, passed=True))
        return await session.write_artifacts(tmp_path)

    asyncio.run(scenario())
    assert (tmp_path / "trace.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()


def test_concurrent_sessions_do_not_interfere():
    """Two async sessions can run concurrently without state bleed."""

    async def run_session(task_id: str, n: int) -> int:
        s = await AsyncControlSession.start("fan_out_parallel", task_id=task_id)
        for i in range(n):
            await s.observe(ExecutionEvent(step=i + 1, tokens_used=50))
        return len(s.decisions)

    async def scenario() -> list[int]:
        return await asyncio.gather(
            run_session("t1", 3),
            run_session("t2", 5),
        )

    results = asyncio.run(scenario())
    assert results[0] == 3
    assert results[1] == 5


def test_config_passthrough():
    async def scenario() -> AsyncControlSession:
        return await AsyncControlSession.start(
            "feedback_loop",
            task_id="cfg-test",
            category="research",
            token_budget=5000,
        )

    session = asyncio.run(scenario())
    assert session.config.task_id == "cfg-test"
    assert session.config.category == "research"
    assert session.config.token_budget == 5000


def test_decide_without_events():
    """decide() with no prior events should still return a ControlDecision."""

    async def scenario() -> ControlDecision:
        session = await AsyncControlSession.start(
            "planner_coder_tester_reviewer",
            task_id="test-decide",
        )
        return await session.decide()

    decision = asyncio.run(scenario())
    assert isinstance(decision, ControlDecision)
