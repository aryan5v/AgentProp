"""Deterministic public demos for the AgentProp control layer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agentprop.integrations.framework_adapters import graph_from_framework_dict, to_langgraph_dict
from agentprop.runtime.control_loop import ExecutionEvent
from agentprop.runtime.session import ControlSession, ControlSessionConfig
from agentprop.workflows import WORKFLOW_TEMPLATES

CONTROL_DEMOS = ("terminal", "multi-agent", "framework")


@dataclass(frozen=True, slots=True)
class ControlDemoResult:
    """Artifacts written by a control-layer demo run."""

    demo: str
    out_dir: Path
    artifacts: dict[str, Path]
    summary: dict[str, object]


def run_control_demo(demo: str, out_dir: str | Path) -> ControlDemoResult:
    """Run a deterministic control-layer demo and write trace/report artifacts."""

    runners: dict[str, Callable[[Path], ControlDemoResult]] = {
        "terminal": _run_terminal_demo,
        "multi-agent": _run_multi_agent_demo,
        "framework": _run_framework_demo,
    }
    try:
        runner = runners[demo]
    except KeyError as exc:
        raise ValueError(f"unknown demo: {demo}") from exc
    return runner(Path(out_dir) / demo)


def _run_terminal_demo(out_dir: Path) -> ControlDemoResult:
    session = ControlSession(
        ControlSessionConfig(
            workflow="tool_use_pipeline",
            task_id="terminal-control-demo",
            category="terminal-repair",
            token_budget=150,
            baseline_tokens=180,
            max_steps_without_verifier=3,
            repeated_error_threshold=2,
        )
    )
    session.observe(
        ExecutionEvent(
            step=1,
            command="inspect failing test",
            progress_made=True,
            tokens_used=35,
            elapsed_s=3.0,
        )
    )
    session.observe(
        ExecutionEvent(
            step=2,
            command="pytest -q",
            exit_code=1,
            verifier_run=True,
            verifier_passed=False,
            tokens_used=42,
            elapsed_s=5.0,
            error_signature="AssertionError:test_mailbox_empty",
        )
    )
    session.observe(
        ExecutionEvent(
            step=3,
            command="pytest -q",
            exit_code=1,
            verifier_run=True,
            verifier_passed=False,
            tokens_used=38,
            elapsed_s=5.0,
            error_signature="AssertionError:test_mailbox_empty",
        )
    )
    session.observe(
        ExecutionEvent(
            step=4,
            command="run focused verifier after strategy switch",
            verifier_run=True,
            verifier_passed=True,
            progress_made=True,
            trusted=True,
            tokens_used=25,
            elapsed_s=4.0,
            final_answer_written=True,
        )
    )
    session.record_outcome(
        passed=True,
        quality_score=1.0,
        metadata={
            "demo": "terminal",
            "claim": "repeated-error control avoids another redundant probe",
        },
    )
    return _write_demo("terminal", out_dir, session)


def _run_multi_agent_demo(out_dir: Path) -> ControlDemoResult:
    session = ControlSession(
        ControlSessionConfig(
            workflow="planner_coder_tester_reviewer",
            task_id="multi-agent-control-demo",
            category="implementation",
            baseline_tokens=220,
            max_steps_without_verifier=2,
        )
    )
    session.observe(
        ExecutionEvent(
            step=1,
            command="planner drafts implementation path",
            progress_made=True,
            tokens_used=45,
            elapsed_s=4.0,
        )
    )
    session.observe(
        ExecutionEvent(
            step=2,
            command="coder reports local pass",
            verifier_run=True,
            verifier_passed=True,
            progress_made=True,
            tokens_used=70,
            elapsed_s=7.0,
            final_answer_written=True,
            trusted=False,
        )
    )
    session.observe(
        ExecutionEvent(
            step=3,
            command="tester runs independent verifier",
            verifier_run=True,
            verifier_passed=True,
            progress_made=True,
            tokens_used=35,
            elapsed_s=5.0,
            final_answer_written=True,
            trusted=True,
        )
    )
    session.record_outcome(
        passed=True,
        quality_score=1.0,
        metadata={
            "demo": "multi-agent",
            "claim": "self-reported passes are not trusted until an independent verifier confirms",
        },
    )
    return _write_demo("multi-agent", out_dir, session)


def _run_framework_demo(out_dir: Path) -> ControlDemoResult:
    source_graph = WORKFLOW_TEMPLATES["planner_coder_tester_reviewer"]()
    framework_graph = graph_from_framework_dict(to_langgraph_dict(source_graph), "langgraph")
    session = ControlSession(
        ControlSessionConfig(
            workflow=framework_graph,
            task_id="framework-control-demo",
            category="framework-integration",
            baseline_tokens=160,
            max_steps_without_verifier=1,
        )
    )
    session.observe(
        ExecutionEvent(
            step=1,
            command="planner node emitted framework state",
            progress_made=True,
            tokens_used=40,
            elapsed_s=4.0,
        )
    )
    session.observe(
        ExecutionEvent(
            step=2,
            command="framework verifier node confirms state",
            verifier_run=True,
            verifier_passed=True,
            progress_made=True,
            tokens_used=28,
            elapsed_s=3.0,
            trusted=True,
        )
    )
    session.record_outcome(
        passed=True,
        quality_score=1.0,
        metadata={
            "demo": "framework",
            "framework": "langgraph-style-dict",
            "claim": "framework graphs can be analyzed and wrapped with the same session API",
        },
    )
    return _write_demo("framework", out_dir, session)


def _write_demo(demo: str, out_dir: Path, session: ControlSession) -> ControlDemoResult:
    artifacts = session.write_artifacts(out_dir)
    return ControlDemoResult(
        demo=demo,
        out_dir=out_dir,
        artifacts=artifacts,
        summary=session.summary(),
    )
