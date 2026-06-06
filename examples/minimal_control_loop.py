"""Minimal ControlSession wrapper — the primary AgentProp integration path."""

from __future__ import annotations

from agentprop import ControlSession, ExecutionEvent


def main() -> None:
    session = ControlSession.start(
        "planner_coder_tester_reviewer",
        task_id="demo-001",
        category="implementation",
        token_budget=50_000,
    )

    steps = [
        ExecutionEvent(step=1, command="plan", tokens_used=2_000, verifier_passed=True),
        ExecutionEvent(
            step=2,
            command="implement",
            tokens_used=8_000,
            verifier_passed=False,
            error_signature="SyntaxError:line_12",
        ),
        ExecutionEvent(step=3, command="pytest -q", verifier_run=True, verifier_passed=True),
    ]

    for event in steps:
        decision = session.observe(event)
        print(f"step {event.step}: {decision.action}")
        if decision.action == "FORCE_VERIFY":
            print("  -> run independent verification before continuing")
        if decision.action == "FINALIZE":
            break

    session.record_outcome(passed=True, quality_score=1.0)
    paths = session.write_artifacts("reports/minimal-control-loop")
    print("Wrote:", ", ".join(str(path) for path in paths.values()))


if __name__ == "__main__":
    main()
