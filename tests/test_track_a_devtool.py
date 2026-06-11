import json
from pathlib import Path

from agentprop import analyze, wrap
from agentprop.integrations import graph_from_langgraph_object
from agentprop.runtime import (
    ArmResult,
    Controller,
    ExecutionEvent,
    JSONLEventStore,
    RegexPIIScrubber,
    TestHarness,
    TrafficSplit,
    rollup_arms,
    scrub_event,
)


def test_analyze_report_contains_track_a_fields() -> None:
    report = analyze("planner_coder_tester_reviewer", trials=3)

    payload = report.to_dict()
    assert payload["verifier_placement"]
    assert payload["resolving_coverage"] >= 0.0
    assert payload["fault_tolerant_coverage"] >= 0.0
    assert payload["constrained_savings"] >= 0.0
    assert payload["recommended_seed_budget"] >= 1
    assert "AgentProp Workflow Analysis" in report.to_markdown()


def test_langgraph_object_import_and_wrap_trace(tmp_path: Path) -> None:
    class GraphView:
        nodes = {"planner": object(), "coder": object(), "tester": object()}
        edges = [("planner", "coder"), ("coder", "tester")]

    class Workflow:
        def get_graph(self) -> GraphView:
            return GraphView()

        def invoke(self, state: dict[str, object]) -> dict[str, object]:
            return {"answer": state["task"], "tokens_used": 123}

    graph = graph_from_langgraph_object(Workflow())
    assert graph.node_count == 3
    controlled = wrap(Workflow(), budget={"tokens": 1000}, trace_path=tmp_path / "trace.jsonl")

    result = controlled.run({"task": "ship"}, run_id="case-1")

    assert result.result["answer"] == "ship"
    assert result.cost_actual == 123
    assert result.decision_trace
    assert (tmp_path / "trace.jsonl").exists()


def test_harness_replays_reference_trace() -> None:
    harness = TestHarness.from_fixture("false_local_pass")

    assert harness.decision_at_step(1).action == "FORCE_VERIFY"
    assert harness.finalized_on_confirmed_pass()
    assert harness.verify_count() == 1


def test_durable_controller_resume_matches_uninterrupted(tmp_path: Path) -> None:
    store = JSONLEventStore(tmp_path)
    events = [
        ExecutionEvent(step=1, progress_made=True, tokens_used=10),
        ExecutionEvent(step=2, error_signature="E", tokens_used=10),
        ExecutionEvent(step=3, error_signature="E", tokens_used=10),
    ]
    uninterrupted = Controller(store=JSONLEventStore(tmp_path / "fresh"), run_id="run-a")
    uninterrupted_actions = [uninterrupted.observe(event).action for event in events]

    first = Controller(store=store, run_id="run-b")
    first.observe(events[0])
    resumed = Controller.resume("run-b", store=store)
    resumed_actions = [resumed.observe(event).action for event in events[1:]]

    step_counts = [first.snapshot().features.step_count, resumed.snapshot().features.step_count]
    resumed_steps = [
        first.tracker.events[0].step,
        *[event.step for event in resumed.tracker.events[1:]],
    ]
    assert step_counts == [1, 3]
    assert resumed_steps == [1, 2, 3]
    assert [*uninterrupted_actions[1:]] == resumed_actions


def test_pii_scrubber_redacts_before_persistence(tmp_path: Path) -> None:
    event = ExecutionEvent(
        step=1,
        command="call email=a@example.com token=abc123",
        error_signature="sk-1234567890abcdefSECRET",
    )
    scrubbed = scrub_event(event, RegexPIIScrubber())
    assert "a@example.com" not in str(scrubbed.command)
    assert "sk-" not in str(scrubbed.error_signature)

    store = JSONLEventStore(tmp_path)
    store.append("run", event)
    content = (tmp_path / "run.jsonl").read_text()
    assert "a@example.com" not in content
    assert "REDACTED" in content


def test_traffic_split_rollup_and_circuit_breaker() -> None:
    split = TrafficSplit({"baseline": 90, "a2": 10})
    assert split.assign("stable-run") in {"baseline", "a2"}

    report = rollup_arms(
        [
            *(ArmResult("baseline", True, 100, 0.10) for _ in range(20)),
            *(ArmResult("a2", False, 120, 0.12) for _ in range(20)),
        ],
        min_pass_rate=0.8,
        min_window=20,
        default_arm="baseline",
    )

    payload = report.to_dict()
    assert "a2" in payload["circuit_breaker_tripped"]
    assert json.dumps(payload)


def test_analyze_report_includes_seed_coverage_interval() -> None:
    report = analyze("planner_coder_tester_reviewer", trials=10)
    interval = report.seed_coverage
    assert interval is not None
    assert 0.0 <= interval.lower <= interval.mean <= interval.upper <= 1.0
    assert interval.samples == 10
    payload = report.to_dict()
    assert payload["seed_coverage"]["mean"] == interval.mean
    assert "Seed propagation coverage" in report.to_markdown()
