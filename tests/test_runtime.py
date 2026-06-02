from agentprop.core import NodeType
from agentprop.rl import CategoryBanditRoutingPolicy
from agentprop.runtime import (
    AgentPropRuntimeController,
    ExecutionEvent,
    ExecutionStateTracker,
    RuntimeControllerConfig,
    RuntimeNodeRequest,
    RuntimeNodeResult,
    RuntimeRewardLogger,
    StoppingController,
    StoppingControllerConfig,
)
from agentprop.workflows import planner_coder_tester_reviewer


def test_runtime_controller_routes_full_and_compressed_context() -> None:
    graph = planner_coder_tester_reviewer()
    seen: dict[str, RuntimeNodeRequest] = {}

    def compressor(context: str, *, task: str, target_ratio: float) -> str:
        return f"SUMMARY({target_ratio}): {context.split()[0]}"

    def executor(request: RuntimeNodeRequest) -> RuntimeNodeResult:
        seen[request.node.id] = request
        return RuntimeNodeResult(node_id=request.node.id, output=f"{request.node.id}-ok")

    controller = AgentPropRuntimeController(
        graph,
        config=RuntimeControllerConfig(seed_budget=2, fixed_seeds=("coder", "tester")),
        compressor=compressor,
    )

    result = controller.run(
        task="Implement edge-case-heavy parser",
        shared_context="FULL_CONTEXT must handle empty strings and invalid input",
        executor=executor,
    )

    assert result.selected_seeds == ("coder", "tester")
    assert seen["coder"].full_context
    assert seen["coder"].visible_context.startswith("FULL_CONTEXT")
    assert seen["tester"].full_context
    assert seen["planner"].visible_context.startswith("SUMMARY")
    assert result.trace_events[0]["context_ratio"] < 1.0


def test_runtime_controller_verifier_can_intercept_compressed_failure() -> None:
    graph = planner_coder_tester_reviewer()
    executed: list[str] = []

    def executor(request: RuntimeNodeRequest) -> RuntimeNodeResult:
        executed.append(request.node.id)
        if request.node.id == "coder":
            output = "handles empty" if request.full_context else "missing empty handling"
            return RuntimeNodeResult(node_id="coder", output=output)
        if request.node.type == NodeType.VERIFIER:
            coder_output = request.upstream_outputs.get("coder", "")
            passed = "handles empty" in coder_output
            return RuntimeNodeResult(
                node_id=request.node.id,
                output="PASS" if passed else "FAIL: empty handling missing",
                passed=passed,
                intercept=not passed,
            )
        return RuntimeNodeResult(node_id=request.node.id, output=f"{request.node.id}-ok")

    controller = AgentPropRuntimeController(
        graph,
        config=RuntimeControllerConfig(seed_budget=1, fixed_seeds=("planner",)),
        compressor=lambda context, *, task, target_ratio: "summary without edge cases",
    )

    result = controller.run(
        task="Implement parser",
        shared_context="Full spec: handles empty input and invalid input.",
        executor=executor,
    )

    assert "tester" in executed
    assert "reviewer" not in executed
    assert result.node_results[-1].node_id == "tester"
    assert result.node_results[-1].intercept
    assert result.passed is False


def test_execution_state_tracker_extracts_real_loop_features() -> None:
    tracker = ExecutionStateTracker()

    tracker.observe(
        ExecutionEvent(step=1, command="pytest", exit_code=1, error_signature="assertion")
    )
    features = tracker.observe(
        ExecutionEvent(step=2, command="pytest", exit_code=1, error_signature="assertion")
    )

    assert features.step_count == 2
    assert features.steps_since_verifier == 2
    assert features.steps_since_progress == 2
    assert features.repeated_error_count == 2
    assert features.last_exit_code == 1


def test_stopping_controller_forces_verify_and_switches_strategy() -> None:
    controller = StoppingController(
        StoppingControllerConfig(max_steps_without_verifier=2, repeated_error_threshold=3)
    )
    stale = ExecutionStateTracker(
        [
            ExecutionEvent(step=1, progress_made=True),
            ExecutionEvent(step=2),
        ]
    ).features()
    repeated = ExecutionStateTracker(
        [
            ExecutionEvent(step=1, error_signature="same"),
            ExecutionEvent(step=2, error_signature="same"),
            ExecutionEvent(step=3, error_signature="same"),
        ]
    ).features()

    assert controller.decide(stale).action == "FORCE_VERIFY"
    assert controller.decide(repeated).action == "SWITCH_STRATEGY"


def test_runtime_reward_logger_updates_bandit_from_real_outcome() -> None:
    logger = RuntimeRewardLogger(
        CategoryBanditRoutingPolicy(
            arms=("baseline", "agentprop_controller"),
            epsilon=0.0,
        )
    )
    features = ExecutionStateTracker(
        [ExecutionEvent(step=1, verifier_run=True, verifier_passed=True)]
    ).features()

    row = logger.record(
        task_id="chess-best-move",
        category="incomplete-answer",
        strategy="agentprop_controller",
        passed=True,
        token_savings=0.12,
        features=features,
    )

    assert row["features"]["evaluator_passed"] is True
    assert logger.bandit.choose("incomplete-answer") == "agentprop_controller"


def test_runtime_controller_can_reject_cycles_in_strict_mode() -> None:
    graph = planner_coder_tester_reviewer()
    controller = AgentPropRuntimeController(
        graph,
        config=RuntimeControllerConfig(allow_cycles=False),
    )

    try:
        controller.run(
            task="demo",
            shared_context="context",
            executor=lambda request: RuntimeNodeResult(
                node_id=request.node.id,
                output="unused",
            ),
        )
    except ValueError as exc:
        assert "contains cycles" in str(exc)
    else:
        raise AssertionError("expected cyclic workflow rejection")


def test_runtime_controller_rejects_none_executor_result() -> None:
    graph = planner_coder_tester_reviewer()
    controller = AgentPropRuntimeController(
        graph,
        config=RuntimeControllerConfig(fixed_seeds=("planner",)),
    )

    try:
        controller.run(task="demo", shared_context="context", executor=lambda request: None)
    except RuntimeError as exc:
        assert "returned None" in str(exc)
    else:
        raise AssertionError("expected None executor rejection")


def test_runtime_truncates_space_free_context() -> None:
    graph = planner_coder_tester_reviewer()
    seen: dict[str, RuntimeNodeRequest] = {}
    controller = AgentPropRuntimeController(
        graph,
        config=RuntimeControllerConfig(fixed_seeds=("coder",)),
    )

    controller.run(
        task="demo",
        shared_context="abcdefghij",
        executor=lambda request: seen.setdefault(
            request.node.id,
            request,
        )
        and RuntimeNodeResult(node_id=request.node.id, output="ok"),
    )

    assert seen["planner"].visible_context.startswith("abc")
    assert len(seen["planner"].visible_context) < len("abcdefghij")
