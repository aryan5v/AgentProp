import json

from agentprop.core import NodeType
from agentprop.rl import CategoryBanditRoutingPolicy
from agentprop.runtime import (
    AgentLoopConfig,
    AgentPropRuntimeController,
    AgentTurnRequest,
    AgentTurnResult,
    ControlledAgentLoop,
    ControlledTerminalLoop,
    ExecutionEvent,
    ExecutionStateTracker,
    RuntimeControllerConfig,
    RuntimeNodeRequest,
    RuntimeNodeResult,
    RuntimeRewardLogger,
    StoppingController,
    StoppingControllerConfig,
    TerminalCommandProposal,
    TerminalCommandResult,
    TerminalLoopConfig,
    TerminalTurnRequest,
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


def test_execution_state_tracker_exposes_normalized_budget_features() -> None:
    tracker = ExecutionStateTracker(
        [
            ExecutionEvent(step=1, tokens_used=25, elapsed_s=2.0),
            ExecutionEvent(step=2, tokens_used=25, elapsed_s=3.0),
        ]
    )

    features = tracker.features(token_budget=100, wall_time_budget_s=20)

    assert features.token_budget_fraction == 0.5
    assert features.wall_time_budget_fraction == 0.25


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


def test_self_reported_pass_forces_independent_verification() -> None:
    controller = StoppingController(StoppingControllerConfig())

    # Agent's own local eval claims a pass but is not a trusted/independent check.
    self_reported = ExecutionStateTracker(
        [ExecutionEvent(step=1, verifier_run=True, verifier_passed=True, trusted=False)]
    ).features()
    assert self_reported.unconfirmed_pass is True
    assert self_reported.evaluator_passed is False
    assert controller.decide(self_reported).action == "FORCE_VERIFY"

    # An independent (trusted) verifier pass is allowed to finalize.
    independent = ExecutionStateTracker(
        [ExecutionEvent(step=1, verifier_run=True, verifier_passed=True, trusted=True)]
    ).features()
    assert independent.evaluator_passed is True
    assert controller.decide(independent).action == "FINALIZE"


def test_independent_verification_can_be_disabled() -> None:
    controller = StoppingController(
        StoppingControllerConfig(require_independent_verification=False)
    )
    self_reported = ExecutionStateTracker(
        [ExecutionEvent(step=1, verifier_run=True, verifier_passed=True, trusted=False)]
    ).features()

    # With the guard off, a self-reported pass still does not finalize on its own,
    # but it is no longer escalated to FORCE_VERIFY either.
    assert controller.decide(self_reported).action == "CONTINUE"


def test_final_answer_without_verification_is_gated() -> None:
    controller = StoppingController(StoppingControllerConfig())
    written = ExecutionStateTracker(
        [ExecutionEvent(step=1, final_answer_written=True)]
    ).features()

    assert controller.decide(written).action == "FORCE_VERIFY"


def test_force_verify_then_finalize_recovers_false_local_pass() -> None:
    """A self-reported pass is escalated, then an independent verifier finalizes."""

    def proposer(request: TerminalTurnRequest) -> TerminalCommandProposal:
        return TerminalCommandProposal(command=f"python eval.py # step {request.step}")

    def executor(
        request: TerminalTurnRequest, proposal: TerminalCommandProposal
    ) -> TerminalCommandResult:
        # The agent runs its own eval and self-reports a pass (untrusted).
        return TerminalCommandResult(
            event=ExecutionEvent(
                step=request.step,
                command=proposal.command,
                verifier_run=True,
                verifier_passed=True,
                trusted=False,
            )
        )

    def verifier(
        request: TerminalTurnRequest,
        blocked_proposal: TerminalCommandProposal | None = None,
    ) -> TerminalCommandResult:
        # The independent verifier confirms the result (trusted).
        return TerminalCommandResult(
            event=ExecutionEvent(
                step=request.step,
                command="agentprop:independent_verify",
                verifier_run=True,
                verifier_passed=True,
                trusted=True,
            )
        )

    loop = ControlledTerminalLoop(
        controller=StoppingController(StoppingControllerConfig(max_steps_without_verifier=4)),
        config=TerminalLoopConfig(max_steps=6),
    )
    result = loop.run(task="t", proposer=proposer, executor=executor, verifier=verifier)

    # The self-reported pass triggered an independent check, which then finalized.
    assert "FORCE_VERIFY" in [d.action for d in result.decisions]
    assert result.decisions[-1].action == "FINALIZE"
    assert result.passed is True


def test_self_reported_pass_without_verifier_is_not_recorded_as_passed() -> None:
    """With no independent verifier, an unconfirmed pass must not finalize as passed."""

    def proposer(request: TerminalTurnRequest) -> TerminalCommandProposal:
        return TerminalCommandProposal(command=f"python eval.py # step {request.step}")

    def executor(
        request: TerminalTurnRequest, proposal: TerminalCommandProposal
    ) -> TerminalCommandResult:
        return TerminalCommandResult(
            event=ExecutionEvent(
                step=request.step,
                command=proposal.command,
                verifier_run=True,
                verifier_passed=True,
                trusted=False,
            )
        )

    loop = ControlledTerminalLoop(
        controller=StoppingController(StoppingControllerConfig(max_steps_without_verifier=4)),
        config=TerminalLoopConfig(max_steps=4),
    )
    # No verifier supplied: the controller keeps requesting independent verification,
    # and the unconfirmed self-report is never credited as a real pass.
    result = loop.run(task="t", proposer=proposer, executor=executor)

    assert result.features.unconfirmed_pass is True
    assert result.features.evaluator_passed is False
    assert result.passed is None


def test_runtime_reward_logger_updates_bandit_from_real_outcome(tmp_path) -> None:
    log_path = tmp_path / "rewards.jsonl"
    logger = RuntimeRewardLogger(
        CategoryBanditRoutingPolicy(arms=("baseline", "agentprop_controller"), epsilon=0.0),
        jsonl_path=log_path,
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
        action="FINALIZE",
    )
    logged = json.loads(log_path.read_text(encoding="utf-8"))

    assert row["features"]["evaluator_passed"] is True
    assert row["state"]["evaluator_passed"] is True
    assert row["action"] == "FINALIZE"
    assert logged["action"] == "FINALIZE"
    assert logged["outcome"]["passed"] is True
    assert logger.bandit.choose("incomplete-answer") == "agentprop_controller"


def test_controlled_agent_loop_forces_stale_verification() -> None:
    loop = ControlledAgentLoop(
        controller=StoppingController(StoppingControllerConfig(max_steps_without_verifier=1)),
        config=AgentLoopConfig(max_steps=4),
    )

    def turn_executor(request: AgentTurnRequest) -> AgentTurnResult:
        return AgentTurnResult(
            event=ExecutionEvent(step=request.step, tokens_used=10),
            output="draft",
        )

    def verifier(request: AgentTurnRequest) -> AgentTurnResult:
        return AgentTurnResult(
            event=ExecutionEvent(
                step=request.step,
                verifier_run=True,
                verifier_passed=True,
                tokens_used=5,
            ),
            output="PASS",
        )

    result = loop.run(task="demo", turn_executor=turn_executor, verifier=verifier)

    assert [decision.action for decision in result.decisions] == [
        "CONTINUE",
        "FORCE_VERIFY",
        "FINALIZE",
    ]
    assert result.passed is True
    assert result.features.total_tokens == 15


def test_controlled_agent_loop_logs_budgeted_state_action_outcome(tmp_path) -> None:
    logger = RuntimeRewardLogger(
        CategoryBanditRoutingPolicy(arms=("agentprop_controller",), epsilon=0.0),
        jsonl_path=tmp_path / "runtime-rewards.jsonl",
    )
    loop = ControlledAgentLoop(
        controller=StoppingController(
            StoppingControllerConfig(max_steps_without_verifier=1, token_budget=20)
        ),
        config=AgentLoopConfig(
            max_steps=3,
            task_id="demo-task",
            category="coding",
            baseline_tokens=40,
        ),
        reward_logger=logger,
    )

    def turn_executor(request: AgentTurnRequest) -> AgentTurnResult:
        return AgentTurnResult(
            event=ExecutionEvent(step=request.step, tokens_used=10, elapsed_s=2.0),
            output="draft",
        )

    def verifier(request: AgentTurnRequest) -> AgentTurnResult:
        return AgentTurnResult(
            event=ExecutionEvent(
                step=request.step,
                verifier_run=True,
                verifier_passed=True,
                tokens_used=5,
                elapsed_s=1.0,
            ),
            output="PASS",
        )

    result = loop.run(task="demo", turn_executor=turn_executor, verifier=verifier)
    logged = json.loads((tmp_path / "runtime-rewards.jsonl").read_text(encoding="utf-8"))

    assert result.reward_row is not None
    assert logged["state"]["token_budget_fraction"] == 0.75
    assert logged["action"] == "FINALIZE"
    assert logged["outcome"]["passed"] is True


def test_controlled_agent_loop_switches_strategy_after_repeated_errors() -> None:
    loop = ControlledAgentLoop(
        controller=StoppingController(StoppingControllerConfig(repeated_error_threshold=2)),
        config=AgentLoopConfig(max_steps=5, fallback_strategy="broadcast"),
    )
    seen_strategies: list[str] = []

    def turn_executor(request: AgentTurnRequest) -> AgentTurnResult:
        seen_strategies.append(request.strategy)
        if request.strategy == "broadcast":
            return AgentTurnResult(
                event=ExecutionEvent(
                    step=request.step,
                    verifier_run=True,
                    verifier_passed=True,
                    final_answer_written=True,
                    tokens_used=20,
                ),
                output="fixed",
            )
        return AgentTurnResult(
            event=ExecutionEvent(
                step=request.step,
                exit_code=1,
                tokens_used=12,
                error_signature="missing-edge-case",
            ),
            output="still failing",
        )

    result = loop.run(task="demo", turn_executor=turn_executor)

    assert "SWITCH_STRATEGY" in {decision.action for decision in result.decisions}
    assert result.strategy == "broadcast"
    assert seen_strategies == ["agentprop_controller", "agentprop_controller", "broadcast"]
    assert result.passed is True


def test_controlled_agent_loop_uses_initial_events_for_resume() -> None:
    loop = ControlledAgentLoop(
        controller=StoppingController(StoppingControllerConfig(repeated_error_threshold=2)),
        config=AgentLoopConfig(max_steps=4, fallback_strategy="broadcast"),
    )
    seen_strategies: list[str] = []

    def turn_executor(request: AgentTurnRequest) -> AgentTurnResult:
        seen_strategies.append(request.strategy)
        return AgentTurnResult(
            event=ExecutionEvent(
                step=request.step,
                verifier_run=True,
                verifier_passed=True,
                final_answer_written=True,
            ),
            output="fixed",
        )

    result = loop.run(
        task="demo",
        turn_executor=turn_executor,
        initial_events=(
            ExecutionEvent(step=1, exit_code=1, error_signature="same-miss"),
            ExecutionEvent(step=2, exit_code=1, error_signature="same-miss"),
        ),
    )

    assert result.decisions[0].action == "SWITCH_STRATEGY"
    assert seen_strategies == ["broadcast"]
    assert result.passed is True


def test_controlled_agent_loop_decides_without_running_turn() -> None:
    loop = ControlledAgentLoop(
        controller=StoppingController(StoppingControllerConfig(repeated_error_threshold=2)),
        config=AgentLoopConfig(fallback_strategy="broadcast"),
    )

    decision = loop.decide(
        task="demo",
        strategy="agentprop_controller",
        initial_events=(
            ExecutionEvent(step=1, exit_code=1, error_signature="same-miss"),
            ExecutionEvent(step=2, exit_code=1, error_signature="same-miss"),
        ),
    )

    assert decision.strategy == "agentprop_controller"
    assert decision.decision.action == "SWITCH_STRATEGY"
    assert decision.request.step == 3
    assert decision.request.features.repeated_error_count == 2
    assert len(decision.request.transcript) == 2


def test_controlled_agent_loop_records_bandit_reward() -> None:
    bandit = CategoryBanditRoutingPolicy(
        arms=("agentprop_controller", "baseline"),
        epsilon=0.0,
    )
    logger = RuntimeRewardLogger(bandit)
    loop = ControlledAgentLoop(
        config=AgentLoopConfig(
            max_steps=2,
            task_id="dna-insert",
            category="constraint-heavy",
            baseline_tokens=100,
        ),
        bandit=bandit,
        reward_logger=logger,
    )

    def turn_executor(request: AgentTurnRequest) -> AgentTurnResult:
        return AgentTurnResult(
            event=ExecutionEvent(
                step=request.step,
                verifier_run=True,
                verifier_passed=True,
                tokens_used=60,
            ),
            output="PASS",
        )

    result = loop.run(task="demo", turn_executor=turn_executor)

    assert result.reward_row is not None
    assert result.reward_row["token_savings"] == 0.4
    assert logger.bandit.choose("constraint-heavy") == "agentprop_controller"


def test_controlled_terminal_loop_executes_allowed_command() -> None:
    loop = ControlledTerminalLoop(
        controller=StoppingController(StoppingControllerConfig(max_steps_without_verifier=4)),
        config=TerminalLoopConfig(max_steps=2),
    )
    executed: list[str] = []

    def proposer(request: TerminalTurnRequest) -> TerminalCommandProposal:
        return TerminalCommandProposal(command=f"pytest -q #{request.step}")

    def executor(
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
    ) -> TerminalCommandResult:
        executed.append(proposal.command)
        return TerminalCommandResult(
            event=ExecutionEvent(
                step=request.step,
                command=proposal.command,
                verifier_run=True,
                verifier_passed=True,
                final_answer_written=True,
            ),
            stdout="PASS\n",
        )

    result = loop.run(task="demo", proposer=proposer, executor=executor)

    assert executed == ["pytest -q #1"]
    assert [decision.action for decision in result.decisions] == ["CONTINUE", "FINALIZE"]
    assert result.stdout == "PASS\n"
    assert result.passed is True


def test_controlled_terminal_loop_force_verify_blocks_pending_command() -> None:
    loop = ControlledTerminalLoop(
        controller=StoppingController(StoppingControllerConfig(max_steps_without_verifier=1)),
        config=TerminalLoopConfig(max_steps=2),
    )
    executed: list[str] = []
    verified_blocked: list[str] = []

    def proposer(request: TerminalTurnRequest) -> TerminalCommandProposal:
        return TerminalCommandProposal(command=f"python solve.py #{request.step}")

    def executor(
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
    ) -> TerminalCommandResult:
        executed.append(proposal.command)
        return TerminalCommandResult(
            event=ExecutionEvent(step=request.step, command=proposal.command),
        )

    def verifier(
        request: TerminalTurnRequest,
        blocked_proposal: TerminalCommandProposal | None = None,
    ) -> TerminalCommandResult:
        assert blocked_proposal is not None
        verified_blocked.append(blocked_proposal.command)
        return TerminalCommandResult(
            event=ExecutionEvent(
                step=request.step,
                command="pytest -q",
                verifier_run=True,
                verifier_passed=True,
                final_answer_written=True,
            ),
            stdout="verified\n",
        )

    result = loop.run(task="demo", proposer=proposer, executor=executor, verifier=verifier)

    assert executed == ["python solve.py #1"]
    assert verified_blocked == ["python solve.py #2"]
    assert [decision.action for decision in result.decisions] == [
        "CONTINUE",
        "FORCE_VERIFY",
    ]
    assert result.passed is True


def test_controlled_terminal_loop_switches_strategy_without_executing_proposal() -> None:
    loop = ControlledTerminalLoop(
        controller=StoppingController(StoppingControllerConfig(repeated_error_threshold=2)),
        config=TerminalLoopConfig(max_steps=4, fallback_strategy="baseline"),
    )
    executed: list[str] = []
    proposed: list[tuple[str, str]] = []

    def proposer(request: TerminalTurnRequest) -> TerminalCommandProposal:
        command = f"{request.strategy}:run-{request.step}"
        proposed.append((request.strategy, command))
        return TerminalCommandProposal(command=command)

    def executor(
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
    ) -> TerminalCommandResult:
        executed.append(proposal.command)
        if request.strategy == "baseline":
            return TerminalCommandResult(
                event=ExecutionEvent(
                    step=request.step,
                    command=proposal.command,
                    verifier_run=True,
                    verifier_passed=True,
                    final_answer_written=True,
                )
            )
        return TerminalCommandResult(
            event=ExecutionEvent(
                step=request.step,
                command=proposal.command,
                exit_code=1,
                error_signature="same",
            )
        )

    result = loop.run(task="demo", proposer=proposer, executor=executor)

    assert proposed[2] == ("agentprop_controller", "agentprop_controller:run-3")
    assert "agentprop_controller:run-3" not in executed
    assert executed == [
        "agentprop_controller:run-1",
        "agentprop_controller:run-2",
        "baseline:run-4",
    ]
    assert result.strategy == "baseline"
    assert "SWITCH_STRATEGY" in {decision.action for decision in result.decisions}


def test_controlled_terminal_loop_logs_state_action_outcome(tmp_path) -> None:
    logger = RuntimeRewardLogger(
        CategoryBanditRoutingPolicy(arms=("agentprop_controller",), epsilon=0.0),
        jsonl_path=tmp_path / "terminal-rewards.jsonl",
    )
    loop = ControlledTerminalLoop(
        controller=StoppingController(
            StoppingControllerConfig(max_steps_without_verifier=2, token_budget=40)
        ),
        config=TerminalLoopConfig(
            max_steps=2,
            task_id="terminal-demo",
            category="coding",
            baseline_tokens=100,
        ),
        reward_logger=logger,
    )

    def proposer(request: TerminalTurnRequest) -> TerminalCommandProposal:
        return TerminalCommandProposal(command="pytest -q")

    def executor(
        request: TerminalTurnRequest,
        proposal: TerminalCommandProposal,
    ) -> TerminalCommandResult:
        return TerminalCommandResult(
            event=ExecutionEvent(
                step=request.step,
                command=proposal.command,
                verifier_run=True,
                verifier_passed=True,
                tokens_used=20,
                elapsed_s=2.0,
            )
        )

    result = loop.run(task="demo", proposer=proposer, executor=executor)
    logged = json.loads((tmp_path / "terminal-rewards.jsonl").read_text(encoding="utf-8"))

    assert result.reward_row is not None
    assert logged["state"]["token_budget_fraction"] == 0.5
    assert logged["action"] == "FINALIZE"
    assert logged["outcome"]["passed"] is True


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
