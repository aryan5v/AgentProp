"""Run offline case-study accounting for AgentProp routing policies."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agentprop.algorithms import greedy_seed_selection, quality_aware_greedy_seed_selection
from agentprop.core import AgentGraph, NodeType
from agentprop.evaluation import (
    HumanLabelScorer,
    LLMExecutionResult,
    OpenAICompatibleChatClient,
    RubricScorer,
    graded_context_allocations,
    openai_compatible_env_status,
    quality_cost_summary,
    routing_risks,
    run_verification_command,
    verification_row_fields,
)
from agentprop.evaluation.metrics import broadcast_cost, seeded_routing_cost
from agentprop.ml import (
    LinearNodeScorer,
    MessagePassingNodeScorer,
    build_seed_utility_example,
    extract_graph_features,
)
from agentprop.propagation import IndependentCascade
from agentprop.rl import AgentRoutingEnv, PPOConfig, train_ppo_policy
from agentprop.workflows import WORKFLOW_TEMPLATES


@dataclass(frozen=True, slots=True)
class CaseStudyTask:
    """A small case-study task specification."""

    id: str
    category: str
    prompt: str
    expected: str
    verification_command: str
    min_coverage: float


class CaseStudyExecutor(Protocol):
    """Protocol for real task execution adapters."""

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> LLMExecutionResult:
        """Execute one task prompt."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AgentProp case-study accounting.")
    parser.add_argument("--tasks", type=Path, default=Path("benchmarks/case_study_tasks.json"))
    parser.add_argument("--workflow", default="planner_coder_tester_reviewer")
    parser.add_argument(
        "--execution-mode",
        choices=["offline-simulated", "llm", "llm-routed", "llm-prompt-only"],
        default="offline-simulated",
    )
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    parser.add_argument("--llm-max-tokens", type=int, default=1200)
    parser.add_argument(
        "--run-verification",
        action="store_true",
        help="run each task verification command after an LLM arm completes",
    )
    parser.add_argument(
        "--verification-workdir",
        type=Path,
        default=None,
        help="working directory for verification commands",
    )
    parser.add_argument("--verification-timeout", type=float, default=120.0)
    parser.add_argument("--verification-output-limit", type=int, default=20_000)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="write a readiness manifest without executing any task arms",
    )
    parser.add_argument("--target-task-count", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("results/case_study_offline"))
    args = parser.parse_args(argv)
    execution_mode = _resolve_execution_mode(args.execution_mode)

    tasks = _load_tasks(args.tasks)
    if args.workflow not in WORKFLOW_TEMPLATES:
        raise ValueError(f"Unknown workflow template: {args.workflow}")
    graph = WORKFLOW_TEMPLATES[args.workflow]()
    policies = _policy_seed_sets(
        graph,
        workflow_name=args.workflow,
        budget=args.budget,
        trials=args.trials,
        episodes=args.episodes,
        epochs=args.epochs,
        seed=args.seed,
    )
    if args.preflight:
        payload = _preflight_payload(
            tasks,
            policies,
            workflow=args.workflow,
            execution_mode=execution_mode,
            target_task_count=args.target_task_count,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "preflight.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(f"Wrote {args.out_dir / 'preflight.json'}")
        return 0

    executor: CaseStudyExecutor | None = None
    if execution_mode.startswith("llm-"):
        executor = OpenAICompatibleChatClient.from_env(
            model=args.llm_model,
            base_url=args.llm_base_url,
            timeout_s=args.llm_timeout,
        )

    rows = []
    traces = []
    outputs = []
    verification_logs = []
    for task in tasks:
        for policy_name, seeds in policies.items():
            if executor is None:
                row, trace = _evaluate_task_arm(
                    task,
                    graph,
                    policy_name=policy_name,
                    seeds=seeds,
                    trials=args.trials,
                    seed=args.seed,
                )
                output = None
            else:
                if execution_mode == "llm-prompt-only":
                    row, trace, output = _evaluate_prompt_only_task_arm(
                        task,
                        graph,
                        policy_name=policy_name,
                        seeds=seeds,
                        executor=executor,
                        trials=args.trials,
                        seed=args.seed,
                        max_tokens=args.llm_max_tokens,
                    )
                else:
                    row, trace, output = _evaluate_routed_llm_task_arm(
                        task,
                        graph,
                        policy_name=policy_name,
                        seeds=seeds,
                        executor=executor,
                        trials=args.trials,
                        seed=args.seed,
                        max_tokens=args.llm_max_tokens,
                    )
            rows.append(row)
            traces.append(trace)
            if output is not None:
                if args.run_verification:
                    verification = run_verification_command(
                        task.verification_command,
                        cwd=args.verification_workdir,
                        timeout_s=args.verification_timeout,
                        output_limit=args.verification_output_limit,
                    )
                    row.update(verification_row_fields(verification))
                    output["verification"] = verification.to_dict()
                    verification_logs.append(
                        {
                            "task_id": task.id,
                            "policy": policy_name,
                            **verification.to_dict(),
                        }
                    )
                outputs.append(output)

    payload = {
        "mode": execution_mode,
        "requested_execution_mode": args.execution_mode,
        "workflow": args.workflow,
        "budget": args.budget,
        "trials": args.trials,
        "task_count": len(tasks),
        "policies": sorted(policies),
        "summary": _summarize(rows),
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(args.out_dir / "results.csv", rows)
    _write_traces(args.out_dir / "traces.jsonl", traces)
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n"
    )
    if outputs:
        _write_traces(args.out_dir / "outputs.jsonl", outputs)
    if verification_logs:
        _write_traces(args.out_dir / "verification_logs.jsonl", verification_logs)
    print(f"Wrote {args.out_dir}")
    return 0


def _resolve_execution_mode(raw_mode: str) -> str:
    if raw_mode == "llm":
        return "llm-routed"
    return raw_mode


def _load_tasks(path: Path) -> list[CaseStudyTask]:
    payload = json.loads(path.read_text())
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("case-study task file must contain a tasks list")
    tasks = []
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise ValueError("each case-study task must be an object")
        tasks.append(
            CaseStudyTask(
                id=str(raw_task["id"]),
                category=str(raw_task["category"]),
                prompt=str(raw_task["prompt"]),
                expected=str(raw_task["expected"]),
                verification_command=str(raw_task["verification_command"]),
                min_coverage=float(raw_task.get("min_coverage", 0.75)),
            )
        )
    return tasks


def _preflight_payload(
    tasks: list[CaseStudyTask],
    policies: dict[str, list[str]],
    *,
    workflow: str,
    execution_mode: str,
    target_task_count: int,
    llm_model: str | None,
    llm_base_url: str | None,
) -> dict[str, Any]:
    task_categories = sorted({task.category for task in tasks})
    env_status = (
        openai_compatible_env_status(model=llm_model, base_url=llm_base_url)
        if execution_mode.startswith("llm-")
        else {
            "ready": True,
            "api_key_env": None,
            "model": None,
            "base_url": None,
            "missing": [],
        }
    )
    total_arms = len(tasks) * len(policies)
    meets_target = len(tasks) >= target_task_count
    ready = bool(meets_target and env_status["ready"])
    return {
        "ready": ready,
        "status": "ready" if ready else "missing_requirements",
        "mode": execution_mode,
        "workflow": workflow,
        "task_count": len(tasks),
        "target_task_count": target_task_count,
        "meets_target_task_count": meets_target,
        "task_categories": task_categories,
        "policy_count": len(policies),
        "policies": {
            policy_name: {
                "selected_seeds": seeds,
                "seed_count": len(seeds),
            }
            for policy_name, seeds in sorted(policies.items())
        },
        "total_task_policy_arms": total_arms,
        "expected_artifacts": [
            "results.json",
            "results.csv",
            "summary.json",
            "traces.jsonl",
            *(["outputs.jsonl"] if execution_mode.startswith("llm-") else []),
        ],
        "llm_environment": env_status,
    }


def _policy_seed_sets(
    graph: AgentGraph,
    *,
    workflow_name: str,
    budget: int,
    trials: int,
    episodes: int,
    epochs: int,
    seed: int,
) -> dict[str, list[str]]:
    return {
        "broadcast": [node.id for node in graph.nodes() if node.type != NodeType.OUTPUT],
        "optimized_greedy": greedy_seed_selection(
            graph,
            budget,
            propagation_model=IndependentCascade(seed=seed),
            trials=trials,
        ),
        "quality_aware_greedy": quality_aware_greedy_seed_selection(
            graph,
            budget,
            propagation_model=IndependentCascade(seed=seed),
            trials=trials,
        ),
        "ml_message_passing": _message_passing_seeds(
            graph,
            workflow_name=workflow_name,
            budget=budget,
            trials=trials,
            epochs=epochs,
        ),
        "rl_ppo": _ppo_seeds(
            graph,
            budget=budget,
            trials=trials,
            episodes=episodes,
            seed=seed,
        ),
    }


def _message_passing_seeds(
    graph: AgentGraph,
    *,
    workflow_name: str,
    budget: int,
    trials: int,
    epochs: int,
) -> list[str]:
    examples = [
        build_seed_utility_example(builder(), budget=budget, trials=trials)
        for name, builder in WORKFLOW_TEMPLATES.items()
        if name != workflow_name
    ]
    feature_count = len(examples[0].features.feature_names)
    scorer = LinearNodeScorer.initialize(feature_count)
    scorer.train(examples, epochs=epochs, learning_rate=0.05)
    message_passing = MessagePassingNodeScorer(scorer)
    features = extract_graph_features(graph)
    neighbors = {
        node_id: sorted({*graph.predecessors(node_id), *graph.successors(node_id)})
        for node_id in features.node_features
    }
    scores = message_passing.score_nodes(features, neighbors)
    eligible = {node.id for node in graph.nodes() if node.type != NodeType.OUTPUT}
    return [
        node_id
        for node_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if node_id in eligible
    ][:budget]


def _ppo_seeds(
    graph: AgentGraph,
    *,
    budget: int,
    trials: int,
    episodes: int,
    seed: int,
) -> list[str]:
    env = AgentRoutingEnv(graph, budget=budget, trials=trials)
    policy, _ = train_ppo_policy(
        env,
        config=PPOConfig(episodes=episodes, learning_rate=0.05, seed=seed, max_steps=budget + 2),
    )
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < budget + 2:
        action = policy.act(env)
        state, _, done, _ = env.step(action)
        steps += 1
    return list(state.selected_seeds)


def _evaluate_task_arm(
    task: CaseStudyTask,
    graph: AgentGraph,
    *,
    policy_name: str,
    seeds: list[str],
    trials: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if policy_name == "broadcast":
        cost = broadcast_cost(graph)
        coverage = 1.0
        propagation_time = 0.0
        activated_nodes = {node.id for node in graph.nodes()}
        context_allocations = {
            node.id: 1.0 for node in graph.nodes() if node.type != NodeType.OUTPUT
        }
        risk_rows: list[dict[str, object]] = []
    else:
        model = IndependentCascade(seed=seed)
        propagation = model.simulate(graph, seeds, trials=trials)
        context_allocations = graded_context_allocations(
            graph,
            seeds=seeds,
            activated_nodes=propagation.activated_nodes,
        )
        cost = seeded_routing_cost(
            graph,
            seeds,
            propagation.activated_nodes,
            context_ratios=context_allocations,
        )
        coverage = propagation.coverage
        propagation_time = propagation.expected_propagation_time or propagation.propagation_time
        activated_nodes = propagation.activated_nodes
        risk_rows = [
            risk.to_dict() for risk in routing_risks(graph, context_ratios=context_allocations)
        ]

    quality_label = _simulated_human_label(coverage, task.min_coverage)
    quality = HumanLabelScorer().from_label(
        quality_label,
        rationale="offline simulated quality label derived from propagation coverage",
    )
    success = coverage >= task.min_coverage and quality.passed
    efficiency = quality_cost_summary(
        success_rate=1.0 if success else 0.0,
        token_cost=cost.total_cost,
        latency=cost.latency,
    )
    row = {
        "task_id": task.id,
        "category": task.category,
        "policy": policy_name,
        "selected_seeds": seeds,
        "context_allocations": context_allocations,
        "routing_risks": risk_rows,
        "activated_node_count": len(activated_nodes),
        "coverage": coverage,
        "token_cost": cost.token_cost,
        "message_cost": cost.message_cost,
        "total_cost": cost.total_cost,
        "message_count": cost.message_count,
        "latency": cost.latency,
        "propagation_time": float(propagation_time),
        "verification_command": task.verification_command,
        "verification_passed": success,
        "quality_score": quality.score,
        "quality_method": quality.method,
        "quality_passed": quality.passed,
        "missed_required_behavior": not success,
        "verifier_or_tester_caught_issue": (
            "tester" in activated_nodes or "reviewer" in activated_nodes
        ),
        "cost_adjusted_success": efficiency.cost_adjusted_success,
        "efficiency_score": efficiency.efficiency_score,
    }
    trace = {
        "task_id": task.id,
        "policy": policy_name,
        "events": [
            {
                "source": "task",
                "target": seed_node,
                "success": seed_node in activated_nodes,
                "token_cost": _node_token_cost(graph, seed_node),
            }
            for seed_node in seeds
        ],
    }
    return row, trace


def _evaluate_prompt_only_task_arm(
    task: CaseStudyTask,
    graph: AgentGraph,
    *,
    policy_name: str,
    seeds: list[str],
    executor: CaseStudyExecutor,
    trials: int,
    seed: int,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    routing = _routing_snapshot(
        graph,
        policy_name=policy_name,
        seeds=seeds,
        trials=trials,
        seed=seed,
    )
    system_prompt = _case_study_system_prompt(policy_name, seeds)
    user_prompt = _case_study_user_prompt(task, routing)
    execution = executor.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=max_tokens,
    )
    quality = _score_real_output(task, execution.response)
    token_cost = float(execution.usage.total_tokens)
    efficiency = quality_cost_summary(
        success_rate=quality.score,
        token_cost=token_cost,
        latency=execution.latency_s,
    )
    row = {
        "task_id": task.id,
        "category": task.category,
        "policy": policy_name,
        "validation_kind": "llm-prompt-only-not-validation",
        "routing_fidelity": "cosmetic_prompt_annotation",
        "selected_seeds": seeds,
        "context_allocations": routing["context_allocations"],
        "routing_risks": routing["routing_risks"],
        "activated_node_count": len(routing["activated_nodes"]),
        "coverage": routing["coverage"],
        "prompt_tokens": execution.usage.prompt_tokens,
        "completion_tokens": execution.usage.completion_tokens,
        "total_llm_tokens": execution.usage.total_tokens,
        "token_cost": token_cost,
        "message_cost": routing["estimated_message_cost"],
        "total_cost": token_cost,
        "message_count": routing["message_count"],
        "latency": execution.latency_s,
        "propagation_time": routing["propagation_time"],
        "model": execution.model,
        "verification_command": task.verification_command,
        "verification_status": "not_run",
        "verification_passed": None,
        "quality_score": quality.score,
        "quality_method": quality.method,
        "quality_passed": quality.passed,
        "missed_required_behavior": not quality.passed,
        "verifier_or_tester_caught_issue": _mentions_verification_role(execution.response),
        "cost_adjusted_success": efficiency.cost_adjusted_success,
        "efficiency_score": efficiency.efficiency_score,
    }
    trace = {
        "task_id": task.id,
        "policy": policy_name,
        "model": execution.model,
        "prompt_tokens": execution.usage.prompt_tokens,
        "completion_tokens": execution.usage.completion_tokens,
        "total_tokens": execution.usage.total_tokens,
        "latency_s": execution.latency_s,
        "events": [
            {
                "source": "task",
                "target": seed_node,
                "success": seed_node in routing["activated_nodes"],
                "token_cost": _node_token_cost(graph, seed_node),
            }
            for seed_node in seeds
        ],
    }
    output = {
        "task_id": task.id,
        "policy": policy_name,
        "model": execution.model,
        "prompt": execution.prompt,
        "response": execution.response,
        "expected": task.expected,
        "quality": {
            "score": quality.score,
            "method": quality.method,
            "passed": quality.passed,
            "rationale": quality.rationale,
            "metadata": quality.metadata,
        },
        "usage": {
            "prompt_tokens": execution.usage.prompt_tokens,
            "completion_tokens": execution.usage.completion_tokens,
            "total_tokens": execution.usage.total_tokens,
        },
        "verification_command": task.verification_command,
    }
    return row, trace, output


def _evaluate_routed_llm_task_arm(
    task: CaseStudyTask,
    graph: AgentGraph,
    *,
    policy_name: str,
    seeds: list[str],
    executor: CaseStudyExecutor,
    trials: int,
    seed: int,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    routing = _routing_snapshot(
        graph,
        policy_name=policy_name,
        seeds=seeds,
        trials=trials,
        seed=seed,
    )
    activated_nodes = {
        node_id
        for node_id in routing["activated_nodes"]
        if graph.node(node_id).type != NodeType.OUTPUT
    }
    node_order = _routed_node_order(graph, seeds, activated_nodes)
    node_outputs: dict[str, str] = {}
    executions = []
    total_latency = 0.0

    for node_id in node_order:
        node = graph.node(node_id)
        predecessor_outputs = {
            predecessor: node_outputs[predecessor]
            for predecessor in graph.predecessors(node_id)
            if predecessor in node_outputs
        }
        context_ratio = float(routing["context_allocations"].get(node_id, 0.0))
        full_context = context_ratio >= 0.80
        execution = executor.chat(
            system_prompt=_node_system_prompt(policy_name, node_id, node.type.value, node.role),
            user_prompt=_node_user_prompt(
                task,
                routing,
                node_id=node_id,
                full_context=full_context,
                context_ratio=context_ratio,
                predecessor_outputs=predecessor_outputs,
            ),
            temperature=0.1,
            max_tokens=max_tokens,
        )
        node_outputs[node_id] = execution.response
        total_latency += execution.latency_s
        executions.append(
            {
                "node_id": node_id,
                "node_type": node.type.value,
                "full_context": full_context,
                "context_ratio": context_ratio,
                "prompt": execution.prompt,
                "response": execution.response,
                "usage": _usage_to_dict(execution),
                "latency_s": execution.latency_s,
            }
        )

    final_execution = executor.chat(
        system_prompt=(
            "You are the AgentProp final output node. Consolidate routed agent outputs "
            "without inventing verification results."
        ),
        user_prompt=_final_user_prompt(task, policy_name, seeds, node_outputs),
        temperature=0.1,
        max_tokens=max_tokens,
    )
    total_latency += final_execution.latency_s
    executions.append(
        {
            "node_id": "final",
            "node_type": NodeType.OUTPUT.value,
            "full_context": False,
            "prompt": final_execution.prompt,
            "response": final_execution.response,
            "usage": _usage_to_dict(final_execution),
            "latency_s": final_execution.latency_s,
        }
    )

    quality = _score_real_output(task, final_execution.response)
    prompt_tokens = sum(item["usage"]["prompt_tokens"] for item in executions)
    completion_tokens = sum(item["usage"]["completion_tokens"] for item in executions)
    total_tokens = sum(item["usage"]["total_tokens"] for item in executions)
    token_cost = float(total_tokens)
    total_cost = token_cost + float(routing["estimated_message_cost"])
    efficiency = quality_cost_summary(
        success_rate=quality.score,
        token_cost=total_cost,
        latency=total_latency,
    )
    row = {
        "task_id": task.id,
        "category": task.category,
        "policy": policy_name,
        "validation_kind": "llm-routed-multi-agent",
        "routing_fidelity": "node_calls_and_context_scope",
        "selected_seeds": seeds,
        "context_allocations": routing["context_allocations"],
        "routing_risks": routing["routing_risks"],
        "executed_nodes": [item["node_id"] for item in executions],
        "activated_node_count": len(activated_nodes),
        "coverage": routing["coverage"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_llm_tokens": total_tokens,
        "token_cost": token_cost,
        "message_cost": routing["estimated_message_cost"],
        "total_cost": total_cost,
        "message_count": routing["message_count"],
        "latency": total_latency,
        "propagation_time": routing["propagation_time"],
        "model": final_execution.model,
        "verification_command": task.verification_command,
        "verification_status": "not_run",
        "verification_passed": None,
        "quality_score": quality.score,
        "quality_method": quality.method,
        "quality_passed": quality.passed,
        "missed_required_behavior": not quality.passed,
        "verifier_or_tester_caught_issue": any(
            _mentions_verification_role(item["response"]) for item in executions
        ),
        "cost_adjusted_success": efficiency.cost_adjusted_success,
        "efficiency_score": efficiency.efficiency_score,
    }
    trace = {
        "task_id": task.id,
        "policy": policy_name,
        "model": final_execution.model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_s": total_latency,
        "events": [
            {
                "source": "task" if item["full_context"] else "predecessor_summaries",
                "target": item["node_id"],
                "success": item["node_id"] in activated_nodes or item["node_id"] == "final",
                "token_cost": item["usage"]["total_tokens"],
                "full_context": item["full_context"],
            }
            for item in executions
        ],
    }
    output = {
        "task_id": task.id,
        "policy": policy_name,
        "model": final_execution.model,
        "validation_kind": "llm-routed-multi-agent",
        "node_executions": executions,
        "response": final_execution.response,
        "expected": task.expected,
        "quality": {
            "score": quality.score,
            "method": quality.method,
            "passed": quality.passed,
            "rationale": quality.rationale,
            "metadata": quality.metadata,
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "verification": {
            "command": task.verification_command,
            "status": "not_run",
            "passed": None,
        },
    }
    return row, trace, output


def _routing_snapshot(
    graph: AgentGraph,
    *,
    policy_name: str,
    seeds: list[str],
    trials: int,
    seed: int,
) -> dict[str, Any]:
    if policy_name == "broadcast":
        cost = broadcast_cost(graph)
        activated_nodes = {node.id for node in graph.nodes()}
        context_allocations = {
            node.id: 1.0 for node in graph.nodes() if node.type != NodeType.OUTPUT
        }
        return {
            "coverage": 1.0,
            "propagation_time": 0.0,
            "activated_nodes": activated_nodes,
            "estimated_message_cost": cost.message_cost,
            "context_allocations": context_allocations,
            "routing_risks": [],
            "message_count": cost.message_count,
        }
    model = IndependentCascade(seed=seed)
    propagation = model.simulate(graph, seeds, trials=trials)
    context_allocations = graded_context_allocations(
        graph,
        seeds=seeds,
        activated_nodes=propagation.activated_nodes,
    )
    cost = seeded_routing_cost(
        graph,
        seeds,
        propagation.activated_nodes,
        context_ratios=context_allocations,
    )
    return {
        "coverage": propagation.coverage,
        "propagation_time": propagation.expected_propagation_time or propagation.propagation_time,
        "activated_nodes": propagation.activated_nodes,
        "estimated_message_cost": cost.message_cost,
        "context_allocations": context_allocations,
        "routing_risks": [
            risk.to_dict() for risk in routing_risks(graph, context_ratios=context_allocations)
        ],
        "message_count": cost.message_count,
    }


def _routed_node_order(
    graph: AgentGraph,
    seeds: list[str],
    activated_nodes: set[str],
) -> list[str]:
    seeded = [node_id for node_id in seeds if node_id in activated_nodes]
    remaining = [
        node.id
        for node in graph.nodes()
        if node.id in activated_nodes and node.id not in seeded
    ]
    return [*seeded, *remaining]


def _node_system_prompt(
    policy_name: str,
    node_id: str,
    node_type: str,
    role: str | None,
) -> str:
    return (
        "You are one node in an AgentProp routed multi-agent case-study run. "
        f"Routing policy: {policy_name}. Node: {node_id}. Type: {node_type}. "
        f"Role: {role or 'unspecified'}. "
        "Return only this node's contribution; do not claim tests were run unless "
        "provided command output proves it."
    )


def _node_user_prompt(
    task: CaseStudyTask,
    routing: dict[str, Any],
    *,
    node_id: str,
    full_context: bool,
    context_ratio: float,
    predecessor_outputs: dict[str, str],
) -> str:
    context_lines = [
        f"Task ID: {task.id}",
        f"Category: {task.category}",
        f"Node receiving context: {node_id}",
        f"Expected outcome: {task.expected}",
        f"Verification command: {task.verification_command}",
        f"Propagation coverage: {routing['coverage']:.3f}",
        f"Routed context allocation: {context_ratio:.0%}",
    ]
    if full_context:
        context_lines.append(f"Full task context: {task.prompt}")
    else:
        context_lines.append("Compressed task context was routed to this node.")
        summary_limit = max(120, int(500 * context_ratio))
        context_lines.append(f"Summary: {_compact_text(task.prompt, limit=summary_limit)}")
        context_lines.append(
            "Use predecessor summaries below and preserve explicit edge-case constraints."
        )
    if predecessor_outputs:
        context_lines.append("Predecessor outputs:")
        for predecessor, output in sorted(predecessor_outputs.items()):
            context_lines.append(f"- {predecessor}: {_compact_text(output)}")
    else:
        context_lines.append("Predecessor outputs: none")
    return "\n".join(context_lines)


def _final_user_prompt(
    task: CaseStudyTask,
    policy_name: str,
    seeds: list[str],
    node_outputs: dict[str, str],
) -> str:
    lines = [
        f"Task ID: {task.id}",
        f"Routing policy: {policy_name}",
        f"Seed agents with full context: {', '.join(seeds) or 'none'}",
        f"Task: {task.prompt}",
        f"Expected outcome: {task.expected}",
        f"Verification command to report, not fabricate: {task.verification_command}",
        "Routed node outputs:",
    ]
    for node_id, output in sorted(node_outputs.items()):
        lines.append(f"- {node_id}: {_compact_text(output, limit=900)}")
    lines.append("Produce the final answer and clearly state that verification was not run.")
    return "\n".join(lines)


def _usage_to_dict(execution: LLMExecutionResult) -> dict[str, int]:
    return {
        "prompt_tokens": execution.usage.prompt_tokens,
        "completion_tokens": execution.usage.completion_tokens,
        "total_tokens": execution.usage.total_tokens,
    }


def _compact_text(text: str, *, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _case_study_system_prompt(policy_name: str, seeds: list[str]) -> str:
    return (
        "You are executing an AgentProp real case-study arm. "
        "Return a concise engineering answer with: plan, implementation summary, "
        "verification, risks, and final answer. "
        f"Routing policy: {policy_name}. Active seed agents: {', '.join(seeds) or 'none'}."
    )


def _case_study_user_prompt(task: CaseStudyTask, routing: dict[str, Any]) -> str:
    activated = ", ".join(sorted(routing["activated_nodes"]))
    return "\n".join(
        [
            f"Task ID: {task.id}",
            f"Category: {task.category}",
            f"Task: {task.prompt}",
            f"Expected outcome: {task.expected}",
            f"Verification command: {task.verification_command}",
            f"Activated workflow nodes: {activated}",
            f"Propagation coverage: {routing['coverage']:.3f}",
            "",
            "Produce the final case-study output for this arm.",
        ]
    )


def _score_real_output(task: CaseStudyTask, actual: str) -> Any:
    normalized_actual = actual.lower()
    expected_terms = [term for term in task.expected.lower().replace(".", "").split() if term]
    expected_hits = sum(1 for term in expected_terms if term in normalized_actual)
    expected_covered = expected_hits >= max(1, len(expected_terms) // 2)
    verification_mentioned = task.verification_command.split()[0].lower() in normalized_actual
    final_answer_present = "final" in normalized_actual or "answer" in normalized_actual
    scorer = RubricScorer(
        {
            "expected_outcome_covered": 0.5,
            "verification_mentioned": 0.3,
            "final_answer_present": 0.2,
        },
        pass_threshold=0.7,
    )
    return scorer.from_criteria(
        {
            "expected_outcome_covered": expected_covered,
            "verification_mentioned": verification_mentioned,
            "final_answer_present": final_answer_present,
        },
        rationale=(
            "automatic case-study rubric; replace or supplement with human labels "
            "for the final public study"
        ),
    )


def _mentions_verification_role(actual: str) -> bool:
    normalized = actual.lower()
    return any(term in normalized for term in ("tester", "reviewer", "verification", "test"))


def _simulated_human_label(coverage: float, min_coverage: float) -> float:
    if coverage >= min_coverage:
        return 5.0
    if coverage >= min_coverage * 0.8:
        return 4.0
    if coverage >= min_coverage * 0.5:
        return 3.0
    return 2.0


def _node_token_cost(graph: AgentGraph, node_id: str) -> float:
    try:
        return graph.node(node_id).token_cost
    except KeyError:
        return 0.0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    policies = sorted({str(row["policy"]) for row in rows})
    summary: dict[str, dict[str, float]] = {}
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        verification_values = [
            bool(row["verification_passed"])
            for row in policy_rows
            if isinstance(row.get("verification_passed"), bool)
        ]
        summary[policy] = {
            "task_count": float(len(policy_rows)),
            "success_rate": _mean([1.0 if passed else 0.0 for passed in verification_values]),
            "verification_observed_count": float(len(verification_values)),
            "mean_quality_score": _mean([float(row["quality_score"]) for row in policy_rows]),
            "mean_total_cost": _mean([float(row["total_cost"]) for row in policy_rows]),
            "mean_token_cost": _mean([float(row["token_cost"]) for row in policy_rows]),
            "mean_message_count": _mean([float(row["message_count"]) for row in policy_rows]),
            "mean_efficiency_score": _mean([float(row["efficiency_score"]) for row in policy_rows]),
        }
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_traces(path: Path, traces: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(trace, sort_keys=True) for trace in traces) + "\n")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
