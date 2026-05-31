"""Run offline case-study accounting for AgentProp routing policies."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentprop.algorithms import greedy_seed_selection
from agentprop.core import AgentGraph, NodeType
from agentprop.evaluation import HumanLabelScorer, quality_cost_summary
from agentprop.evaluation.metrics import broadcast_cost, seeded_routing_cost
from agentprop.ml import (
    LinearNodeScorer,
    MessagePassingNodeScorer,
    build_seed_selection_example,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline AgentProp case-study accounting.")
    parser.add_argument("--tasks", type=Path, default=Path("benchmarks/case_study_tasks.json"))
    parser.add_argument("--workflow", default="planner_coder_tester_reviewer")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/case_study_offline"))
    args = parser.parse_args(argv)

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

    rows = []
    traces = []
    for task in tasks:
        for policy_name, seeds in policies.items():
            row, trace = _evaluate_task_arm(
                task,
                graph,
                policy_name=policy_name,
                seeds=seeds,
                trials=args.trials,
                seed=args.seed,
            )
            rows.append(row)
            traces.append(trace)

    payload = {
        "mode": "offline-simulated",
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
    print(f"Wrote {args.out_dir}")
    return 0


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
        build_seed_selection_example(builder(), budget=budget, trials=trials)
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
    else:
        model = IndependentCascade(seed=seed)
        propagation = model.simulate(graph, seeds, trials=trials)
        cost = seeded_routing_cost(graph, seeds, propagation.activated_nodes)
        coverage = propagation.coverage
        propagation_time = propagation.expected_propagation_time or propagation.propagation_time
        activated_nodes = propagation.activated_nodes

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
        summary[policy] = {
            "task_count": float(len(policy_rows)),
            "success_rate": _mean(
                [1.0 if row["verification_passed"] else 0.0 for row in policy_rows]
            ),
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
