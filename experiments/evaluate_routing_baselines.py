"""Compare classical, ML, and RL routing baselines on workflow graphs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentprop.algorithms import (
    betweenness_seed_selection,
    celf_seed_selection,
    closeness_seed_selection,
    degree_seed_selection,
    greedy_seed_selection,
    k_core_seed_selection,
    pagerank_seed_selection,
    random_seed_selection,
)
from agentprop.core import AgentGraph, NodeType
from agentprop.evaluation.metrics import (
    CostSummary,
    broadcast_cost,
    quality_cost_summary,
    seeded_routing_cost,
)
from agentprop.ml import (
    EmpiricalRoutingExample,
    EmpiricalVerifierPlacementExample,
    LinearNodeRegressor,
    LinearNodeScorer,
    MessagePassingNodeScorer,
    MLPNodeScorer,
    PairwiseNodeRanker,
    SeedSelectionExample,
    VerifierPlacementExample,
    build_seed_ranking_example,
    build_seed_selection_example,
    extract_graph_features,
)
from agentprop.propagation import IndependentCascade
from agentprop.rl import (
    AgentRoutingEnv,
    FeaturePolicyConfig,
    NodeScorerRoutingPolicy,
    PPOConfig,
    QLearningConfig,
    ReinforceConfig,
    RoutingAction,
    RoutingState,
    train_feature_policy,
    train_ppo_policy,
    train_q_policy,
    train_reinforce_policy,
)
from agentprop.workflows import WORKFLOW_TEMPLATES

SeedSelector = Callable[[AgentGraph], list[str]]
NodeTrainingExample = (
    SeedSelectionExample
    | VerifierPlacementExample
    | EmpiricalRoutingExample
    | EmpiricalVerifierPlacementExample
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate classical, ML, and RL routing baselines."
    )
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--workflows",
        default=",".join(WORKFLOW_TEMPLATES),
        help="Comma-separated workflow template names.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/rl/routing_baseline_comparison.json"),
    )
    args = parser.parse_args(argv)

    workflow_names = _parse_workflow_names(args.workflows)
    rows: list[dict[str, Any]] = []
    for workflow_name in workflow_names:
        graph = WORKFLOW_TEMPLATES[workflow_name]()
        rows.append(_broadcast_row(workflow_name, graph))
        rows.extend(
            _evaluate_policy(
                workflow_name,
                graph,
                policy_name,
                selector(graph),
                trials=args.trials,
                seed=args.seed,
            )
            for policy_name, selector in _classical_selectors(
                budget=args.budget,
                trials=args.trials,
                seed=args.seed,
            ).items()
        )
        learned_node_score_maps = _learned_node_score_maps(
            workflow_name=workflow_name,
            budget=args.budget,
            trials=args.trials,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        rows.extend(
            _evaluate_policy(
                workflow_name,
                graph,
                policy_name,
                seeds,
                trials=args.trials,
                seed=args.seed,
            )
            for policy_name, seeds in _learned_seed_sets(
                learned_node_score_maps,
                budget=args.budget,
            ).items()
        )
        rows.extend(
            _evaluate_learned_scorer_policies(
                workflow_name=workflow_name,
                graph=graph,
                node_score_maps=learned_node_score_maps,
                budget=args.budget,
                trials=args.trials,
                max_steps=args.max_steps,
            )
        )
        rows.extend(
            _evaluate_rl_policies(
                workflow_name,
                graph,
                budget=args.budget,
                trials=args.trials,
                episodes=args.episodes,
                learning_rate=args.learning_rate,
                seed=args.seed,
                max_steps=args.max_steps,
            )
        )

    payload = {
        "budget": args.budget,
        "trials": args.trials,
        "episodes": args.episodes,
        "epochs": args.epochs,
        "workflows": workflow_names,
        "summary": _summarize(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


def _parse_workflow_names(raw_names: str) -> list[str]:
    names = [name.strip() for name in raw_names.split(",") if name.strip()]
    unknown = sorted(name for name in names if name not in WORKFLOW_TEMPLATES)
    if unknown:
        raise ValueError(f"Unknown workflow templates: {', '.join(unknown)}")
    return names


def _classical_selectors(
    *,
    budget: int,
    trials: int,
    seed: int,
) -> dict[str, SeedSelector]:
    return {
        "random": lambda graph: random_seed_selection(graph, budget, seed=seed),
        "degree": lambda graph: degree_seed_selection(graph, budget),
        "pagerank": lambda graph: pagerank_seed_selection(graph, budget),
        "betweenness": lambda graph: betweenness_seed_selection(graph, budget),
        "closeness": lambda graph: closeness_seed_selection(graph, budget),
        "k_core": lambda graph: k_core_seed_selection(graph, budget),
        "greedy": lambda graph: greedy_seed_selection(
            graph,
            budget,
            propagation_model=IndependentCascade(seed=seed),
            trials=trials,
        ),
        "celf": lambda graph: celf_seed_selection(
            graph,
            budget,
            propagation_model=IndependentCascade(seed=seed),
            trials=trials,
        ),
    }


def _learned_seed_sets(
    node_score_maps: dict[str, dict[str, float]],
    *,
    budget: int,
) -> dict[str, list[str]]:
    return {
        policy_name: _top_k(node_scores, budget)
        for policy_name, node_scores in node_score_maps.items()
    }


def _learned_node_score_maps(
    *,
    workflow_name: str,
    budget: int,
    trials: int,
    epochs: int,
    learning_rate: float,
) -> dict[str, dict[str, float]]:
    examples: list[NodeTrainingExample] = [
        build_seed_selection_example(builder(), budget=budget, trials=trials)
        for name, builder in WORKFLOW_TEMPLATES.items()
        if name != workflow_name
    ]
    ranking_examples = [
        build_seed_ranking_example(builder(), budget=budget, trials=trials)
        for name, builder in WORKFLOW_TEMPLATES.items()
        if name != workflow_name
    ]
    if not examples:
        return {}

    graph = WORKFLOW_TEMPLATES[workflow_name]()
    features = extract_graph_features(graph)
    feature_count = len(examples[0].features.feature_names)

    mlp = MLPNodeScorer.initialize(feature_count)
    mlp.train(examples, epochs=epochs, learning_rate=learning_rate)

    linear = LinearNodeScorer.initialize(feature_count)
    linear.train(examples, epochs=epochs, learning_rate=learning_rate)
    message_passing = MessagePassingNodeScorer(linear)
    ranker = PairwiseNodeRanker.initialize(feature_count)
    ranker.train(ranking_examples, epochs=epochs, learning_rate=learning_rate)
    regressor = LinearNodeRegressor.initialize(feature_count)
    regressor.train(ranking_examples, epochs=epochs, learning_rate=learning_rate)

    neighbors = {
        node_id: sorted({*graph.predecessors(node_id), *graph.successors(node_id)})
        for node_id in features.node_features
    }
    return {
        "mlp": _seed_eligible_scores(graph, mlp.score_nodes(features)),
        "message_passing_gnn": _seed_eligible_scores(
            graph,
            message_passing.score_nodes(features, neighbors),
        ),
        "pairwise_ranker": _seed_eligible_scores(graph, ranker.score_nodes(features)),
        "marginal_gain_regressor": _seed_eligible_scores(graph, regressor.score_nodes(features)),
    }


def _evaluate_learned_scorer_policies(
    *,
    workflow_name: str,
    graph: AgentGraph,
    node_score_maps: dict[str, dict[str, float]],
    budget: int,
    trials: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    rows = []
    for policy_name, node_scores in node_score_maps.items():
        env = AgentRoutingEnv(graph, budget=budget, trials=trials)
        policy = NodeScorerRoutingPolicy(node_scores)
        state, actions, reward_trace = _rollout_routing_policy(
            env,
            policy.act,
            max_steps=max_steps,
        )
        rows.append(
            _evaluate_rl_state(
                workflow_name,
                graph,
                f"{policy_name}_routing_policy",
                state,
                actions,
                reward_trace,
            )
        )
    return rows


def _evaluate_rl_policies(
    workflow_name: str,
    graph: AgentGraph,
    *,
    budget: int,
    trials: int,
    episodes: int,
    learning_rate: float,
    seed: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    rows = []
    for expanded_actions in (False, True):
        suffix = "_expanded" if expanded_actions else ""
        rows.extend(
            _evaluate_rl_policy_family(
                workflow_name,
                graph,
                suffix=suffix,
                budget=budget,
                trials=trials,
                episodes=episodes,
                learning_rate=learning_rate,
                seed=seed,
                max_steps=max_steps,
                expanded_actions=expanded_actions,
            )
        )
    return rows


def _evaluate_rl_policy_family(
    workflow_name: str,
    graph: AgentGraph,
    *,
    suffix: str,
    budget: int,
    trials: int,
    episodes: int,
    learning_rate: float,
    seed: int,
    max_steps: int,
    expanded_actions: bool,
) -> list[dict[str, Any]]:
    q_env = AgentRoutingEnv(graph, budget=budget, trials=trials)
    q_policy, _ = train_q_policy(
        q_env,
        config=QLearningConfig(
            episodes=episodes,
            learning_rate=learning_rate,
            epsilon=0.2,
            seed=seed,
            expanded_actions=expanded_actions,
        ),
    )
    reinforce_env = AgentRoutingEnv(graph, budget=budget, trials=trials)
    reinforce_policy, _ = train_reinforce_policy(
        reinforce_env,
        config=ReinforceConfig(
            episodes=episodes,
            learning_rate=learning_rate,
            seed=seed,
            max_steps=max_steps,
            expanded_actions=expanded_actions,
        ),
    )
    ppo_env = AgentRoutingEnv(graph, budget=budget, trials=trials)
    ppo_policy, _ = train_ppo_policy(
        ppo_env,
        config=PPOConfig(
            episodes=episodes,
            learning_rate=learning_rate,
            seed=seed,
            max_steps=max_steps,
            expanded_actions=expanded_actions,
        ),
    )
    policy_rows: list[tuple[str, AgentRoutingEnv, Callable[[AgentRoutingEnv], str]]] = [
        (f"q_learning{suffix}", q_env, q_policy.act),
        (f"reinforce{suffix}", reinforce_env, reinforce_policy.act),
        (f"ppo{suffix}", ppo_env, ppo_policy.act),
    ]
    if not expanded_actions:
        feature_env = AgentRoutingEnv(graph, budget=budget, trials=trials)
        feature_policy, _ = train_feature_policy(
            feature_env,
            config=FeaturePolicyConfig(
                episodes=episodes,
                learning_rate=learning_rate,
                epsilon=0.2,
                seed=seed,
                max_steps=max_steps,
            ),
        )
        policy_rows.append(("feature_policy", feature_env, feature_policy.act))

    rows = []
    for policy_name, env, action_fn in policy_rows:
        state, actions, reward_trace = _rollout_routing_policy(
            env,
            action_fn,
            max_steps=max_steps,
        )
        rows.append(
            _evaluate_rl_state(
                workflow_name,
                graph,
                policy_name,
                state,
                actions,
                reward_trace,
            )
        )
    return rows


def _rollout_routing_policy(
    env: AgentRoutingEnv,
    action_fn: Callable[[AgentRoutingEnv], str],
    *,
    max_steps: int,
) -> tuple[RoutingState, list[str], list[dict[str, Any]]]:
    state = env.reset()
    actions = []
    reward_trace = []
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = action_fn(env)
        actions.append(action)
        state, reward, done, info = env.step(action)
        reward_trace.append(
            {
                "action": action,
                "reward": reward,
                "propagation_reward": info.get("propagation_reward", 0.0),
                "control_reward": info.get("control_reward", {}),
            }
        )
        steps += 1
        if action == RoutingAction.STOP.value:
            break
    return state, actions, reward_trace


def _evaluate_rl_state(
    workflow_name: str,
    graph: AgentGraph,
    policy_name: str,
    state: RoutingState,
    actions: list[str],
    reward_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    broadcast = broadcast_cost(graph)
    cost = CostSummary(
        token_cost=state.token_cost,
        message_cost=state.message_cost,
        latency=state.propagation_time,
        message_count=len(state.used_edges),
    )
    quality = quality_cost_summary(
        success_rate=state.coverage,
        token_cost=cost.total_cost,
        latency=cost.latency,
    )
    row = _row(
        workflow_name=workflow_name,
        policy_name=policy_name,
        seeds=list(state.selected_seeds),
        coverage=state.coverage,
        full_activation_probability=None,
        propagation_time=state.propagation_time,
        cost=cost,
        broadcast=broadcast,
        efficiency_score=quality.efficiency_score,
        cost_adjusted_success=quality.cost_adjusted_success,
    )
    row.update(
        {
            "actions": actions,
            "reward_trace": reward_trace,
            "activated_verifiers": list(state.activated_verifiers),
            "used_edges": [list(edge) for edge in state.used_edges],
            "pruned_edges": [list(edge) for edge in state.pruned_edges],
            "called_tools": list(state.called_tools),
            "summary_nodes": list(state.summary_nodes),
        }
    )
    return row


def _evaluate_policy(
    workflow_name: str,
    graph: AgentGraph,
    policy_name: str,
    seeds: list[str],
    *,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    model = IndependentCascade(seed=seed)
    propagation = model.simulate(graph, seeds, trials=trials)
    cost = seeded_routing_cost(graph, seeds, propagation.activated_nodes)
    broadcast = broadcast_cost(graph)
    propagation_time = propagation.expected_propagation_time or propagation.propagation_time
    quality = quality_cost_summary(
        success_rate=propagation.coverage,
        token_cost=cost.total_cost,
        latency=cost.latency,
    )
    return _row(
        workflow_name=workflow_name,
        policy_name=policy_name,
        seeds=seeds,
        coverage=propagation.coverage,
        full_activation_probability=propagation.full_activation_probability,
        propagation_time=float(propagation_time),
        cost=cost,
        broadcast=broadcast,
        efficiency_score=quality.efficiency_score,
        cost_adjusted_success=quality.cost_adjusted_success,
    )


def _broadcast_row(workflow_name: str, graph: AgentGraph) -> dict[str, Any]:
    cost = broadcast_cost(graph)
    quality = quality_cost_summary(
        success_rate=1.0,
        token_cost=cost.total_cost,
        latency=cost.latency,
    )
    return _row(
        workflow_name=workflow_name,
        policy_name="broadcast",
        seeds=[node.id for node in graph.nodes()],
        coverage=1.0,
        full_activation_probability=1.0,
        propagation_time=0.0,
        cost=cost,
        broadcast=cost,
        efficiency_score=quality.efficiency_score,
        cost_adjusted_success=quality.cost_adjusted_success,
    )


def _row(
    *,
    workflow_name: str,
    policy_name: str,
    seeds: list[str],
    coverage: float,
    full_activation_probability: float | None,
    propagation_time: float,
    cost: CostSummary,
    broadcast: CostSummary,
    efficiency_score: float,
    cost_adjusted_success: float,
) -> dict[str, Any]:
    estimated_savings = 0.0
    if broadcast.total_cost > 0:
        estimated_savings = (broadcast.total_cost - cost.total_cost) / broadcast.total_cost
    return {
        "workflow": workflow_name,
        "policy": policy_name,
        "seeds": seeds,
        "coverage": coverage,
        "full_activation_probability": full_activation_probability,
        "propagation_time": propagation_time,
        "token_cost": cost.token_cost,
        "message_cost": cost.message_cost,
        "total_cost": cost.total_cost,
        "latency": cost.latency,
        "message_count": cost.message_count,
        "estimated_savings": estimated_savings,
        "cost_adjusted_success": cost_adjusted_success,
        "efficiency_score": efficiency_score,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    policies = sorted({str(row["policy"]) for row in rows})
    summary = {}
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            "workflows": float(len(policy_rows)),
            "mean_coverage": _mean([float(row["coverage"]) for row in policy_rows]),
            "mean_total_cost": _mean([float(row["total_cost"]) for row in policy_rows]),
            "mean_estimated_savings": _mean(
                [float(row["estimated_savings"]) for row in policy_rows]
            ),
            "mean_efficiency_score": _mean(
                [float(row["efficiency_score"]) for row in policy_rows]
            ),
        }
    return summary


def _top_k(scores: dict[str, float], budget: int) -> list[str]:
    return [
        node_id
        for node_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:budget]
    ]


def _seed_eligible_scores(graph: AgentGraph, scores: dict[str, float]) -> dict[str, float]:
    eligible = {node.id for node in graph.nodes() if node.type != NodeType.OUTPUT}
    return {node_id: score for node_id, score in scores.items() if node_id in eligible}


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
