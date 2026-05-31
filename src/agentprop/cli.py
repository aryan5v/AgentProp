"""Command-line interface for AgentProp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentprop.algorithms import (
    betweenness_seed_selection,
    bottleneck_nodes,
    degree_seed_selection,
    greedy_seed_selection,
    low_weight_edges,
    pagerank_seed_selection,
    random_seed_selection,
    risk_aware_verifier_placement,
)
from agentprop.core import AgentGraph
from agentprop.evaluation import compare_routing
from agentprop.propagation import (
    BootstrapPercolation,
    IndependentCascade,
    LinearThreshold,
    RandomizedZeroForcing,
)


def main(argv: list[str] | None = None) -> int:
    """Run the AgentProp CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "optimize":
        return _optimize(args)
    if args.command == "analyze":
        return _analyze(args)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentprop")
    subparsers = parser.add_subparsers(dest="command")

    optimize = subparsers.add_parser("optimize", help="recommend seed nodes for a workflow graph")
    optimize.add_argument("workflow", type=Path)
    optimize.add_argument("--budget", "-k", type=int, default=2)
    optimize.add_argument(
        "--algorithm",
        choices=["random", "degree", "pagerank", "betweenness", "greedy"],
        default="greedy",
    )
    optimize.add_argument(
        "--model",
        choices=["independent-cascade", "linear-threshold", "bootstrap", "rzf"],
        default="independent-cascade",
    )
    optimize.add_argument("--trials", type=int, default=100)
    optimize.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    analyze = subparsers.add_parser("analyze", help="show graph diagnostics")
    analyze.add_argument("workflow", type=Path)
    analyze.add_argument("--json", action="store_true")

    return parser


def _optimize(args: argparse.Namespace) -> int:
    graph = AgentGraph.from_json(args.workflow)
    model = _propagation_model(args.model)
    seeds = _select_seeds(graph, args.algorithm, args.budget, model, args.trials)
    propagation = model.simulate(graph, seeds, trials=args.trials)
    report = compare_routing(
        graph,
        seeds,
        model.name,
        propagation,
        bottlenecks=bottleneck_nodes(graph),
        pruning_candidates=low_weight_edges(graph),
        verifier_candidates=risk_aware_verifier_placement(
            graph,
            min(args.budget, graph.node_count),
        ),
    )

    if args.json:
        print(json.dumps(_report_to_dict(report), indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0


def _analyze(args: argparse.Namespace) -> int:
    graph = AgentGraph.from_json(args.workflow)
    payload = {
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "bottlenecks": bottleneck_nodes(graph),
        "pruning_candidates": low_weight_edges(graph),
        "verifier_candidates": risk_aware_verifier_placement(graph, min(3, graph.node_count)),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Nodes: {payload['nodes']}")
        print(f"Edges: {payload['edges']}")
        print("Bottlenecks:")
        for node_id, score in payload["bottlenecks"]:
            print(f"  - {node_id}: {score:.3f}")
        print("Pruning candidates:")
        for source, target in payload["pruning_candidates"]:
            print(f"  - {source} -> {target}")
        print("Verifier candidates:")
        for node_id in payload["verifier_candidates"]:
            print(f"  - {node_id}")
    return 0


def _propagation_model(name: str) -> Any:
    if name == "independent-cascade":
        return IndependentCascade(seed=0)
    if name == "linear-threshold":
        return LinearThreshold()
    if name == "bootstrap":
        return BootstrapPercolation()
    if name == "rzf":
        return RandomizedZeroForcing(seed=0)
    raise ValueError(f"Unknown propagation model: {name}")


def _select_seeds(
    graph: AgentGraph,
    algorithm: str,
    budget: int,
    model: Any,
    trials: int,
) -> list[str]:
    if algorithm == "random":
        return random_seed_selection(graph, budget, seed=0)
    if algorithm == "degree":
        return degree_seed_selection(graph, budget)
    if algorithm == "pagerank":
        return pagerank_seed_selection(graph, budget)
    if algorithm == "betweenness":
        return betweenness_seed_selection(graph, budget)
    if algorithm == "greedy":
        return greedy_seed_selection(graph, budget, propagation_model=model, trials=trials)
    raise ValueError(f"Unknown seed algorithm: {algorithm}")


def _print_report(report: Any) -> None:
    print("AgentProp Optimization Report")
    print("=============================")
    print(f"Propagation model: {report.propagation_model}")
    print(f"Recommended seeds: {', '.join(report.seeds)}")
    print(f"Coverage: {report.propagation.coverage:.1%}")
    print(f"Expected propagation time: {report.propagation.expected_propagation_time:.2f}")
    print(f"Full activation probability: {report.propagation.full_activation_probability:.1%}")
    print("")
    print("Cost")
    print("----")
    print(f"Broadcast total: {report.broadcast_cost.total_cost:.0f}")
    print(f"Optimized total: {report.optimized_cost.total_cost:.0f}")
    print(f"Estimated savings: {report.estimated_savings:.1%}")
    print("")
    print("Bottlenecks")
    print("-----------")
    for node_id, score in report.bottlenecks:
        print(f"- {node_id}: {score:.3f}")
    print("")
    print("Pruning candidates")
    print("------------------")
    for source, target in report.pruning_candidates:
        print(f"- {source} -> {target}")
    print("")
    print("Verifier candidates")
    print("-------------------")
    for node_id in report.verifier_candidates:
        print(f"- {node_id}")


def _report_to_dict(report: Any) -> dict[str, Any]:
    return {
        "seeds": report.seeds,
        "propagation_model": report.propagation_model,
        "coverage": report.propagation.coverage,
        "expected_propagation_time": report.propagation.expected_propagation_time,
        "full_activation_probability": report.propagation.full_activation_probability,
        "activated_nodes": sorted(report.propagation.activated_nodes),
        "broadcast_cost": {
            "token_cost": report.broadcast_cost.token_cost,
            "message_cost": report.broadcast_cost.message_cost,
            "total_cost": report.broadcast_cost.total_cost,
            "latency": report.broadcast_cost.latency,
            "message_count": report.broadcast_cost.message_count,
        },
        "optimized_cost": {
            "token_cost": report.optimized_cost.token_cost,
            "message_cost": report.optimized_cost.message_cost,
            "total_cost": report.optimized_cost.total_cost,
            "latency": report.optimized_cost.latency,
            "message_count": report.optimized_cost.message_count,
        },
        "estimated_savings": report.estimated_savings,
        "bottlenecks": report.bottlenecks,
        "pruning_candidates": report.pruning_candidates,
        "verifier_candidates": report.verifier_candidates,
    }


if __name__ == "__main__":
    raise SystemExit(main())
