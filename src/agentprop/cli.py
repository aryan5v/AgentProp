"""Command-line interface for AgentProp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentprop.algorithms import (
    bottleneck_nodes,
    high_cost_low_relevance_edges,
    low_weight_edges,
    risk_aware_verifier_placement,
)
from agentprop.core import AgentGraph
from agentprop.evaluation import compare_routing, evaluate_pruning
from agentprop.evaluation.reporting import report_to_dict, write_report
from agentprop.evaluation.runner import make_propagation_model, run_benchmark, select_seeds
from agentprop.integrations import graph_from_trace
from agentprop.visualization import write_dot
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    """Run the AgentProp CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "optimize":
        return _optimize(args)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "report":
        return _report(args)
    if args.command == "simulate":
        return _simulate(args)
    if args.command == "prune":
        return _prune(args)
    if args.command == "trace":
        return _trace(args)
    if args.command == "viz":
        return _viz(args)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentprop")
    subparsers = parser.add_subparsers(dest="command")
    algorithm_choices = [
        "random",
        "degree",
        "in-degree",
        "out-degree",
        "pagerank",
        "betweenness",
        "closeness",
        "k-core",
        "greedy",
        "celf",
        "cost-aware-greedy",
    ]

    optimize = subparsers.add_parser("optimize", help="recommend seed nodes for a workflow graph")
    optimize.add_argument("workflow", type=Path)
    optimize.add_argument("--budget", "-k", type=int, default=2)
    optimize.add_argument(
        "--algorithm",
        choices=algorithm_choices,
        default="greedy",
    )
    optimize.add_argument(
        "--model",
        choices=[
            "independent-cascade",
            "linear-threshold",
            "bootstrap",
            "rzf",
            "zero-forcing",
            "learned",
        ],
        default="independent-cascade",
    )
    optimize.add_argument("--trials", type=int, default=100)
    optimize.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    analyze = subparsers.add_parser("analyze", help="show graph diagnostics")
    analyze.add_argument("workflow", type=Path)
    analyze.add_argument("--json", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="compare algorithms and propagation models")
    benchmark.add_argument("workflow", help="workflow JSON path or built-in workflow name")
    benchmark.add_argument("--budget", "-k", type=int, default=2)
    benchmark.add_argument("--trials", type=int, default=100)
    benchmark.add_argument(
        "--algorithms",
        nargs="+",
        default=["random", "degree", "pagerank", "greedy"],
        choices=algorithm_choices,
    )
    benchmark.add_argument(
        "--models",
        nargs="+",
        default=["independent-cascade", "rzf"],
        choices=[
            "independent-cascade",
            "linear-threshold",
            "bootstrap",
            "rzf",
            "zero-forcing",
            "learned",
        ],
    )
    benchmark.add_argument("--json", action="store_true")

    report = subparsers.add_parser("report", help="write a Markdown, JSON, or HTML report")
    report.add_argument("workflow", help="workflow JSON path or built-in workflow name")
    report.add_argument("--budget", "-k", type=int, default=2)
    report.add_argument(
        "--algorithm",
        choices=algorithm_choices,
        default="greedy",
    )
    report.add_argument(
        "--model",
        choices=[
            "independent-cascade",
            "linear-threshold",
            "bootstrap",
            "rzf",
            "zero-forcing",
            "learned",
        ],
        default="independent-cascade",
    )
    report.add_argument("--trials", type=int, default=100)
    report.add_argument("--out", type=Path, default=Path("reports/agentprop_report.md"))
    report.add_argument(
        "--format",
        dest="report_format",
        choices=["auto", "markdown", "json", "html"],
        default="auto",
        help="report format; auto infers from --out extension",
    )

    simulate = subparsers.add_parser("simulate", help="simulate propagation from seed nodes")
    simulate.add_argument("workflow", help="workflow JSON path or built-in workflow name")
    simulate.add_argument("--seeds", nargs="+", required=True)
    simulate.add_argument(
        "--model",
        choices=[
            "independent-cascade",
            "linear-threshold",
            "bootstrap",
            "rzf",
            "zero-forcing",
            "learned",
        ],
        default="independent-cascade",
    )
    simulate.add_argument("--trials", type=int, default=100)
    simulate.add_argument("--json", action="store_true")

    prune = subparsers.add_parser("prune", help="recommend and evaluate edge pruning")
    prune.add_argument("workflow", help="workflow JSON path or built-in workflow name")
    prune.add_argument("--target-token-reduction", type=float, default=0.3)
    prune.add_argument(
        "--strategy",
        choices=["low-weight", "high-cost-low-relevance"],
        default="low-weight",
    )
    prune.add_argument("--budget", "-k", type=int, default=2)
    prune.add_argument(
        "--model",
        choices=[
            "independent-cascade",
            "linear-threshold",
            "bootstrap",
            "rzf",
            "zero-forcing",
            "learned",
        ],
        default="independent-cascade",
    )
    prune.add_argument("--trials", type=int, default=100)
    prune.add_argument("--json", action="store_true")

    trace = subparsers.add_parser("trace", help="convert a trace JSON file into workflow JSON")
    trace.add_argument("trace_file", type=Path)
    trace.add_argument("--out", type=Path, default=Path("results/trace_workflow.json"))

    viz = subparsers.add_parser("viz", help="export a workflow graph as Graphviz DOT")
    viz.add_argument("workflow", help="workflow JSON path or built-in workflow name")
    viz.add_argument("--out", type=Path, default=Path("reports/workflow.dot"))

    return parser


def _optimize(args: argparse.Namespace) -> int:
    _, graph = _load_workflow(str(args.workflow))
    report = _build_recommendation_report(
        graph,
        algorithm=args.algorithm,
        model_name=args.model,
        budget=args.budget,
        trials=args.trials,
    )

    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    workflow_name, graph = _load_workflow(args.workflow)
    rows = run_benchmark(
        graph,
        workflow_name=workflow_name,
        algorithms=args.algorithms,
        models=args.models,
        budget=args.budget,
        trials=args.trials,
    )
    if args.json:
        print(json.dumps([row.to_dict() for row in rows], indent=2, sort_keys=True))
        return 0

    print("workflow,algorithm,model,seeds,coverage,expected_time,full_activation,savings")
    for row in rows:
        print(
            f"{row.workflow},{row.algorithm},{row.propagation_model},"
            f"{'|'.join(row.seeds)},{row.coverage:.3f},"
            f"{row.expected_propagation_time:.3f},"
            f"{row.full_activation_probability:.3f},{row.estimated_savings:.3f}"
        )
    return 0


def _report(args: argparse.Namespace) -> int:
    workflow_name, graph = _load_workflow(args.workflow)
    report = _build_recommendation_report(
        graph,
        algorithm=args.algorithm,
        model_name=args.model,
        budget=args.budget,
        trials=args.trials,
    )
    output_path = write_report(
        report,
        args.out,
        workflow_name=workflow_name,
        report_format=args.report_format,
    )
    print(f"Wrote {output_path}")
    return 0


def _simulate(args: argparse.Namespace) -> int:
    workflow_name, graph = _load_workflow(args.workflow)
    model = make_propagation_model(args.model)
    result = model.simulate(graph, args.seeds, trials=args.trials)
    payload = {
        "workflow": workflow_name,
        "model": model.name,
        "seeds": args.seeds,
        "activated_nodes": sorted(result.activated_nodes),
        "coverage": result.coverage,
        "propagation_time": result.propagation_time,
        "expected_propagation_time": result.expected_propagation_time,
        "full_activation_probability": result.full_activation_probability,
        "coverage_by_round": result.coverage_by_round,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Workflow: {workflow_name}")
        print(f"Model: {model.name}")
        print(f"Seeds: {', '.join(args.seeds)}")
        print(f"Coverage: {result.coverage:.1%}")
        print(f"Expected propagation time: {result.expected_propagation_time}")
        print(f"Full activation probability: {result.full_activation_probability}")
        print(f"Activated nodes: {', '.join(sorted(result.activated_nodes))}")
    return 0


def _prune(args: argparse.Namespace) -> int:
    workflow_name, graph = _load_workflow(args.workflow)
    if not 0 <= args.target_token_reduction <= 1:
        raise ValueError("--target-token-reduction must be between 0 and 1")
    model = make_propagation_model(args.model)
    seeds = select_seeds(graph, "cost-aware-greedy", args.budget, model, args.trials)
    candidate_edges = _rank_pruning_edges(graph, args.strategy)
    selected_edges = _edges_to_reduction_target(
        graph,
        candidate_edges,
        target_reduction=args.target_token_reduction,
    )
    evaluation = evaluate_pruning(
        graph,
        selected_edges,
        seeds=seeds,
        propagation_model=model,
        trials=args.trials,
    )
    payload = {
        "workflow": workflow_name,
        "strategy": args.strategy,
        "target_token_reduction": args.target_token_reduction,
        "seeds": seeds,
        "removed_edges": [
            {"source": source, "target": target} for source, target in evaluation.removed_edges
        ],
        "baseline_coverage": evaluation.baseline_coverage,
        "pruned_coverage": evaluation.pruned_coverage,
        "coverage_delta": evaluation.coverage_delta,
        "baseline_cost": evaluation.baseline_cost,
        "pruned_cost": evaluation.pruned_cost,
        "cost_delta": evaluation.cost_delta,
        "achieved_cost_reduction": (
            0.0
            if evaluation.baseline_cost == 0
            else (evaluation.baseline_cost - evaluation.pruned_cost) / evaluation.baseline_cost
        ),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Workflow: {workflow_name}")
        print(f"Strategy: {args.strategy}")
        print(f"Seeds used for evaluation: {', '.join(seeds)}")
        print("Removed edges:")
        for source, target in evaluation.removed_edges:
            print(f"  - {source} -> {target}")
        print(f"Coverage delta: {evaluation.coverage_delta:.3f}")
        print(f"Cost delta: {evaluation.cost_delta:.0f}")
        print(f"Achieved cost reduction: {payload['achieved_cost_reduction']:.1%}")
    return 0


def _trace(args: argparse.Namespace) -> int:
    result = graph_from_trace(args.trace_file)
    result.graph.to_json(args.out)
    print(
        f"Wrote {args.out} "
        f"from {result.message_count} messages "
        f"({result.total_token_cost:.0f} tokens)"
    )
    return 0


def _viz(args: argparse.Namespace) -> int:
    workflow_name, graph = _load_workflow(args.workflow)
    output_path = write_dot(graph, args.out, name=workflow_name)
    print(f"Wrote {output_path}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    graph = AgentGraph.from_json(args.workflow)
    bottlenecks = bottleneck_nodes(graph)
    pruning_candidates = low_weight_edges(graph)
    verifier_candidates = risk_aware_verifier_placement(graph, min(3, graph.node_count))
    payload = {
        "nodes": graph.node_count,
        "edges": graph.edge_count,
        "bottlenecks": bottlenecks,
        "pruning_candidates": pruning_candidates,
        "verifier_candidates": verifier_candidates,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Nodes: {payload['nodes']}")
        print(f"Edges: {payload['edges']}")
        print("Bottlenecks:")
        for node_id, score in bottlenecks:
            print(f"  - {node_id}: {score:.3f}")
        print("Pruning candidates:")
        for source, target in pruning_candidates:
            print(f"  - {source} -> {target}")
        print("Verifier candidates:")
        for node_id in verifier_candidates:
            print(f"  - {node_id}")
    return 0


def _load_workflow(workflow: str) -> tuple[str, AgentGraph]:
    if workflow in WORKFLOW_TEMPLATES:
        return workflow, WORKFLOW_TEMPLATES[workflow]()

    path = Path(workflow)
    return path.stem, AgentGraph.from_json(path)


def _build_recommendation_report(
    graph: AgentGraph,
    *,
    algorithm: str,
    model_name: str,
    budget: int,
    trials: int,
) -> Any:
    model = make_propagation_model(model_name)
    seeds = select_seeds(graph, algorithm, budget, model, trials)
    propagation = model.simulate(graph, seeds, trials=trials)
    return compare_routing(
        graph,
        seeds,
        model.name,
        propagation,
        bottlenecks=bottleneck_nodes(graph),
        pruning_candidates=low_weight_edges(graph),
        verifier_candidates=risk_aware_verifier_placement(graph, min(budget, graph.node_count)),
    )


def _rank_pruning_edges(graph: AgentGraph, strategy: str) -> list[tuple[str, str]]:
    if strategy == "low-weight":
        return low_weight_edges(graph, fraction=1.0)
    if strategy == "high-cost-low-relevance":
        return high_cost_low_relevance_edges(graph, limit=graph.edge_count)
    raise ValueError(f"Unknown pruning strategy: {strategy}")


def _edges_to_reduction_target(
    graph: AgentGraph,
    ranked_edges: list[tuple[str, str]],
    *,
    target_reduction: float,
) -> list[tuple[str, str]]:
    baseline_cost = sum(edge.message_cost for edge in graph.edges())
    if baseline_cost == 0:
        return []
    selected: list[tuple[str, str]] = []
    removed_cost = 0.0
    for source, target in ranked_edges:
        selected.append((source, target))
        removed_cost += graph.edge(source, target).message_cost
        if removed_cost / baseline_cost >= target_reduction:
            break
    return selected


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


if __name__ == "__main__":
    raise SystemExit(main())
