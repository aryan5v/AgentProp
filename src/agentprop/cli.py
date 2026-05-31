"""Command-line interface for AgentProp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentprop.algorithms import bottleneck_nodes, low_weight_edges, risk_aware_verifier_placement
from agentprop.core import AgentGraph
from agentprop.evaluation import compare_routing
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
        choices=["independent-cascade", "linear-threshold", "bootstrap", "rzf", "zero-forcing"],
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
        choices=["independent-cascade", "linear-threshold", "bootstrap", "rzf", "zero-forcing"],
    )
    benchmark.add_argument("--json", action="store_true")

    report = subparsers.add_parser("report", help="write a markdown or JSON optimization report")
    report.add_argument("workflow", help="workflow JSON path or built-in workflow name")
    report.add_argument("--budget", "-k", type=int, default=2)
    report.add_argument(
        "--algorithm",
        choices=algorithm_choices,
        default="greedy",
    )
    report.add_argument(
        "--model",
        choices=["independent-cascade", "linear-threshold", "bootstrap", "rzf", "zero-forcing"],
        default="independent-cascade",
    )
    report.add_argument("--trials", type=int, default=100)
    report.add_argument("--out", type=Path, default=Path("reports/agentprop_report.md"))

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
    output_path = write_report(report, args.out, workflow_name=workflow_name)
    print(f"Wrote {output_path}")
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
