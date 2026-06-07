"""Command-line interface for AgentProp."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from agentprop.algorithms import (
    bottleneck_nodes,
    high_cost_low_relevance_edges,
    low_weight_edges,
    risk_aware_verifier_placement,
)
from agentprop.core import AgentGraph
from agentprop.core.validation import WorkflowValidationError
from agentprop.evaluation import compare_routing, evaluate_pruning, summarize_pruning_risk
from agentprop.evaluation.constants import (
    PROPAGATION_MODEL_CHOICES,
    SEED_ALGORITHM_CHOICES,
)
from agentprop.evaluation.readiness import (
    build_v1_readiness_report,
    render_v1_readiness_markdown,
)
from agentprop.evaluation.reporting import report_to_dict, write_report
from agentprop.evaluation.runner import make_propagation_model, run_benchmark, select_seeds
from agentprop.evaluation.terminal_bench import (
    HarborWatchdogConfig,
    TerminalBenchLaunchConfig,
    write_terminal_bench_launch_bundle,
    write_terminal_bench_summary_report,
)
from agentprop.integrations import (
    aggregate_session_stats,
    graph_from_trace,
    render_coding_agent_instructions,
    render_session_stats_markdown,
)
from agentprop.runtime.demos import CONTROL_DEMOS, run_control_demo
from agentprop.visualization import write_dot
from agentprop.workflows import WORKFLOW_DESCRIPTIONS, WORKFLOW_TEMPLATES
from agentprop.workflows.scaffolder import TOPOLOGIES, scaffold_workflow

_CLI_EPILOG = """
examples:
  agentprop analyze planner_coder_tester_reviewer --json
  agentprop optimize planner_coder_tester_reviewer --budget 2
  agentprop control-demo --demo terminal --out-dir reports/control-demo

See docs/index.md for the full tutorial.
"""


def main(argv: list[str] | None = None) -> int:
    """Run the AgentProp CLI."""

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if getattr(args, "version", False):
            try:
                print(importlib.metadata.version("agentprop"))
            except importlib.metadata.PackageNotFoundError:
                print("unknown (package not installed)")
            return 0
        return _dispatch(args, parser)
    except WorkflowValidationError as error:
        print("Workflow validation failed:", file=sys.stderr)
        for issue in error.issues:
            print(f"  {issue.format()}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.command == "optimize":
        return _optimize(args)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "run-evidence":
        return _run_evidence(args)
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
    if args.command == "agent-instructions":
        return _agent_instructions(args)
    if args.command == "readiness":
        return _readiness(args)
    if args.command == "terminal-bench":
        return _terminal_bench(args)
    if args.command == "control-demo":
        return _control_demo(args)
    if args.command == "workflows":
        return _workflows(args)
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "ingest-trace":
        return _ingest_trace(args)
    if args.command == "trace-replay":
        return _trace_replay(args)
    if args.command == "init":
        return _init(args)
    if args.command == "sessions":
        return _sessions(args)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentprop",
        description="Graph control for AI-agent workflows.",
        epilog=_CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print package version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")
    algorithm_choices = ["auto", *SEED_ALGORITHM_CHOICES]
    model_choices = list(PROPAGATION_MODEL_CHOICES)

    optimize = subparsers.add_parser("optimize", help="recommend seed nodes for a workflow graph")
    optimize.add_argument("workflow", type=Path)
    optimize.add_argument("--budget", "-k", type=int, default=2)
    optimize.add_argument(
        "--algorithm",
        choices=algorithm_choices,
        default="auto",
        help=(
            "Seed algorithm. 'auto' uses greedy (n≤15), rzf-centrality (15<n≤60), "
            "or imm (n>60). "
            "and greedy for small graphs."
        ),
    )
    optimize.add_argument(
        "--model",
        choices=model_choices,
        default="independent-cascade",
    )
    optimize.add_argument("--trials", type=int, default=100)
    optimize.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    analyze = subparsers.add_parser("analyze", help="show graph diagnostics")
    analyze.add_argument("workflow", help="workflow JSON path or built-in workflow name")
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
        choices=model_choices,
    )
    benchmark.add_argument("--json", action="store_true")

    run_evidence = subparsers.add_parser(
        "run-evidence",
        help="run multi-workflow routing evidence matrix and write docs/results artifacts",
    )
    run_evidence.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/results/scale_quality_evidence"),
    )
    run_evidence.add_argument("--tasks-per-arm", type=int, default=30)
    run_evidence.add_argument("--repeats", type=int, default=3)
    run_evidence.add_argument("--seed-budget", type=int, default=3)
    run_evidence.add_argument("--trials", type=int, default=50)

    report = subparsers.add_parser("report", help="write a Markdown, JSON, or HTML report")
    report.add_argument("workflow", help="workflow JSON path or built-in workflow name")
    report.add_argument("--budget", "-k", type=int, default=2)
    report.add_argument(
        "--algorithm",
        choices=algorithm_choices,
        default="auto",
        help="Seed algorithm. 'auto' picks greedy/rzf-centrality/imm by graph size.",
    )
    report.add_argument(
        "--model",
        choices=model_choices,
        default="independent-cascade",
    )
    report.add_argument("--trials", type=int, default=100)
    report.add_argument("--out", type=Path, default=Path("reports/agentprop_report.md"))
    report.add_argument("--pruning-target-token-reduction", type=float, default=0.3)
    report.add_argument(
        "--pruning-strategy",
        choices=["low-weight", "high-cost-low-relevance"],
        default="low-weight",
    )
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
        choices=model_choices,
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
        choices=model_choices,
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

    instructions = subparsers.add_parser(
        "agent-instructions",
        help="write a Claude Code/Codex-ready workflow brief",
    )
    instructions.add_argument("workflow", help="workflow JSON path or built-in workflow name")
    instructions.add_argument("--budget", "-k", type=int, default=2)
    instructions.add_argument(
        "--algorithm",
        choices=algorithm_choices,
        default="auto",
        help="Seed algorithm. 'auto' picks greedy/rzf-centrality/imm by graph size.",
    )
    instructions.add_argument(
        "--model",
        choices=model_choices,
        default="independent-cascade",
    )
    instructions.add_argument("--trials", type=int, default=100)
    instructions.add_argument(
        "--target",
        choices=["claude-code", "codex", "generic"],
        default="generic",
    )
    instructions.add_argument(
        "--out",
        type=Path,
        default=Path("reports/agent_instructions.md"),
    )

    readiness = subparsers.add_parser(
        "readiness",
        help="show implementation maturity by component area",
    )
    readiness.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    readiness.add_argument(
        "--out",
        type=Path,
        help="write maturity report to a file (use docs/local/ for private notes)",
    )

    terminal_bench = subparsers.add_parser(
        "terminal-bench",
        help="prepare or summarize external Terminal-Bench runs",
    )
    terminal_bench_subparsers = terminal_bench.add_subparsers(dest="terminal_bench_command")
    terminal_bench_prepare = terminal_bench_subparsers.add_parser(
        "prepare",
        help="write a Terminal-Bench 2.1 + Terminus-2 launch bundle without running it",
    )
    terminal_bench_prepare.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark-results/terminal-bench-2.1"),
    )
    terminal_bench_prepare.add_argument("--dataset", default="terminal-bench/terminal-bench-2-1")
    terminal_bench_prepare.add_argument("--agent", default="terminus-2")
    terminal_bench_prepare.add_argument("--model", default="google/gemini-3.1-pro-preview")
    terminal_bench_prepare.add_argument("--environment", default="modal")
    terminal_bench_prepare.add_argument("--run-name", default="agentprop-tbench-21-terminus2")
    terminal_bench_prepare.add_argument("--task-count", type=int, default=None)
    terminal_bench_prepare.add_argument(
        "--output-root",
        default="benchmark-results/terminal-bench-2.1/terminus-2-agentprop",
    )
    terminal_bench_prepare.add_argument("--timeout", type=int, default=21_600)
    terminal_bench_prepare.add_argument("--idle-timeout", type=int, default=1_800)
    terminal_bench_prepare.add_argument("--registry-root", type=Path, default=None)
    terminal_bench_prepare.add_argument("--json", action="store_true")

    terminal_bench_summarize = terminal_bench_subparsers.add_parser(
        "summarize",
        help="summarize saved Harbor result.json artifacts",
    )
    terminal_bench_summarize.add_argument("--results-root", type=Path, required=True)
    terminal_bench_summarize.add_argument("--out-dir", type=Path, required=True)
    terminal_bench_summarize.add_argument(
        "--title",
        default="AgentProp Terminal-Bench Result Summary",
    )
    terminal_bench_summarize.add_argument("--registry-root", type=Path, default=None)
    terminal_bench_summarize.add_argument("--json", action="store_true")

    control_demo = subparsers.add_parser(
        "control-demo",
        help="run a key-free analysis + runtime-control demo",
    )
    control_demo.add_argument("--demo", choices=CONTROL_DEMOS, default="terminal")
    control_demo.add_argument("--out-dir", type=Path, default=Path("reports/control-demo"))
    control_demo.add_argument("--json", action="store_true")

    workflows = subparsers.add_parser(
        "workflows",
        help="list built-in workflow template names",
    )
    workflows_sub = workflows.add_subparsers(dest="workflows_command")
    workflows_list = workflows_sub.add_parser("list", help="print built-in workflow names")
    workflows_list.add_argument("--json", action="store_true")

    doctor = subparsers.add_parser(
        "doctor",
        help="check install, optional deps, and environment for a usage tier",
    )
    doctor.add_argument(
        "--tier",
        choices=["graph", "dev", "llm", "terminal-bench"],
        default="graph",
        help="usage tier to validate (default: graph — no API keys required)",
    )
    doctor.add_argument("--json", action="store_true")

    ingest = subparsers.add_parser(
        "ingest-trace",
        help="convert a trace to workflow JSON, optimize seeds, and write a control brief",
    )
    ingest.add_argument("trace_file", type=Path)
    ingest.add_argument("--out-workflow", type=Path, default=Path("results/ingested_workflow.json"))
    ingest.add_argument("--out-brief", type=Path, default=Path("reports/ingested_brief.md"))
    ingest.add_argument("--budget", "-k", type=int, default=2)
    ingest.add_argument(
        "--algorithm",
        choices=algorithm_choices,
        default="auto",
    )
    ingest.add_argument(
        "--model",
        choices=model_choices,
        default="quality-cascade",
    )
    ingest.add_argument("--trials", type=int, default=50)
    ingest.add_argument("--json", action="store_true")

    trace_replay = subparsers.add_parser(
        "trace-replay",
        help="replay a saved trace.jsonl and compare A0 vs A2 token usage",
    )
    trace_replay.add_argument("trace_file", type=Path, help="path to a trace.jsonl file")
    trace_replay.add_argument(
        "--no-control",
        action="store_true",
        help="treat all A0 decisions as CONTINUE (pure baseline replay)",
    )
    trace_replay.add_argument(
        "--baseline-tokens",
        type=int,
        default=None,
        help=(
            "baseline/A0 token total for savings math; if omitted, trace metadata "
            "is used when available, otherwise observed tokens are reused"
        ),
    )
    trace_replay.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")

    init = subparsers.add_parser(
        "init",
        help="scaffold a starter workflow JSON file from a node list and topology",
    )
    init.add_argument("name", help="workflow name (used for the output file stem)")
    init.add_argument(
        "--nodes",
        required=True,
        help="comma-separated node ids, e.g. planner,coder,tester",
    )
    init.add_argument(
        "--type",
        dest="topology",
        choices=list(TOPOLOGIES),
        default="pipeline",
        help="graph topology to wire the nodes into (default: pipeline)",
    )
    init.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path (default: ./<name>.json)",
    )
    init.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    sessions = subparsers.add_parser(
        "sessions",
        help="inspect saved control sessions",
    )
    sessions_sub = sessions.add_subparsers(dest="sessions_command")
    sessions_stats = sessions_sub.add_parser(
        "stats",
        help="aggregate analytics across saved session records",
    )
    sessions_stats.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="session directory (default: $AGENTPROP_SESSION_DIR or ~/.agentprop/sessions)",
    )
    sessions_stats.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the report to a file instead of stdout",
    )
    sessions_stats.add_argument("--json", action="store_true", help="emit machine-readable JSON")

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


def _run_evidence(args: argparse.Namespace) -> int:
    from agentprop.evaluation.evidence_harness import (
        EvidenceHarnessConfig,
        write_evidence_artifacts,
    )

    config = EvidenceHarnessConfig(
        tasks_per_arm=args.tasks_per_arm,
        repeats=args.repeats,
        seed_budget=args.seed_budget,
        trials=args.trials,
    )
    results_path = write_evidence_artifacts(config, args.out_dir)
    print(f"Wrote {results_path}")
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
        pruning_target_token_reduction=args.pruning_target_token_reduction,
        pruning_strategy=args.pruning_strategy,
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


def _agent_instructions(args: argparse.Namespace) -> int:
    workflow_name, graph = _load_workflow(args.workflow)
    report = _build_recommendation_report(
        graph,
        algorithm=args.algorithm,
        model_name=args.model,
        budget=args.budget,
        trials=args.trials,
    )
    markdown = render_coding_agent_instructions(
        report,
        workflow_name=workflow_name,
        target=args.target,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown)
    print(f"Wrote {args.out}")
    return 0


def _readiness(args: argparse.Namespace) -> int:
    report = build_v1_readiness_report()
    if args.json:
        content = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    else:
        content = render_v1_readiness_markdown(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content)
        print(f"Wrote {args.out}")
    else:
        print(content, end="")
    return 0


def _terminal_bench(args: argparse.Namespace) -> int:
    if args.terminal_bench_command == "prepare":
        config = TerminalBenchLaunchConfig(
            dataset=args.dataset,
            agent=args.agent,
            model=args.model,
            environment=args.environment,
            run_name=args.run_name,
            task_count=args.task_count,
            output_root=args.output_root,
            watchdog=HarborWatchdogConfig(
                timeout_s=args.timeout,
                idle_timeout_s=args.idle_timeout,
            ),
        )
        paths = write_terminal_bench_launch_bundle(
            args.out_dir,
            config,
            registry_root=args.registry_root,
        )
    elif args.terminal_bench_command == "summarize":
        paths = write_terminal_bench_summary_report(
            args.results_root,
            args.out_dir,
            title=args.title,
            registry_root=args.registry_root,
        )
    else:
        raise SystemExit("Error: terminal-bench requires a subcommand: prepare or summarize")

    payload = {name: str(path) for name, path in paths.items()}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for name, path in payload.items():
            print(f"{name}: {path}")
    return 0


def _workflows(args: argparse.Namespace) -> int:
    if args.workflows_command != "list":
        raise SystemExit("Error: workflows requires a subcommand: list")

    rows = [
        {"name": name, "description": WORKFLOW_DESCRIPTIONS.get(name, "")}
        for name in sorted(WORKFLOW_TEMPLATES)
    ]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print("Built-in workflow templates:")
        for row in rows:
            print(f"  {row['name']}: {row['description']}")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, object]] = []
    ok = True

    def record(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        if not passed:
            ok = False
        checks.append({"name": name, "ok": passed, "detail": detail})

    try:
        version = importlib.metadata.version("agentprop")
        record("agentprop", True, f"version {version}")
    except importlib.metadata.PackageNotFoundError:
        record("agentprop", False, "package not installed; run pip install -e .")

    record("networkx", _optional_import("networkx"), "core graph dependency")

    if args.tier in {"dev", "llm", "terminal-bench"}:
        record("pytest", _optional_import("pytest"), "dev extra: pip install -e '.[dev]'")
        record("ruff", _optional_import("ruff"), "dev extra")
        record("mypy", _optional_import("mypy"), "dev extra")

    if args.tier in {"llm", "terminal-bench"}:
        import os

        has_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("TOKEN_ROUTER_API_KEY"))
        record(
            "llm_credentials",
            has_key,
            "set OPENAI_API_KEY or TOKEN_ROUTER_API_KEY (see docs/environment.md)",
        )

    if args.tier == "terminal-bench":
        record("harbor", shutil.which("harbor") is not None, "Harbor CLI for Terminal-Bench runs")
        record("dot", shutil.which("dot") is not None, "Graphviz dot for viz rendering")

    graphviz_ok = shutil.which("dot") is not None
    if args.tier == "graph":
        checks.append(
            {
                "name": "graphviz_dot",
                "ok": graphviz_ok,
                "optional": True,
                "detail": "optional; required only to render .dot files",
            }
        )
    else:
        record("graphviz_dot", graphviz_ok, "optional; required only to render .dot files")

    payload = {
        "tier": args.tier,
        "ok": ok,
        "checks": checks,
        "next_step": _doctor_next_step(args.tier, ok),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"AgentProp doctor (tier={args.tier}): {'PASS' if ok else 'FAIL'}")
        for check in checks:
            status = "ok" if check["ok"] else ("warn" if check.get("optional") else "FAIL")
            print(f"  [{status}] {check['name']}: {check['detail']}")
        print(f"Next: {payload['next_step']}")
    return 0 if ok else 1


def _doctor_next_step(tier: str, ok: bool) -> str:
    if ok and tier == "graph":
        return "agentprop analyze planner_coder_tester_reviewer --json"
    if ok and tier == "dev":
        return "pytest"
    if not ok and tier == "graph":
        return "python -m pip install -e ."
    if not ok and tier == "dev":
        return "python -m pip install -e '.[dev]'"
    return "see docs/environment.md"


def _optional_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _ingest_trace(args: argparse.Namespace) -> int:
    result = graph_from_trace(args.trace_file)
    args.out_workflow.parent.mkdir(parents=True, exist_ok=True)
    result.graph.to_json(args.out_workflow)

    report = _build_recommendation_report(
        result.graph,
        algorithm=args.algorithm,
        model_name=args.model,
        budget=args.budget,
        trials=args.trials,
    )
    brief = render_coding_agent_instructions(
        report,
        workflow_name=args.out_workflow.stem,
        target="generic",
    )
    args.out_brief.parent.mkdir(parents=True, exist_ok=True)
    args.out_brief.write_text(brief)

    payload = {
        "trace_file": str(args.trace_file),
        "workflow": str(args.out_workflow),
        "brief": str(args.out_brief),
        "message_count": result.message_count,
        "total_token_cost": result.total_token_cost,
        "seeds": report.seeds,
        "estimated_savings": report.estimated_savings,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Wrote workflow {args.out_workflow}")
        print(f"Wrote brief {args.out_brief}")
        print(f"Seeds: {', '.join(report.seeds)}")
        print(f"Estimated savings: {report.estimated_savings:.1%}")
    return 0


def _trace_replay(args: argparse.Namespace) -> int:
    from agentprop.runtime.trace_replay import format_replay_table, replay_trace

    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"error: trace file not found: {trace_path}", file=sys.stderr)
        return 1
    result = replay_trace(
        trace_path,
        no_control=args.no_control,
        baseline_tokens=args.baseline_tokens,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "task_id": result.task_id,
                    "workflow": result.workflow,
                    "baseline_tokens": result.baseline_tokens,
                    "total_tokens_no_control": result.total_tokens_no_control,
                    "total_tokens_with_control": result.total_tokens_with_control,
                    "token_delta": result.token_delta,
                    "reduction_pct": result.reduction_pct,
                    "replay_warning": result.replay_warning,
                    "rows": [
                        {
                            "step": r.step,
                            "command": r.command,
                            "tokens_used": r.tokens_used,
                            "decision_no_control": r.decision_no_control,
                            "decision_with_control": r.decision_with_control,
                        }
                        for r in result.rows
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_replay_table(result))
    return 0


def _init(args: argparse.Namespace) -> int:
    node_ids = [node.strip() for node in args.nodes.split(",") if node.strip()]
    if not node_ids:
        raise ValueError("--nodes must list at least one node id")
    graph = scaffold_workflow(node_ids, topology=args.topology)
    out_path = args.out if args.out is not None else Path(f"{args.name}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph.to_json(out_path)
    payload = {
        "name": args.name,
        "topology": args.topology,
        "nodes": node_ids,
        "out": str(out_path),
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Wrote starter workflow to {out_path}")
        print(f"  Topology: {args.topology}")
        print(f"  Nodes ({graph.node_count}): {', '.join(node_ids)}")
        print(f"  Edges: {graph.edge_count}")
        print(f"Next: agentprop analyze {out_path} --json")
    return 0


def _resolve_sessions_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    import os

    env = os.environ.get("AGENTPROP_SESSION_DIR")
    if env:
        return Path(env)
    return Path.home() / ".agentprop" / "sessions"


def _sessions(args: argparse.Namespace) -> int:
    if args.sessions_command != "stats":
        raise SystemExit("Error: sessions requires a subcommand: stats")

    sessions_dir = _resolve_sessions_dir(args.dir)
    report = aggregate_session_stats(sessions_dir)

    if args.json:
        content = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    else:
        content = render_session_stats_markdown(report, root=sessions_dir)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(content, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(content, end="")
    return 0


def _control_demo(args: argparse.Namespace) -> int:
    result = run_control_demo(args.demo, args.out_dir)
    payload = {
        "demo": result.demo,
        "out_dir": str(result.out_dir),
        "artifacts": {name: str(path) for name, path in result.artifacts.items()},
        "summary": result.summary,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Wrote AgentProp control demo to {result.out_dir}")
        for name, path in result.artifacts.items():
            print(f"{name}: {path}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    _, graph = _load_workflow(args.workflow)
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
    pruning_target_token_reduction: float = 0.3,
    pruning_strategy: str = "low-weight",
) -> Any:
    # Phase 0 cheap-defaults: for CLI/MCP "analysis/optimize/report" paths,
    # default expensive seed selection (greedy family) to the cheap rzf-centrality
    # when the workflow is larger than ~15 nodes. This makes interactive use and
    # MCP tools scale; exact greedy/CELF remain available (and are still the default
    # for tiny graphs and for paper-grade exact results).
    from agentprop.algorithms.seed_selection import auto_seed_algorithm

    if algorithm in {"auto", "default"} or (
        algorithm == "greedy" and graph.node_count > 15
    ):
        algorithm = auto_seed_algorithm(graph, requested=algorithm)

    model = make_propagation_model(model_name)
    seeds = select_seeds(graph, algorithm, budget, model, trials)
    propagation = model.simulate(graph, seeds, trials=trials)
    candidate_edges = _rank_pruning_edges(graph, pruning_strategy)
    pruning_candidates = _edges_to_reduction_target(
        graph,
        candidate_edges,
        target_reduction=pruning_target_token_reduction,
    )
    pruning_evaluation = evaluate_pruning(
        graph,
        pruning_candidates,
        seeds=seeds,
        propagation_model=make_propagation_model(model_name),
        trials=trials,
    )
    report = compare_routing(
        graph,
        seeds,
        model.name,
        propagation,
        bottlenecks=bottleneck_nodes(graph),
        pruning_candidates=pruning_candidates,
        verifier_candidates=risk_aware_verifier_placement(graph, min(budget, graph.node_count)),
        pruning_risk=summarize_pruning_risk(
            pruning_evaluation,
            target_cost_reduction=pruning_target_token_reduction,
        ),
    )
    from agentprop.evaluation.metrics import build_what_if_k_curve

    graph.warm_analysis_cache()
    report.what_if_k = build_what_if_k_curve(
        graph,
        model=model,
        candidate_seeds=[node.id for node in graph.nodes()],
        max_k=min(budget + 2, graph.node_count),
        trials=max(20, trials // 2),
    )
    return report


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
    if report.pruning_risk is not None:
        print("Pruning risk")
        print("------------")
        print(f"Target cost reduction: {report.pruning_risk.target_cost_reduction:.1%}")
        print(f"Achieved cost reduction: {report.pruning_risk.achieved_cost_reduction:.1%}")
        print(f"Coverage delta: {report.pruning_risk.coverage_delta:.1%}")
        print(f"Risk score: {report.pruning_risk.risk_score:.3f}")
        print("")
    if report.robustness is not None:
        print("Robustness")
        print("----------")
        print(f"Worst node-failure loss: {report.robustness.worst_node_failure_loss:.1%}")
        print(f"Worst edge-failure loss: {report.robustness.worst_edge_failure_loss:.1%}")
        print("")
    print("Verifier candidates")
    print("-------------------")
    for node_id in report.verifier_candidates:
        print(f"- {node_id}")


if __name__ == "__main__":
    raise SystemExit(main())
