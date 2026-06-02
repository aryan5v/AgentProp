"""Train an optional torch GNN seed scorer on built-in workflow templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentprop.dl import (
    GraphEncoderConfig,
    TorchBackendUnavailable,
    train_torch_seed_scorer,
)
from agentprop.ml import (
    build_empirical_routing_examples,
    build_empirical_verifier_placement_examples,
    build_seed_selection_example,
    build_verifier_placement_example,
)
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a torch GNN seed-selection scorer.")
    parser.add_argument(
        "--architecture",
        choices=[
            "gcn",
            "graphsage",
            "gat",
            "gin",
            "graph_transformer",
            "heterogeneous",
            "edge_conditioned",
        ],
        default="graphsage",
    )
    parser.add_argument("--task", choices=["seed", "verifier"], default="seed")
    parser.add_argument("--workflow", default="planner_coder_tester_reviewer")
    parser.add_argument("--empirical-results", type=Path, default=None)
    parser.add_argument(
        "--allow-heuristic-labels",
        action="store_true",
        help=(
            "Allow topology/heuristic labels when --empirical-results is absent. "
            "Use this only for baseline imitation runs."
        ),
    )
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=Path("results/dl/torch_gnn_seed_scorer.json"))
    args = parser.parse_args(argv)

    examples, label_source = _build_training_examples(
        task=args.task,
        workflow=args.workflow,
        empirical_results=args.empirical_results,
        allow_heuristic_labels=args.allow_heuristic_labels,
        budget=args.budget,
        trials=args.trials,
    )
    config = GraphEncoderConfig(
        input_dim=len(examples[0].features.feature_names),
        hidden_dim=args.hidden_dim,
        architecture=args.architecture,
        task=args.task,
    )

    try:
        scorer, result = train_torch_seed_scorer(
            examples,
            config=config,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
    except TorchBackendUnavailable as exc:
        print(str(exc))
        return 2

    evaluations = []
    for workflow_name, builder in WORKFLOW_TEMPLATES.items():
        graph = builder()
        scores = scorer.score_nodes(graph)
        top_nodes = [
            node_id for node_id, _ in sorted(scores.items(), key=lambda item: -item[1])
        ][: args.budget]
        evaluations.append(
            {
                "workflow": workflow_name,
                "recommended_seeds": top_nodes,
                "scores": scores,
            }
        )

    payload = {
        "architecture": result.architecture,
        "edge_feature_dim": config.edge_feature_dim,
        "epochs": result.epochs,
        "final_loss": result.final_loss,
        "label_source": label_source,
        "task": args.task,
        "training_example_count": len(examples),
        "losses": result.losses,
        "evaluations": evaluations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


def _build_training_examples(
    *,
    task: str,
    workflow: str,
    empirical_results: Path | None,
    allow_heuristic_labels: bool,
    budget: int,
    trials: int,
) -> tuple[list[Any], str]:
    if empirical_results is not None:
        if workflow not in WORKFLOW_TEMPLATES:
            raise ValueError(f"Unknown workflow template: {workflow}")
        graph = WORKFLOW_TEMPLATES[workflow]()
        rows = _load_empirical_rows(empirical_results)
        examples: list[Any]
        if task == "verifier":
            examples = build_empirical_verifier_placement_examples(
                graph,
                rows,
                default_budget=budget,
            )
        else:
            examples = build_empirical_routing_examples(
                graph,
                rows,
                default_budget=budget,
            )
        if not examples:
            raise ValueError(f"No usable empirical {task} examples found")
        return examples, "empirical-outcome"

    if not allow_heuristic_labels:
        raise ValueError(
            "--empirical-results is required for training; pass --allow-heuristic-labels "
            "to run an explicit heuristic baseline."
        )

    if task == "verifier":
        return [
            build_verifier_placement_example(builder(), budget=budget)
            for builder in WORKFLOW_TEMPLATES.values()
        ], "heuristic-baseline"
    return [
        build_seed_selection_example(builder(), budget=budget, trials=trials)
        for builder in WORKFLOW_TEMPLATES.values()
    ], "heuristic-baseline"


def _load_empirical_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "tasks", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)]
    raise ValueError("Empirical results must be a list or contain rows/tasks/results")


if __name__ == "__main__":
    raise SystemExit(main())
