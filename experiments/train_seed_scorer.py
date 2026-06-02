"""Train lightweight AgentProp node-policy scorers on built-in workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentprop.evaluation import register_artifact, safe_artifact_id
from agentprop.ml import (
    LinearNodeRegressor,
    LinearNodeScorer,
    MLPNodeScorer,
    PairwiseNodeRanker,
    build_empirical_routing_examples,
    build_seed_ranking_example,
    build_seed_selection_example,
    build_verifier_placement_example,
    extract_graph_features,
    save_ml_model,
)
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight node-policy scorer.")
    parser.add_argument(
        "--model",
        choices=["linear", "mlp", "pairwise", "regression"],
        default="linear",
    )
    parser.add_argument("--task", choices=["seed", "verifier"], default="seed")
    parser.add_argument("--workflow", default="planner_coder_tester_reviewer")
    parser.add_argument("--empirical-results", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("results/ml/linear_seed_scorer.json"))
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    if args.task == "verifier" and args.model in {"pairwise", "regression"}:
        parser.error("pairwise and regression models are currently seed-task only")
    if args.empirical_results is not None and args.model in {"pairwise", "regression"}:
        parser.error("empirical training currently supports linear and mlp node scorers")
    if args.empirical_results is not None and args.task != "seed":
        parser.error("empirical training currently targets seed/context routing")

    label_source = "heuristic"
    examples: list[Any]
    if args.empirical_results is not None:
        if args.workflow not in WORKFLOW_TEMPLATES:
            raise ValueError(f"Unknown workflow template: {args.workflow}")
        examples = build_empirical_routing_examples(
            WORKFLOW_TEMPLATES[args.workflow](),
            _load_empirical_rows(args.empirical_results),
            default_budget=args.budget,
        )
        if not examples:
            raise ValueError("No usable empirical routing examples found")
        label_source = "empirical-outcome"
    elif args.model in {"pairwise", "regression"}:
        examples = [
            build_seed_ranking_example(builder(), budget=args.budget, trials=args.trials)
            for builder in WORKFLOW_TEMPLATES.values()
        ]
    elif args.task == "verifier":
        examples = [
            build_verifier_placement_example(builder(), budget=args.budget)
            for builder in WORKFLOW_TEMPLATES.values()
        ]
    else:
        examples = [
            build_seed_selection_example(builder(), budget=args.budget, trials=args.trials)
            for builder in WORKFLOW_TEMPLATES.values()
        ]
    feature_count = len(examples[0].features.feature_names)
    scorer: Any
    if args.model == "mlp":
        scorer = MLPNodeScorer.initialize(feature_count)
    elif args.model == "pairwise":
        scorer = PairwiseNodeRanker.initialize(feature_count)
    elif args.model == "regression":
        scorer = LinearNodeRegressor.initialize(feature_count)
    else:
        scorer = LinearNodeScorer.initialize(feature_count)
    scorer.train(examples, epochs=args.epochs, learning_rate=args.learning_rate)

    evaluations = []
    for workflow_name, builder in WORKFLOW_TEMPLATES.items():
        graph = builder()
        features = extract_graph_features(graph)
        scores = scorer.score_nodes(features)
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
        "feature_names": examples[0].features.feature_names,
        "model": args.model,
        "task": args.task,
        "label_source": label_source,
        "evaluations": evaluations,
    }
    if isinstance(scorer, LinearNodeScorer | LinearNodeRegressor):
        payload["weights"] = scorer.weights
        payload["bias"] = scorer.bias
    if isinstance(scorer, PairwiseNodeRanker):
        payload["weights"] = scorer.weights
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    run_id = safe_artifact_id(args.run_id or f"{args.task}-{args.model}-node-scorer")
    checkpoint_path = args.checkpoint_out
    if checkpoint_path is None and args.registry_root is not None:
        checkpoint_path = args.registry_root / "checkpoints" / f"{run_id}.json"
    if checkpoint_path is not None:
        checkpoint_path = save_ml_model(
            scorer,
            checkpoint_path,
            metadata={
                "task": args.task,
                "model": args.model,
                "label_source": label_source,
                "budget": args.budget,
                "trials": args.trials,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "feature_names": examples[0].features.feature_names,
            },
        )
    if args.registry_root is not None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be available when registry_root is set")
        register_artifact(
            args.registry_root,
            artifact_id=run_id,
            kind="ml-model",
            path=checkpoint_path,
            source="experiments.train_seed_scorer",
            metrics_path=args.out,
            tags=("node-scorer", args.task, args.model),
            metadata={
                "budget": args.budget,
                "label_source": label_source,
                "trials": args.trials,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
            },
        )
    print(f"Wrote {args.out}")
    return 0


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
