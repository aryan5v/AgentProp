"""Train a lightweight edge-pruning scorer on built-in workflow templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from agentprop.evaluation import register_artifact, safe_artifact_id
from agentprop.ml import (
    EdgePruningExample,
    EmpiricalEdgePruningExample,
    LinearEdgeScorer,
    build_edge_pruning_example,
    build_empirical_edge_pruning_examples,
    extract_edge_features,
    save_ml_model,
)
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an edge-pruning scorer.")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--workflow", default="planner_coder_tester_reviewer")
    parser.add_argument("--empirical-results", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-penalty", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("results/ml/edge_pruning_scorer.json"))
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    label_source = "heuristic"
    examples: list[EdgePruningExample | EmpiricalEdgePruningExample]
    if args.empirical_results is not None:
        if args.workflow not in WORKFLOW_TEMPLATES:
            raise ValueError(f"Unknown workflow template: {args.workflow}")
        examples = cast(
            list[EdgePruningExample | EmpiricalEdgePruningExample],
            build_empirical_edge_pruning_examples(
                WORKFLOW_TEMPLATES[args.workflow](),
                _load_empirical_rows(args.empirical_results),
            ),
        )
        if not examples:
            raise ValueError("No usable empirical edge-pruning examples found")
        label_source = "empirical-outcome"
    else:
        examples = [
            build_edge_pruning_example(builder(), fraction=args.fraction)
            for builder in WORKFLOW_TEMPLATES.values()
        ]
    feature_count = len(examples[0].features.feature_names)
    scorer = LinearEdgeScorer.initialize(feature_count)
    scorer.train(
        examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_penalty=args.l2_penalty,
    )

    evaluations = []
    for workflow_name, builder in WORKFLOW_TEMPLATES.items():
        graph = builder()
        features = extract_edge_features(graph)
        scores = scorer.score_edges(features)
        ranked_edges = [
            {"source": source, "target": target, "score": score}
            for (source, target), score in sorted(scores.items(), key=lambda item: -item[1])
        ]
        evaluations.append({"workflow": workflow_name, "ranked_edges": ranked_edges})

    payload = {
        "feature_names": examples[0].features.feature_names,
        "weights": scorer.weights,
        "bias": scorer.bias,
        "fraction": args.fraction,
        "label_source": label_source,
        "l2_penalty": args.l2_penalty,
        "evaluations": evaluations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    run_id = safe_artifact_id(args.run_id or "edge-pruning-scorer")
    checkpoint_path = args.checkpoint_out
    if checkpoint_path is None and args.registry_root is not None:
        checkpoint_path = args.registry_root / "checkpoints" / f"{run_id}.json"
    if checkpoint_path is not None:
        checkpoint_path = save_ml_model(
            scorer,
            checkpoint_path,
            metadata={
                "task": "edge-pruning",
                "label_source": label_source,
                "fraction": args.fraction,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "l2_penalty": args.l2_penalty,
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
            source="experiments.train_edge_pruning_scorer",
            metrics_path=args.out,
            tags=("edge-pruning", "linear"),
            metadata={
                "fraction": args.fraction,
                "label_source": label_source,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "l2_penalty": args.l2_penalty,
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
