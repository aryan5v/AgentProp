"""Evaluate lightweight ML policy generalization on held-out workflow graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.ml import (
    LinearNodeScorer,
    MLPNodeScorer,
    build_seed_utility_example,
    extract_graph_features,
)
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate held-out graph generalization.")
    parser.add_argument("--model", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("results/ml/generalization.json"))
    args = parser.parse_args(argv)

    rows = []
    for held_out_name, held_out_builder in WORKFLOW_TEMPLATES.items():
        training_examples = [
            build_seed_utility_example(builder(), budget=args.budget, trials=args.trials)
            for name, builder in WORKFLOW_TEMPLATES.items()
            if name != held_out_name
        ]
        feature_count = len(training_examples[0].features.feature_names)
        if args.model == "mlp":
            scorer = MLPNodeScorer.initialize(feature_count)
        else:
            scorer = LinearNodeScorer.initialize(feature_count)
        scorer.train(training_examples, epochs=args.epochs, learning_rate=args.learning_rate)

        held_out_graph = held_out_builder()
        held_out_example = build_seed_utility_example(
            held_out_graph,
            budget=args.budget,
            trials=args.trials,
        )
        scores = scorer.score_nodes(extract_graph_features(held_out_graph))
        predicted = [
            node_id for node_id, _ in sorted(scores.items(), key=lambda item: -item[1])
        ][: args.budget]
        positives = set(held_out_example.positive_seeds)
        recall = len(positives.intersection(predicted)) / max(len(positives), 1)
        rows.append(
            {
                "held_out_workflow": held_out_name,
                "model": args.model,
                "predicted": predicted,
                "positive_seeds": held_out_example.positive_seeds,
                "top_k_recall": recall,
            }
        )

    payload = {
        "model": args.model,
        "label_source": "marginal-utility",
        "budget": args.budget,
        "mean_top_k_recall": sum(row["top_k_recall"] for row in rows) / len(rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
