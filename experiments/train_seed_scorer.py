"""Train the lightweight AgentProp seed scorer on built-in workflow templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.ml import LinearNodeScorer, build_seed_selection_example, extract_graph_features
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight seed-selection scorer.")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("results/ml/linear_seed_scorer.json"))
    args = parser.parse_args(argv)

    examples = [
        build_seed_selection_example(builder(), budget=args.budget, trials=args.trials)
        for builder in WORKFLOW_TEMPLATES.values()
    ]
    feature_count = len(examples[0].features.feature_names)
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
        "weights": scorer.weights,
        "bias": scorer.bias,
        "evaluations": evaluations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
