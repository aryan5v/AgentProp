"""Train a lightweight edge-pruning scorer on built-in workflow templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.ml import LinearEdgeScorer, build_edge_pruning_example, extract_edge_features
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train an edge-pruning scorer.")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("results/ml/edge_pruning_scorer.json"))
    args = parser.parse_args(argv)

    examples = [
        build_edge_pruning_example(builder(), fraction=args.fraction)
        for builder in WORKFLOW_TEMPLATES.values()
    ]
    feature_count = len(examples[0].features.feature_names)
    scorer = LinearEdgeScorer.initialize(feature_count)
    scorer.train(examples, epochs=args.epochs, learning_rate=args.learning_rate)

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
        "evaluations": evaluations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
