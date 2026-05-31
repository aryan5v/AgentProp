"""Train an optional torch GNN seed scorer on built-in workflow templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.dl import (
    GraphEncoderConfig,
    TorchBackendUnavailable,
    train_torch_seed_scorer,
)
from agentprop.ml import build_seed_selection_example
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a torch GNN seed-selection scorer.")
    parser.add_argument("--architecture", choices=["gcn", "graphsage", "gat"], default="graphsage")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=Path("results/dl/torch_gnn_seed_scorer.json"))
    args = parser.parse_args(argv)

    examples = [
        build_seed_selection_example(builder(), budget=args.budget, trials=args.trials)
        for builder in WORKFLOW_TEMPLATES.values()
    ]
    config = GraphEncoderConfig(
        input_dim=len(examples[0].features.feature_names),
        hidden_dim=args.hidden_dim,
        architecture=args.architecture,
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
        "epochs": result.epochs,
        "final_loss": result.final_loss,
        "losses": result.losses,
        "evaluations": evaluations,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
