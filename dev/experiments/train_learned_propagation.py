"""Fit a learned propagation model from trace JSON and write diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.integrations import graph_from_trace_dict
from agentprop.propagation import LearnedPropagation, fit_learned_propagation_from_trace_dicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a trace-calibrated propagation model.")
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--seed-node", default=None)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("results/propagation/learned.json"))
    args = parser.parse_args(argv)

    traces = [json.loads(path.read_text()) for path in args.trace]
    fit = fit_learned_propagation_from_trace_dicts(traces)
    model = LearnedPropagation(edge_probabilities=fit.edge_probabilities, seed=0)
    graph = graph_from_trace_dict(traces[0]).graph
    seed_node = args.seed_node or next(iter(sorted(node.id for node in graph.nodes())))
    result = model.simulate(graph, [seed_node], trials=args.trials)

    payload = {
        "edge_probabilities": {
            f"{source}->{target}": probability
            for (source, target), probability in sorted(fit.edge_probabilities.items())
        },
        "edge_counts": {
            f"{source}->{target}": count
            for (source, target), count in sorted(fit.edge_counts.items())
        },
        "source_counts": fit.source_counts,
        "seed_node": seed_node,
        "coverage": result.coverage,
        "expected_propagation_time": result.expected_propagation_time,
        "full_activation_probability": result.full_activation_probability,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
