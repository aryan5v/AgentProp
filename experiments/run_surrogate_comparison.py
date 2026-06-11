#!/usr/bin/env python3
"""Compare submodular-surrogate placement against exact greedy resolving coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.algorithms.submodular_placement import submodular_verifier_placement
from agentprop.algorithms.verifier_placement import (
    metric_dimension_verifier_placement,
    resolving_coverage,
)
from agentprop.evaluation.fault_injection import GRAPH_FAMILIES
from agentprop.workflows import WORKFLOW_TEMPLATES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/submodular_surrogate"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()

    graphs = {name: build(0) for name, build in GRAPH_FAMILIES.items()}
    graphs.update({name: factory() for name, factory in WORKFLOW_TEMPLATES.items()})

    rows = []
    for name, graph in graphs.items():
        for k in args.budgets:
            exact = list(metric_dimension_verifier_placement(graph, k))
            surrogate = submodular_verifier_placement(graph, k)
            rows.append(
                {
                    "graph": name,
                    "budget": k,
                    "exact_resolving_coverage": resolving_coverage(graph, exact),
                    "surrogate_resolving_coverage": resolving_coverage(
                        graph, list(surrogate.verifiers)
                    ),
                    "surrogate_objective_fraction": surrogate.objective_fraction,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Submodular Surrogate vs Exact Greedy Resolving Coverage",
        "",
        "Resolving coverage achieved by exact greedy (no guarantee) vs the",
        "(1-1/e)-guaranteed submodular surrogate, same verifier budgets.",
        "",
        "| Graph | k | Exact greedy | Surrogate greedy | Gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        gap = row["surrogate_resolving_coverage"] - row["exact_resolving_coverage"]
        lines.append(
            f"| {row['graph']} | {row['budget']} | "
            f"{row['exact_resolving_coverage']:.1%} | "
            f"{row['surrogate_resolving_coverage']:.1%} | {gap:+.1%} |"
        )
    lines.append("")
    (args.out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_dir} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
