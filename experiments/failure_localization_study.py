"""Failure localization study: signature collisions vs verifier placement method.

Deterministic script. Injects a synthetic fault at each node and measures whether
the observable verifier-distance signature is unique (localizable) under different
placement methods.

Run (after `pip install -e ".[dev]"`):

    python experiments/failure_localization_study.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import networkx as nx

from agentprop.algorithms import (
    betweenness_verifier_placement,
    metric_dimension_verifier_placement,
    resolving_coverage,
    risk_aware_verifier_placement,
)
from agentprop.workflows import WORKFLOW_TEMPLATES

KS = [1, 2, 3, 4, 5]
METHODS = {
    "metric_dim": lambda g, k: metric_dimension_verifier_placement(g, k),
    "risk_aware": lambda g, k: risk_aware_verifier_placement(g, k),
    "betweenness": lambda g, k: betweenness_verifier_placement(g, k),
}


def signature(
    distances: dict[str, dict[str, int]],
    node: str,
    verifiers: list[str],
) -> tuple[int, ...]:
    return tuple(distances.get(node, {}).get(v, -1) for v in verifiers)


def collision_rate(
    graph_nodes: list[str],
    distances: dict[str, dict[str, int]],
    verifiers: list[str],
) -> float:
    seen: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for node in graph_nodes:
        seen[signature(distances, node, verifiers)].append(node)
    colliding_nodes = sum(len(nodes) for nodes in seen.values() if len(nodes) > 1)
    return colliding_nodes / max(len(graph_nodes), 1)


def main() -> None:
    rows: list[dict[str, object]] = []

    print("=" * 78)
    print("Failure localization — signature collision rate by placement method")
    print("=" * 78)
    header = f"{'workflow':28s}" + "".join(f"  {m:12s}" for m in METHODS) + "  cov@k=3(md)"
    print(header)

    for workflow_name, builder in sorted(WORKFLOW_TEMPLATES.items()):
        graph = builder()
        nodes = [node.id for node in graph.nodes()]
        distances = dict(nx.all_pairs_shortest_path_length(graph.to_networkx().to_undirected()))
        row_print = f"{workflow_name:28s}"
        method_collisions: dict[str, float] = {}

        for method_name, placer in METHODS.items():
            verifiers = placer(graph, 3)
            rate = collision_rate(nodes, distances, verifiers)
            method_collisions[method_name] = rate
            row_print += f"  {rate:12.3f}"

        md_cov = resolving_coverage(graph, metric_dimension_verifier_placement(graph, 3))
        row_print += f"  {md_cov:8.3f}"
        print(row_print)

        for method_name in METHODS:
            for k in KS:
                verifiers = METHODS[method_name](graph, k)
                rows.append(
                    {
                        "workflow": workflow_name,
                        "method": method_name,
                        "k": k,
                        "collision_rate": collision_rate(nodes, distances, verifiers),
                        "resolving_coverage": resolving_coverage(graph, verifiers),
                        "verifiers": verifiers,
                    }
                )

    print()
    print("Mean collision rate by method at k=3:")
    for method_name in METHODS:
        subset = [row for row in rows if row["method"] == method_name and row["k"] == 3]
        mean_collision = st.mean(float(row["collision_rate"]) for row in subset)
        mean_cov = st.mean(float(row["resolving_coverage"]) for row in subset)
        print(
            f"  {method_name:15s} collision={mean_collision:.3f}  "
            f"resolving_coverage={mean_cov:.3f}"
        )

    out_dir = Path("docs/results/failure_localization")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
