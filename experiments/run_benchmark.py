"""Run AgentProp benchmark tables and write JSON/CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agentprop.core import AgentGraph
from agentprop.evaluation import run_benchmark
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AgentProp graph optimization benchmarks.")
    parser.add_argument(
        "--workflows",
        nargs="+",
        default=list(WORKFLOW_TEMPLATES),
        help="Built-in workflow names or JSON paths.",
    )
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[
            "random",
            "degree",
            "in-degree",
            "out-degree",
            "pagerank",
            "betweenness",
            "closeness",
            "k-core",
            "greedy",
            "celf",
            "cost-aware-greedy",
        ],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "independent-cascade",
            "linear-threshold",
            "bootstrap",
            "rzf",
            "zero-forcing",
            "learned",
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/benchmark"))
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for workflow in args.workflows:
        workflow_name, graph = _load_workflow(workflow)
        rows.extend(
            run_benchmark(
                graph,
                workflow_name=workflow_name,
                algorithms=args.algorithms,
                models=args.models,
                budget=args.budget,
                trials=args.trials,
            )
        )

    payload = [row.to_dict() for row in rows]
    json_path = args.out_dir / "results.json"
    csv_path = args.out_dir / "results.csv"
    svg_path = args.out_dir / "savings_by_algorithm.svg"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0]) if payload else [])
        writer.writeheader()
        writer.writerows(payload)
    svg_path.write_text(_render_savings_svg(payload))

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {svg_path}")
    return 0


def _load_workflow(workflow: str) -> tuple[str, AgentGraph]:
    if workflow in WORKFLOW_TEMPLATES:
        return workflow, WORKFLOW_TEMPLATES[workflow]()
    path = Path(workflow)
    return path.stem, AgentGraph.from_json(path)


def _render_savings_svg(rows: list[dict[str, object]]) -> str:
    width = 960
    height = 360
    margin = 48
    grouped: dict[str, list[float]] = {}
    for row in rows:
        algorithm = str(row["algorithm"])
        grouped.setdefault(algorithm, []).append(float(row["estimated_savings"]))
    averages = {
        algorithm: sum(values) / len(values)
        for algorithm, values in sorted(grouped.items())
        if values
    }
    if not averages:
        return "<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n"

    max_value = max(max(averages.values()), 0.01)
    bar_width = (width - 2 * margin) / max(len(averages), 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="48" y="28" font-size="18" font-family="sans-serif">',
        "Average Estimated Savings By Algorithm",
        "</text>",
    ]
    for index, (algorithm, value) in enumerate(averages.items()):
        x = margin + index * bar_width + 8
        bar_height = (height - 120) * (value / max_value)
        y = height - margin - bar_height
        lines.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 16:.1f}" '
                f'height="{bar_height:.1f}" fill="#3267d6"/>',
                f'<text x="{x:.1f}" y="{height - 26}" font-size="11" '
                f'font-family="sans-serif" transform="rotate(30 {x:.1f},{height - 26})">',
                algorithm,
                "</text>",
                f'<text x="{x:.1f}" y="{y - 6:.1f}" font-size="12" font-family="sans-serif">',
                f"{value:.1%}",
                "</text>",
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
