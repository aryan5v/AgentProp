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
        default=["random", "degree", "pagerank", "betweenness", "greedy"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["independent-cascade", "linear-threshold", "bootstrap", "rzf"],
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
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0]) if payload else [])
        writer.writeheader()
        writer.writerows(payload)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


def _load_workflow(workflow: str) -> tuple[str, AgentGraph]:
    if workflow in WORKFLOW_TEMPLATES:
        return workflow, WORKFLOW_TEMPLATES[workflow]()
    path = Path(workflow)
    return path.stem, AgentGraph.from_json(path)


if __name__ == "__main__":
    raise SystemExit(main())
