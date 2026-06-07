"""Quality cascade vs Independent Cascade: routing Pareto on built-in workflows.

Deterministic benchmark comparing quality-aware seeds + quality-cascade propagation
against PageRank + IC on token savings and coverage.

Run (after `pip install -e ".[dev]"`):

    python dev/experiments/quality_cascade_vs_ic.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from agentprop.evaluation.runner import make_propagation_model, run_benchmark
from agentprop.workflows import WORKFLOW_TEMPLATES

WORKFLOWS = [
    "chain",
    "planner_coder_tester_reviewer",
    "research_writer_verifier",
    "rag_pipeline",
    "dense_graph",
    "layered_pipeline",
]

ARMS = [
    ("qc_quality_aware", "quality-aware-greedy", "quality-cascade"),
    ("ic_pagerank", "pagerank", "independent-cascade"),
    ("ic_greedy", "greedy", "independent-cascade"),
]


def main() -> None:
    rows: list[dict[str, object]] = []

    for workflow_name in WORKFLOWS:
        graph = WORKFLOW_TEMPLATES[workflow_name]()
        for arm_name, algorithm, model_name in ARMS:
            model = make_propagation_model(model_name)
            bench_rows = run_benchmark(
                graph,
                workflow_name=workflow_name,
                algorithms=[algorithm],
                models=[model_name],
                budget=2,
                trials=30,
            )
            row = bench_rows[0]
            rows.append(
                {
                    "arm": arm_name,
                    "workflow": workflow_name,
                    "algorithm": algorithm,
                    "model": model.name,
                    "seeds": row.seeds,
                    "coverage": row.coverage,
                    "estimated_savings": row.estimated_savings,
                    "mean_output_quality": row.mean_output_quality,
                    "optimized_cost": row.optimized_cost,
                    "broadcast_cost": row.broadcast_cost,
                }
            )

    print("=" * 78)
    print("Quality cascade vs IC — mean estimated savings by arm")
    print("=" * 78)
    for arm_name, _, _ in ARMS:
        subset = [row for row in rows if row["arm"] == arm_name]
        savings = st.mean(float(row["estimated_savings"]) for row in subset)
        quality = st.mean(float(row["mean_output_quality"]) for row in subset)
        coverage = st.mean(float(row["coverage"]) for row in subset)
        print(
            f"  {arm_name:18s} savings={savings:.3f}  coverage={coverage:.3f}  "
            f"quality={quality:.3f}"
        )

    out_dir = Path("docs/results/quality_cascade_vs_ic")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
