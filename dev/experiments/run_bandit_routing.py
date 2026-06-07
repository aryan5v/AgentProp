"""Replay empirical rows through a category-conditioned routing bandit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentprop.rl import CategoryBanditRoutingPolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Update a per-task-category routing-policy bandit from empirical rows."
    )
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument(
        "--arms",
        default="broadcast,optimized_greedy,quality_aware_greedy,ml_message_passing,rl_ppo,agentprop",
        help="Comma-separated routing policies the bandit may recommend.",
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/rl/category_bandit.json"))
    args = parser.parse_args(argv)

    rows = _load_rows(args.rows)
    policy = CategoryBanditRoutingPolicy(
        arms=tuple(_split_csv(args.arms)),
        epsilon=args.epsilon,
        seed=args.seed,
    )
    updates: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        observed_arm = _optional_string(row.get("policy") or row.get("arm"))
        passed = _row_passed(row)
        if observed_arm is None or passed is None or observed_arm not in policy.arms:
            skipped += 1
            continue
        category = _optional_string(row.get("category") or row.get("task_category")) or "default"
        recommended_before = policy.choose(category)
        savings = _row_token_savings(row)
        policy.update(
            category,
            observed_arm,
            passed=passed,
            token_savings=savings,
            quality_score=_optional_float(row.get("quality_score")),
        )
        updates.append(
            {
                "task_id": _optional_string(row.get("task_id") or row.get("task_name")),
                "category": category,
                "recommended_before_update": recommended_before,
                "observed_policy": observed_arm,
                "passed": passed,
                "token_savings": savings,
                "values_after_update": policy.values(category),
            }
        )

    categories = sorted(policy.stats)
    payload = {
        "row_count": len(rows),
        "update_count": len(updates),
        "skipped_count": skipped,
        "arms": list(policy.arms),
        "epsilon": args.epsilon,
        "categories": {
            category: {
                "recommended_policy": policy.choose(category),
                "values": policy.values(category),
                "counts": {
                    arm: stats.count for arm, stats in policy.stats[category].items()
                },
            }
            for category in categories
        },
        "updates": updates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "tasks", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, dict)]
    raise ValueError("Rows must be a list or contain rows/tasks/results")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _row_passed(row: dict[str, Any]) -> bool | None:
    for key in ("verification_passed", "quality_passed", "passed"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
    return None


def _row_token_savings(row: dict[str, Any]) -> float:
    for key in ("token_savings", "token_savings_pct", "measured_saving"):
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    baseline = _optional_float(row.get("baseline_token_cost") or row.get("broadcast_token_cost"))
    current = _optional_float(row.get("token_cost") or row.get("total_tokens"))
    if baseline and current is not None:
        return (baseline - current) / baseline
    return 0.0


def _optional_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
