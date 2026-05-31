"""Replay exported routing trajectories on built-in workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentprop.propagation import IndependentCascade
from agentprop.rl import AgentRoutingEnv, actions_from_exported_trajectory, replay_actions
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay exported AgentProp RL trajectories.")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--workflow", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/rl/replayed_trajectory.json"))
    args = parser.parse_args(argv)

    rows = _load_rows(args.trajectory)
    replay_rows = []
    for row in rows:
        workflow_name = str(row.get("workflow", ""))
        policy_name = str(row.get("policy", ""))
        if args.workflow is not None and workflow_name != args.workflow:
            continue
        if args.policy is not None and policy_name != args.policy:
            continue
        if workflow_name not in WORKFLOW_TEMPLATES:
            raise ValueError(f"Unknown workflow in trajectory export: {workflow_name}")

        trajectory = row.get("trajectory")
        if not isinstance(trajectory, list):
            raise ValueError(f"Row for {workflow_name} is missing a trajectory list")
        graph = WORKFLOW_TEMPLATES[workflow_name]()
        env = AgentRoutingEnv(
            graph,
            budget=args.budget,
            propagation_model=IndependentCascade(seed=args.seed),
            trials=args.trials,
        )
        actions = actions_from_exported_trajectory(trajectory)
        replay = replay_actions(env, actions)
        replay_rows.append(
            {
                "workflow": workflow_name,
                "policy": policy_name,
                "source_action_count": len(actions),
                **replay.to_dict(),
            }
        )

    payload = {
        "source": str(args.trajectory),
        "budget": args.budget,
        "trials": args.trials,
        "seed": args.seed,
        "rows": replay_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("trajectory export must be a list of workflow rows")
    rows = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"trajectory row {index} must be an object")
        rows.append(row)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
