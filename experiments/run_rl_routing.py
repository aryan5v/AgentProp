"""Run the lightweight sequential routing baseline on built-in workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop.rl import AgentRoutingEnv, GreedyCoveragePolicy, RoutingAction
from agentprop.workflows import WORKFLOW_TEMPLATES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AgentProp sequential routing baseline.")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("results/rl/greedy_policy.json"))
    args = parser.parse_args(argv)

    policy = GreedyCoveragePolicy()
    rows = []
    for workflow_name, builder in WORKFLOW_TEMPLATES.items():
        env = AgentRoutingEnv(builder(), budget=args.budget, trials=args.trials)
        trajectory = []
        done = False
        while not done:
            action = policy.act(env)
            if action == RoutingAction.STOP.value:
                state, reward, done, info = env.step(action)
            else:
                state, reward, done, info = env.step(action)
            trajectory.append(
                {
                    "action": action,
                    "reward": reward,
                    "coverage": state.coverage,
                    "token_cost": state.token_cost,
                    "message_cost": state.message_cost,
                    "done": done,
                    "info": dict(info),
                }
            )
        rows.append({"workflow": workflow_name, "trajectory": trajectory})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
