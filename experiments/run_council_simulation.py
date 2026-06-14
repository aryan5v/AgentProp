#!/usr/bin/env python3
"""Zero-cost Council vs ensemble vs single-model economics simulation.

Produces the cost-vs-accuracy figure that justifies live DRACO spend, using
the OpenRouter "budget panel" shape (Gemini 3 Flash + Kimi K2.6 + DeepSeek V4
Pro) plus a frontier reference. No API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentprop.council.simulator import SimConfig, SimModel, simulate_strategies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/council_simulation"))
    parser.add_argument("--trials", type=int, default=500)
    args = parser.parse_args()

    if args.trials <= 0:
        print("Error: --trials must be greater than 0", file=sys.stderr)
        return 1

    # Illustrative competence/price values (NOT measured; the live runs supply
    # real numbers). Prices are rough per-Ktok blended rates.
    pool = [
        SimModel("gemini-3-flash", competence=0.55, price_per_mtok=0.10, tier=1),
        SimModel("kimi-k2.6", competence=0.58, price_per_mtok=0.15, tier=2),
        SimModel("deepseek-v4-pro", competence=0.60, price_per_mtok=0.20, tier=2),
    ]
    frontier = SimModel("frontier", competence=0.70, price_per_mtok=1.20, tier=3)

    outcomes = simulate_strategies(
        pool, cfg=SimConfig(), frontier=frontier, trials=args.trials, seed=0
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "simulation.json").write_text(
        json.dumps([o.to_dict() for o in outcomes], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Council Economics Simulation (zero API cost)",
        "",
        "Analytical model of single / ensemble (Fusion) / council strategies on",
        "the OpenRouter budget-panel shape. Illustrative competence/price inputs;",
        "the *shape* is the claim (council below-and-right of ensemble on",
        "cost-vs-accuracy), not absolute numbers — live DRACO supplies those.",
        "",
        "| Strategy | Accuracy | Cost (USD) | $/accuracy-point |",
        "| --- | ---: | ---: | ---: |",
    ]
    for o in outcomes:
        ratio = o.mean_cost_usd / max(o.mean_accuracy, 1e-9)
        lines.append(
            f"| {o.strategy} | {o.mean_accuracy:.3f} | {o.mean_cost_usd:.4f} | {ratio:.4f} |"
        )
    council = next(o for o in outcomes if o.strategy == "council")
    ensemble = next(o for o in outcomes if o.strategy == "ensemble")
    cost_reduction = 1.0 - council.mean_cost_usd / max(ensemble.mean_cost_usd, 1e-9)
    acc_delta = council.mean_accuracy - ensemble.mean_accuracy
    lines += [
        "",
        f"**Council vs ensemble:** {cost_reduction:.0%} lower cost, "
        f"accuracy {acc_delta:+.3f}.",
        "",
    ]
    (args.out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nArtifacts: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
