#!/usr/bin/env python3
"""Run the four arms on a DRACO subset and grade them (needs OPENROUTER_API_KEY).

Arms:
  A0  single budget model + web search
  A1  single frontier model + web search
  B0  Fusion-parity ensemble (whole pool answers the whole task -> synth)
  B1  AgentProp Council (decompose + assign + claim-check + synth)

Writes per-task scores, token/cost/latency, and a cost-vs-accuracy summary to
docs/results/draco_council/. Without a key it prints setup instructions and
exits 0 so CI stays green.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentprop.council import (
    Council,
    LLMPlanner,
    ModelPool,
    ModelSpec,
    OpenRouterWebSearch,
    Synthesizer,
)
from agentprop.council.model_pool import ModelResponse
from agentprop.evaluation.draco import (
    DracoTask,
    JudgeFn,
    grade_response,
    load_draco_jsonl,
)
from agentprop.evaluation.intervals import bootstrap_mean_interval

# Budget panel (OpenRouter slugs) + a frontier reference. Prices are per-Mtok.
BUDGET_SPECS = (
    ModelSpec("google/gemini-3-flash-preview", 0.10, 0.40, capability_tier=1, tags=("search",)),
    ModelSpec("moonshotai/kimi-k2.6", 0.30, 0.50, capability_tier=2, tags=("search",)),
    ModelSpec("deepseek/deepseek-v4-pro", 0.40, 0.80, capability_tier=2, tags=("search",)),
)
FRONTIER_SPEC = ModelSpec(
    "anthropic/claude-opus-4-6", 5.0, 25.0, capability_tier=3, tags=("search",)
)
JUDGE_MODEL = "google/gemini-3-pro-preview"
JUDGE_SPEC = ModelSpec(JUDGE_MODEL, 0.50, 1.50, capability_tier=2)

_JUDGE_SYSTEM = (
    "You are a strict grader. Given a QUERY, a RESPONSE, and a single CRITERION, "
    "answer with exactly MET or UNMET. MET means the criterion is satisfied by "
    "the response. Answer only the single word."
)


def make_judge(pool: ModelPool) -> JudgeFn:
    def judge(query: str, response: str, criterion: str) -> bool:
        result = pool.call(
            JUDGE_MODEL,
            system_prompt=_JUDGE_SYSTEM,
            user_prompt=f"QUERY:\n{query}\n\nRESPONSE:\n{response}\n\nCRITERION:\n{criterion}",
            temperature=0.0,
            max_tokens=4,
        )
        # A failed judge call would silently corrupt every score it touches.
        if result.error:
            raise RuntimeError(f"judge call failed: {result.error}")
        return result.text.strip().upper().startswith("MET")

    return judge


def _single(pool: ModelPool, model: str, task: DracoTask) -> tuple[str, float, float, int]:
    web = OpenRouterWebSearch().for_subtask(task.query)
    resp: ModelResponse = pool.call(
        model,
        system_prompt="You are a deep-research analyst. Produce a cited report.",
        user_prompt=task.query,
        extra_body=web.extra_body or None,
    )
    if resp.error:
        raise RuntimeError(f"model call to {model} failed: {resp.error}")
    return resp.text, resp.cost_usd, resp.latency_s, resp.usage.total_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=False,
                        help="DRACO tasks JSONL (download from HF first)")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/results/draco_council"))
    parser.add_argument("--arms", nargs="+", default=["A0", "A1", "B0", "B1"])
    args = parser.parse_args()

    if not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("Set OPENROUTER_API_KEY to run live DRACO arms. Nothing to do.")
        print("Download tasks first:  hf.co/datasets/perplexity-ai/draco -> JSONL")
        return 0
    if not args.tasks or not args.tasks.exists():
        print("Provide --tasks pointing at a DRACO JSONL export.")
        return 0

    tasks = load_draco_jsonl(args.tasks)[: args.limit]
    pool = ModelPool(specs=(*BUDGET_SPECS, FRONTIER_SPEC, JUDGE_SPEC))
    budget_pool = ModelPool(specs=BUDGET_SPECS, api_key=pool.api_key)
    judge = make_judge(pool)

    council = Council(
        pool=budget_pool,
        planner=LLMPlanner(model="google/gemini-3-flash-preview"),
        synthesizer=Synthesizer(model="deepseek/deepseek-v4-pro"),
        retrieval=OpenRouterWebSearch(),
    )
    ensemble = Council(
        pool=budget_pool,
        planner=LLMPlanner(model="google/gemini-3-flash-preview"),
        synthesizer=Synthesizer(model="deepseek/deepseek-v4-pro"),
        retrieval=OpenRouterWebSearch(),
        confidence_threshold=2.0,  # force the ensemble path always
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_file = args.out_dir / "rows.jsonl"
    rows_file.write_text("", encoding="utf-8")  # fresh run
    rows: list[dict] = []
    for task in tasks:
        for arm in args.arms:
            if arm == "A0":
                text, cost, latency, tokens = _single(
                    pool, "google/gemini-3-flash-preview", task
                )
            elif arm == "A1":
                text, cost, latency, tokens = _single(pool, FRONTIER_SPEC.name, task)
            elif arm in ("B0", "B1"):
                result = (ensemble if arm == "B0" else council).run(
                    task.query, task_id=f"{arm}-{task.task_id}"
                )
                text, cost, latency, tokens = (
                    result.answer, result.total_cost_usd, result.wall_latency_s,
                    result.total_tokens,
                )
            else:
                continue
            score = grade_response(task, text, judge)
            row = {
                "arm": arm, "task_id": task.task_id, "domain": task.domain,
                "normalized_score": score.normalized_score, "pass_rate": score.pass_rate,
                "cost_usd": cost, "latency_s": latency, "tokens": tokens,
            }
            rows.append(row)
            # Append per result so a long run survives interruptions.
            with rows_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"{arm} {task.task_id}: score={score.normalized_score:.1f} "
                  f"cost=${cost:.3f} {latency:.0f}s")

    _write_summary(rows, args.out_dir)
    print(f"\nArtifacts: {args.out_dir}")
    return 0


def _write_summary(rows: list[dict], out_dir: Path) -> None:
    arms = sorted({r["arm"] for r in rows})
    lines = ["# DRACO subset: cost vs accuracy", "",
             "| Arm | Norm score | Pass rate | Cost (USD) | Latency (s) |",
             "| --- | ---: | ---: | ---: | ---: |"]
    summary = []
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        score = bootstrap_mean_interval([r["normalized_score"] for r in ar], seed=0)
        pass_rate = bootstrap_mean_interval([r["pass_rate"] for r in ar], seed=0)
        cost = bootstrap_mean_interval([r["cost_usd"] for r in ar], seed=0)
        latency = bootstrap_mean_interval([r["latency_s"] for r in ar], seed=0)
        lines.append(
            f"| {arm} | {score.mean:.1f} [{score.lower:.1f},{score.upper:.1f}] | "
            f"{pass_rate.mean:.1f} | {cost.mean:.3f} | {latency.mean:.0f} |"
        )
        summary.append({"arm": arm, "norm_score": score.mean, "cost_usd": cost.mean})
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
