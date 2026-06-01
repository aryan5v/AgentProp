"""Instruction rendering for coding-agent integrations."""

from __future__ import annotations

from typing import Literal

from agentprop.evaluation.metrics import RecommendationReport

CodingAgentTarget = Literal["claude-code", "codex", "generic"]


def render_coding_agent_instructions(
    report: RecommendationReport,
    *,
    workflow_name: str,
    target: CodingAgentTarget = "generic",
) -> str:
    """Render a Markdown brief that coding agents can use as operating context."""

    target_label = {
        "claude-code": "Claude Code",
        "codex": "Codex",
        "generic": "Coding agent",
    }[target]
    lines = [
        f"# AgentProp Brief For {target_label}",
        "",
        f"Workflow: `{workflow_name}`",
        f"Propagation model: `{report.propagation_model}`",
        "",
        "## Routing Decision",
        "",
        "Give full task context first to these seed agents:",
        "",
        *_bullet_code(report.seeds),
        "",
        (
            "Use selective context passing for the rest of the workflow. Avoid broadcast "
            "routing unless the task is safety-critical, ambiguous, or the optimized "
            "coverage is too low."
        ),
        "",
        "## Why This Routing Plan",
        "",
        f"- Estimated coverage: `{report.propagation.coverage:.1%}`",
        f"- Estimated savings vs broadcast: `{report.estimated_savings:.1%}`",
        f"- Optimized total cost: `{report.optimized_cost.total_cost:.0f}`",
        f"- Broadcast total cost: `{report.broadcast_cost.total_cost:.0f}`",
        "",
        "## Verifier Placement",
        "",
        (
            "Prefer these verifier/checker nodes when asking the coding agent to review, "
            "test, or intercept mistakes:"
        ),
        "",
        *_bullet_code(report.verifier_candidates),
        "",
        "Verifier semantics for the coding agent:",
        "",
        "- Observe: task context, selected seed outputs, changed files, and test output.",
        (
            "- Correct: implementation mistakes, missing tests, wrong assumptions, and "
            "final-answer gaps."
        ),
        (
            "- Intercept: stop propagation when a verifier finds a failing test, unsafe "
            "change, or contradicted requirement."
        ),
        "",
        "## Bottlenecks And Pruning",
        "",
        "Treat these bottlenecks as high-attention handoff points:",
        "",
        *_ranked_bullets(report.bottlenecks),
        "",
        "Candidate communication edges to prune or summarize:",
        "",
        *_edge_bullets(report.pruning_candidates),
        "",
        (
            "Do not prune an edge if it is the only path carrying verification, user "
            "constraints, or tool output into the final answer."
        ),
        "",
        "## ML/DL/RL Follow-Up",
        "",
        (
            "When improving this workflow rather than executing one task, run the "
            "reproducible ML/RL suite:"
        ),
        "",
        "```bash",
        "PYTHONPATH=src:. python experiments/run_experiment_suite.py \\",
        "  --config configs/experiment_suites/ml_core.json \\",
        "  --artifact-root results/ml_core",
        "```",
        "",
        (
            "Compare learned policies against PageRank, CELF, greedy, message-passing "
            "GNN-style scoring, Q-learning, REINFORCE, and PPO before changing defaults."
        ),
        "",
        "## Suggested Agent Prompt",
        "",
        _suggested_prompt(report, workflow_name=workflow_name, target_label=target_label),
        "",
        "## Required Evidence Before Finishing",
        "",
        "- State which seed agents received full context.",
        "- State which verifier/checker reviewed the work.",
        "- Include command output or saved artifact paths.",
        "- Report whether any pruning/summarization changed task quality.",
        "- If using LLM execution, save traces and token counts before claiming a cost win.",
        "",
    ]
    return "\n".join(lines)


def _suggested_prompt(
    report: RecommendationReport,
    *,
    workflow_name: str,
    target_label: str,
) -> str:
    seeds = ", ".join(report.seeds) if report.seeds else "none"
    verifiers = ", ".join(report.verifier_candidates) if report.verifier_candidates else "none"
    return (
        f"{target_label}, execute this task using AgentProp workflow `{workflow_name}`. "
        f"Send full context first to `{seeds}`. Use verifier/checker nodes `{verifiers}` "
        "before finalizing. Preserve user requirements, run the relevant verification command, "
        "and summarize token/cost-sensitive routing decisions in the final response."
    )


def _bullet_code(values: list[str]) -> list[str]:
    if not values:
        return ["- `none`"]
    return [f"- `{value}`" for value in values]


def _ranked_bullets(values: list[tuple[str, float]]) -> list[str]:
    if not values:
        return ["- `none`"]
    return [f"- `{node}`: `{score:.3f}`" for node, score in values]


def _edge_bullets(values: list[tuple[str, str]]) -> list[str]:
    if not values:
        return ["- `none`"]
    return [f"- `{source}` -> `{target}`" for source, target in values]
