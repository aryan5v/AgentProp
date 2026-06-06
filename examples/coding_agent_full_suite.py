"""Full-suite coding-agent wrapper example.

This is the beta path for developers who want AgentProp to help with day-to-day
Codex or Claude Code runs, not just benchmark experiments.

AgentProp does not execute Codex or Claude here. Your host loop executes the
agent and emits one ExecutionEvent per meaningful step; AgentProp analyzes the
workflow graph, recommends context/verifier structure, returns control
decisions, and writes auditable artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentprop import ControlSession, ExecutionEvent
from agentprop.algorithms import (
    bottleneck_nodes,
    greedy_seed_selection,
    low_weight_edges,
    metric_dimension_verifier_placement,
    resolving_coverage,
)
from agentprop.evaluation import compare_routing, report_to_dict, write_report
from agentprop.integrations.context_advisor import ContextExpansionAdvisor, bundle_from_advice
from agentprop.propagation import IndependentCascade
from agentprop.runtime.critical_facts import CriticalFactStore, extract_critical_facts
from agentprop.workflows import planner_coder_tester_reviewer

TASK = "Implement parser edge-case handling and prove it with pytest."

SHARED_CONTEXT = """
The parser must preserve the public function signature `parse_payload(raw: str) -> dict`.
Never silently drop malformed fields; return an `errors` list instead.
The verifier is `pytest tests/test_parser_edges.py -q` and must pass before finalization.
If the implementation changes the schema, update the tests and explain the migration risk.
"""


def run(out_dir: str | Path = "reports/beta-coding-agent-full-suite") -> dict[str, Path]:
    """Run a deterministic, key-free full-suite AgentProp wrapper demo."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    graph = planner_coder_tester_reviewer()
    propagation_model = IndependentCascade(seed=0)

    seeds = greedy_seed_selection(
        graph,
        2,
        propagation_model=propagation_model,
        trials=30,
        protect_critical_nodes=True,
    )
    propagation = propagation_model.simulate(graph, seeds, trials=30)
    verifiers = metric_dimension_verifier_placement(graph, 2)
    recommendation = compare_routing(
        graph,
        seeds,
        propagation_model.name,
        propagation,
        bottlenecks=bottleneck_nodes(graph, limit=3),
        pruning_candidates=low_weight_edges(graph, fraction=0.4),
        verifier_candidates=verifiers,
    )

    routing_report = write_report(
        recommendation,
        output / "routing_report.md",
        workflow_name="planner_coder_tester_reviewer",
    )
    (output / "routing_summary.json").write_text(
        json.dumps(report_to_dict(recommendation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fact_store = CriticalFactStore()
    fact_store.global_facts.extend(extract_critical_facts(SHARED_CONTEXT, task=TASK)[:4])
    advisor = ContextExpansionAdvisor(graph, fact_store=fact_store)
    target_node = "coder" if graph.has_node("coder") else seeds[0]
    current_ratio = recommendation.context_allocations.get(target_node, 0.35)
    advice = advisor.should_expand(target_node, current_ratio=current_ratio, task=TASK)
    bundle = bundle_from_advice(
        shared_context=SHARED_CONTEXT,
        task=TASK,
        node_id=target_node,
        advice=advice,
        fact_store=fact_store,
    )
    (output / "context_advice.json").write_text(
        json.dumps(
            {
                "target_node": target_node,
                "advice": advice.to_dict(),
                "visible_context": bundle.render(ratio=advice.target_ratio),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    session = ControlSession.start(
        graph,
        task_id="codex-or-claude-demo",
        category="implementation",
        token_budget=35_000,
        wall_time_budget_s=900,
        baseline_tokens=int(recommendation.broadcast_cost.token_cost),
    )
    session.enable_dynamic_graph()
    session.mutate_add_conditional_edge(
        "coder",
        "tester",
        condition_key="tests_changed",
        condition_value=True,
        relevance=0.95,
        reliability=0.92,
        message_cost=120,
    )

    events = [
        ExecutionEvent(
            step=1,
            command="codex plan parser edge cases",
            progress_made=True,
            tokens_used=2_200,
            elapsed_s=45,
        ),
        ExecutionEvent(
            step=2,
            command="codex edit parser implementation",
            progress_made=True,
            tokens_used=9_600,
            elapsed_s=210,
        ),
        ExecutionEvent(
            step=3,
            command="pytest tests/test_parser_edges.py -q",
            verifier_run=True,
            verifier_passed=False,
            error_signature="AssertionError:test_malformed_field_keeps_error",
            tokens_used=1_400,
            elapsed_s=40,
        ),
        ExecutionEvent(
            step=4,
            command="codex fix failing edge-case assertion",
            progress_made=True,
            tokens_used=5_500,
            elapsed_s=130,
        ),
        ExecutionEvent(
            step=5,
            command="pytest tests/test_parser_edges.py -q",
            verifier_run=True,
            verifier_passed=True,
            final_answer_written=True,
            trusted=True,
            tokens_used=1_000,
            elapsed_s=35,
        ),
    ]

    decisions = []
    for event in events:
        decision = session.observe(event)
        decisions.append({"step": event.step, "action": decision.action, "reason": decision.reason})

    session.record_outcome(
        passed=True,
        quality_score=1.0,
        metadata={
            "seeds": seeds,
            "verifiers": verifiers,
            "resolving_coverage": resolving_coverage(graph, verifiers),
            "routing_savings": recommendation.estimated_savings,
        },
    )
    control_paths = session.write_artifacts(output / "control_session")

    prompt = _host_agent_prompt(seeds=seeds, verifiers=verifiers, advice=advice.to_dict())
    prompt_path = output / "host_agent_prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    decisions_path = output / "decisions.json"
    decisions_path.write_text(
        json.dumps(decisions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "routing_report": routing_report,
        "routing_summary": output / "routing_summary.json",
        "context_advice": output / "context_advice.json",
        "host_agent_prompt": prompt_path,
        "decisions": decisions_path,
        **{f"control_{name}": path for name, path in control_paths.items()},
    }


def _host_agent_prompt(*, seeds: list[str], verifiers: list[str], advice: dict[str, object]) -> str:
    return "\n".join(
        [
            "# AgentProp Host-Agent Brief",
            "",
            "Use this with Codex CLI or Claude Code after normal login.",
            "",
            f"- Full-context seed agents: `{', '.join(seeds)}`",
            f"- Required verifier/checkpoint nodes: `{', '.join(verifiers)}`",
            f"- Context expansion advice: `{json.dumps(advice, sort_keys=True)}`",
            "- Treat verifier success as a hard gate before finalization.",
            "- Emit one `ExecutionEvent` after each plan/edit/test/review step.",
            "- If AgentProp returns `FORCE_VERIFY`, run the independent verifier.",
            "- If AgentProp returns `SWITCH_STRATEGY`, change approach before "
            "spending more tokens.",
            "- Save `trace.jsonl`, `summary.json`, `report.md`, and the "
            "verification command output.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the key-free AgentProp full-suite coding-agent demo."
    )
    parser.add_argument(
        "--out-dir",
        default="reports/beta-coding-agent-full-suite",
        help="Directory where demo artifacts will be written.",
    )
    args = parser.parse_args()

    paths = run(args.out_dir)
    print("AgentProp full-suite coding-agent artifacts:")
    for name, path in sorted(paths.items()):
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
