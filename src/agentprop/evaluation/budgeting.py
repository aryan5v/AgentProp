"""Budget policies for benchmark and coding-agent execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentBudgetPolicy:
    """Bounded execution policy for a task category."""

    category: str
    planning_steps: int
    probe_commands: int
    candidate_sweeps: int
    verification_runs: int
    repair_cycles: int
    max_wall_time_s: int
    stop_rule: str

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "planning_steps": self.planning_steps,
            "probe_commands": self.probe_commands,
            "candidate_sweeps": self.candidate_sweeps,
            "verification_runs": self.verification_runs,
            "repair_cycles": self.repair_cycles,
            "max_wall_time_s": self.max_wall_time_s,
            "stop_rule": self.stop_rule,
        }


DEFAULT_BUDGET_POLICIES: tuple[AgentBudgetPolicy, ...] = (
    AgentBudgetPolicy(
        category="direct-answer",
        planning_steps=1,
        probe_commands=2,
        candidate_sweeps=3,
        verification_runs=1,
        repair_cycles=0,
        max_wall_time_s=300,
        stop_rule="Enumerate required outputs, verify exact format, write the answer, then stop.",
    ),
    AgentBudgetPolicy(
        category="setup-build",
        planning_steps=2,
        probe_commands=5,
        candidate_sweeps=2,
        verification_runs=2,
        repair_cycles=2,
        max_wall_time_s=1_800,
        stop_rule="Switch from discovery to implementation once the failing command is known.",
    ),
    AgentBudgetPolicy(
        category="code-repair",
        planning_steps=2,
        probe_commands=6,
        candidate_sweeps=2,
        verification_runs=2,
        repair_cycles=2,
        max_wall_time_s=1_800,
        stop_rule=(
            "Run the verifier once, repair only the failing delta, re-run once, then finalize."
        ),
    ),
    AgentBudgetPolicy(
        category="numerical-scientific",
        planning_steps=2,
        probe_commands=4,
        candidate_sweeps=6,
        verification_runs=2,
        repair_cycles=1,
        max_wall_time_s=1_500,
        stop_rule="Validate units/schema first; stop when the evaluator accepts a valid candidate.",
    ),
    AgentBudgetPolicy(
        category="reverse-engineering",
        planning_steps=2,
        probe_commands=8,
        candidate_sweeps=3,
        verification_runs=2,
        repair_cycles=1,
        max_wall_time_s=2_400,
        stop_rule="After bounded probes, commit to the simplest falsifiable hypothesis and verify.",
    ),
    AgentBudgetPolicy(
        category="repo-hygiene",
        planning_steps=1,
        probe_commands=4,
        candidate_sweeps=1,
        verification_runs=2,
        repair_cycles=1,
        max_wall_time_s=1_200,
        stop_rule="Prefer minimal diffs and stop once repository checks pass.",
    ),
)


def budget_policy_by_category(category: str) -> AgentBudgetPolicy:
    """Return the named policy, falling back to setup/build for unknown categories."""

    normalized = category.strip().lower().replace("_", "-")
    for policy in DEFAULT_BUDGET_POLICIES:
        if policy.category == normalized:
            return policy
    return DEFAULT_BUDGET_POLICIES[1]


def render_budget_policy_markdown(
    policies: tuple[AgentBudgetPolicy, ...] = DEFAULT_BUDGET_POLICIES,
) -> str:
    """Render budget policies as compact benchmark-agent instructions."""

    lines = ["## Budget-Aware Stop Conditions", ""]
    lines.extend(
        [
            "Before acting, classify the task and pick the closest budget row. These are",
            "upper bounds, not targets; stop earlier when the verifier signal is already clear.",
            "",
            "| Category | Plan | Probes | Sweeps | Verify | Repairs | Wall time | Stop rule |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for policy in policies:
        lines.append(
            f"| `{policy.category}` | {policy.planning_steps} | {policy.probe_commands} | "
            f"{policy.candidate_sweeps} | {policy.verification_runs} | "
            f"{policy.repair_cycles} | {policy.max_wall_time_s}s | {policy.stop_rule} |"
        )
    lines.extend(
        [
            "",
            "Finalizer rule: if executable checks exist, run the narrowest relevant check once,",
            "repair only the failing delta, and avoid broad re-exploration after a passing check.",
        ]
    )
    return "\n".join(lines)
