# Real LLM Case-Study Protocol

This protocol defines the first real LLM workflow study for AgentProp.

## Goal

Measure whether AgentProp can reduce context/message cost in a multi-agent LLM
workflow while preserving output quality.

## Workflow

Use a planner-coder-tester-reviewer software task workflow:

1. Planner decomposes the issue and routes task context.
2. Coder implements the change.
3. Tester runs or proposes verification.
4. Reviewer checks correctness, maintainability, and missed edge cases.
5. Final node summarizes the accepted answer.

This maps directly to `benchmarks/workflows/planner_coder_tester_reviewer.json`.

## Study Arms

- Broadcast baseline: every non-output agent receives full task context.
- AgentProp training-free routing: use `agentprop optimize` with budget 2.
- AgentProp learned routing: use the torch GNN scorer or Q-learning policy once
  trained on benchmark templates.

## Task Set

Use 20 small engineering tasks:

- 8 bug-fix tasks with clear failing behavior.
- 6 feature-addition tasks touching one or two files.
- 3 documentation or developer-experience tasks.
- 3 refactor tasks with tests preserved.

Each task should fit in one working session and have an objective verification
command such as `pytest`, a CLI smoke test, or expected output diff.

## Measurements

Record these fields per task and study arm:

- selected seed agents
- total prompt/input tokens
- total output tokens
- message count
- wall-clock latency
- verification command and result
- human quality score from 1 to 5
- whether the final answer missed required behavior
- whether reviewer/tester caught a real issue

Use [../quality_scoring.md](../quality_scoring.md) to record exact-match,
human-label, rubric, or LLM-as-judge quality scores with method metadata.

## Acceptance Criteria

AgentProp is considered successful if, relative to broadcast:

- median total token cost is at least 20% lower
- verification pass rate is not worse by more than 5 percentage points
- median human quality score is not worse by more than 0.25
- at least one workflow shows a clear verifier-placement or pruning insight

## Reproducibility Artifacts

For each run, save:

- workflow JSON
- trace JSONL
- AgentProp report Markdown
- benchmark row CSV/JSON
- final LLM output
- verification command output

Suggested directory:

```text
docs/results/case_study_001/
```

## Commands

```bash
agentprop optimize benchmarks/workflows/planner_coder_tester_reviewer.json --budget 2 --trials 50
agentprop report planner_coder_tester_reviewer --budget 2 --out docs/results/case_study_001/report.md
agentprop benchmark planner_coder_tester_reviewer --budget 2 --trials 50 --json
```

## Public-Release Gate

Do not make the repository public until this protocol has at least one completed
case-study result directory or the release notes clearly label the project as an
alpha framework pending real LLM validation.
