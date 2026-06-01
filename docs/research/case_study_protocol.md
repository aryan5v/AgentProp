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

Offline accounting dry run:

```bash
PYTHONPATH=src:. python experiments/run_case_study.py \
  --tasks benchmarks/case_study_tasks.json \
  --workflow planner_coder_tester_reviewer \
  --out-dir docs/results/case_study_offline
```

The offline runner does not call an LLM. It validates the study schema by
writing `results.json`, `results.csv`, `summary.json`, and `traces.jsonl` for
the broadcast, optimized-greedy, ML message-passing, and PPO routing arms.

Real LLM execution:

```bash
export TOKEN_ROUTER_API_KEY=...
export TOKEN_ROUTER_BASE_URL=...
export TOKEN_ROUTER_MODEL=...

PYTHONPATH=src:. python experiments/run_case_study.py \
  --execution-mode llm \
  --tasks benchmarks/case_study_tasks.json \
  --workflow planner_coder_tester_reviewer \
  --out-dir docs/results/case_study_001
```

The LLM mode uses an OpenAI-compatible chat endpoint. It records prompt tokens,
completion tokens, total LLM tokens, latency, selected seeds, quality rubric
scores, traces, and `outputs.jsonl` with prompts and final model responses.
Keep the result directory private until a redaction pass confirms no secrets or
private task data are present.

Analyze saved results:

```bash
PYTHONPATH=src:. python experiments/analyze_case_study.py \
  --results docs/results/case_study_001/results.json \
  --out-dir docs/results/case_study_001
```

The analyzer writes `analysis.json`, `policy_comparison.csv`, `analysis.md`,
`token_savings_by_policy.svg`, and `quality_by_policy.svg`.

## Public-Release Gate

Do not make the repository public until this protocol has at least one completed
case-study result directory or the release notes clearly label the project as an
alpha framework pending real LLM validation.
