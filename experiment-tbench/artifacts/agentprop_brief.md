# AgentProp Brief For Claude Code

Workflow: `planner_coder_tester_reviewer`
Propagation model: `independent-cascade`

## Routing Decision

Give full task context first to these seed agents:

- `coder`
- `tester`

Use selective context passing for the rest of the workflow. Avoid broadcast routing unless the task is safety-critical, ambiguous, or the optimized coverage is too low.

## Why This Routing Plan

- Estimated coverage: `80.0%`
- Estimated savings vs broadcast: `37.5%`
- Optimized total cost: `4811`
- Broadcast total cost: `7700`

## Verifier Placement

Prefer these verifier/checker nodes when asking the coding agent to review, test, or intercept mistakes:

- `tester`
- `planner`

Verifier semantics for the coding agent:

- Observe: task context, selected seed outputs, changed files, and test output.
- Correct: implementation mistakes, missing tests, wrong assumptions, and final-answer gaps.
- Intercept: stop propagation when a verifier finds a failing test, unsafe change, or contradicted requirement.

## Bottlenecks And Pruning

Treat these bottlenecks as high-attention handoff points:

- `reviewer`: `0.443`
- `tester`: `0.367`
- `coder`: `0.328`
- `planner`: `0.160`
- `final`: `0.063`

Candidate communication edges to prune or summarize:

- `planner` -> `reviewer`
- `tester` -> `coder`
- `tester` -> `reviewer`

Do not prune an edge if it is the only path carrying verification, user constraints, or tool output into the final answer.

## ML/DL/RL Follow-Up

When improving this workflow rather than executing one task, run the reproducible ML/RL suite:

```bash
PYTHONPATH=src:. python experiments/run_experiment_suite.py \
  --config configs/experiment_suites/ml_core.json \
  --artifact-root results/ml_core
```

Compare learned policies against PageRank, CELF, greedy, message-passing GNN-style scoring, Q-learning, REINFORCE, and PPO before changing defaults.

## Suggested Agent Prompt

Claude Code, execute this task using AgentProp workflow `planner_coder_tester_reviewer`. Send full context first to `coder, tester`. Use verifier/checker nodes `tester, planner` before finalizing. Preserve user requirements, run the relevant verification command, and summarize token/cost-sensitive routing decisions in the final response.

## Required Evidence Before Finishing

- State which seed agents received full context.
- State which verifier/checker reviewed the work.
- Include command output or saved artifact paths.
- Report whether any pruning/summarization changed task quality.
- If using LLM execution, save traces and token counts before claiming a cost win.
