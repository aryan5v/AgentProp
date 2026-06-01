---
name: agentprop-workflow-optimizer
description: Use when optimizing, implementing, reviewing, or debugging a multi-agent workflow with AgentProp. Generates workflow routing briefs, analyzes bottlenecks, places verifiers, checks pruning risk, and runs ML/RL policy suites before changing workflow defaults.
---

# AgentProp Workflow Optimizer

Use AgentProp before changing a multi-agent workflow, agent handoff graph, verifier
layout, or routing policy.

## Quick Workflow

1. Identify the workflow JSON or built-in workflow name.
2. Generate a coding-agent brief:

```bash
agentprop agent-instructions <workflow> --target claude-code --out reports/claude_code_agent_brief.md
```

3. Read the generated brief and follow its seed, verifier, bottleneck, and
   pruning guidance.
4. Implement or review the task.
5. Run the relevant verification command.
6. Save or mention the AgentProp artifacts used as evidence.

## Commands

Analyze graph structure:

```bash
agentprop analyze <workflow.json> --json
```

Optimize context seeds:

```bash
agentprop optimize <workflow.json> --budget 2 --trials 50 --json
```

Write a report:

```bash
agentprop report <workflow> --out reports/agentprop_report.html --format html
```

Run the ML/RL recipe suite before changing policy defaults:

```bash
PYTHONPATH=src:. python experiments/run_experiment_suite.py \
  --config configs/experiment_suites/ml_core.json \
  --artifact-root results/ml_core
```

Analyze saved case-study results:

```bash
PYTHONPATH=src:. python experiments/analyze_case_study.py \
  --results docs/results/case_study_001/results.json \
  --out-dir docs/results/case_study_001
```

Prepare an external Terminal-Bench run without launching it:

```bash
agentprop terminal-bench prepare \
  --dataset terminal-bench/terminal-bench-2-1 \
  --agent terminus-2 \
  --model google/gemini-3.1-pro-preview \
  --environment modal \
  --out-dir benchmark-results/terminal-bench-2.1
```

## Rules For The Coding Agent

- Do not claim a learned ML/RL policy is better unless a saved artifact compares
  it against training-free baselines.
- Do not prune edges that carry verification, tool output, or user constraints
  into the final answer unless the report shows acceptable risk.
- For benchmark work, prepare manifests and watchdog wrappers before launching
  expensive external runs.
- Use verifier nodes to intercept errors before finalizing.
- Mention selected seed agents and verification evidence in the final response.
- Keep Token Router, Modal, OpenAI, and Hugging Face credentials in environment
  variables or ignored local config only.
