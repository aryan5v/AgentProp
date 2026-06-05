---
name: agentprop-workflow-optimizer
description: Use when installing or using AgentProp to analyze, optimize, implement, review, or debug multi-agent workflows. Guides agents to install the AgentProp CLI, generate routing briefs, analyze graph bottlenecks, place verifiers, assess pruning risk, use MCP/control demos, and preserve evidence before changing workflow defaults.
---

# AgentProp Workflow Optimizer

Use this skill when a task involves a multi-agent workflow, agent handoff graph,
context-routing policy, verifier layout, framework adapter, coding-agent team,
or benchmark harness.

AgentProp is not the agent runtime. It is the graph analysis and control layer:
analyze the workflow, recommend where context and verifiers should go, then use
those recommendations while implementing or reviewing the workflow.

## 1. Install Or Locate AgentProp

First check whether the CLI is available:

```bash
agentprop --help
```

If missing, install the package:

```bash
python -m pip install agentprop
```

For a source checkout:

```bash
python -m pip install -e ".[dev]"
```

For MCP/editor-agent tools:

```bash
python -m pip install "agentprop[mcp]"
agentprop-mcp
```

If working inside the AgentProp repo without an editable install, prefix commands
with `PYTHONPATH=src`.

## 2. Identify The Workflow

Use a built-in workflow name when possible:

- `planner_coder_tester_reviewer`
- `research_writer_verifier`
- `rag_pipeline`
- `tool_use_pipeline`
- `chain`, `tree`, `star`, `dense_graph`, `small_world_graph`, `generic_dag`

Otherwise use a workflow JSON path. If no graph exists yet, sketch one before
optimizing: nodes should be agents, tools, verifiers, memory stores, or outputs;
edges should be handoffs, context packets, tool outputs, or verification paths.

## 3. Generate The Agent Brief

For Claude Code:

```bash
agentprop agent-instructions <workflow> \
  --target claude-code \
  --budget 2 \
  --trials 50 \
  --out reports/claude_code_agent_brief.md
```

For Codex:

```bash
agentprop agent-instructions <workflow> \
  --target codex \
  --budget 2 \
  --trials 50 \
  --out reports/codex_agent_brief.md
```

Read the generated brief before editing. Treat selected seed agents as the first
full-context recipients. Treat verifier candidates as required check points.

## 4. Analyze And Report

Run structure diagnostics:

```bash
agentprop analyze <workflow> --json
```

Optimize context routing:

```bash
agentprop optimize <workflow> \
  --budget 2 \
  --algorithm greedy \
  --model rzf \
  --trials 50 \
  --json
```

Write a durable report:

```bash
agentprop report <workflow> \
  --out reports/agentprop_report.html \
  --format html
```

Before pruning or summarizing edges, run:

```bash
agentprop prune <workflow> \
  --target-token-reduction 0.3 \
  --model rzf \
  --trials 50 \
  --json
```

## 5. Use The Guidance While Building

- Give full context first to seed agents or high-sensitivity roles such as
  coder, tester, verifier, planner, or domain specialist.
- Do not starve a high-risk node with summary-only context unless the report
  says the quality risk is acceptable.
- Place verifiers downstream of compressed, pruned, or high-error handoffs.
- If an agent claims a local pass, require an independent verifier before
  finalizing.
- If the same error repeats, switch strategy instead of running the same probe
  again.
- Save the brief, report, trace, and verification output as evidence.

## 6. Optional Runtime Control

Use the key-free demos to confirm the control layer is available:

```bash
agentprop control-demo --demo terminal --out-dir reports/control-demo
agentprop control-demo --demo multi-agent --out-dir reports/control-demo
agentprop control-demo --demo framework --out-dir reports/control-demo
```

For real wrappers, emit `ExecutionEvent` rows from the host agent/runtime into
`ControlSession`. AgentProp should observe; the host agent executes.

## 7. Final Response Checklist

Before claiming done, state:

- which workflow was analyzed
- which seed agents received full context
- which verifier/checker was used
- what was pruned, summarized, or left untouched
- where the AgentProp artifacts were saved
- what verification command or independent check passed

Do not claim a learned ML/RL policy improved performance unless saved artifacts
compare it against training-free baselines.
