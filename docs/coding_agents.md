# Coding Agent Integration

AgentProp should be used by coding agents as an analysis layer before they
change a multi-agent workflow. It does not replace Claude Code, Codex, or an
orchestrator. It tells those agents which workflow nodes should receive full
context, which verifiers should intercept mistakes, and which edges are safe
candidates for pruning or summarization.

## One-Shot Workflow Brief

Generate a Claude Code or Codex-ready brief:

```bash
agentprop agent-instructions planner_coder_tester_reviewer \
  --target codex \
  --budget 2 \
  --trials 50 \
  --out reports/codex_agent_brief.md
```

For Claude Code:

```bash
agentprop agent-instructions planner_coder_tester_reviewer \
  --target claude-code \
  --out reports/claude_code_agent_brief.md
```

Give the generated Markdown to the coding agent before it starts work. The
brief includes:

- seed agents that should receive full context first
- expected coverage and cost savings
- verifier/checker placement guidance
- bottleneck nodes and pruning candidates
- ML/RL experiment commands for workflow-improvement tasks
- required evidence before the coding agent claims the task is done

## Recommended Agent Instructions

Use this style when asking a coding agent to work on an AgentProp-modeled
workflow:

```text
Before implementing, run AgentProp on the workflow or read the attached
AgentProp brief. Send full context first only to the selected seed agents.
Use the verifier nodes before finalizing. If you prune or summarize a handoff,
state the quality risk and run the relevant verification command. Save any
trace, report, or benchmark artifact that supports the final claim.
```

For ML/DL/RL work:

```text
Use the AgentProp ML core suite before changing policy defaults. Compare any
learned scorer or RL policy against training-free baselines. Do not claim a
learned policy is better unless the saved artifact compares it against
PageRank, CELF, greedy, message-passing GNN-style scoring, and RL baselines
where applicable.
```

## Claude Code Skill Shape

A Claude Code skill should be small and command-oriented:

- Read or generate a workflow JSON file.
- Run `agentprop analyze`, `agentprop report`, and `agentprop agent-instructions`.
- Attach the generated brief to the working context.
- After implementation, run the verification command and write trace/report
  artifacts.

Suggested skill name: `agentprop-workflow-optimizer`.

## Codex Plugin Shape

A Codex plugin should expose the same core actions:

- `analyze_workflow(workflow)`
- `optimize_routing(workflow, budget, model)`
- `generate_agent_brief(workflow, target)`
- `run_ml_core_suite(artifact_root)`
- `analyze_case_study(results_path)`

The current CLI is intentionally enough to back these actions without adding a
runtime dependency.

## MCP Server Shape

An MCP server is the best long-term integration for editor agents because it
can expose AgentProp as tools instead of requiring agents to shell out.

AgentProp includes a lightweight stdio JSON-RPC server:

```bash
agentprop-mcp
```

Recommended tools:

- `agentprop_analyze`: returns graph diagnostics, bottlenecks, pruning
  candidates, and verifier candidates.
- `agentprop_optimize`: returns seed recommendations and cost/coverage deltas.
- `agentprop_report`: writes Markdown/JSON/HTML reports.
- `agentprop_agent_instructions`: returns the coding-agent brief as Markdown.
- `agentprop_run_suite`: launches configured ML/DL/RL experiment recipes.
- `agentprop_case_study_analysis`: turns saved case-study results into tables
  and plots.

The MCP server should keep secrets out of tool arguments. Token Router, OpenAI,
Modal, and Hugging Face credentials should be read from local environment
variables or uncommitted config files.

Current implemented MCP-style tools:

- `agentprop_analyze`
- `agentprop_optimize`
- `agentprop_report`
- `agentprop_agent_instructions`

Claude Code skill template:

```text
integrations/claude-code/agentprop-workflow-optimizer/SKILL.md
```

Codex instruction template:

```text
integrations/codex/AGENTPROP.md
```

## What This Enables

For a developer using Claude Code or Codex, the ideal loop is:

1. Model or import the workflow graph.
2. Generate an AgentProp brief.
3. Give the brief to the coding agent with the implementation task.
4. Let the agent implement while respecting seed/verifier/pruning guidance.
5. Run verification.
6. Save reports, traces, and case-study outputs.
7. Use ML/DL/RL suites only when optimizing policy quality, not for every
   ordinary task.
