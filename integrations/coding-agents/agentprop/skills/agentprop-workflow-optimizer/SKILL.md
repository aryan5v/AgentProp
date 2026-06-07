---
name: agentprop-workflow-optimizer
description: Use when installing or using AgentProp to improve multi-agent workflows. Covers CLI setup, workflow graph analysis, context routing, verifier placement, pruning risk, Codex/Claude Code briefs, FastMCP tools, runtime ControlSession wrapping, and evidence preservation.
---

# AgentProp Workflow Optimizer

Guide the user through AgentProp-powered workflow improvement: install the CLI,
model or locate the workflow graph, analyze routing and verifier structure, use
the recommendations while building, and save evidence before claiming success.

AgentProp is a graph analysis and control layer for agent workflows. It is not a
replacement for Codex, Claude Code, LangGraph, CrewAI, AutoGen, or a custom
agent runtime; it helps those systems spend context more carefully and verify at
the right structural points.

## Source Of Truth

Prefer current project docs and CLI output over memory:

- Repository: `https://github.com/aryan5v/AgentProp`
- Package: `agentprop`
- CLI help: `agentprop --help`
- Docs index: `docs/index.md`
- Coding-agent guide: `docs/coding_agents.md`
- Control-layer guide: `docs/control_layer_quickstart.md`

If a command fails because the package is not installed, install or locate
AgentProp before continuing.

## Fast Path

```bash
agentprop --help || python -m pip install agentprop

agentprop agent-instructions planner_coder_tester_reviewer \
  --target claude-code \
  --budget 2 \
  --trials 50 \
  --out reports/claude_code_agent_brief.md

agentprop analyze planner_coder_tester_reviewer --json
agentprop report planner_coder_tester_reviewer \
  --out reports/agentprop_report.html \
  --format html
```

Read the generated brief before editing. Treat selected seed agents as the first
full-context recipients and verifier candidates as required check points.

## Reference Map

Load the smallest relevant reference file for the task:

Area | Resource | When to use
--- | --- | ---
Install and configure | `references/install-and-configure.md` | AgentProp CLI is missing, MCP is requested, or the user asks how to set it up
Workflow analysis | `references/workflow-analysis.md` | Need to analyze, optimize, prune, report, or model an agent workflow graph
Coding agents | `references/coding-agents.md` | Codex, Claude Code, Cursor, Copilot, or another coding agent should use AgentProp guidance
Runtime control | `references/runtime-control.md` | Need live wrapping with `ControlSession`, repeated-error control, force-verify, stop, switch, or traces
Framework builders | `references/framework-builders.md` | User is building LangGraph, CrewAI, AutoGen, OpenAI Agents, LlamaIndex, or custom multi-agent systems
Benchmarks and evidence | `references/benchmarks-and-evidence.md` | User wants benchmark setup, claims, reports, or saved evidence

## Core Rules

- Always identify the workflow graph or built-in workflow name before running
  optimization.
- Do not prune, summarize, or deprioritize verifier/tool-output/user-constraint
  edges without checking pruning risk.
- Do not trust an agent's self-reported pass as final; require an independent
  verifier when possible.
- Mention the seed agents, verifier/checker, saved artifacts, and verification
  command in the final response.
- Keep provider keys, benchmark secrets, Modal credentials, and local machine
  config out of committed files.
