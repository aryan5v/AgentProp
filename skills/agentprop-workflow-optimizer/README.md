# AgentProp Workflow Optimizer

[![skills.sh](https://www.skills.sh/b/aryan5v/AgentProp)](https://www.skills.sh/aryan5v/AgentProp)

AgentProp Workflow Optimizer is an agent skill for improving multi-agent
systems with graph analysis and runtime-control guidance.

It teaches coding agents how to install the AgentProp CLI, generate workflow
briefs, analyze bottlenecks, place verifiers, assess pruning risk, use FastMCP
tools, and preserve evidence before changing a multi-agent workflow.

## Install

Install into the current project for all detected agents:

```bash
npx skills add https://github.com/aryan5v/AgentProp --skill agentprop-workflow-optimizer
```

If your skills CLI supports shorthand:

```bash
npx skills add aryan5v/AgentProp --skill agentprop-workflow-optimizer
```

Install globally for Codex and Claude Code:

```bash
npx skills add aryan5v/AgentProp \
  --skill agentprop-workflow-optimizer \
  --agent codex \
  --agent claude-code \
  --global
```

List available skills in the repository without installing:

```bash
npx skills add aryan5v/AgentProp --list
```

Try the skill without installing it:

```bash
npx skills use aryan5v/AgentProp \
  --skill agentprop-workflow-optimizer \
  --agent codex
```

Install from the direct skill path:

```bash
npx skills add \
  https://github.com/aryan5v/AgentProp/tree/main/skills/agentprop-workflow-optimizer
```

## What It Covers

- AgentProp CLI installation and sanity checks
- built-in and JSON workflow graph analysis
- context seed selection and routing briefs
- verifier placement and local-pass distrust
- pruning and summarization risk checks
- Codex and Claude Code usage
- FastMCP server setup
- `ControlSession` runtime wrapping
- benchmark evidence hygiene

## Typical Agent Flow

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

The agent should read the generated brief before editing and report the seed
agents, verifier/checker, saved artifacts, and verification result before
claiming the workflow is improved.

## Skill Files

- [`SKILL.md`](SKILL.md): entry point loaded by agent skill systems.
- [`references/install-and-configure.md`](references/install-and-configure.md):
  installation, extras, MCP, and sanity checks.
- [`references/workflow-analysis.md`](references/workflow-analysis.md):
  analysis, optimization, pruning, and reporting commands.
- [`references/coding-agents.md`](references/coding-agents.md):
  Codex and Claude Code usage.
- [`references/runtime-control.md`](references/runtime-control.md):
  `ControlSession`, demos, and decision meanings.
- [`references/framework-builders.md`](references/framework-builders.md):
  LangGraph/CrewAI/AutoGen/custom workflow integration.
- [`references/benchmarks-and-evidence.md`](references/benchmarks-and-evidence.md):
  benchmark setup and responsible claim language.

## Source

AgentProp is public alpha research software. See the main repository README and
docs for current capabilities, caveats, and evidence.
