# AgentProp Workflow Optimizer Skill

[![skills.sh](https://www.skills.sh/b/aryan5v/AgentProp)](https://www.skills.sh/aryan5v/AgentProp)

This skill teaches coding agents how to install and use AgentProp as a graph
analysis and control layer for multi-agent workflows.

Use it when an agent is implementing, reviewing, benchmarking, or debugging:

- agent handoff graphs
- context-routing policies
- verifier placement
- pruning or summarization choices
- LangGraph/CrewAI/AutoGen-style workflow adapters
- Codex, Claude Code, or other coding-agent team workflows

## What The Skill Does

The skill instructs agents to:

1. Install or locate the `agentprop` CLI.
2. Generate a workflow brief with `agentprop agent-instructions`.
3. Analyze graph bottlenecks, verifier candidates, and pruning risk.
4. Use the guidance while changing the workflow.
5. Run verification and preserve artifacts before claiming success.
6. Optionally use FastMCP tools or `ControlSession` demos for runtime control.

## Install AgentProp

```bash
python -m pip install agentprop
```

For MCP/editor-agent tools:

```bash
python -m pip install "agentprop[mcp]"
agentprop-mcp
```

For source development:

```bash
git clone https://github.com/aryan5v/AgentProp.git
cd AgentProp
python -m pip install -e ".[dev,mcp]"
```

## Core Agent Commands

Generate a Claude Code brief:

```bash
agentprop agent-instructions planner_coder_tester_reviewer \
  --target claude-code \
  --budget 2 \
  --trials 50 \
  --out reports/claude_code_agent_brief.md
```

Generate a Codex brief:

```bash
agentprop agent-instructions planner_coder_tester_reviewer \
  --target codex \
  --budget 2 \
  --trials 50 \
  --out reports/codex_agent_brief.md
```

Analyze and report:

```bash
agentprop analyze planner_coder_tester_reviewer --json
agentprop report planner_coder_tester_reviewer \
  --out reports/agentprop_report.html \
  --format html
```

Run a key-free runtime-control demo:

```bash
agentprop control-demo --demo terminal --out-dir reports/control-demo
```

## Skill Files

- [`SKILL.md`](SKILL.md): the skill definition loaded by compatible coding
  agents.
- [`agents/openai.yaml`](agents/openai.yaml): UI metadata for Codex-compatible
  skill listings.

## Listing

skills.sh documents GitHub-hosted skills with a `SKILL.md` and README. The badge
above follows the documented install-count badge format for `owner/repo`.
