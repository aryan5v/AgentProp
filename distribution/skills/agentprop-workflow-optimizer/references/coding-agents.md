# Coding-Agent Use

Use this when Codex, Claude Code, Cursor, Copilot, or another coding agent should
use AgentProp guidance while implementing or reviewing a workflow.

## Generate A Brief

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

Give the generated Markdown to the coding agent before it starts work.

## How The Agent Should Use The Brief

- Send full context first to the selected seed agents.
- Preserve high-risk context for coder/tester/verifier-style roles.
- Use verifier candidates as required review or execution checkpoints.
- If summarizing a handoff, state the quality risk and run verification.
- If the same error repeats, switch strategy instead of repeating the same
  command.
- If an agent claims a local pass, independently verify before finalizing.

## Codex Login

AgentProp does not replace Codex auth:

```bash
codex login
codex exec "Use reports/codex_agent_brief.md while implementing this workflow change."
```

## Claude Code

Claude Code can use either the generated brief or the MCP server:

```bash
python -m pip install "agentprop[mcp]"
agentprop-mcp
```

Use MCP when the agent should call AgentProp tools live rather than only reading
a static brief.

## Final Response

Include:

- workflow analyzed
- seed agents
- verifier/checker used
- artifact paths
- verification command and result
