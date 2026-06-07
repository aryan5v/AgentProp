# AgentProp Coding-Agent Plugin

This bundle packages AgentProp for Codex and Claude Code users. It includes:

- the `agentprop-workflow-optimizer` skill
- the local `agentprop-mcp` stdio server configuration
- a Codex plugin manifest
- a Claude Code plugin manifest
- starter prompts for graph analysis, routing, verifier placement, and runtime
  control

The plugin expects AgentProp to be installed in the environment where the host
starts MCP servers:

```bash
python -m pip install "agentprop[mcp]"
agentprop doctor --tier graph
```

Install in Codex:

```bash
codex plugin marketplace add aryan5v/AgentProp \
  --sparse .agents \
  --sparse integrations/coding-agents/agentprop
codex plugin add agentprop@agentprop
```

Then start a new Codex thread and ask Codex to use `@agentprop` or the bundled
`$agentprop-workflow-optimizer` skill.

Install in Claude Code:

```bash
claude plugin marketplace add aryan5v/AgentProp \
  --sparse .claude-plugin \
  --sparse integrations/coding-agents/agentprop
claude plugin install agentprop
```

In a new Claude Code session, ask Claude to use the AgentProp skill and MCP
tools while analyzing or wrapping the workflow.

For direct MCP setup without the plugin wrapper:

```bash
codex mcp add agentprop -- agentprop-mcp
claude mcp add agentprop -- agentprop-mcp
codex mcp list
claude mcp list
```
