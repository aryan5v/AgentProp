# AgentProp Codex Plugin

This plugin packages AgentProp for Codex users. It includes:

- the `agentprop-workflow-optimizer` skill
- the local `agentprop-mcp` stdio server configuration
- starter prompts for graph analysis, routing, verifier placement, and runtime
  control

The plugin expects AgentProp to be installed in the environment where Codex
starts MCP servers:

```bash
python -m pip install "agentprop[mcp]"
agentprop doctor --tier graph
```

Install the repo marketplace from this repository:

```bash
codex plugin marketplace add aryan5v/AgentProp --sparse .agents --sparse plugins
```

Then open Codex, browse the AgentProp marketplace, and install the AgentProp
plugin. In a new thread, ask Codex to use `@agentprop` or the bundled
`$agentprop-workflow-optimizer` skill.

For direct MCP setup without the plugin wrapper:

```bash
codex mcp add agentprop -- agentprop-mcp
codex mcp list
```
