# Editor-agent distribution

Install surfaces for Codex, Claude Code, skills.sh, and related coding agents.

| Path | Purpose |
| --- | --- |
| [skills/agentprop-workflow-optimizer/](skills/agentprop-workflow-optimizer/) | **Canonical** skill — edit here first |
| [plugins/agentprop/](plugins/agentprop/) | Same-repo plugin bundle (skill copy + MCP config) |
| [wrappers/claude-code/](wrappers/claude-code/) | Thin Claude Code skill path |
| [wrappers/codex/](wrappers/codex/) | Codex agent brief template |

Repo marketplaces (must stay at repository root): `.agents/plugins/marketplace.json`,
`.claude-plugin/marketplace.json`.

```bash
codex plugin marketplace add aryan5v/AgentProp --sparse .agents --sparse distribution/plugins
claude plugin marketplace add aryan5v/AgentProp --sparse .claude-plugin distribution/plugins
```

Details: [plugin distribution](../docs/plugin_distribution.md).
