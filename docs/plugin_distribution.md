# Plugin Distribution

AgentProp has three install surfaces:

- **Main repo beta bundle:** this repository contains `integrations/coding-agents/agentprop/` with
  Codex and Claude Code plugin manifests, the packaged skill, and `.mcp.json`.
  The repo also contains `.agents/plugins/marketplace.json` for Codex and
  `.claude-plugin/marketplace.json` for Claude Code.
- **Portable skill:** `skills/agentprop-workflow-optimizer/` can be installed
  through skills.sh for Codex, Claude Code, or other skill-aware agents.
- **Future dedicated plugin repo:** before broader beta distribution, create a
  smaller repo focused only on coding-agent installation and daily use.

The same-repo bundle is useful while AgentProp's Python package, MCP tools, and
skill references are changing quickly. A separate repo becomes useful once the
install story is stable enough to hand to people who do not need the research
or library source.

## Recommended Dedicated Repo Shape

Name: `agentprop-plugin` or `agentprop-coding-agent-plugin`.

Suggested layout:

```text
agentprop-plugin/
  README.md
  LICENSE
  .mcp.json
  .claude-plugin/
  .codex-plugin/plugin.json
  .agents/plugins/marketplace.json
  integrations/coding-agents/agentprop/
  skills/agentprop-workflow-optimizer/
```

The README should be installer-first:

1. Install AgentProp:
   `python -m pip install "agentprop[mcp]"`
2. Install the Claude Code plugin or skill.
3. Install the Codex marketplace/plugin.
4. Verify `agentprop-mcp` through the host's MCP list command.
5. Run one key-free demo and save `trace.jsonl`, `summary.json`, and
   `report.md`.

## Split Criteria

Create the dedicated repo when all of these are true:

- PR #38 or its successor has landed on `main`.
- The clean-install beta smoke in Linear AGE-30 passes.
- The PyPI release includes the current `agentprop-mcp` entrypoint.
- The skill and plugin docs no longer need daily changes from benchmark work.

Until then, keep the same-repo plugin bundle as the beta distribution path and
sync any major skill changes into `integrations/coding-agents/agentprop/skills/`.
