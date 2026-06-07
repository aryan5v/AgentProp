# Claude Code Skill Install Path

This directory is a **thin wrapper** for skill-specific metadata. The canonical
skill content lives at
[`skills/agentprop-workflow-optimizer/`](../../../../skills/agentprop-workflow-optimizer/).

For the full Claude Code plugin, use
[`integrations/coding-agents/agentprop`](../../agentprop), which includes the
Claude plugin manifest, the skill, and the MCP server config.

## What lives here

- [`agents/openai.yaml`](agents/openai.yaml) — UI metadata for Codex-compatible
  skill discovery
- [`SKILL.md`](SKILL.md) — same entry point as the canonical skill (keep in sync)

## Install the full plugin

```bash
python -m pip install "agentprop[mcp]"
claude plugin marketplace add aryan5v/AgentProp \
  --sparse .claude-plugin \
  --sparse integrations/coding-agents/agentprop
claude plugin install agentprop
```

## Install only the skill

```bash
npx skills add aryan5v/AgentProp --skill agentprop-workflow-optimizer
```

For a source checkout, point Claude Code at either:

- `skills/agentprop-workflow-optimizer/SKILL.md` (canonical), or
- this directory if you need the bundled `agents/openai.yaml`

See the [canonical skill README](../../../../skills/agentprop-workflow-optimizer/README.md)
for full install options and reference files.
