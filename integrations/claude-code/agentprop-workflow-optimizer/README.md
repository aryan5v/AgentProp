# Claude Code Install Path

This directory is a **thin wrapper** for Claude Code and Codex-compatible agent
metadata. The canonical skill content lives at
[`skills/agentprop-workflow-optimizer/`](../../../skills/agentprop-workflow-optimizer/).

## What lives here

- [`agents/openai.yaml`](agents/openai.yaml) — UI metadata for Codex-compatible
  skill discovery
- [`SKILL.md`](SKILL.md) — same entry point as the canonical skill (keep in sync)

## Install (preferred)

```bash
npx skills add https://github.com/aryan5v/AgentProp --skill agentprop-workflow-optimizer
```

For a source checkout, point Claude Code at either:

- `skills/agentprop-workflow-optimizer/SKILL.md` (canonical), or
- this directory if you need the bundled `agents/openai.yaml`

See the [canonical skill README](../../../skills/agentprop-workflow-optimizer/README.md)
for full install options and reference files.
