# AgentProp 0.1.0-alpha.4 Release Notes

Release date: 2026-06-06

AgentProp `0.1.0-alpha.4` turns the control-layer work into a more complete
coding-agent beta path for Codex CLI, Claude Code, MCP-capable hosts, and
multi-agent framework builders.

## Highlights

- Full-suite coding-agent wrapper example:
  - graph analysis and routing recommendations,
  - metric-dimension verifier placement,
  - context-risk advice for high-sensitivity roles,
  - `ControlSession` runtime decisions,
  - saved `trace.jsonl`, `summary.json`, and `report.md` artifacts.
- New beta quickstart:
  - Codex CLI normal `codex login` flow,
  - Claude Code skill setup,
  - `agentprop-mcp` registration,
  - evidence-to-save checklist,
  - troubleshooting for missing CLI/MCP/plugin artifacts.
- Same-repo Codex plugin beta bundle:
  - `plugins/agentprop/.codex-plugin/plugin.json`,
  - `plugins/agentprop/.mcp.json`,
  - packaged `agentprop-workflow-optimizer` skill,
  - repo marketplace metadata under `.agents/plugins/marketplace.json`.
- Plugin distribution plan for a future dedicated Vanta-style plugin repo once
  the PyPI artifact and clean-install smoke are stable.
- Better first-run UX:
  - `examples/coding_agent_full_suite.py --out-dir ...`,
  - optional Graphviz now displays as `[warn] graphviz_dot` in
    `agentprop doctor --tier graph`.

## Validation Scope

This release is still public alpha/beta-path software. The source checkout has
been smoke-tested from a fresh temp venv with `agentprop[mcp]`, control demos,
the full-suite coding-agent example, coding-agent brief generation, and Codex
marketplace registration under a temporary `HOME`.

The next release gate is a published-artifact smoke from PyPI/TestPyPI.

## Suggested Checks Before Tagging

```bash
ruff check .
mypy src
pytest
python -m build
twine check dist/*
python -m pip install "dist/agentprop-0.1.0a4-py3-none-any.whl[mcp]"
agentprop doctor --tier graph
agentprop control-demo --demo terminal --out-dir /tmp/agentprop-control-demo
python examples/coding_agent_full_suite.py --out-dir /tmp/agentprop-full-suite
```
