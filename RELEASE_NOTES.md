# AgentProp 0.1.0-alpha.3 Release Notes

Release date: 2026-06-05

AgentProp `0.1.0-alpha.3` adds the first beta-ready control-layer surface: a
small Python SDK facade, key-free demos, MCP tools, and an installable agent
skill package.

## Highlights

- `ControlSession` facade for analysis-backed runtime wrapping:
  - starts with graph analysis,
  - observes `ExecutionEvent` rows,
  - returns `CONTINUE`, `FORCE_VERIFY`, `SWITCH_STRATEGY`, or `FINALIZE`,
  - writes `trace.jsonl`, `summary.json`, and `report.md`.
- New CLI demos:
  - `agentprop control-demo --demo terminal`
  - `agentprop control-demo --demo multi-agent`
  - `agentprop control-demo --demo framework`
- FastMCP-backed MCP server with optional extra:
  - `python -m pip install "agentprop[mcp]"`
  - `agentprop-mcp`
- MCP tools for analysis, optimization, reports, coding-agent briefs, and live
  control sessions.
- Installable skills.sh-style package:
  - `skills/agentprop-workflow-optimizer/SKILL.md`
  - task-specific references for install/config, workflow analysis, coding
    agents, runtime control, framework builders, and evidence.
- README badges and docs now surface PyPI, skills.sh, and MCP usage.

## Validation Scope

This release improves the user-facing control-layer experience. The demos are
deterministic and key-free; they are onboarding evidence, not benchmark claims.
Live benchmark claims still require repeated matched runs with saved traces,
tokens, cost, elapsed time, and verifier outcomes.

## Suggested Checks Before Tagging

```bash
python -m ruff check .
python -m mypy src
python -m pytest
PYTHONPATH=src python -m agentprop.cli control-demo --demo terminal --out-dir /tmp/agentprop-control-demo
python -m build
twine check dist/*
```
