# Install And Configure AgentProp

## Check For The CLI

```bash
agentprop --help
```

If the command exists, use it. If it is missing, install AgentProp.

## Install From PyPI

```bash
python -m pip install agentprop
```

Use this for normal users who only need the CLI, reports, graph analysis, and
coding-agent briefs.

## Install From A Source Checkout

```bash
git clone https://github.com/aryan5v/AgentProp.git
cd AgentProp
python -m pip install -e ".[dev]"
```

If commands are run directly from a checkout before editable install, prefix
with `PYTHONPATH=src`.

## Optional Extras

```bash
python -m pip install "agentprop[mcp]" # FastMCP tools
python -m pip install "agentprop[dl]"  # torch-backed graph models
python -m pip install "agentprop[rl]"  # Gymnasium-compatible RL experiments
```

## MCP Server

AgentProp uses FastMCP when installed:

```bash
python -m pip install "agentprop[mcp]"
agentprop-mcp
```

The MCP server exposes analysis tools and live control-session tools. It should
not receive API keys in tool arguments; credentials belong in local environment
variables or secret managers.

## Sanity Checks

```bash
agentprop analyze planner_coder_tester_reviewer --json
agentprop control-demo --demo terminal --out-dir reports/control-demo
```

The control demo should write `trace.jsonl`, `summary.json`, and `report.md`.
