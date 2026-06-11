# AgentProp

<p align="center">
  <img src="docs/assets/agentprop-logo.png" alt="AgentProp logo" width="160" />
</p>

<p align="center">
  <strong>Observability tools watch. Orchestrators run. Nobody supervises.</strong><br />
  AgentProp is the control layer for agent workflows.
</p>

<p align="center">
  <a href="https://github.com/aryan5v/AgentProp/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/aryan5v/AgentProp/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://pypi.org/project/agentprop/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agentprop.svg" /></a>
  <a href="https://www.skills.sh/aryan5v/AgentProp"><img alt="skills.sh" src="https://www.skills.sh/b/aryan5v/AgentProp" /></a>
  <a href="docs/coding_agents.md#mcp-server-shape"><img alt="MCP" src="https://img.shields.io/badge/MCP-FastMCP-12c95b" /></a>
  <a href="https://github.com/aryan5v/AgentProp/security"><img alt="Security" src="https://img.shields.io/badge/security-policy-black" /></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0b1-black" />
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-black" />
  <img alt="Status" src="https://img.shields.io/badge/status-public_beta-12c95b" />
</p>

AgentProp models AI-agent workflows as directed weighted graphs—agents, tools,
context, verifiers, and failures become nodes and edges you can analyze,
simulate, and supervise. It answers two questions no orchestrator or
observability tool does:

- **Where do I put checks?** Verifier placement is solved as a resolving-set
  problem: at full resolving coverage, any single failing node produces a
  unique signature and is exactly localizable — a guarantee, not a heuristic.
- **Should this run keep going?** Your harness emits one `ExecutionEvent` per
  step; AgentProp returns continue, verify, switch strategy, or finalize —
  backed by forward cascade simulation, calibrated risk gates, and budget
  control.

It is a **control layer**, not an orchestrator: it sits beside LangGraph,
CrewAI, OpenAI Agents, Claude Code, or Codex rather than replacing them.
Full documentation: **[aryan5v.github.io/AgentProp](https://aryan5v.github.io/AgentProp/)** ·
core ideas: [overview](docs/overview.md).

## Get started — pick your path

### Path A: you use a coding agent (Claude Code / Codex)

Install the package and register the plugin or MCP server so your agent can
analyze its own workflow and run under budget control:

```bash
python -m pip install "agentprop[mcp]"
agentprop doctor --tier graph

codex mcp add agentprop -- agentprop-mcp   # or the plugin bundle below
```

Then follow the [beta quickstart](docs/beta_quickstart.md). No API keys are
needed for graph analysis.

### Path B: you build multi-agent systems (LangGraph and friends)

```bash
python -m pip install agentprop
```

Analyze a LangGraph workflow in a few lines:

```python
from agentprop import analyze

report = analyze(my_langgraph_workflow)
print(report.to_markdown())
```

Wrap it with runtime control:

```python
from agentprop import wrap

controlled = wrap(my_langgraph_workflow, budget={"tokens": 100_000, "cost": 0.50})
result = controlled.run({"task": "ship the workflow"})
print(result.decision_trace)
```

Start with the [tutorial](docs/tutorial.md) and the
[framework status matrix](docs/framework_integrations.md#integration-status)
(LangGraph, CrewAI, and OpenAI Agents have native builders; AutoGen and
LlamaIndex are dict-interchange only).

### Try it without any framework

```bash
agentprop analyze planner_coder_tester_reviewer
agentprop control-demo --demo terminal --out-dir reports/control-demo
agentprop view planner_coder_tester_reviewer --out reports/view.html
```

More: [docs/index.md](docs/index.md) ·
[troubleshooting](docs/troubleshooting.md) ·
[contributing](docs/project/CONTRIBUTING.md)

Development checkout:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
make test
```

## Early signal

On one Terminal-Bench 2.1 task (`regex-log`, Codex + gpt-5.5), AgentProp control
preserved pass while reducing tokens and cost versus the raw agent:

| Arm | Result | Tokens | Cost |
| --- | --- | ---: | ---: |
| A0 raw Codex | pass | 123,731 | $0.33 |
| A2 AgentProp control | pass | 81,949 | $0.20 |

That is a **single-task early signal**, not a benchmark claim. Multi-task
replication is documented in
[Terminal-Bench multi-task protocol](docs/results/terminal_bench_multi/README.md).

## Coding-agent integration

AgentProp ships a same-repo plugin bundle at [`plugins/agentprop`](plugins/agentprop)
(Codex + Claude Code manifests, packaged skill, MCP config). Install the Python
package, then register the plugin:

```bash
python -m pip install "agentprop[mcp]"

codex plugin marketplace add aryan5v/AgentProp --sparse .agents --sparse plugins
codex plugin add agentprop@agentprop
```

Portable skill-only install:

```bash
npx skills add aryan5v/AgentProp --skill agentprop-workflow-optimizer
```

Full setup, MCP registration, and troubleshooting:
[coding agents](docs/coding_agents.md) · [plugin distribution](docs/plugin_distribution.md).

## Examples

```bash
python examples/coding_agent_full_suite.py --out-dir reports/full-suite
python examples/minimal_control_loop.py
```

More: [examples/README.md](examples/README.md).

## Repository map

| Path | Contents |
| --- | --- |
| `src/agentprop/` | Library and CLI |
| `docs/` | Guides, reference, and [public artifacts](docs/results/ARTIFACTS.md) |
| `experiments/` | Repro scripts — [catalog](experiments/README.md) |
| `examples/` | Integration templates |
| `plugins/agentprop/` | Editor-agent plugin bundle |
| `skills/` | Canonical skills.sh skill source |

Details: [repository layout](docs/repository_layout.md).

## Status

Public beta. Graph analysis, propagation, runtime control, the interactive
`view` report, and key-free demos all work without API keys. Live-agent
numbers are labeled **directional** until the multi-task study with saved
artifacts lands under `docs/results/`.

## License

Apache 2.0. See [SECURITY.md](SECURITY.md) and [CHANGELOG.md](docs/project/CHANGELOG.md).
