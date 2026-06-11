# Troubleshooting

Quick recovery paths for the most common first-hour problems. If you are
stuck after this page, open an issue with the output of
`agentprop doctor --tier dev`.

## Install and import

**`pip install agentprop` succeeds but `agentprop` is not found.**
The console script lands in your environment's `bin/`. Activate the
virtualenv you installed into, or run `python -m agentprop.cli`.

**`ImportError: fastmcp`** — you ran `agentprop-mcp` without the MCP extra.
MCP is optional; install it only if you want a coding agent to call AgentProp
tools directly:

```bash
python -m pip install "agentprop[mcp]"
```

Graph analysis (`analyze`, `optimize`, `report`, `viz`, `control-demo`) never
needs fastmcp or any API key.

**`TorchBackendUnavailable` / numpy or gymnasium import errors.**
The `ml`, `rl`, `dl`, and `otel` extras are opt-in. Install only what you use,
e.g. `pip install "agentprop[ml]"`. Core analysis depends only on `networkx`.

## Which doctor tier do I run?

```bash
agentprop doctor --tier graph
```

| Tier | Checks | Use when |
| --- | --- | --- |
| `graph` | Core install, built-in workflows, propagation | First install — start here |
| `dev` | Adds dev tooling (pytest, ruff, mypy) | Contributing from a checkout |
| `llm` | Adds live-model wiring | Running live-agent experiments |
| `terminal-bench` | Adds benchmark harness checks | Reproducing Terminal-Bench runs |

A failing higher tier does not mean the library is broken — it means that
optional capability is not configured.

## Graph and workflow errors

**"Unknown workflow" from `analyze` / `optimize`.**
The positional argument must be a built-in name (`agentprop workflows list`),
a path to a workflow JSON file, or — from Python — an `AgentGraph` or
LangGraph object.

**Workflow JSON fails validation.**
Check it against the [workflow schema](workflow_schema.md). The most common
issues are edges referencing undeclared node ids and non-numeric
weights.

**`viz` produces a `.dot` file but no image.**
Graphviz rendering needs the system `dot` binary (`apt install graphviz` /
`brew install graphviz`), or paste the DOT output into an online viewer.

## MCP server

**Server starts then exits immediately.**
Run `agentprop-mcp` in a terminal to see the error. With `fastmcp` missing it
falls back to a dependency-free JSON-RPC mode; if your client requires
FastMCP semantics, install the `[mcp]` extra.

**Codex / Claude Code does not list the tools.**
Re-register and verify:

```bash
codex mcp add agentprop -- agentprop-mcp
codex mcp list
```

For Claude Code, check the `.mcp.json` shipped in `plugins/agentprop/` and the
[coding agents guide](coding_agents.md#mcp-server-shape).

**Sessions look stale.** MCP session state persists under
`~/.agentprop/sessions`. Deleting that directory resets all sessions and the
shared graph-analysis cache.

## Runtime control

**`wrap()` runs but never returns anything other than `CONTINUE`.**
The controller needs real signals: populate `ExecutionEvent` fields
(errors, verifier outcomes, token counts) on every step. With empty events
there is nothing to react to.

**Numbers differ between runs.**
Propagation models are stochastic. Pass a `seed` to the model or raise
`--trials`; analysis reports average over `trials` simulations.

## Glossary

- **Seed** — node where information/influence is injected first.
- **Propagation model** — rule for how activation/quality spreads along edges
  (IC, LT, zero forcing, quality cascade, ...).
- **Verifier** — node that checks outputs; placement is chosen so failure
  signatures stay distinguishable (see [verifier semantics](verifier_semantics.md)).
- **Resolving coverage** — fraction of node pairs the verifier set can
  distinguish; 1.0 means any single failing node is uniquely identifiable.
- **Quality cascade** — continuous quality degradation along edges, used for
  context routing.
