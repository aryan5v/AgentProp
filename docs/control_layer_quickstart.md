# Control Layer Quickstart

AgentProp is both an analysis layer and a runtime wrapping layer. The usual flow
is:

1. Analyze the workflow graph: bottlenecks, pruning risk, verifier candidates,
   and context-critical nodes.
2. Wrap execution with a `ControlSession`.
3. Emit `ExecutionEvent` rows from each agent, tool, terminal command, or
   verifier step.
4. Let AgentProp return structured decisions: `CONTINUE`, `FORCE_VERIFY`,
   `SWITCH_STRATEGY`, or `FINALIZE`.
5. Save `trace.jsonl`, `summary.json`, and `report.md` as evidence.

## Try It In 10 Minutes

These demos require no model API keys:

```bash
agentprop control-demo --demo terminal --out-dir reports/control-demo
agentprop control-demo --demo multi-agent --out-dir reports/control-demo
agentprop control-demo --demo framework --out-dir reports/control-demo
```

Each demo writes:

- `trace.jsonl`: analysis rows, observed events, features, decisions, and final
  outcome.
- `summary.json`: compact machine-readable state.
- `report.md`: human-readable explanation of analysis and decisions.

## Python SDK

Use `ControlSession` when you are building a real executor:

```python
from agentprop.runtime import ControlSession, ExecutionEvent

session = ControlSession.start(
    "planner_coder_tester_reviewer",
    task_id="task-123",
    category="implementation",
    token_budget=120_000,
    baseline_tokens=180_000,
)

decision = session.observe(
    ExecutionEvent(
        step=1,
        command="pytest -q",
        verifier_run=True,
        verifier_passed=False,
        error_signature="AssertionError:test_edge_case",
        tokens_used=18_000,
        elapsed_s=42.0,
    )
)

if decision.action == "SWITCH_STRATEGY":
    # Ask the host agent to change approach before spending another attempt.
    pass

session.record_outcome(passed=True, quality_score=1.0)
session.write_artifacts("reports/task-123")
```

The first trace row is an analysis snapshot. Subsequent rows connect real
execution features to decisions, so a user can audit why AgentProp intervened.

## Codex And Claude Code

For everyday coding-agent use, keep using the normal agent login path:

```bash
codex login
agentprop agent-instructions planner_coder_tester_reviewer \
  --target codex \
  --out reports/codex_agent_brief.md
```

Give the generated brief to Codex or Claude Code when the agent should reason
about context routing, verifier placement, and risky handoffs.

When you want live tool calls instead of a static brief, install the MCP extra:

```bash
python -m pip install "agentprop[mcp]"
agentprop-mcp
```

The MCP server uses [FastMCP](https://github.com/PrefectHQ/fastmcp) when the
extra is installed, and exposes these live-control tools:

- `agentprop_control_start`
- `agentprop_control_observe`
- `agentprop_control_decide`
- `agentprop_control_finish`

These tools do not execute user code. They observe events and return structured
decisions for the host agent to obey.

For a complete key-free reference that combines graph analysis, context advice,
verifier placement, runtime decisions, budgets, and trace output, run:

```bash
python examples/coding_agent_full_suite.py
```

Use that example as the template for a Codex or Claude Code harness: the host
agent does the work, and AgentProp supplies analysis, intervention decisions,
and evidence.

## Framework Builders

AgentProp can wrap custom, LangGraph-style, CrewAI-style, AutoGen-style,
OpenAI Agents-style, or LlamaIndex-style workflow dictionaries through the
framework adapters. The important integration point is to emit an
`ExecutionEvent` from each node/tool/verifier step.

```python
from agentprop.integrations.framework_adapters import graph_from_framework_dict
from agentprop.runtime import ControlSession

graph = graph_from_framework_dict(workflow_dict, "langgraph")
session = ControlSession.start(
    graph,
    task_id="framework-task",
    category="research-pipeline",
)
```

The same trace/report format is produced whether the workflow came from a JSON
graph, a built-in template, a coding-agent run, or a framework adapter.
