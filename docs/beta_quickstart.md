# Beta Quickstart For Coding Agents

This guide is for early users who want AgentProp in the loop with Codex CLI,
Claude Code, or a custom agent harness. It takes you from install to saved
AgentProp artifacts without requiring external model API keys.

AgentProp has two jobs:

- Analyze the workflow graph before the agent starts: seed roles, verifier
  placement, context risk, bottlenecks, and pruning risk.
- Improve execution when the host loop can emit runtime events: verify, retry,
  stop, switch strategy, or preserve critical context.

Standard Codex and Claude Code sessions can use the analysis layer immediately
through generated briefs and MCP tools. Full runtime control requires a host
harness or Python wrapper that emits one `ExecutionEvent` per agent step.

## Install

```bash
python -m pip install "agentprop[mcp]"
agentprop doctor --tier graph
```

For local development from a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,mcp]"
agentprop doctor --tier dev
```

## Codex CLI

Use your normal Codex login. AgentProp does not need your OpenAI API key for
local graph analysis.

```bash
codex login
agentprop agent-instructions planner_coder_tester_reviewer \
  --target codex \
  --budget 2 \
  --trials 50 \
  --out reports/codex_agent_brief.md
```

Add the MCP server if you want Codex to call AgentProp tools directly:

```bash
codex mcp add agentprop -- agentprop-mcp
codex mcp list
```

Or install the same-repo Codex plugin bundle, which packages the AgentProp
skill and MCP configuration:

```bash
codex plugin marketplace add aryan5v/AgentProp --sparse .agents --sparse plugins
codex plugin add agentprop@agentprop
```

Then start a new Codex session and ask it to use the AgentProp plugin or MCP
tools while it works.

Then start Codex with the brief in context:

```bash
codex exec --cd /path/to/project \
  "Use reports/codex_agent_brief.md and AgentProp MCP tools while implementing <task>. Preserve critical facts, run the recommended verifier, and save trace/report artifacts."
```

## Claude Code

Use Claude Code's normal authentication path. Install the AgentProp plugin,
which packages the AgentProp skill and MCP configuration:

```bash
claude plugin marketplace add aryan5v/AgentProp --sparse .claude-plugin plugins
claude plugin install agentprop
```

Or install the portable skill and MCP server directly:

```bash
npx skills add aryan5v/AgentProp \
  --skill agentprop-workflow-optimizer \
  --agent claude-code \
  --global

claude mcp add agentprop -- agentprop-mcp
claude mcp list
```

Generate a Claude-oriented brief:

```bash
agentprop agent-instructions planner_coder_tester_reviewer \
  --target claude-code \
  --budget 2 \
  --trials 50 \
  --out reports/claude_code_agent_brief.md
```

Start Claude Code with the AgentProp skill and brief:

```bash
claude "Use the AgentProp skill and reports/claude_code_agent_brief.md while implementing <task>. Follow the verifier and context-risk guidance before finalizing."
```

## Full Runtime Wrapper

For a real control layer, wire AgentProp into the host loop that executes the
agent. The host loop should emit one `ExecutionEvent` after each plan, edit,
tool call, test, verifier, or review step.

Run the key-free reference example:

```bash
python examples/coding_agent_full_suite.py
```

The example shows the full beta shape:

1. Build or import an `AgentGraph`.
2. Run seed selection, metric-dimension verifier placement, routing analysis,
   context-risk analysis, and bottleneck checks.
3. Give Codex, Claude Code, or another host agent a small prompt plus the
   generated AgentProp brief.
4. Observe real execution through `ControlSession.observe(...)`.
5. Obey `FORCE_VERIFY`, `SWITCH_STRATEGY`, `FINALIZE`, and budget decisions.
6. Save the trace and report before claiming an improvement.

The minimal runtime facade looks like this:

```python
from agentprop.runtime import ControlSession, ExecutionEvent

session = ControlSession.start(
    "planner_coder_tester_reviewer",
    task_id="codex-task-123",
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
    )
)

session.write_artifacts("reports/codex-task-123")
```

## Evidence To Save

Before saying AgentProp improved a workflow, save the artifacts that let someone
inspect the claim:

- `routing_report.md`
- `routing_summary.json`
- `context_advice.json`
- `host_agent_prompt.md`
- `control_session/trace.jsonl`
- `control_session/summary.json`
- `control_session/report.md`
- verifier command output

Commit only public-safe summaries. Keep raw model prompts, private benchmark
state, `.env` files, `jobs/`, `benchmark-results/`, and machine-local reports
out of git unless they have been intentionally sanitized.

## Troubleshooting

If `agentprop` is not found, activate the virtual environment or reinstall:

```bash
python -m pip install "agentprop[mcp]"
command -v agentprop
```

If `agentprop-mcp` is not found, make sure the MCP extra is installed:

```bash
python -m pip install "agentprop[mcp]"
command -v agentprop-mcp
```

If Codex or Claude Code cannot see the MCP server, list registered servers and
re-add AgentProp with the full executable path:

```bash
command -v agentprop-mcp
codex mcp list
claude mcp list
```

`agentprop-mcp` is a stdio server. It is normally launched by Codex, Claude
Code, or another MCP host; if you run it directly in a terminal and stdin
closes, it may exit immediately.

If no `trace.jsonl` appears, the coding agent probably used only the static
brief or MCP analysis tools. Runtime traces require `ControlSession` or another
wrapper that emits `ExecutionEvent` rows.

If the Codex plugin marketplace does not appear, refresh marketplaces and start
a new Codex thread:

```bash
codex plugin marketplace list
codex plugin marketplace upgrade agentprop
```

If the Claude Code plugin marketplace does not appear, refresh marketplaces and
start a new Claude Code session:

```bash
claude plugin marketplace list
claude plugin marketplace update
```
