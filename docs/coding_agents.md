# Coding Agent Integration

AgentProp can be used by coding agents in two complementary ways.

If you are setting this up for the first time, start with the
[beta quickstart for coding agents](beta_quickstart.md). It includes exact Codex
CLI, Claude Code, MCP, artifact, and troubleshooting commands.

For everyday coding tasks, AgentProp generates a workflow brief that tells Codex
CLI, Claude Code, or another coding agent where to concentrate context, where to
verify, and which handoffs are risky to summarize. The agent still uses the
developer's normal authentication path, such as `codex login` or Claude Code's
own login.

For teams building multi-agent systems, AgentProp can also run as a tool layer:
the CLI and `agentprop-mcp` expose graph diagnostics, verifier placement,
routing recommendations, and benchmark/report generation to editor agents.

For controlled runtime experiments, AgentProp can wrap an execution loop through
`ControlSession` and record `ExecutionEvent` traces for stop/retry/verify
decisions.

## Everyday Codex And Claude Code Use

Use this when you want a coding agent to implement or review a workflow with
AgentProp's graph guidance in context, without running a benchmark.

Generate a Claude Code or Codex-ready brief:

```bash
agentprop agent-instructions planner_coder_tester_reviewer \
  --target codex \
  --budget 2 \
  --trials 50 \
  --out reports/codex_agent_brief.md
```

For Claude Code:

```bash
agentprop agent-instructions planner_coder_tester_reviewer \
  --target claude-code \
  --out reports/claude_code_agent_brief.md
```

Give the generated Markdown to the coding agent before it starts work. The
brief includes:

- seed agents that should receive full context first
- expected coverage and cost savings
- verifier/checker placement guidance
- bottleneck nodes and pruning candidates
- ML/RL experiment commands for workflow-improvement tasks
- required evidence before the coding agent claims the task is done

For Codex CLI, keep using the normal Codex session:

```bash
codex login
codex exec "Use reports/codex_agent_brief.md while implementing the requested workflow change."
```

For a fuller Codex integration, install the same-repo AgentProp plugin
marketplace. The plugin packages the AgentProp skill plus the `agentprop-mcp`
stdio server config:

```bash
python -m pip install "agentprop[mcp]"
codex plugin marketplace add aryan5v/AgentProp --sparse .agents --sparse distribution/plugins
codex plugin add agentprop@agentprop
```

Then start a new Codex session and ask it to use the AgentProp plugin or MCP
tools. A separate repository is not required for this beta path; this repo can
host the skill, plugin bundle, and marketplace metadata together while the
integration is moving quickly. See
[plugin distribution](plugin_distribution.md) for the dedicated-repo plan.

For Claude Code, install the same bundle as a Claude plugin:

```bash
python -m pip install "agentprop[mcp]"
claude plugin marketplace add aryan5v/AgentProp --sparse .claude-plugin distribution/plugins
claude plugin install agentprop
```

The canonical skill lives at `distribution/skills/agentprop-workflow-optimizer/`. The
`distribution/wrappers/claude-code/agentprop-workflow-optimizer/` path is a thin wrapper
for skill-specific metadata (`agents/openai.yaml`) and points at the same skill
content.

Install only the public skill package with:

```bash
npx skills add aryan5v/AgentProp --skill agentprop-workflow-optimizer
```

List skills before installing:

```bash
npx skills add aryan5v/AgentProp --list
```

Install globally for Codex and Claude Code:

```bash
npx skills add aryan5v/AgentProp \
  --skill agentprop-workflow-optimizer \
  --agent codex \
  --agent claude-code \
  --global
```

Try it without installing:

```bash
npx skills use aryan5v/AgentProp \
  --skill agentprop-workflow-optimizer \
  --agent codex
```

See the Claude Code docs for
[Agent Skills](https://docs.claude.com/en/docs/claude-code/skills) and
[MCP](https://code.claude.com/docs/en/mcp). `agentprop-mcp` is the path when
Claude should call AgentProp directly instead of only reading a generated brief.

## Full-Suite Coding-Agent Wrapper

For beta users, the preferred path is not instructions-only. Use the full suite
when the host loop can observe real steps from Codex CLI, Claude Code, or a
custom agent runner:

```bash
python dev/examples/coding_agent_full_suite.py
```

The example writes:

- `routing_report.md` and `routing_summary.json`: seed selection, quality/risk,
  cost comparison, verifier candidates, bottlenecks, and pruning risk.
- `context_advice.json`: critical-fact and graded-context advice for a
  context-sensitive node.
- `host_agent_prompt.md`: the small prompt you give Codex or Claude Code after
  normal login.
- `control_session/trace.jsonl`, `summary.json`, and `report.md`: the runtime
  control evidence from observed `ExecutionEvent` rows.

This is the shape to copy for real use:

1. Run graph analysis before the coding agent starts.
2. Give full context to selected seed roles and preserve critical facts.
3. Place verifiers using metric-dimension/resolving coverage candidates.
4. Let the coding agent work normally.
5. Emit an `ExecutionEvent` after each plan/edit/test/review step.
6. Obey `FORCE_VERIFY`, `SWITCH_STRATEGY`, `FINALIZE`, and budget decisions.
7. Save trace/report artifacts before claiming the workflow improved.

## Recommended Agent Instructions

Use this style when asking a coding agent to work on an AgentProp-modeled
workflow:

```text
Before implementing, run AgentProp on the workflow or read the attached
AgentProp brief. Send full context first only to the selected seed agents.
Use the verifier nodes before finalizing. If you prune or summarize a handoff,
state the quality risk and run the relevant verification command. Save any
trace, report, or benchmark artifact that supports the final claim.
```

For ML/DL/RL work:

```text
Use the AgentProp ML core suite before changing policy defaults. Compare any
learned scorer or RL policy against training-free baselines. Do not claim a
learned policy is better unless the saved artifact compares it against
PageRank, CELF, greedy, message-passing GNN-style scoring, and RL baselines
where applicable.
```

## Claude Code Skill Shape

A Claude Code skill should be small and command-oriented:

- Read or generate a workflow JSON file.
- Run `agentprop analyze`, `agentprop report`, and `agentprop agent-instructions`.
- Attach the generated brief to the working context.
- After implementation, run the verification command and write trace/report
  artifacts.

Suggested skill name: `agentprop-workflow-optimizer`.

## Codex Plugin Shape

A Codex plugin should expose the same core actions:

- `analyze_workflow(workflow)`
- `optimize_routing(workflow, budget, model)`
- `generate_agent_brief(workflow, target)`
- `run_ml_core_suite(artifact_root)`
- `analyze_case_study(results_path)`

The CLI is enough for analysis-mode integrations. Runtime-controller integrations
should call `agentprop.runtime.AgentPropRuntimeController` and inject their own
model/tool executor.

## Runtime Controller Shape

Use this path when benchmarking or running a real workflow. The coding agent is
no longer just reading an AgentProp brief; the executor receives a concrete
`RuntimeNodeRequest` with the context that AgentProp selected for that node.

```python
from agentprop.runtime import AgentPropRuntimeController, RuntimeControllerConfig

controller = AgentPropRuntimeController(
    workflow_graph,
    config=RuntimeControllerConfig(seed_budget=2),
    compressor=my_context_compressor,
)
result = controller.run(
    task=task_prompt,
    shared_context=full_task_context,
    executor=my_node_executor,
)
```

For terminal-style agents, wrap the outer execution loop as well. This lets
AgentProp react to real progress, repeated errors, verifier failures, and token
use instead of only selecting an initial prompt shape.

```python
from agentprop.runtime import (
    AgentLoopConfig,
    AgentTurnResult,
    ControlledAgentLoop,
    ExecutionEvent,
    StoppingController,
    StoppingControllerConfig,
)

loop = ControlledAgentLoop(
    controller=StoppingController(
        StoppingControllerConfig(max_steps_without_verifier=4),
    ),
    config=AgentLoopConfig(task_id=task_id, category=task_category),
)

result = loop.run(
    task=task_prompt,
    turn_executor=my_agent_turn,
    verifier=my_verifier,
    strategy_switcher=my_strategy_switcher,
)
```

Benchmark adapters should translate `RuntimeNodeRequest` into the local agent
or Harbor/Terminal-Bench action, translate terminal/model observations into
`ExecutionEvent`, then persist runtime traces next to the benchmark outcome.
Keep model keys, task sandboxes, and machine-local retry state in local
configuration rather than committed artifacts.

For a smaller public facade, use `ControlSession`. It starts with graph analysis
and then wraps real execution events:

```python
from agentprop.runtime import ControlSession, ExecutionEvent

session = ControlSession.start(
    "planner_coder_tester_reviewer",
    task_id="codex-task-123",
    category="implementation",
    token_budget=120_000,
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

Run the key-free demos before wiring a real coding agent:

```bash
agentprop control-demo --demo terminal --out-dir reports/control-demo
agentprop control-demo --demo multi-agent --out-dir reports/control-demo
agentprop control-demo --demo framework --out-dir reports/control-demo
```

## MCP Server Shape

An MCP server is the best long-term integration for editor agents because it
can expose AgentProp as tools instead of requiring agents to shell out.

AgentProp uses [FastMCP](https://github.com/PrefectHQ/fastmcp) when installed,
with a dependency-free JSON-RPC fallback for tests and minimal local setups:

```bash
python -m pip install "agentprop[mcp]"
agentprop-mcp
```

Recommended tools:

- `agentprop_analyze`: returns graph diagnostics, bottlenecks, pruning
  candidates, and verifier candidates.
- `agentprop_optimize`: returns seed recommendations and cost/coverage deltas.
- `agentprop_report`: writes Markdown/JSON/HTML reports.
- `agentprop_agent_instructions`: returns the coding-agent brief as Markdown.
- `agentprop_control_start`: starts an analysis-backed control session.
- `agentprop_control_observe`: records one runtime event and returns a decision.
- `agentprop_control_decide`: returns the current decision without adding an
  event.
- `agentprop_control_finish`: records the final outcome and closes the session.

The MCP server should keep secrets out of tool arguments. Provider credentials
should be read from local environment variables, a secret manager, or
uncommitted config files.

Current implemented MCP-style tools:

- `agentprop_analyze`
- `agentprop_optimize`
- `agentprop_report`
- `agentprop_agent_instructions`
- `agentprop_control_start`
- `agentprop_control_observe`
- `agentprop_control_decide`
- `agentprop_control_finish`

Claude Code skill template:

```text
distribution/wrappers/claude-code/agentprop-workflow-optimizer/SKILL.md
```

Codex instruction template:

```text
distribution/wrappers/codex/AGENTPROP.md
```

## What This Enables

For a developer using Claude Code or Codex, the ideal loop is:

1. Model or import the workflow graph.
2. Generate an AgentProp brief.
3. Give the brief to the coding agent with the implementation task.
4. Let the agent implement while respecting seed/verifier/pruning guidance.
5. Run verification.
6. Save reports, traces, and case-study outputs.
7. Use ML/DL/RL suites only when optimizing policy quality, not for every
   ordinary task.
