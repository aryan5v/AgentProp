# AgentProp

<p align="center">
  <img src="docs/assets/agentprop-logo.png" alt="AgentProp logo" width="160" />
</p>

<p align="center">
  <strong>Graph control for agent workflows.</strong>
</p>

<p align="center">
  <a href="https://github.com/aryan5v/AgentProp/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/aryan5v/AgentProp/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://pypi.org/project/agentprop/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agentprop.svg" /></a>
  <a href="https://github.com/aryan5v/AgentProp/security"><img alt="Security" src="https://img.shields.io/badge/security-policy-black" /></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0a2-black" />
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-black" />
  <img alt="Status" src="https://img.shields.io/badge/status-public_alpha-12c95b" />
</p>

AgentProp studies AI-agent workflows as directed weighted graphs. Agents,
tools, context packets, verifier calls, terminal commands, and failure states
become nodes and edges in a graph that can be measured, simulated, and
controlled.

The research wedge is simple:

- **Metric dimension** is the core contribution: framing verifier placement as a
  resolving set makes failure localization a *provable* property — if resolving
  coverage is 1.0, every distinct failure produces a unique signature and any
  single faulty node is uniquely identifiable. With fault-tolerant metric dimension,
  this holds even if one verifier itself fails. No weighted-heuristic placement can promise this.
- **Quality cascade** models how correctness and compression propagate, so
  context allocation follows the quality actually reaching each node.
- **Randomized Zero Forcing (RZF)** is a *secondary, scoped* result: process-based
  RZF centrality helps on **large** workflows where static centrality misjudges
  reachability; on small graphs (under ~15 nodes) classical centrality is
  competitive. Reported honestly, not as a universal win.
- **Runtime control** turns those ideas into actions: verify, retry, stop, switch
  strategy, or send more context.

AgentProp is not another agent orchestrator. It wraps a workflow you already
have: each step your agent proposes work, the controller inspects the accumulated
`ExecutionEvent` history, and decides what happens next.

```
   task ─► ┌─ AgentProp control loop ───────────────────────┐
           │  ┌────────┐  propose   ┌─────────────────────┐  │
           │  │  your  │ ─────────► │ Stopping Controller │  │
           │  │ agent  │ ◄───────── │ CONTINUE/VERIFY/    │  │ ─► result +
           │  └────────┘  decision  │ SWITCH/FINALIZE     │  │    decision trace
           │      └─ ExecutionEvent ┴─────────────────────┘  │
           │     (tokens, exit code, verifier_passed, ...)   │
           └─────────────────────────────────────────────────┘
```

Every decision is logged, so the trace is auditable. The only contract your
harness must satisfy is emitting one `ExecutionEvent` per step. AgentProp ships
dependency-light adapters for **LangGraph, AutoGen, CrewAI, OpenAI Agents, and
LlamaIndex** (see [framework integrations](docs/framework_integrations.md)), and
controls any other harness that can return an `ExecutionEvent`.

**Why metric dimension matters (intuition):** a workflow only fails *usefully* if
you can tell *which* node failed. With verifiers placed badly, a bad planner
output and a bad tester output can produce the *same* observable signature — so
you cannot route a fix. A resolving set guarantees each node's vector of distances
to the verifiers is unique, giving every distinct failure a distinct fingerprint.
See [verifier semantics](docs/verifier_semantics.md).

## Early Signal

On one Terminal-Bench 2.1 smoke task using Harbor's `codex` agent with
`gpt-5.5`, the AgentProp A2 controller preserved success while reducing spend:

| Task | Arm | Result | Tokens | Cost | Time |
| --- | --- | --- | ---: | ---: | ---: |
| `regex-log` | A0 raw Codex | pass | 123,731 | $0.333551 | 203.8s |
| `regex-log` | A2 AgentProp control | pass | 81,949 | $0.196834 | 173.6s |

That is **33.8% fewer tokens**, **41.0% lower cost**, and **14.8% less wall
time** on a pass-preserving comparison. This is a single-task early signal, not
a benchmark claim; the point is that AgentProp can already act as a spend-aware
controller around live coding-agent execution.

## What Is Implemented

- Directed weighted `AgentGraph` with JSON validation, NetworkX conversion, and
  Graphviz export.
- Propagation models: Independent Cascade, Linear Threshold, Bootstrap
  Percolation, deterministic Zero Forcing, Randomized Zero Forcing, learned
  propagation, and Quality Cascade.
- Graph algorithms for seed selection, pruning, bottlenecks, k-core, bridges,
  articulation points, centrality, verifier placement, and resolving coverage.
- Metric-dimension verifier placement, including fault-tolerant resolving
  coverage for single-verifier failure.
- RZF process-based centrality for seed selection and scaling studies.
- Runtime controllers for graph-node execution, terminal-loop control,
  verifier forcing, local-pass distrust, retry/stop/switch decisions, and
  category-conditioned bandit policies.
- Optional ML/DL/RL baselines: learned seed scorers, torch GNNs, Q-learning,
  REINFORCE, PPO, and artifact/checkpoint tooling.
- Coding-agent integration helpers for Codex, Claude Code, MCP-style tools,
  and framework adapters.

## Install

```bash
python -m pip install agentprop
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Optional extras:

```bash
python -m pip install -e ".[dl]"  # torch-backed graph models
python -m pip install -e ".[rl]"  # Gymnasium-compatible RL experiments
```

## Quick Start

Analyze a built-in workflow:

```bash
agentprop analyze planner_coder_tester_reviewer
```

Recommend context seed nodes under the RZF propagation model:

```bash
agentprop optimize planner_coder_tester_reviewer \
  --budget 2 \
  --algorithm greedy \
  --model rzf
```

Compare graph propagation policies:

```bash
PYTHONPATH=src:. python experiments/run_benchmark.py \
  --workflows chain planner_coder_tester_reviewer research_writer_verifier \
  --algorithms rzf-centrality greedy betweenness pagerank random \
  --models quality-cascade independent-cascade \
  --budget 2 --trials 50 --decay --decay-seed 0 \
  --out-dir results/my_run
```

Generate verifier-placement evidence:

```bash
PYTHONPATH=src:. python experiments/verifier_placement_evidence.py
```

Run the RZF scaling study:

```bash
PYTHONPATH=src:. python experiments/rzf_scaling_study.py
```

Both scripts are deterministic and print an expected-output block at the top of
the source so you can confirm you reproduced the published numbers (metric
dimension reaching a resolving set at lower budget `k`, and RZF leading on large
graphs). The headline figures are summarized in
[reproducible results](docs/research/reproducible_results.md).

Use the runtime controller from Python:

```python
from agentprop.runtime import AgentPropRuntimeController, RuntimeControllerConfig
from agentprop.workflows import planner_coder_tester_reviewer

graph = planner_coder_tester_reviewer()
controller = AgentPropRuntimeController(
    graph,
    config=RuntimeControllerConfig(seed_budget=2, fixed_seeds=("coder", "tester")),
)
```

## Coding-Agent Integration

AgentProp can be used with Codex CLI, Claude Code, or any MCP-capable editor
agent as a workflow-analysis layer. It does not need model API keys to generate
briefs or run local graph analysis; Codex can keep using `codex login`, and
Claude Code can use the included skill/MCP-style integration.

```bash
agentprop agent-instructions planner_coder_tester_reviewer \
  --target codex \
  --out reports/codex_agent_brief.md

agentprop agent-instructions planner_coder_tester_reviewer \
  --target claude-code \
  --out reports/claude_code_agent_brief.md
```

Use these briefs for everyday implementation/review tasks, or run
`agentprop-mcp` when a coding agent should call AgentProp tools directly while
designing or debugging a multi-agent workflow.

## Research Position

AgentProp sits between graph theory, diffusion models, and agent evaluation. The
core hypothesis is that agent workflows should be optimized as communication
graphs under quality, cost, and observability constraints, rather than treated
as opaque prompt loops.

Key inspirations:

- Jesse Geneson et al., [Randomized Zero Forcing](https://arxiv.org/abs/2602.16300):
  stochastic propagation on directed weighted graphs.
- Jesse Geneson, [Metric dimension and pattern avoidance in graphs](https://arxiv.org/abs/1807.08334):
  resolving sets and graph observability.
- Jesse Geneson and Leslie Hogben,
  [Propagation time for probabilistic zero forcing](https://arxiv.org/abs/1812.10476):
  expected propagation time as a graph parameter.
- Kempe, Kleinberg, and Tardos,
  [Maximizing the Spread of Influence through a Social Network](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf):
  influence maximization under cascade models.
- [GPTSwarm](https://openreview.net/pdf?id=uTC9AFXIhg),
  [DyLAN](https://arxiv.org/abs/2310.02170), and
  [AgentPrune](https://arxiv.org/abs/2410.02506):
  agent workflows as optimizable, sparse, task-adaptive communication graphs.

See [the documentation index](docs/index.md),
[research references](docs/research/references.md), and the
[literature review](docs/research/literature_review.md) for more detail.

## Status

AgentProp is public alpha research software. The graph backbone, propagation
models, runtime-control APIs, CLI, tests, and experiment scripts are usable, but
the benchmark evidence is still early. Treat live-agent results as directional
until larger repeated studies are published.

## Development

```bash
ruff check .
mypy src
pytest
```

CI runs the same gates on pull requests. AgentProp is released under the Apache
2.0 license.
