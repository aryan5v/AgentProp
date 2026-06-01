# AgentProp

<p align="center">
  <img src="docs/assets/agentprop-logo.png" alt="AgentProp logo" width="180" />
</p>

<p align="center">
  <strong>Graph optimization for multi-agent LLM workflows.</strong>
</p>

<p align="center">
  <a href="https://github.com/aryan5v/AgentProp/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/aryan5v/AgentProp/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://pypi.org/project/agentprop/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agentprop.svg" /></a>
  <a href="https://github.com/aryan5v/AgentProp/security"><img alt="Security" src="https://img.shields.io/badge/security-policy-black" /></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0a1-black" />
  <img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-black" />
  <img alt="Status" src="https://img.shields.io/badge/status-public_alpha-12c95b" />
</p>

AgentProp models agents, tools, memories, documents, and verifiers as nodes in a
directed weighted graph. It then uses propagation models, classical graph
algorithms, optional GNN-style policies, and reinforcement learning to optimize:

- which agents receive full context first
- which edges are redundant or risky to prune
- where verifier agents should observe or intercept failures
- how much cost is saved versus broadcast routing
- how learned routing policies compare with graph-theoretic baselines

AgentProp is not an agent orchestrator. It is an analysis and optimization layer
for workflows you already have, want to inspect, or want to study.

## Status

AgentProp is usable as a public alpha framework. The graph backbone, CLI,
reports, workflow templates, ML/RL baselines, MCP/coding-agent briefs,
checkpoints, and experiment artifact registry are implemented and tested.

## First Benchmark

We ran an AgentProp-guided Terminal-Bench comparison using Harbor on a frozen
27-task subset completed by the local Gemini CLI baseline. The AgentProp-guided
run completed 26 tasks; one task hung in the external harness and is excluded
from the headline.

On the 26 completed matched tasks, AgentProp-guided routing improved pass rate
from **17/26 (65.4%)** to **18/26 (69.2%)**. The run produced three wins, two
regressions, and twenty-one ties. On the 24 completed tasks with token data in
both arms, AgentProp-guided used **2.4% fewer input+output tokens** with **1.1%
higher reported cost** due to cache and completion mix.

The first value signal is the recovery of tasks that the initial local baseline
failed. The baseline failed nine of the 26 completed matched tasks;
AgentProp-guided routing converted three of those older failures into passes:
`build-pov-ray`, `caffe-cifar-10`, and `sanitize-git-repo`. Those recoveries are
the clearest early evidence for the framework: explicit planning, full context
for implementation-sensitive work, and executable verification can change task
outcomes instead of only reshuffling token usage.

This is a first directional benchmark, not a leaderboard submission. It shows a
small positive success-rate signal and several useful failure modes for future
budget-aware routing. See the full artifact note:
[Terminal-Bench guided benchmark](docs/results/terminal_bench_guided/README.md).

The main limitation is evidence depth: real routed LLM validation should be
treated as directional until larger, repeated studies are published. The current
library prioritizes reproducible artifacts and conservative claims.

Check the current rollout state:

```bash
agentprop readiness
```

## Install

```bash
python -m pip install agentprop
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[dl]"  # torch-backed GNN experiments
python -m pip install -e ".[rl]"  # optional Gymnasium ecosystem compatibility
```

CUDA/GPU is not required for the current dependency-light alpha workflows.
Modal/GPU becomes useful for larger torch sweeps and hyperparameter searches.

## First Recipes

Analyze a workflow:

```bash
agentprop analyze benchmarks/workflows/planner_coder_tester_reviewer.json
```

Recommend seed agents for context routing:

```bash
agentprop optimize planner_coder_tester_reviewer --budget 2 --algorithm greedy
```

Use quality-aware routing when correctness-sensitive roles should be protected:

```bash
agentprop optimize planner_coder_tester_reviewer \
  --budget 2 \
  --algorithm quality-aware-greedy
```

Simulate propagation:

```bash
agentprop simulate chain --seeds node_0 --model zero-forcing
```

Prune toward a token-reduction target:

```bash
agentprop prune planner_coder_tester_reviewer --target-token-reduction 0.3
```

Write an HTML report:

```bash
agentprop report planner_coder_tester_reviewer --out reports/demo.html --format html
```

Generate a Codex or Claude Code brief:

```bash
agentprop agent-instructions planner_coder_tester_reviewer \
  --target codex \
  --out reports/codex_agent_brief.md
```

## Experiment Recipes

Run the benchmark table and SVG plot:

```bash
PYTHONPATH=src:. python experiments/run_benchmark.py \
  --workflows planner_coder_tester_reviewer,chain,tree \
  --budget 2 \
  --trials 20 \
  --out-dir results/benchmark
```

Run a small ML/RL sweep:

```bash
PYTHONPATH=src:. python experiments/run_ml_rl_sweep.py \
  --config configs/sweeps/ml_rl_smoke.json \
  --artifact-root results/ml_rl_smoke
```

Dry-run the full recipe suite:

```bash
PYTHONPATH=src:. python experiments/run_experiment_suite.py \
  --config configs/experiment_suites/ml_core.json \
  --artifact-root results/ml_core \
  --dry-run
```

Preflight the real LLM case study without making LLM calls:

```bash
PYTHONPATH=src:. python experiments/run_case_study.py \
  --execution-mode llm \
  --preflight \
  --out-dir docs/results/case_study_001
```

Prepare a Terminal-Bench 2.1 + Terminus-2 launch bundle without running the
benchmark:

```bash
agentprop terminal-bench prepare \
  --dataset terminal-bench/terminal-bench-2-1 \
  --agent terminus-2 \
  --model google/gemini-3.1-pro-preview \
  --environment modal \
  --out-dir benchmark-results/terminal-bench-2.1
```

## Artifacts

AgentProp writes plain, inspectable artifacts:

- `results.json` / `results.csv` for benchmark and case-study rows
- `summary.json` for aggregate metrics
- `traces.jsonl` and `outputs.jsonl` for routed LLM execution traces
- `verification_logs.jsonl` when command verification is enabled
- `registry.json` for ML/RL checkpoints and metric artifacts
- `manifest.json`, `RUNBOOK.md`, and watchdog status JSON for prepared external
  benchmark runs
- `*.svg` plots for benchmark and case-study summaries
- Markdown, JSON, or HTML optimization reports

This recipe-first layout is intentional: every claim should point to a command
and a saved artifact.

## What Is Implemented

- Directed weighted `AgentGraph` with JSON, validation, NetworkX conversion, and
  Graphviz DOT export.
- Propagation models: Independent Cascade, Linear Threshold, Bootstrap
  Percolation, Randomized Zero Forcing, deterministic Zero Forcing, and learned
  trace-calibrated propagation.
- Classical baselines: random, degree, in-degree, out-degree, PageRank,
  betweenness, closeness, k-core, greedy, CELF, cost-aware greedy, and
  quality-aware greedy.
- Bottleneck, articulation, bridge, low-reliability, failure-sensitive, pruning,
  observability, and verifier-placement diagnostics.
- Role-critical routing with context-sensitivity scores, graded context
  allocation, calibrated compression ratios, risk annotations, and
  verifier-placement coupling.
- Workflow templates for agent-inspired workflows and synthetic graph families.
- Quality scorers for exact match, human labels, rubrics, and injected
  LLM-as-judge adapters.
- Dependency-light ML baselines, optional torch GNNs, Q-learning, REINFORCE, PPO,
  expanded workflow-control actions, a category-conditioned online bandit,
  checkpoints, and artifact registry.
- Framework interchange adapters and optional native hooks for LangGraph, CrewAI,
  and OpenAI Agents SDK.
- Claude Code/Codex instructions and a lightweight stdio MCP server.

## Documentation

- [Documentation index](docs/index.md)
- [Tutorial](docs/tutorial.md)
- [Quality-aware routing](docs/routing_quality.md)
- [Case-study protocol](docs/research/case_study_protocol.md)
- [ML/DL/RL guide](docs/deep_learning.md)
- [Coding-agent integration](docs/coding_agents.md)
- [Framework integrations](docs/framework_integrations.md)
- [Publishing](docs/publishing.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)

## Development

```bash
ruff check .
mypy src
pytest
```

CI runs the same gates on every push and pull request.
