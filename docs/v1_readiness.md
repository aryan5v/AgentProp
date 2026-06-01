# AgentProp V1 Readiness

AgentProp is ready for a serious private alpha / v1-candidate rollout, but not
for unqualified public claims until the real routed LLM case study has saved
results.

Run the audit at any time:

```bash
agentprop readiness
agentprop readiness --json
agentprop readiness --out docs/results/readiness.md
```

## Current Rollup

- Overall score: 81.1%
- Private alpha ready: yes
- Public ready: no
- Status counts: complete=8, alpha=4, blocked=2, missing=0

## What Is Solid

- Core graph backbone: directed weighted graph abstraction, metadata, validation,
  JSON import/export, NetworkX conversion, trace ingestion, templates, and DOT
  visualization.
- Classical algorithms: centrality, greedy, CELF, cost-aware greedy, bottleneck
  diagnostics, cut-point/bridge/failure-sensitive analysis, pruning, and verifier
  placement.
- Propagation models: Independent Cascade, Linear Threshold, Bootstrap
  Percolation, Randomized Zero Forcing, Zero Forcing, and trace-calibrated
  learned propagation.
- Product surface: analyze, optimize, benchmark, report, simulate, prune, trace,
  viz, and agent-instructions CLI commands, with Markdown/JSON/HTML reports.
- Quality surface: exact-match, human-label, rubric, and injected LLM-as-judge
  scorer interfaces.
- Packaging: license, changelog, contributing guide, release notes, CI, docs
  index, and private repository decision.

## What Is Alpha

- Optional torch DL backend: GCN, GraphSAGE, GAT, GIN, Graph Transformer,
  heterogeneous GNN, and edge-conditioned GNN exist, but need larger held-out
  sweeps before we claim model quality.
- Expanded-action RL: Q-learning, REINFORCE, PPO-style baselines, and auditable
  action traces exist, but real-task comparisons are still pending.
- Framework adapters: dependency-light LangGraph, AutoGen, CrewAI, OpenAI Agents
  SDK, and LlamaIndex interchange adapters exist, but native runtime adapters and
  real examples are still pending.
- Coding-agent integrations: Claude Code/Codex briefs and MCP server exist, but
  need more dogfooding against real coding-agent workflows.

## Blockers For Public Claims

1. Real routed LLM case-study results.
   The 20-task protocol, task set, routed LLM harness, preflight, and analysis
   scripts exist. We still need Token Router or OpenAI-compatible credentials,
   real routed multi-node runs, cost/quality/trace artifacts, and verification
   logs from an environment that actually applies and tests generated code.

2. Public launch proof package.
   The public release decision is documented. We still need the first case-study
   result directory, public-facing README plots/screenshots, and the GitHub
   release once the evidence package is ready.

## Practical Decision

Treat the project as roughly 80% to a proper v1 rollout. That last 20% is not a
large amount of code; it is evidence, dogfooding, and packaging:

- run the real LLM case study
- inspect the results for honest savings and quality claims
- do one pass of native framework/coding-agent dogfooding
- turn the saved evidence into README plots and release artifacts
