# Changelog

## 0.1.0-alpha.4 - 2026-06-06

Coding-agent beta distribution refresh:

- Added a full-suite coding-agent wrapper example that combines graph analysis,
  seed selection, metric-dimension verifier placement, context-risk advice,
  `ControlSession`, runtime decisions, and saved trace/report artifacts.
- Added `docs/beta_quickstart.md` for Codex CLI, Claude Code, MCP setup,
  runtime wrapping, evidence preservation, and troubleshooting.
- Added a same-repo Codex plugin bundle at `plugins/agentprop/` with packaged
  AgentProp skill content and `agentprop-mcp` stdio configuration.
- Added repo marketplace metadata at `.agents/plugins/marketplace.json` and
  documented the future Vanta-style dedicated plugin repo path in
  `docs/plugin_distribution.md`.
- Added tests for plugin manifest/marketplace integrity and example artifact
  generation.
- Added `--out-dir` support to `examples/coding_agent_full_suite.py` so smoke
  tests and users can keep generated artifacts outside the repository.
- Clarified `agentprop doctor --tier graph` output by marking optional Graphviz
  as a warning instead of a failed install check.
- Updated MCP server metadata to derive its version from installed package
  metadata instead of a hardcoded string.

## 0.1.0-alpha.3 - 2026-06-05

Beta-control-layer and skill distribution refresh:

- Added `ControlSession`, a public analysis-backed runtime facade that records
  events, returns control decisions, and writes `trace.jsonl`, `summary.json`,
  and `report.md`.
- Added key-free `agentprop control-demo` flows for terminal, multi-agent, and
  framework-style workflows.
- Added FastMCP-backed MCP tools with a dependency-free JSON-RPC fallback:
  analysis, optimization, report, coding-agent brief, and live control-session
  lifecycle tools.
- Added the optional `agentprop[mcp]` package extra.
- Added the installable `agentprop-workflow-optimizer` skill package for
  skills.sh-style distribution, including task-specific references for install,
  workflow analysis, coding agents, runtime control, framework builders, and
  benchmark evidence.
- Updated public docs and README with control-layer quickstarts, MCP install
  guidance, skills.sh install commands, and responsible evidence language.

## 0.1.0-alpha.2 - 2026-06-04

Research-facing public alpha refresh:

- Reframed AgentProp around graph observability, quality propagation, RZF
  scaling, and runtime control.
- Added metric-dimension verifier placement, resolving coverage, and
  fault-tolerant resolving coverage.
- Added Quality Cascade propagation and quality-decay experiment support.
- Added RZF process-based centrality/scaling evidence scripts.
- Added runtime-control improvements for terminal and agent loops, including
  verifier forcing, stale-verifier avoidance, local-pass distrust, deferred
  command handling, and pass-preserving finalization.
- Added a category-conditioned bandit policy with safer reward shaping.
- Documented an early Codex CLI A0-vs-A2 pass-preserving token/cost reduction
  signal.
- Cleaned the README and documentation index for public research review.

## 0.1.0-alpha.1 - 2026-05-31

Initial AgentProp foundation:

- Directed weighted `AgentGraph` with JSON and NetworkX conversion.
- Workflow JSON validation.
- Propagation models:
  - Independent Cascade
  - Linear Threshold
  - Bootstrap Percolation
  - Randomized Zero Forcing
  - deterministic Zero Forcing
- Training-free seed-selection baselines:
  - random
  - degree
  - PageRank-style influence score
  - betweenness
  - greedy influence maximization
  - CELF
  - cost-aware greedy
- Bottleneck, pruning, and verifier-placement heuristics.
- Observability metrics and pruning evaluation.
- Broadcast vs optimized routing cost comparison.
- CLI commands:
  - `agentprop analyze`
  - `agentprop optimize`
  - `agentprop benchmark`
  - `agentprop report`
- Built-in workflow templates and benchmark fixtures.
- Markdown/JSON report generation.
- Lightweight ML foundations for seed-scoring experiments.
- Optional torch GNN seed scorers for GCN, GraphSAGE, and GAT.
- Lightweight sequential RL routing environment and Q-learning policy.
- Reproducible experiment scripts for benchmark, ML, and RL runs.
- Saved benchmark result table and first SVG plot.
- Literature review, case-study protocol, tutorial, and research docs.

## Unreleased

- Continue repeated Terminal-Bench and SWE-style control evaluations.
