# Changelog

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
