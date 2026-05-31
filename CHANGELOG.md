# Changelog

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
- Product PRD, literature review, case-study protocol, and paper outline docs.
