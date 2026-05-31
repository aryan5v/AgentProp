# AgentProp v1 Release Checklist

AgentProp is not v1-complete until every required item below has evidence.

## Product Surface

- [x] Package installs locally.
- [x] CLI has `analyze`, `optimize`, `benchmark`, and `report`.
- [x] README explains the core differentiation.
- [x] Workflow JSON schema is documented.
- [x] Reports can be written as Markdown or JSON.
- [x] Public documentation site or equivalent docs index exists.
- [x] At least one notebook-style tutorial exists.

## Core Graph Framework

- [x] Directed weighted graph abstraction.
- [x] Node and edge metadata.
- [x] JSON import/export.
- [x] NetworkX conversion.
- [x] Workflow validation.
- [x] Built-in workflow templates.
- [x] Trace ingestion from message logs.
- [x] Graph visualization export.

## Propagation and Algorithms

- [x] Independent Cascade.
- [x] Linear Threshold.
- [x] Bootstrap Percolation.
- [x] Randomized Zero Forcing.
- [x] Random, degree, PageRank, betweenness baselines.
- [x] Greedy influence maximization.
- [x] CELF.
- [x] Cost-aware greedy.
- [x] Bottleneck detection.
- [x] Pruning candidates.
- [x] Verifier placement.
- [x] Classical zero forcing completeness model.
- [x] More rigorous pruning evaluation.

## ML / DL / RL

- [x] ML feature extraction.
- [x] Greedy-labeled dataset builder.
- [x] Lightweight trainable node scorer.
- [x] Message-passing-style scorer.
- [x] Sequential routing environment.
- [x] Reproducible ML/RL experiment scripts.
- [x] Optional deep-learning backend interface.
- [x] Optional torch-based GNN implementation.
- [x] RL training loop beyond greedy one-step policy.

## Research Readiness

- [x] PRD.
- [x] Literature review.
- [x] Benchmark runner.
- [x] Benchmark fixtures.
- [x] Result artifact scripts.
- [x] First saved benchmark result table committed or attached to release.
- [x] Real LLM case-study protocol.
- [x] Paper outline / research memo.

## Open Source Readiness

- [x] License.
- [x] Changelog.
- [x] Contributing guide.
- [x] CI runs lint, type-check, and tests.
- [x] Current `main` has green CI.
- [x] Version tag.
- [x] Release notes.
- [x] Repository visibility decision.
