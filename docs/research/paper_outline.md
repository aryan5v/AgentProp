# AgentProp Paper Outline

## Working Title

AgentProp: Graph Propagation and Learned Routing for Multi-Agent LLM Workflows

## Abstract Claim

Multi-agent LLM workflows are usually designed as orchestration code, but their
communication patterns can be modeled as directed weighted graphs. AgentProp
compares training-free graph algorithms, propagation models, GNN seed scorers,
and reinforcement-learning routing policies for reducing communication cost
while preserving task quality.

## Research Questions

1. Can graph propagation models identify compact context-seed sets for agent
   workflows?
2. Do training-free graph algorithms provide strong baselines against learned
   GNN and RL policies?
3. Can verifier placement and pruning metrics surface useful workflow-design
   recommendations?
4. How much token/message cost can be reduced before output quality degrades?

## Method

Represent each workflow as an `AgentGraph`:

- nodes: agents, tools, memories, documents, verifiers, outputs
- edges: directed communication paths with message cost, latency, reliability,
  and propagation weight
- propagation models: Independent Cascade, Linear Threshold, Bootstrap
  Percolation, Randomized Zero Forcing, deterministic Zero Forcing
- algorithms: random, degree, PageRank, betweenness, greedy, CELF,
  cost-aware greedy, GNN scorer, Q-learning routing

## Experiments

1. Synthetic workflow benchmarks using the six built-in workflow templates.
2. Ablations across propagation models and seed-selection algorithms.
3. Pruning and verifier-placement diagnostic examples.
4. Real LLM case study using the protocol in
   [case_study_protocol.md](case_study_protocol.md).

## Primary Metrics

- coverage
- expected propagation time
- full activation probability
- broadcast cost
- optimized routing cost
- estimated savings
- task verification pass rate
- human quality score

## Expected Contributions

- Open-source graph abstraction for multi-agent LLM workflow analysis.
- Reproducible benchmark fixtures and result artifacts.
- Training-free baselines for context routing and topology diagnosis.
- Optional torch GNN and dependency-light RL policy baselines.
- Real LLM case-study protocol for validating cost-quality trade-offs.

## Threats To Validity

- Synthetic propagation does not guarantee real task success.
- Token/message cost is easier to measure than semantic quality.
- Small workflow graphs may favor classical algorithms over learned policies.
- Human quality scoring can be noisy without blinded review.
- Different LLM providers may change cost/latency behavior over time.

## Current Evidence

Committed v1 benchmark artifacts are in [../results/v1](../results/v1).

The evidence is sufficient for an alpha release, but not yet sufficient for a
strong paper claim about real LLM task quality. That requires completing the
case-study protocol.
