# AgentProp

AgentProp is an open-source framework for graph optimization in multi-agent LLM workflows.

It models agents, tools, memories, documents, and verifiers as nodes in a directed weighted graph, then uses graph propagation models, classical graph algorithms, graph neural networks, and reinforcement learning to optimize context routing, verifier placement, and topology pruning.

## Why AgentProp Is Different

AgentProp is not another agent orchestrator. It is an analysis and optimization layer for workflows you already have or want to study.

- **Training-free first:** strong graph baselines before expensive learning.
- **Graph-theoretic grounding:** propagation time, influence maximization, randomized zero forcing, verifier placement, and bottleneck analysis.
- **Cost-aware by design:** every recommendation compares against broadcast routing with token/message cost estimates.
- **Research-ready:** benchmark runner, propagation models, workflow templates, and ML/RL extension points share one graph abstraction.
- **Developer-usable:** CLI commands produce immediate seed, pruning, verifier, and savings recommendations.

## What It Helps With

- Select which agents receive full context first.
- Identify redundant communication edges.
- Place verifier agents where corrections spread quickly.
- Simulate information and correction propagation.
- Benchmark training-free, GNN, and RL routing policies.

## Quickstart

Install locally:

```bash
python -m pip install -e .
```

Run the first optimization demo:

```bash
agentprop optimize benchmarks/workflows/planner_coder_tester_reviewer.json --budget 2
```

Compare algorithms and propagation models:

```bash
agentprop benchmark planner_coder_tester_reviewer --budget 2 --trials 50
```

Use AgentProp as a library:

```python
from agentprop import AgentGraph
from agentprop.algorithms import greedy_seed_selection
from agentprop.propagation import IndependentCascade

graph = AgentGraph()
graph.add_agent("planner", token_cost=1000)
graph.add_agent("coder", token_cost=1500)
graph.add_agent("tester", token_cost=900)
graph.add_agent("reviewer", token_cost=800)

graph.add_edge("planner", "coder", message_cost=500, weight=0.9)
graph.add_edge("coder", "tester", message_cost=400, weight=0.8)
graph.add_edge("tester", "reviewer", message_cost=300, weight=0.7)

model = IndependentCascade(seed=0)
seeds = greedy_seed_selection(graph, k=2, propagation_model=model, trials=100)

print(seeds)
```

## Core Features

- Directed weighted `AgentGraph` abstraction with JSON and NetworkX conversion.
- Propagation models:
  - Independent Cascade
  - Linear Threshold
  - Bootstrap Percolation
  - Randomized Zero Forcing
- Training-free seed selection:
  - Random
  - Degree
  - PageRank-style influence score
  - Betweenness
  - Greedy influence maximization
  - CELF lazy greedy
  - Cost-aware greedy
- Diagnostics:
  - Bottleneck nodes
  - Low-weight pruning candidates
  - Risk-aware verifier placement
- Evaluation:
  - Broadcast vs optimized cost comparison
  - Coverage
  - Expected propagation time
  - Full activation probability
  - Estimated token/message savings
- ML/RL foundations:
  - Graph feature extraction
  - Greedy-labeled seed-selection examples
  - Lightweight linear node scorer
  - Message-passing node scorer
  - Sequential routing environment

## Built-In Workflow Templates

- `planner_coder_tester_reviewer`
- `research_writer_verifier`
- `debate_judge`
- `rag_pipeline`
- `tool_use_pipeline`
- `hub_and_spoke_supervisor`

## Research Direction

AgentProp studies whether training-free graph algorithms and learned GNN/RL policies can reduce communication cost in multi-agent LLM workflows while preserving task success, consistency, and reliability.

- [docs/index.md](docs/index.md) for the documentation index.
- [docs/tutorial.md](docs/tutorial.md) for the first full walkthrough.
- [docs/PRD.md](docs/PRD.md) for the cleaned product and research plan.
- [docs/research/literature_review.md](docs/research/literature_review.md) for the academic framing and related-work map.
- [docs/trace_ingestion.md](docs/trace_ingestion.md) for converting message logs into workflow graphs.
- [docs/visualization.md](docs/visualization.md) for Graphviz DOT exports.
- [docs/deep_learning.md](docs/deep_learning.md) for the optional DL roadmap.
- [docs/release_checklist.md](docs/release_checklist.md) for the v1 readiness checklist.
- [CONTRIBUTING.md](CONTRIBUTING.md) for development and contribution guidance.
