# AgentProp

AgentProp is an open-source framework for graph optimization in multi-agent LLM workflows.

It models agents, tools, memories, documents, and verifiers as nodes in a directed weighted graph, then uses graph propagation models, classical graph algorithms, graph neural networks, and reinforcement learning to optimize context routing, verifier placement, and topology pruning.

## What It Helps With

- Select which agents receive full context first.
- Identify redundant communication edges.
- Place verifier agents where corrections spread quickly.
- Simulate information and correction propagation.
- Benchmark training-free, GNN, and RL routing policies.

## Quickstart

```python
from agentprop import AgentGraph

graph = AgentGraph()
graph.add_agent("planner", token_cost=1000)
graph.add_agent("coder", token_cost=1500)
graph.add_agent("tester", token_cost=900)
graph.add_agent("reviewer", token_cost=800)

graph.add_edge("planner", "coder", message_cost=500, weight=0.9)
graph.add_edge("coder", "tester", message_cost=400, weight=0.8)
graph.add_edge("tester", "reviewer", message_cost=300, weight=0.7)

print(graph.node_count, graph.edge_count)
```

## Current Status

This repository is in initial setup. The first implementation milestone is:

1. Core `AgentGraph` abstraction.
2. JSON import/export.
3. NetworkX conversion.
4. Propagation model interfaces.
5. Classical seed-selection baselines.

See [docs/PRD.md](docs/PRD.md) for the cleaned product and research plan.
