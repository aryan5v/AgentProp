# AgentProp PRD

## One-Line Summary

AgentProp is an open-source framework for modeling multi-agent LLM systems as directed weighted graphs, then optimizing context routing, verifier placement, communication topology, and token-cost tradeoffs with classical graph algorithms, graph neural networks, and reinforcement learning.

## Project Purpose

AgentProp helps developers and researchers answer:

> In a multi-agent LLM workflow, which agents, tools, memory nodes, or verifier nodes should receive context first, communicate with each other, or be pruned, so that the workflow remains accurate while using fewer tokens, fewer messages, and less latency?

The project should not be positioned as "zero forcing for LLM agents." The stronger framing is:

> A graph-theoretic optimization and benchmarking framework for multi-agent LLM workflows, using influence maximization, topology pruning, randomized zero forcing, graph neural networks, and reinforcement learning.

## Research Framing

The central research question is:

> Can training-free graph algorithms and learned GNN/RL policies reduce communication cost in multi-agent LLM workflows while preserving task success, consistency, and reliability?

Specific questions:

- Can graph algorithms reduce token cost compared with broadcast-style communication?
- Can classical methods compete with learned topology-optimization methods?
- Can randomized zero forcing or related graph propagation models help analyze agent communication?
- Can GNNs learn better seed-selection or pruning policies than hand-designed centrality methods?
- Can RL learn sequential communication policies that decide when to route context, call tools, activate verifiers, or stop?

## Connection to Dr. Jesse Geneson's Work

This project connects to Dr. Geneson's work through zero forcing, randomized zero forcing, graph propagation, propagation time, directed and weighted graph processes, graph invariants, and network dynamics.

AgentProp should not claim that LLM agents literally follow classical zero forcing. A better statement:

> Dr. Geneson's work studies graph processes where activation, information, or control spreads through networks. AgentProp applies related graph-process ideas to modern multi-agent AI systems, where agents, tools, memories, and verifiers form communication graphs.

Zero forcing should be treated as one inspiration and one optional propagation model, not the entire brand.

The framework should compare multiple propagation models:

- Independent Cascade
- Linear Threshold
- Bootstrap Percolation
- Randomized Zero Forcing
- Deterministic zero forcing
- Learned propagation models from traces

## Problem

Modern agent workflows often use hardcoded communication patterns:

- Supervisor-worker
- Planner-coder-tester-reviewer
- Debate
- Research-writer-verifier
- Tool-using agents
- Graph-of-thought workflows

These systems often suffer from:

- High token cost
- Repeated context
- Redundant communication
- Unclear verifier placement
- Slow propagation of corrections
- Fragile agent dependencies
- Hard-to-debug communication graphs

Most frameworks help developers wire agents together, but they do not deeply answer:

- Which agents actually need full context?
- Which edges are redundant?
- Where should verifiers be placed?
- Which communication paths are bottlenecks?
- How does an error or correction propagate through the workflow?

AgentProp answers these questions using graph algorithms and learned policies.

## Positioning

### Do Not Claim

- Nothing like this exists.
- AgentProp invented seed selection.
- Zero forcing directly models LLM communication.
- AgentProp replaces LangGraph, AutoGen, CrewAI, or other agent frameworks.
- AgentProp solves multi-agent reliability.

### Do Claim

- Multi-agent LLM workflows can be represented as directed weighted graphs.
- Existing workflows often use ad hoc context routing and communication patterns.
- Classical graph algorithms may provide cheap, training-free optimization baselines.
- GNNs and RL can improve routing, pruning, and verifier-placement decisions.
- Randomized zero forcing and related graph propagation models provide useful structural analysis tools.
- The framework offers a unified benchmark and open-source toolkit for comparing these methods.

## Name

Recommended name: **AgentProp**

Full title:

> AgentProp: Graph Optimization for Multi-Agent LLM Workflows

Possible subtitle:

> Training-Free and Learned Propagation Models for Context Routing, Verifier Placement, and Topology Pruning

Recommended package name: `agentprop`

## Target Users

### Multi-Agent LLM Developers

People building with LangGraph, AutoGen, CrewAI, OpenAI Agents SDK, LlamaIndex workflows, or custom orchestration. They want to reduce token cost, latency, redundant messages, and brittle routing.

### ML Researchers

Researchers studying agent communication, graph neural networks, reinforcement learning, multi-agent systems, topology optimization, and evaluation benchmarks.

### Graph Theory / Network Science Researchers

Researchers interested in propagation models, influence maximization, zero forcing, randomized zero forcing, graph dynamics, diffusion, and controllability.

### Agent Observability / Reliability Researchers

People studying debugging traces, communication redundancy, error propagation, verifier placement, and safety monitoring.

## Core User Stories

### Developer Stories

- As a developer, I have 8 agents in my workflow. I do not want to send the full prompt to every agent. I want AgentProp to recommend which 2-3 agents should receive full context first.
- As a developer, I have a dense communication graph. I want to identify which communication edges are redundant so I can reduce token usage.
- As a developer, I want to know where to place verifier agents so corrections spread quickly through the workflow.
- As a developer, I want to upload or log a workflow trace and receive graph metrics, bottleneck analysis, and pruning suggestions.
- As a developer, I want to compare broadcast routing, supervisor-only routing, centrality-based routing, influence-maximization routing, GNN routing, and RL routing.

### Researcher Stories

- As a researcher, I want to benchmark classical graph algorithms against learned GNN and RL policies on agent workflow graphs.
- As a researcher, I want to compare Independent Cascade, Linear Threshold, Bootstrap Percolation, Randomized Zero Forcing, and learned propagation models.
- As a researcher, I want to train a GNN policy on synthetic workflows and test whether it generalizes to unseen workflow graphs.
- As a researcher, I want an RL environment where an agent learns when to send context, call tools, prune communication, activate verifiers, or stop.

## Core Data Model

Represent every workflow as a directed weighted graph:

```python
G = (V, E)
```

Where:

- `V` = agents, tools, memory nodes, verifier nodes, documents, subtasks
- `E` = communication paths, dependencies, retrieval links, tool-call dependencies

Supported node types:

```text
AGENT
TOOL
MEMORY
DOCUMENT
VERIFIER
PLANNER
EXECUTOR
REVIEWER
OUTPUT
CUSTOM
```

Example node attributes:

```python
{
    "name": "coder",
    "type": "AGENT",
    "role": "writes code",
    "token_cost": 1200,
    "latency": 2.3,
    "reliability": 0.82,
    "error_rate": 0.11,
    "context_capacity": 8000,
    "tool_access": ["python", "github"],
    "importance_score": None,
}
```

Example edge attributes:

```python
{
    "source": "planner",
    "target": "coder",
    "message_cost": 600,
    "latency": 1.1,
    "relevance": 0.85,
    "reliability": 0.9,
    "activation_probability": 0.7,
    "dependency_strength": 0.8,
}
```

## Product Modules

```text
agentprop/
  core/
  graph/
  propagation/
  algorithms/
  ml/
  rl/
  workflows/
  integrations/
  evaluation/
  visualization/
  experiments/
  examples/
  docs/
```

## Architecture

```text
User Workflow / Synthetic Workflow / Trace
              |
              v
       Agent Graph Builder
              |
              v
   Directed Weighted Agent Graph
              |
              v
+-------------+--------------+--------------+
| Classical   | GNN / ML     | RL Policy    |
| Algorithms  | Policies     | Learning     |
+-------------+--------------+--------------+
              |
              v
      Propagation Simulator
              |
              v
        Evaluation Metrics
              |
              v
 Recommendations + Plots + Reports
```

## Level 1: Classical Graph Algorithms

This is the training-free layer and should be the stable product core.

Primary tasks:

- Seed selection
- Verifier placement
- Topology pruning
- Bottleneck detection
- Propagation simulation

Algorithms to include:

- Random
- Degree, in-degree, and out-degree centrality
- PageRank
- Betweenness centrality
- Closeness centrality
- K-core / K-shell
- Greedy influence maximization
- CELF
- Cost-aware greedy
- Randomized zero forcing heuristic
- Bootstrap percolation heuristic

Edge pruning methods:

- Low-weight pruning
- Low-usage pruning
- Betweenness-preserving pruning
- Reachability-preserving pruning
- Cost-aware edge pruning
- Redundancy-based pruning

Verifier placement methods:

- Betweenness verifier placement
- High-risk node verifier placement
- Error-propagation centrality
- Greedy correction-coverage placement
- PageRank verifier placement

Bottleneck detection methods:

- Articulation points
- Bridges
- High-betweenness nodes
- High-cost / high-traffic edges
- Low-reliability cut points
- Failure-sensitive nodes

## Level 2: ML / Deep Learning

The learned static policy layer should train models that learn from graph structure and workflow traces.

Main task:

> Given an agent workflow graph, score each node or edge for seed selection, verifier placement, or pruning.

GNN tasks:

- Seed node scoring
- Edge pruning scoring
- Verifier placement scoring

Model options:

- GCN
- GraphSAGE
- GAT
- GIN
- MLP baseline

Later:

- Graph Transformer
- Heterogeneous GNN
- Edge-conditioned GNN

Training strategies:

- Imitation learning from greedy/CELF decisions
- Supervised regression on propagation time or marginal gain
- Pairwise ranking
- RL-based fine-tuning later

## Level 3: Reinforcement Learning

The RL layer handles sequential decision-making.

Example decisions:

- Which agent gets context next?
- Should a verifier be activated now?
- Should an edge be used or pruned?
- Should the workflow continue or stop?
- Should a tool be called?
- Should the system broadcast or route selectively?

Create a Gymnasium-compatible environment:

```python
class AgentRoutingEnv(gym.Env):
    ...
```

Initial V1 action space:

```text
SELECT_NEXT_SEED_NODE
STOP
```

Expanded action space:

```text
SEND_CONTEXT(node)
ACTIVATE_VERIFIER(node)
SEND_MESSAGE(edge)
PRUNE_EDGE(edge)
CALL_TOOL(node)
REQUEST_SUMMARY(node)
STOP
```

Base reward:

```text
reward = task_success
         - lambda_1 * token_cost
         - lambda_2 * latency
         - lambda_3 * message_count
         - lambda_4 * error_count
         - lambda_5 * correction_delay
```

Simplified early reward:

```text
reward = full_activation_success
         - lambda_1 * propagation_time
         - lambda_2 * communication_cost
```

## Propagation Models

No single graph propagation model is obviously correct for agent workflows. AgentProp should compare several:

- Independent Cascade
- Linear Threshold
- Bootstrap Percolation
- Randomized Zero Forcing
- Classical zero forcing
- Learned propagation model from traces

Classical zero forcing must be documented carefully:

> Classical zero forcing is not assumed to be a literal model of LLM communication.

## Integrations

Phase 1 starts with manual graph definition:

```python
from agentprop import AgentGraph

graph = AgentGraph()
graph.add_agent("planner")
graph.add_agent("coder")
graph.add_agent("tester")
graph.add_agent("reviewer")
graph.add_edge("planner", "coder", message_cost=500)
graph.add_edge("coder", "tester", message_cost=300)
graph.add_edge("tester", "reviewer", message_cost=250)
```

Phase 2 adapters:

- LangGraph
- AutoGen
- CrewAI
- OpenAI Agents SDK
- LlamaIndex workflows

Trace ingestion should support logs with:

```json
{
  "nodes": [],
  "messages": [],
  "token_costs": [],
  "latencies": [],
  "success": true
}
```

## Metrics

Core metrics:

- Task success rate
- Token cost
- Message count
- Latency
- Propagation time
- Coverage
- Activation probability
- Correction propagation delay
- Verifier coverage
- Edge redundancy score
- Node bottleneck score
- Robustness under node failure
- Robustness under edge failure

Cost-quality metrics:

```text
Cost-adjusted success = task_success / token_cost
Efficiency score = success_rate - lambda * normalized_token_cost
```

Propagation metrics:

- Expected propagation time
- Full activation probability
- Average activation round
- Activation coverage after `t` rounds
- Number of unreachable nodes

Reliability metrics:

- Correction delay
- Error persistence
- Compromised node spread
- Verifier interception rate
- Failure recovery rate

## Benchmark Workflows

Synthetic templates:

- Chain
- Star
- Tree
- DAG
- Dense graph
- Small-world graph
- Hub-and-spoke
- Layered pipeline
- Random directed graph

Agent-inspired templates:

- Planner-coder-tester-reviewer
- Research-writer-verifier
- Debate
- RAG pipeline
- Tool-use workflow

## First Real LLM Evaluation

Goal:

> Validate that graph optimization helps real workflows, not only simulations.

Minimal workflow:

```text
Planner -> Researcher -> Writer -> Verifier -> Finalizer
```

Compare:

- Broadcast full context to all agents
- Planner-only context
- Random seed agents
- PageRank seed agents
- Greedy influence seed agents
- GNN-selected seed agents
- RL-selected routing

Tasks:

- Short research QA
- Document summarization
- Coding problem explanation
- Bug localization
- Planning task

Metrics:

- Final answer quality
- Token cost
- Number of messages
- Latency
- Verifier correction success
- Consistency across agents

## Developer CLI

Target commands:

```bash
agentprop analyze workflow.json
agentprop optimize workflow.json --budget 3
agentprop simulate workflow.json --model independent-cascade
agentprop prune workflow.json --target-token-reduction 0.3
agentprop report workflow.json --out report.html
```

Reports should include:

- Recommended seed agents
- Redundant edges
- Bottleneck nodes
- Verifier placement suggestions
- Estimated token savings
- Propagation simulation
- Risk analysis
- Visual graph

## Development Phases

### Phase 0: Project Setup

Duration: 1-2 days

Deliverables:

- Working install
- Passing test skeleton
- Package import works
- README draft
- Basic docs
- CI

### Phase 1: Core Graph Abstraction

Duration: 2-4 days

Deliverables:

- `AgentGraph`
- Directed weighted graph support
- Node and edge attributes
- NetworkX conversion
- JSON import/export
- Workflow templates

### Phase 2: Propagation Engine

Duration: 4-7 days

Deliverables:

- Base propagation interface
- Independent Cascade
- Linear Threshold
- Bootstrap Percolation
- Randomized Zero Forcing
- Deterministic zero forcing

### Phase 3: Classical Algorithms

Duration: 5-8 days

Deliverables:

- Random, degree, PageRank, and betweenness seed selection
- Greedy influence maximization
- CELF
- Cost-aware greedy
- Edge pruning
- Verifier placement

### Phase 4: Metrics and Evaluation Runner

Duration: 4-6 days

Deliverables:

- Token cost
- Message count
- Propagation time
- Coverage
- Latency
- Robustness
- Correction delay
- Benchmark runner

### Phase 5: GNN / ML Layer

Duration: 1-2 weeks

Deliverables:

- Graph dataset generator
- Greedy/CELF labels
- GCN / GraphSAGE / GAT
- Node-scoring policy
- Top-k seed evaluation

### Phase 6: RL Layer

Duration: 1-2 weeks

Deliverables:

- `AgentRoutingEnv`
- State/action/reward definitions
- Simple seed-selection RL
- PPO or REINFORCE training
- Evaluation against greedy and GNN

### Phase 7: Workflow Templates

Duration: 4-6 days

Deliverables:

- Chain
- Star
- Tree
- Dense
- Planner-coder-tester-reviewer
- Research-writer-verifier
- Debate
- RAG
- Tool-use

### Phase 8: CLI and Reports

Duration: 4-7 days

Deliverables:

- Analyze command
- Optimize command
- Prune command
- Simulate command
- Report command
- Markdown/HTML reports
- CSV results

### Phase 9: LLM Case Study

Duration: 1-2 weeks

Deliverables:

- Real LLM workflow
- Broadcast vs optimized comparisons
- At least 20 tasks
- Results table

## MVP Scope

### Must Include

- `AgentGraph` abstraction
- JSON workflow schema
- Independent Cascade
- Linear Threshold
- Randomized Zero Forcing
- Random seed baseline
- Degree baseline
- PageRank baseline
- Greedy baseline
- CELF baseline
- Token/message cost metrics
- Synthetic workflow templates
- Benchmark runner
- CLI optimize command
- Basic report generation

### Should Include

- GNN seed selector
- RL seed-selection environment
- Planner-coder-tester workflow
- Research-writer-verifier workflow
- Simple LLM case study

### Does Not Need Yet

- Full LangGraph integration
- Full AutoGen integration
- Advanced multi-agent RL
- Large-scale benchmark suite
- Perfect learned baselines
- Production-grade UI

## Paper Plan

Recommended title:

> Training-Free Graph Optimization for Multi-Agent LLM Workflows

Subtitle:

> A Benchmark of Propagation Models, Seed Selection, and Topology Pruning

Main claim:

> Classical graph algorithms and lightweight learned policies can reduce communication cost in multi-agent LLM workflows while preserving task success, providing a practical alternative or complement to learned topology-optimization systems.

Contributions:

1. A formal graph model for multi-agent LLM workflows.
2. An open-source benchmark for context propagation, verifier placement, and topology pruning.
3. A comparison of influence maximization, randomized zero forcing, centrality, GNN, and RL methods.
4. An empirical study showing cost-quality tradeoffs on synthetic and real agent workflows.

## Success Criteria

Product success:

- A developer can define a workflow graph.
- The package recommends seed nodes or pruning decisions.
- The package simulates propagation.
- The package outputs useful reports.
- The package runs benchmarks reproducibly.

Research success:

- Training-free graph methods reduce token/message cost by 20%+ with little quality loss.
- GNN policies outperform centrality baselines on propagation or cost-quality metrics.
- RL policies learn useful sequential routing strategies.
- Randomized zero forcing gives useful structural insight into at least one workflow class.

Minimum publishable result:

> On synthetic and agent-inspired workflows, AgentProp shows that training-free graph algorithms can reduce communication cost while preserving propagation coverage, and provides a benchmark for comparing classical, GNN, and RL routing policies.

## Key Risks

### Propagation Models May Not Match Real Agents

Mitigation:

- Compare multiple propagation models.
- Include a real LLM case study.
- Avoid overclaiming zero forcing.

### Classical Methods May Be Too Weak

Mitigation:

- Position them as cheap baselines.
- Compare cost vs quality.
- Add GNN/RL methods.

### Developers May Not Need This

Mitigation:

- Build practical CLI/reporting.
- Integrate with LangGraph.
- Focus on token cost and debugging.
- Validate with real workflows.

### Existing Literature May Already Cover Too Much

Mitigation:

- Explicitly cite influence maximization and agent topology optimization.
- Focus on training-free + unified benchmark + production trace analysis.
- Avoid novelty overclaims.

### Building All Three Levels May Cause Scope Creep

Mitigation:

- Keep clear interfaces.
- Build MVP first.
- Keep ML/RL optional.
- Require tests and examples for every module.

## Non-Negotiable Design Principles

- Be honest about theory.
- Be useful to developers.
- Make everything benchmarkable.
- Keep training-free methods strong.
- Make ML/RL optional but supported.

## First Seven-Day Plan

### Day 1

- Create repo.
- Set up package.
- Define `AgentGraph`.
- Define JSON schema.
- Create README.

### Day 2

- Implement NetworkX conversion.
- Add workflow templates.
- Add chain/star/DAG/planner-coder-tester examples.

### Day 3

- Implement Independent Cascade.
- Implement Linear Threshold.
- Implement Randomized Zero Forcing.
- Add simulation result object.

### Day 4

- Implement random/degree/PageRank/betweenness seed selection.
- Add propagation metrics.
- Add first benchmark script.

### Day 5

- Implement greedy influence maximization.
- Implement CELF.
- Save results to CSV.
- Generate first plots.

### Day 6

- Implement edge pruning.
- Implement verifier placement.
- Implement CLI prototype.

### Day 7

- Run first full benchmark.
- Write first research memo draft.
- Create 2-3 plots.
- Open issues for GNN/RL agents.

## Obvious Gaps / Decisions Needed

These are the highest-signal gaps to close before the project becomes larger:

1. **Workflow trace schema**: The PRD names trace ingestion, but it needs a concrete event format for messages, tool calls, verifier corrections, token counts, and latency.
2. **Quality metric definition**: "Task success" and "final answer quality" need a scorer: human labels, LLM-as-judge, exact-match tests, or task-specific metrics.
3. **Ground truth for pruning**: Edge pruning needs a way to know whether removing an edge harmed task quality, not just propagation coverage.
4. **LLM evaluation budget**: The real LLM case study needs a fixed model choice, task count, retry policy, and cost ceiling.
5. **Literature baseline list**: The paper plan should explicitly track related work such as influence maximization, learned agent topology optimization, agent pruning, debate routing, graph-of-thoughts, and multi-agent orchestration frameworks.
6. **Package dependency tiers**: Core users should not install ML/RL stacks by default. Keep `agentprop` core light, with extras such as `agentprop[ml]`, `agentprop[rl]`, and `agentprop[viz]`.
7. **First demo path**: The MVP should choose one end-to-end demo first: planner-coder-tester-reviewer with PageRank/CELF vs broadcast routing.
8. **Verifier semantics**: Verifier placement depends on what a verifier can observe and correct. This needs a clear model before algorithms can be evaluated honestly.
9. **Directed vs undirected algorithm behavior**: Some classical graph algorithms behave differently or have restrictions on directed graphs. The implementation should document each choice.
10. **Reproducibility policy**: Simulations and LLM evaluations need seeds, saved configs, result artifacts, and environment metadata from day one.

## Immediate Next Step

Implement:

```text
AgentGraph + Independent Cascade + PageRank/CELF seed selection + benchmark runner
```

The first meaningful demo should be:

> Given a planner-coder-tester-reviewer workflow, AgentProp recommends seed agents, simulates propagation, estimates token/message cost, and compares broadcast vs optimized routing.
