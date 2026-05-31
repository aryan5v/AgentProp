# AgentProp Literature Review

## Executive Summary

The strongest academic framing for AgentProp is not "zero forcing for LLM agents." That framing is too easy to attack because classical zero forcing has a specific forcing rule and influence maximization already covers seed selection on networks.

The stronger framing is:

> AgentProp is a benchmark and open-source framework for optimizing information flow in multi-agent LLM workflows, comparing classical graph-propagation algorithms, randomized-zero-forcing-style models, GNN policies, and RL routing policies.

Dr. Jesse Geneson's work gives AgentProp a graph-process foundation: zero forcing, probabilistic zero forcing, randomized zero forcing, expected propagation time, graph centrality from propagation dynamics, metric and broadcast dimension, and graph invariants.

The broader literature gives AgentProp its competitive baseline: influence maximization, Independent Cascade, Linear Threshold, CELF, TIM/SKIM-style scalable algorithms, GNN/RL influence maximization, and recent LLM-agent topology optimization work such as GPTSwarm, DyLAN, AgentPrune, G-Designer, AFlow, MaAS, ARG-Designer, and AgentSquare.

The paper should claim:

> We test whether training-free graph algorithms and learned graph policies can reduce token/message cost in multi-agent LLM workflows while preserving task success and reliability.

## Core Academic Thesis

Research question:

> Can graph-theoretic propagation models and learned graph policies optimize communication in multi-agent LLM workflows?

Specific questions:

- Which agents should receive initial task context?
- Which communication edges are redundant?
- Where should verifier agents be placed?
- How fast do context, corrections, or errors propagate through the workflow?
- Can training-free graph algorithms compete with learned topology optimizers?

This connects to Geneson's work because his research studies how activation spreads through graphs and how graph structure controls propagation time. It connects to ML because AgentProp can use GNNs to learn seed, pruning, and verifier policies, and RL to learn sequential routing decisions.

## Dr. Geneson's Most Relevant Work

### Randomized Zero Forcing

Paper: **Randomized Zero Forcing** - Jesse Geneson, Illya Hicks, Noah Lichtenberg, Alvin Moon, Nicolas Robles, 2026.

This is the most important Geneson paper for AgentProp. It introduces randomized zero forcing on directed graphs. A white vertex becomes blue with probability equal to the fraction of its incoming neighbors that are blue. The weighted version replaces neighbor counts with incoming weight proportions. The paper studies expected propagation time, monotonicity under larger initial blue sets and increased outgoing weights, exact/asymptotic values for graph families, and an application to empirical input-output networks as a process-based centrality measure.

How AgentProp uses it:

- Use Randomized Zero Forcing as one propagation model.
- Use directed weighted graphs because agent workflows are naturally directed and weighted.
- Use expected propagation time as an analogue for context propagation time, correction propagation time, information coverage time, and coordination latency.
- Use process-based centrality to motivate AgentProp's centrality scores for agent workflows.
- Be honest: RZF is a model AgentProp compares, not the whole project.

Implementation implication:

```text
src/agentprop/propagation/randomized_zero_forcing.py
```

Core rule:

```text
P(v activates) =
  sum(weight(u, v) for active incoming neighbors u)
  /
  sum(weight(u, v) for all incoming neighbors u)
```

### Propagation Time for Probabilistic Zero Forcing

Paper: **Propagation time for probabilistic zero forcing** - Jesse Geneson and Leslie Hogben, 2018.

This paper studies probabilistic zero forcing as a discrete dynamical system. Since any single blue vertex in a connected graph can eventually color the whole graph blue under probabilistic zero forcing, expected propagation time becomes the main parameter. The paper gives exact values for paths and cycles, asymptotic values for stars, and upper/lower bounds in terms of radius and order.

How AgentProp uses it:

- It justifies expected propagation time as a serious graph parameter, not just an engineering metric.
- It motivates measuring expected rounds to full activation, confidence intervals over stochastic trials, and propagation-time distributions.
- It gives AgentProp a mathematical language for comparing graph structures.

Implementation implication:

Every propagation model should report:

- `expected_propagation_time`
- `activation_probability`
- `coverage_after_t`
- `confidence_interval`

### Markov Chains for Expected Propagation Time

Paper: **Using Markov chains to determine expected propagation time for probabilistic zero forcing** - Yu Chan, Emelie Curl, Jesse Geneson, Leslie Hogben, Kevin Liu, Issac Odegard, Michael S. Ross, 2019.

This paper treats probabilistic zero forcing as a Markov chain and gives an exact formula for expected propagation time from the transition matrix. It applies Markov-chain methods to graph families and propagation-time bounds.

How AgentProp uses it:

- For small workflow graphs, compute exact or near-exact propagation metrics using Markov-chain state enumeration.
- For larger graphs, use Monte Carlo simulation.
- This gives AgentProp two modes: exact mode for small graphs and simulation mode for realistic workflows.

Implementation implication:

```text
src/agentprop/propagation/markov_exact.py
src/agentprop/propagation/monte_carlo.py
```

Research angle:

> For small agent graphs, exact propagation analysis is possible; for large agent graphs, simulation and learned approximations are needed.

### Reconfiguration Graphs of Zero Forcing Sets

Paper: **Reconfiguration graphs of zero forcing sets** - Jesse Geneson, Ruth Haas, Leslie Hogben, 2020.

This paper studies the "zero forcing graph" whose vertices are minimum zero forcing sets of a base graph, with edges connecting sets that differ by one vertex. It shows that zero forcing reconfiguration can be connected or disconnected depending on the base graph, and that computing the zero forcing graph can take exponential time in the worst case.

How AgentProp uses it:

- Seed sets are not just individual choices; they form a search space.
- The landscape of good seed sets can be complex.
- This motivates local search over seed sets, replacing one seed agent at a time, comparing greedy vs. GNN vs. RL search, and studying whether good context-seed sets are clustered or disconnected.

Implementation implication:

```text
src/agentprop/algorithms/local_search.py
```

Example:

> Given seed set `S`, swap one seed agent with another node if propagation improves.

### Broadcast Dimension of Graphs

Paper: **Broadcast Dimension of Graphs** - Jesse Geneson and Eunjeong Yi, 2020.

This paper introduces broadcast dimension, a variant of metric dimension where vertices can act like transmitters of different strengths. The cost is the sum of broadcast strengths, and the goal is to resolve the graph with minimum total transmitter cost.

How AgentProp uses it:

- Not every node needs full context.
- Some nodes can receive high-fidelity context.
- Some nodes can receive summaries.
- Some nodes can receive no context.
- Each context level has a cost.

This motivates variable-strength context seeding:

```text
0 = no context
1 = short summary
2 = task-specific context
3 = full context
```

Optimization objective:

```text
minimize sum(context_strength(v) for v in V)
subject to preserving task success or propagation coverage
```

Implementation implication:

```text
NO_CONTEXT
TASK_HINT
SUMMARY_CONTEXT
FULL_CONTEXT
```

This may become one of the most original bridges to Geneson's work.

### Metric Dimension and Pattern Avoidance in Graphs

Paper: **Metric dimension and pattern avoidance in graphs** - Jesse Geneson, 2018.

This paper studies graph structure under bounded metric dimension and edge metric dimension. Metric dimension is about resolving vertices using distance vectors to selected landmarks.

How AgentProp uses it:

- Metric dimension connects to observability and agent state disambiguation.
- In agent workflows, AgentProp can ask which agents/verifiers need to observe outputs so failures can be localized.
- This is analogous to resolving vertices in a graph.

Possible extension:

- Choose verifier nodes so that error sources are distinguishable.
- Choose logging points so failures can be traced.
- Measure workflow observability dimension.

Implementation implication:

```text
src/agentprop/algorithms/observability.py
```

### Extremal Results for Graphs of Bounded Metric Dimension

Paper: **Extremal results for graphs of bounded metric dimension** - Jesse Geneson, Suchir Kaustav, Antoine Labelle, 2020.

This is lower priority, but it can support theory around workflow observability:

- What workflow structures remain observable with few verifier/logging nodes?
- How dense can a workflow get while still being easy to resolve?
- What topologies create high ambiguity in failure attribution?

## Core Non-Geneson Graph and Diffusion Literature

### Influence Maximization

Foundational paper: **Maximizing the Spread of Influence through a Social Network** - Kempe, Kleinberg, and Tardos, KDD 2003.

Influence maximization asks for `k` seed nodes that maximize expected spread under diffusion models such as Independent Cascade and Linear Threshold. These models are the closest classical match to "choose initial agents so information spreads."

How AgentProp uses it:

- Cite influence maximization directly.
- Do not claim seed selection is new.
- Make Independent Cascade and Linear Threshold first-class propagation models.
- Make greedy influence maximization a required baseline.

Implementation implication:

```text
src/agentprop/propagation/independent_cascade.py
src/agentprop/propagation/linear_threshold.py
src/agentprop/algorithms/greedy_im.py
```

### Submodularity of Influence

Paper: **On the Submodularity of Influence in Social Networks** - Mossel and Roch, 2006.

This paper proves and extends submodularity results for influence processes, supporting the theory behind greedy approximation methods for diffusion-style influence spread.

How AgentProp uses it:

> Under standard IC/LT assumptions, the expected coverage objective is monotone submodular, so greedy seed selection gives a principled approximation baseline.

Important caveat:

> This applies to the diffusion model, not automatically to LLM task success.

### TIM and Scalable Influence Maximization

Paper: **Influence Maximization: Near-Optimal Time Complexity Meets Practical Efficiency** - Tang, Xiao, Shi, 2014.

TIM gives a scalable influence maximization algorithm with near-optimal time complexity and a `(1 - 1/e - epsilon)` approximation guarantee under IC/LT-style models.

How AgentProp uses it:

- Classical baselines should not stop at PageRank and degree.
- V1 should include greedy and CELF.
- A serious paper submission should add TIM/IMM-style reverse reachable set methods.

### SKIM and Sketch-Based Influence Maximization

Paper: **Sketch-based Influence Maximization and Computation** - Cohen, Delling, Pajor, Werneck, 2014.

This paper develops sketch-based influence computation and influence maximization that scale to very large graphs while keeping approximation guarantees.

How AgentProp uses it:

- Relevant if AgentProp supports large production traces.
- Early work can focus on smaller workflow graphs.
- Scalable influence-maximization algorithms become important for large agent ecosystems.

### CyNetDiff

Paper: **CyNetDiff: A Python Library for Accelerated Implementation of Network Diffusion Models**, 2024.

CyNetDiff is a Python/Cython library for accelerated diffusion simulations, especially Independent Cascade and Linear Threshold models.

How AgentProp differentiates:

- CyNetDiff accelerates diffusion simulation.
- AgentProp targets agent workflow optimization, including context cost, verifier placement, topology pruning, and LLM case studies.
- CyNetDiff may later become a backend.

### Bootstrap Percolation

Bootstrap percolation is a threshold-style process where nodes become active depending on active-neighbor thresholds. It is relevant to fault tolerance and distributed computing.

How AgentProp uses it:

- Bootstrap percolation may be more realistic than classical zero forcing for verifier-based workflows.
- Example: an agent becomes "verified" only if at least two trusted upstream agents agree.

Implementation implication:

```text
src/agentprop/propagation/bootstrap_percolation.py
```

Useful for:

- Consensus
- Multi-agent agreement
- Redundancy
- Robust activation

## ML, DL, and RL Influence-Maximization Literature

### DeepIM

Paper: **Deep Graph Representation Learning and Optimization for Influence Maximization**, 2023.

DeepIM frames influence maximization with deep graph representation learning. It learns latent representations of seed sets and diffusion patterns, and supports flexible node-centrality-based budget constraints.

How AgentProp uses it:

- DeepIM is a direct baseline for the GNN layer.
- AgentProp's ML layer should include a GNN node scorer, seed-set representation, supervised/imitation learning from greedy/CELF, and possibly generative seed-set modeling later.

AgentProp distinction:

> AgentProp focuses on agent workflow graphs with token cost, verifier placement, topology pruning, and LLM-specific metrics, not generic social-network influence maximization.

### DREIM / Deep RL for Influence Maximization

Paper: **Finding Influencers in Complex Networks: An Effective Deep Reinforcement Learning Approach**, 2023.

This paper uses deep RL with a GNN encoder and RL decoder to solve influence maximization.

How AgentProp uses it:

- Supports the RL layer.
- AgentProp can implement a simplified version where state is selected seeds plus graph features, action is choosing the next seed node, and reward is spread/cost tradeoff.
- Later, AgentProp extends this to send context, activate verifier, prune edge, and stop.

Novelty caveat:

> RL for seed selection already exists; AgentProp's novelty is the LLM-agent workflow domain and metrics.

### DeepSN and Neural Diffusion

Paper: **DeepSN: A Sheaf Neural Framework for Influence Maximization**, 2024.

This paper argues that ordinary GNNs may fail to capture complex influence diffusion and uses sheaf neural diffusion for richer influence modeling.

How AgentProp uses it:

- Stretch baseline.
- V1 should use GCN, GraphSAGE, and GAT.
- Later paper polish can mention richer neural diffusion models as future backends.

## Multi-Agent LLM Topology and Workflow Papers

This is the most important competitive literature cluster. AgentProp must cite it clearly.

### GPTSwarm / Language Agents as Optimizable Graphs

Paper: **Language Agents as Optimizable Graphs**, 2024.

GPTSwarm represents LLM agents as computational graphs. Nodes are LLM/tool functions, and edges define information flow. It optimizes both node-level prompts and graph connectivity.

AgentProp distinction:

- GPTSwarm learns/optimizes graph structures.
- AgentProp focuses on training-free graph algorithms, propagation analysis, seed selection, verifier placement, pruning, and benchmark comparison across classical, GNN, and RL methods.
- AgentProp should be usable as an analysis layer over existing workflows, not only as a graph generator.

### DyLAN

Paper: **A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration**, 2023.

DyLAN dynamically selects agent teams using an unsupervised Agent Importance Score, then lets selected agents collaborate dynamically.

How AgentProp uses it:

- DyLAN is directly relevant to seed/agent selection.
- DyLAN's Agent Importance Score can become a related baseline if implementable.

AgentProp distinction:

- DyLAN selects agents for collaboration.
- AgentProp studies graph propagation, cost-aware context routing, pruning, verifier placement, and multiple diffusion models.

### AgentPrune

Paper: **Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems**, 2024.

AgentPrune identifies communication redundancy in LLM multi-agent pipelines and prunes redundant or malicious messages. It reports comparable performance at much lower cost, including large token reductions and adversarial robustness gains.

How AgentProp uses it:

- Must-cite and must-benchmark paper.
- AgentPrune validates the core pain point: token overhead exists, communication redundancy exists, and pruning can reduce cost.

AgentProp distinction:

- Context routing
- Propagation modeling
- Verifier placement
- Classical graph algorithms
- Learned policies
- Trace analysis

### G-Designer

Paper: **G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**, 2024.

G-Designer uses a variational graph autoencoder to design task-aware multi-agent communication topologies.

AgentProp distinction:

- G-Designer learns task-specific communication topologies.
- AgentProp compares classical training-free methods, GNN methods, RL methods, and propagation models.
- AgentProp is positioned as a benchmark/toolkit and analysis layer, not only a learned topology model.

### AFlow

Paper: **AFlow: Automating Agentic Workflow Generation**, 2024.

AFlow formulates workflow optimization as a search problem over code-represented workflows and uses Monte Carlo Tree Search to refine workflows.

AgentProp distinction:

- AFlow searches workflows.
- AgentProp analyzes and optimizes graph communication inside workflows using graph propagation and cost metrics.
- AgentProp can be a lightweight pre-optimization or diagnostic layer before expensive MCTS-style search.

### MaAS

Paper: **Multi-agent Architecture Search via Agentic Supernet**, 2025.

MaAS treats multi-agent architectures as a supernet and samples query-dependent agentic systems.

AgentProp distinction:

- MaAS is architecture search.
- AgentProp is graph analysis, propagation, training-free algorithms, and optional ML/RL.

Strong claim:

> Can cheaper classical algorithms get much of the benefit without expensive architecture search?

### ARG-Designer / Assemble Your Crew

Paper: **Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation**, 2025.

This paper frames MAS design as conditional autoregressive graph generation, selecting both agent roles and communication links based on the task.

AgentProp positioning:

- Do not compete head-on at first.
- Offer interpretable graph analysis and cheaper training-free optimization for existing workflows.

### AgentSquare

Paper: **AgentSquare: Automatic LLM Agent Search in Modular Design Space**, 2024.

AgentSquare abstracts LLM agent designs into modules such as planning, reasoning, tool use, and memory, then searches over modular agent designs.

AgentProp's narrower niche:

- Communication graph optimization
- Context propagation
- Verifier placement
- Topology pruning

## Agent Evaluation, Reliability, and Observability Literature

### AI Agents That Matter

Paper: **AI Agents That Matter**, 2024.

This paper argues that agent benchmarks focus too much on accuracy and not enough on cost, causing overly complex and expensive agents. It recommends jointly optimizing accuracy and cost, separating model benchmarks from downstream developer needs, preventing overfitting, and improving reproducibility.

How AgentProp uses it:

- Report task success.
- Report token cost.
- Report latency.
- Report cost-adjusted success.
- Make runs reproducible.

Evaluation principle:

> Accuracy alone is insufficient; agent workflows must be evaluated on cost-quality tradeoffs.

### Why Do Multi-Agent LLM Systems Fail?

Paper: **Why Do Multi-Agent LLM Systems Fail?**, 2025.

This paper analyzes multiple MAS frameworks across 150+ tasks and identifies 14 failure modes, grouped into system design failures, inter-agent misalignment, and task verification/termination.

How AgentProp uses it:

- Supports verifier-placement and correction-propagation modules.

Relevant metrics:

- Inter-agent misalignment
- Correction propagation delay
- Verifier coverage
- Termination reliability
- Error persistence

### LumiMAS

Paper: **LumiMAS: A Comprehensive Framework for Real-Time Monitoring and Enhanced Observability in Multi-Agent Systems**, 2025.

LumiMAS focuses on monitoring, anomaly detection, and root-cause analysis across multi-agent systems.

AgentProp distinction:

- LumiMAS monitors and detects anomalies.
- AgentProp optimizes graph structure, seed/context routing, pruning, and verifier placement.
- The systems could be complementary.

### AgenTracer

Paper: **AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?**, 2025.

AgenTracer studies failure attribution in agentic systems using counterfactual replay and fault injection.

How AgentProp uses it:

- Connects to Geneson's metric dimension work.
- AgentProp can ask where to place logging/verifier nodes so failures are distinguishable.
- This can become an agent workflow observability extension.

## How The Literature Changes AgentProp

### Claims To Avoid

Do not claim:

- Seed selection is new.
- Influence propagation is new.
- Agent graphs are new.
- Topology optimization for LLM agents is new.
- Zero forcing literally models agent communication.
- No tools exist for diffusion simulation.

### Credible Claims

AgentProp can credibly claim:

- There is no widely adopted open-source framework focused specifically on training-free graph analysis, learned graph policies, and propagation benchmarking for existing LLM-agent workflows.
- Existing LLM-agent topology papers often focus on learned search/design; AgentProp evaluates whether cheaper graph-theoretic methods are competitive.
- Existing diffusion libraries are not built around LLM-agent workflow metrics like token cost, verifier placement, correction delay, and task success.
- Randomized zero forcing provides a useful directed weighted propagation model and process-based centrality lens, but it is one model among several.
- AgentProp's contribution is the benchmark, evaluation suite, and comparative study.

## Proposed Paper Framing

Title:

> Training-Free Graph Optimization for Multi-Agent LLM Workflows

Subtitle:

> A Benchmark of Propagation Models, Seed Selection, Verifier Placement, and Topology Pruning

Abstract-level claim:

> Multi-agent LLM systems often improve task-solving ability but introduce high token cost, redundant communication, and reliability failures. We model agent workflows as directed weighted graphs and evaluate whether classical graph algorithms, randomized-zero-forcing-style propagation models, GNN policies, and RL routing policies can reduce communication cost while preserving task success. We introduce AgentProp, an open-source benchmark and toolkit for context seeding, verifier placement, propagation simulation, and topology pruning in multi-agent LLM workflows.

## Literature-To-Implementation Map

| Literature area | What it gives AgentProp | How AgentProp uses it |
| --- | --- | --- |
| Geneson RZF | Directed weighted stochastic propagation | `RandomizedZeroForcing` model |
| Geneson probabilistic ZF | Expected propagation time | Propagation-time metrics |
| Geneson Markov chains | Exact expected time on small graphs | Exact analysis mode |
| Geneson broadcast dimension | Costed transmitter strength | Variable context levels |
| Geneson metric dimension | Resolving/observability | Verifier/logging placement |
| Influence maximization | Seed selection foundation | IC/LT + greedy/CELF baselines |
| Scalable IM | Efficient large-graph optimization | Future TIM/SKIM/IMM backends |
| DeepIM/DREIM | GNN/RL seed selection | Learned seed/routing policies |
| GPTSwarm | Agent graphs as optimizable graphs | Main related work |
| DyLAN | Agent importance / dynamic teams | Baseline for agent selection |
| AgentPrune | Communication redundancy pruning | Main pruning competitor |
| G-Designer/MaAS/AFlow | Learned topology/workflow search | Learned baseline landscape |
| AI Agents That Matter | Cost-aware evaluation | Cost-quality metrics |
| MAS failure taxonomy | Reliability failure modes | Verifier/correction metrics |
| LumiMAS/AgenTracer | Observability/failure attribution | Trace diagnostics module |

## Implementation Priorities Informed By Literature

### Propagation Models

Priority order:

1. Independent Cascade
2. Linear Threshold
3. Bootstrap Percolation
4. Randomized Zero Forcing
5. Classical Zero Forcing

### Algorithms

Priority order:

1. Random seed baseline
2. Degree/PageRank/Betweenness
3. Greedy influence maximization
4. CELF
5. Cost-aware greedy
6. RZF centrality heuristic
7. GNN seed selector
8. RL routing policy

### Metrics

Priority order:

- `task_success`
- `token_cost`
- `message_count`
- `latency`
- `expected_propagation_time`
- `activation_coverage`
- `full_activation_probability`
- `correction_delay`
- `verifier_coverage`
- `cost_adjusted_success`

## Defensible Academic Gap

Existing influence-maximization work gives strong algorithms for seed selection, but it is not tailored to LLM-agent workflows. Existing LLM-agent topology papers optimize workflows or communication graphs, but they usually emphasize learned or search-based methods. AgentProp provides a unified benchmark and toolkit that compares training-free graph algorithms, randomized-zero-forcing-style models, GNN policies, and RL routing policies under LLM-agent-specific metrics: token cost, message count, correction propagation, verifier placement, and task success.

## Suggested Related Work Section

### Graph Propagation and Zero Forcing

Discuss deterministic zero forcing, probabilistic zero forcing, randomized zero forcing, expected propagation time, broadcast dimension, and metric dimension. Use Geneson's work heavily here.

### Influence Maximization

Discuss Kempe-Kleinberg-Tardos, Independent Cascade, Linear Threshold, submodularity, greedy approximation, CELF, and TIM/SKIM/IMM-style scalable methods.

### Learning-Based Influence Maximization

Discuss DeepIM, DREIM, DeepSN, and GNN/RL seed selection.

### Multi-Agent LLM Topology Optimization

Discuss GPTSwarm, DyLAN, AgentPrune, G-Designer, AFlow, MaAS, ARG-Designer, and AgentSquare.

### Agent Evaluation and Reliability

Discuss AI Agents That Matter, Why Do Multi-Agent LLM Systems Fail?, LumiMAS, and AgenTracer.

## What To Take To Dr. Geneson

Tell him:

> I reviewed the relevant graph-theory and LLM-agent literature. The strongest connection to your work is not claiming that zero forcing directly models LLM agents, but using zero forcing, randomized zero forcing, propagation time, broadcast dimension, and metric dimension as graph-theoretic tools inside a broader framework for studying information propagation in agent workflow graphs.

Then say:

> The broader CS literature already has influence maximization and recent LLM-agent topology optimization, so the project must be framed carefully. The contribution is a benchmark and toolkit that compares training-free graph algorithms, randomized-zero-forcing-style models, GNN policies, and RL policies for context routing, verifier placement, and topology pruning in multi-agent LLM workflows.

## Recommended Academic Direction

### Paper 1: Practical Benchmark / Tool Paper

Title:

> AgentProp: Training-Free and Learned Graph Optimization for Multi-Agent LLM Workflows

Core result:

> Show that graph algorithms can reduce token/message cost while preserving task performance on synthetic and real agent workflows.

### Paper 2: Graph Theory Extension

Possible direction:

> Variable-strength randomized zero forcing on directed weighted graphs and its application to workflow observability or propagation-time centrality.

This connects more directly to Geneson's mathematical work.

## Priority Reading List

1. Geneson et al., Randomized Zero Forcing.
2. Geneson and Hogben, Propagation time for probabilistic zero forcing.
3. Chan et al., Using Markov chains to determine expected propagation time for probabilistic zero forcing.
4. Geneson and Yi, Broadcast Dimension of Graphs.
5. Kempe, Kleinberg, Tardos, Maximizing the Spread of Influence through a Social Network.
6. Tang, Xiao, Shi, Influence Maximization: Near-Optimal Time Complexity Meets Practical Efficiency.
7. DeepIM.
8. DREIM.
9. GPTSwarm.
10. DyLAN.
11. AgentPrune.
12. G-Designer.
13. AFlow.
14. MaAS.
15. AI Agents That Matter.
16. Why Do Multi-Agent LLM Systems Fail?

## Bottom Line

This project is academically viable only if framed honestly.

Bad framing:

> We invented zero-forcing-based context seeding for agents.

Good framing:

> We build and evaluate AgentProp, a graph-optimization framework for multi-agent LLM workflows that compares classical influence-maximization algorithms, randomized-zero-forcing-inspired propagation models, GNN policies, and RL routing policies under cost, propagation, and task-success metrics.
