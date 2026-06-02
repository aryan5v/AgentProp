# AgentProp Research Backlog

This backlog turns the current benchmark work into research tracks that keep
AgentProp anchored in graph theory, influence maximization, and learned control.

## Thesis

AgentProp should treat multi-agent workflows as weighted communication graphs
whose nodes, edges, verifier placements, and context budgets are jointly
optimized under quality, cost, and reliability constraints.

The mathematical backbone is:

- diffusion models: Independent Cascade, Linear Threshold, bootstrap
  percolation, zero forcing, and learned propagation;
- graph optimization: influence maximization, cut points, bridges, k-core,
  centrality, observability, and sparsification;
- learned policy search: supervised risk models, GNN scoring, contextual
  bandits, and reinforcement learning over routing/verifier actions.

## Evidence Tracks To Pursue

### 1. Task-Adaptive Agent Topologies

Recent multi-agent LLM work increasingly models agent systems as graphs whose
topology can be optimized, not fixed by hand.

Evidence to study:

- GPTSwarm, "Language Agents as Optimizable Graphs":
  https://openreview.net/pdf/fcd7b79c216e39b694d44951f287447276351249.pdf
- AFlow, automated workflow generation and optimization:
  https://arxiv.org/abs/2410.10762
- Multi-Agent System Search (MASS):
  https://arxiv.org/abs/2502.02533
- DynaSwarm, dynamic graph structure selection:
  https://arxiv.org/abs/2507.23261
- Dynamic topology generation with graph diffusion models:
  https://arxiv.org/abs/2510.07799

AgentProp opportunity:

- Make routing recommendations task-conditioned instead of static.
- Compare chain, star, tree, DAG, small-world, and dense topologies on real
  benchmark categories.
- Optimize edges and context budgets together, not just seed nodes.

### 2. Influence Maximization For Context Propagation

Classical influence maximization gives the theory for selecting seed agents
under propagation models. It also gives useful constraints: NP-hardness,
submodular greedy approximations, and scalable heuristics.

Evidence to study:

- Kempe, Kleinberg, and Tardos, "Maximizing the Spread of Influence through a
  Social Network": https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf
- 2024 influence maximization survey:
  https://www.sciencedirect.com/science/article/pii/S095741742400294X
- Temporal influence maximization review:
  https://link.springer.com/article/10.1007/s41109-024-00625-3

AgentProp opportunity:

- Map "full context" to seed activation and "summarized handoff" to attenuated
  edge transmission.
- Add quality-aware marginal gain: `expected_success - lambda_cost * cost`.
- Use articulation points, bridges, and verifier nodes as reliability-critical
  structure, not only centrality features.

### 3. Learned Diffusion And GNN Scoring

Topology alone did not explain the benchmark regressions. Learned models should
estimate which nodes are context-sensitive and which edges carry critical
information.

Evidence to study:

- GLIE, GNN estimation for influence maximization:
  https://arxiv.org/abs/2108.04623
- Influence maximization via graph neural bandits:
  https://arxiv.org/abs/2406.10521
- Temporal influence maximization with continuous-time GNNs and deep RL:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12992706/

AgentProp opportunity:

- Train expected-success and timeout-risk models from benchmark traces.
- Learn edge compression ratios and edge failure probabilities.
- Add calibration reports so recommendations expose uncertainty, not just rank.

### 4. Token-Efficient Multi-Agent Pruning

The benchmark goal is not "more agents"; it is better outcomes at lower cost.
AgentProp should prune redundant roles and low-value communication while
protecting verifier and implementation-sensitive paths.

Evidence to study:

- AgentDropout, dynamic agent and edge elimination:
  https://aclanthology.org/2025.acl-long.1170.pdf
- ResMAS, resilience optimization in LLM-based multi-agent systems:
  https://ojs.aaai.org/index.php/AAAI/article/download/40824/44785

AgentProp opportunity:

- Couple pruning with risk-aware verifier placement.
- Preserve high-sensitivity roles even when topology-only centrality would prune
  their context.
- Report "saves X, risks Y" for every recommendation.

### 5. Runtime Scheduling And Budget Control

Budget-aware stopping belongs in the research core because over-exploration is a
control problem on a graph. Each probe, verifier call, and context expansion has
expected information gain and cost.

Evidence to study:

- AI Metropolis, out-of-order scheduling for LLM-agent simulations:
  https://mast.stanford.edu/pubs/ai_metropolis_scaling_large_language_model_based_multi_agent_simulation_with_out_of_order_execution/

AgentProp opportunity:

- Model commands and verifier calls as budgeted actions.
- Add contextual bandits over task-category policies.
- Penalize timeout risk directly in routing and benchmark scorecards.

## Near-Term Research Build Order

1. Finish trace-to-training-data for Terminal-Bench outcomes.
2. Add calibrated expected-success, timeout-risk, and cost models.
3. Feed predictions into quality-aware seed/edge/verifier optimization.
4. Run held-out ablations: static guidance vs topology-only vs learned routing.
5. Publish a transparent result card with pass rate, token/cost, timeout rate,
   regressions, and confidence intervals.
