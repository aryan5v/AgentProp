# Research References

AgentProp is inspired by graph observability, stochastic propagation, influence
maximization, and recent work on optimizing multi-agent LLM communication. This
page lists the core references that shape the current alpha.

## Graph Observability And Propagation

- Jesse Geneson, Illya Hicks, Noah Lichtenberg, Alvin Moon, and Nicolas Robles.
  [Randomized Zero Forcing](https://arxiv.org/abs/2602.16300), 2026.
  AgentProp uses RZF as a directed weighted propagation model and as motivation
  for process-based centrality and expected propagation time.
- Jesse Geneson.
  [Metric dimension and pattern avoidance in graphs](https://arxiv.org/abs/1807.08334),
  2018. AgentProp uses metric-dimension ideas to frame verifier placement as an
  observability problem: choose landmarks that make failure sources
  distinguishable.
- Jesse Geneson and Leslie Hogben.
  [Propagation time for probabilistic zero forcing](https://arxiv.org/abs/1812.10476),
  2018. AgentProp uses expected propagation time as a graph-level analogue for
  context, correction, and error-spread latency.
- Jesse Geneson and Eunjeong Yi.
  [Broadcast Dimension of Graphs](https://arxiv.org/abs/2005.07311), 2020.
  AgentProp's graded context levels mirror the idea that different transmitters
  can have different strengths and costs.

## Influence Maximization And Diffusion

- David Kempe, Jon Kleinberg, and Eva Tardos.
  [Maximizing the Spread of Influence through a Social Network](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf),
  KDD 2003. AgentProp treats context seeding as related to influence
  maximization, while adding agent-specific cost, quality, and verifier
  constraints.

## Multi-Agent LLM Topology And Pruning

- GPTSwarm:
  [Language Agents as Optimizable Graphs](https://openreview.net/pdf?id=uTC9AFXIhg).
  This work motivates viewing agent systems as graph structures that can be
  optimized rather than fixed by hand.
- DyLAN:
  [Dynamic LLM-Agent Network](https://arxiv.org/abs/2310.02170).
  DyLAN motivates task-adaptive agent selection and topology changes.
- AgentPrune:
  [Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2410.02506).
  AgentPrune motivates token-aware communication pruning in LLM multi-agent
  systems.

## AgentProp's Position

The current research direction is:

> AgentProp models agent workflows as directed weighted graphs and studies
> verifier observability, quality propagation, seed/context selection, pruning,
> and runtime control under token-cost and task-success constraints.

The goal is not to claim that graph theory alone solves agent reliability. The
goal is to make the structure of agent execution measurable enough to compare
training-free graph algorithms, stochastic propagation models, learned policies,
and runtime controllers on the same workflows.
