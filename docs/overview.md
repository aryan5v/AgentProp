# AgentProp Overview

AgentProp models multi-agent workflows as directed weighted graphs. The library
measures how information, quality, failures, and cost propagate—and provides a
runtime control layer that supervises execution without replacing your agent
orchestrator.

## Core ideas

**Metric dimension (verifier placement).** Verifier placement is framed as a
resolving set problem. When resolving coverage is 1.0, each distinct failure
produces a unique distance signature, so a single faulty node is identifiable.
Fault-tolerant variants preserve this property when one verifier is unavailable.
No weighted heuristic alone guarantees unique localization. See
[verifier semantics](verifier_semantics.md).

**Quality cascade.** Output quality degrades along edges. Context routing can
follow the quality that actually reaches each downstream node rather than
broadcasting full context everywhere.

**Randomized Zero Forcing (RZF).** RZF centrality is a scoped scaling tool for
larger workflows where static centrality underestimates reachability. On small
graphs (under ~15 nodes), classical centrality remains competitive. Treat RZF as
a conditional advantage, not a universal default.

**Runtime control.** Graph analysis drives runtime actions: force verification,
retry, stop, switch strategy, or expand context. The harness emits one
`ExecutionEvent` per step; AgentProp returns `CONTINUE`, `FORCE_VERIFY`,
`SWITCH_STRATEGY`, or `FINALIZE`.

## Control loop

AgentProp wraps the loop you already run:

```text
task ─► your agent proposes work
     ─► AgentProp inspects ExecutionEvent history
     ─► StoppingController returns a decision
     ─► harness continues, verifies, switches, or finalizes
```

Adapters exist for LangGraph, AutoGen, CrewAI, OpenAI Agents, and LlamaIndex.
Any harness that can emit `ExecutionEvent` rows can integrate. See
[framework integrations](framework_integrations.md) and
[control layer quickstart](control_layer_quickstart.md).

## Performance (v0.1.0a4+)

| Area | Improvement |
| --- | --- |
| Verifier placement | Memoized distances + incremental resolving tracker |
| Seed selection | Lazy CELF + candidate sampling on large graphs |
| Default optimize | `auto` selects greedy / RZF / IMM by graph size |
| Propagation | Integer-indexed adjacency; optional batch simulation |
| Runtime tracker | Incremental `ExecutionStateTracker` (O(1) per step) |
| Context compression | Critical-fact slices for convention-sensitive tasks |

Microbenchmarks live in `benchmarks/perf_micro.py` (CI-gated via
`tests/test_perf_micro.py`). MCP sessions persist under `~/.agentprop/sessions`
with a shared graph-analysis cache.

## Capabilities

- Directed weighted `AgentGraph`, JSON validation, NetworkX export, Graphviz DOT
- Propagation: IC, LT, bootstrap percolation, zero forcing, RZF, quality cascade, learned models
- Algorithms: seeds, pruning, bottlenecks, centrality, metric-dimension verifiers, IMM backend
- Runtime: `ControlSession`, terminal loop, verifier forcing, bandit policies, dynamic graphs
- Optional ML/DL/RL baselines and reproducible experiment scripts
- Coding-agent briefs, FastMCP tools, and plugin/skill distribution bundles

Run `agentprop readiness --json` for a public-safe maturity report.

## Research context

AgentProp treats agent workflows as communication graphs under quality, cost,
and observability constraints—not opaque prompt chains.

Primary references:

- [Randomized Zero Forcing](https://arxiv.org/abs/2602.16300) — stochastic propagation on directed graphs
- [Metric dimension](https://arxiv.org/abs/1807.08334) — resolving sets and observability
- [Influence maximization](https://www.cs.cornell.edu/home/kleinber/kdd03-inf.pdf) — cascade seeding
- Agent workflow literature: [GPTSwarm](https://openreview.net/pdf?id=uTC9AFXIhg), [DyLAN](https://arxiv.org/abs/2310.02170), [AgentPrune](https://arxiv.org/abs/2410.02506)

Public benchmark artifacts: [docs/results/ARTIFACTS.md](results/ARTIFACTS.md).
