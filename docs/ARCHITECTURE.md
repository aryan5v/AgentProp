# AgentProp Architecture

AgentProp is a graph-control layer for AI agent workflows. It is **not** an
orchestrator: it analyzes, simulates, and supervises workflows you already run.

## Layer diagram

```text
Inputs                Core                 Analysis
──────                ────                 ────────
Workflow JSON    →    AgentGraph      →    algorithms/ (seeds, verifiers, pruning)
Trace JSON       →    validation      →    propagation/ (IC, QC, RZF, learned)
Framework dict   →    NetworkX export →    evaluation/ (benchmarks, reports)

Runtime               Integrations         Surfaces
───────               ────────────         ────────
ControlSession   ←    trace_loader    →    CLI (agentprop)
StoppingController    framework_adapters   MCP (agentprop-mcp)
RuntimeController     agent_instructions   Python SDK
```

## When to use which runtime API

| API | Use when |
| --- | --- |
| `ControlSession` | You have a **live agent loop** (Codex, Claude Code, custom harness). Emit one `ExecutionEvent` per step; AgentProp returns CONTINUE / FORCE_VERIFY / SWITCH_STRATEGY / FINALIZE. |
| `AgentPropRuntimeController` | You want AgentProp to **execute graph nodes** with seed-based context routing inside a simulated or injected executor. |

Most integrations should start with `ControlSession`. See
[control layer quickstart](control_layer_quickstart.md).

## Concept glossary

| Term | Meaning |
| --- | --- |
| **Propagation** | How context, quality, or activation spreads from seed nodes along directed edges. |
| **Seeds** | Nodes that receive full task context first; chosen by centrality, greedy, or RZF algorithms. |
| **Quality cascade** | Propagation model where output quality degrades along edges; drives quality-aware routing. |
| **Verifier (graph)** | Node placement problem: where to put checkers so failures are uniquely localizable. |
| **Resolving coverage** | Fraction of failure pairs distinguishable by verifier distance signatures (metric dimension). |
| **ExecutionEvent** | One observed runtime step: command, tokens, verifier result, error signature, etc. |

## Package map

| Path | Role |
| --- | --- |
| `src/agentprop/core/` | `AgentGraph`, nodes, edges, JSON validation |
| `src/agentprop/propagation/` | Diffusion and quality models |
| `src/agentprop/algorithms/` | Seed selection, verifier placement, bottlenecks |
| `src/agentprop/evaluation/` | Benchmarks, reports, readiness, Terminal-Bench helpers |
| `src/agentprop/runtime/` | Control loop, session facade, demos |
| `src/agentprop/integrations/` | Traces, framework adapters, MCP, coding-agent briefs |
| `src/agentprop/workflows/` | Built-in template graphs |
| `experiments/` | Reproducible research scripts (run from repo checkout) |

## What's real today

Run `agentprop readiness` for a public-safe component maturity report. Graph
analysis, propagation simulation, verifier placement, and key-free control demos
work without API keys. Real LLM validation and multi-task Terminal-Bench arms
require credentials documented in [environment.md](environment.md).

## Related docs

- [Tutorial](tutorial.md) — hands-on walkthrough
- [Workflow JSON schema](workflow_schema.md) — graph contract
- [Verifier semantics](verifier_semantics.md) — metric dimension contribution
- [Framework integrations](framework_integrations.md) — LangGraph, CrewAI, etc.
- [AGENTS.md](../AGENTS.md) — quick guide for coding agents
