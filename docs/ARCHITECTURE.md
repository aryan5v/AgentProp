# AgentProp Architecture

AgentProp is a graph-control layer for AI agent workflows. It is **not** an
orchestrator: it analyzes, simulates, and supervises workflows you already run.

## Layer diagram

```text
Inputs                Core                 Analysis
──────                ────                 ────────
Workflow JSON    →    AgentGraph      →    algorithms/ (seeds, verifiers, IMM, pruning)
Trace JSON       →    validation      →    propagation/ (IC, QC, RZF, fast kernel, learned)
Framework dict   →    dynamic_graph   →    evaluation/ (benchmarks, evidence harness, reports)
                     propagation_index

Runtime               Integrations         Surfaces
───────               ────────────         ────────
ControlSession   ←    trace_loader    →    CLI (agentprop, run-evidence)
StoppingController    framework_adapters   MCP (agentprop-mcp + session_store)
RuntimeController     agent_instructions   Python SDK
critical_facts        context_advisor
hierarchical_context
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
| `src/agentprop/core/` | `AgentGraph`, `dynamic_graph`, `propagation_index`, JSON validation |
| `src/agentprop/propagation/` | IC, QC, RZF, learned models, fast propagation kernel |
| `src/agentprop/algorithms/` | Seeds, verifier placement, IMM/TIM backend, bottlenecks |
| `src/agentprop/evaluation/` | Benchmarks, `evidence_harness`, reports, readiness, Terminal-Bench |
| `src/agentprop/runtime/` | Control loop, `ControlSession`, critical facts, hierarchical context |
| `src/agentprop/integrations/` | Traces, adapters, MCP, `session_store`, coding-agent briefs |
| `src/agentprop/workflows/` | Built-in template graphs (incl. dynamic/conditional) |
| `experiments/` | Reproducible scripts — [experiments/README.md](../experiments/README.md) |
| `examples/` | Minimal integrations — [examples/README.md](../examples/README.md) |
| `docs/results/` | Sanitized public artifacts — [ARTIFACTS.md](results/ARTIFACTS.md) |

## What's real today

Run `agentprop readiness` for a public-safe component maturity report. Graph
analysis, propagation simulation, verifier placement, and key-free control demos
work without API keys. Real LLM validation and multi-task Terminal-Bench arms
require credentials documented in [environment.md](environment.md).

## Recent capabilities (v0.1.0b1+)

| Feature | Entry point |
| --- | --- |
| Influence maximization (IMM) for large graphs | `algorithm=auto` or `imm` in benchmark/optimize |
| Dynamic graph mutations at runtime | `ControlSession.enable_dynamic_graph()` |
| Scale/quality evidence matrix | `agentprop run-evidence` or `experiments/run_evidence_harness.py` |
| MCP session persistence | `SessionStore` under `~/.agentprop/sessions` (see coding agents doc) |
| Critical-fact routing hints | `runtime/critical_facts.py`, `integrations/context_advisor.py` |

## Related docs

- [Overview](overview.md) — core ideas and capability summary
- [Repository layout](repository_layout.md) — where artifacts and scripts live
- [Tutorial](tutorial.md) — hands-on walkthrough
- [Workflow JSON schema](workflow_schema.md) — graph contract
- [Verifier semantics](verifier_semantics.md) — metric dimension contribution
- [Framework integrations](framework_integrations.md) — LangGraph, CrewAI, etc.
- [AGENTS.md](project/AGENTS.md) — quick guide for coding agents
