# AgentProp Documentation

AgentProp is a graph-control framework for AI-agent workflows. It models
agents, tools, context, verifiers, and execution attempts as directed weighted
graphs, then studies how information quality, failures, and cost propagate.

AgentProp wraps your existing agent loop rather than replacing it. The only
contract a harness must satisfy is emitting one `ExecutionEvent` per step; the
controller then decides whether to continue, force an independent verification,
switch strategy, or finalize. Adapters for LangGraph, AutoGen, CrewAI, OpenAI
Agents, and LlamaIndex are in [framework integrations](framework_integrations.md).

## Start Here

- [Tutorial walkthrough](tutorial.md)
- [Workflow JSON schema](workflow_schema.md)
- [Quality-aware routing](routing_quality.md)
- [Verifier semantics](verifier_semantics.md) — the core metric-dimension
  contribution: place verifiers so every distinct failure is uniquely localizable
- [Quality scoring](quality_scoring.md)
- [Framework integrations](framework_integrations.md) — wrap an existing
  LangGraph, AutoGen, CrewAI, OpenAI Agents, or LlamaIndex workflow
- [Coding agent integration](coding_agents.md)
- [Control layer quickstart](control_layer_quickstart.md)

## Research and Evaluation

- [Reproducible results](research/reproducible_results.md) — the headline
  verifier-placement and RZF-scaling tables, with the scripts that produce them
- [Research references](research/references.md)
- [Literature review](research/literature_review.md)
- [Real LLM case-study protocol](research/case_study_protocol.md)
- [Trace ingestion](trace_ingestion.md)
- [Learned propagation](learned_propagation.md)
- [Routing baseline evaluation](routing_baseline_evaluation.md)
- [Reinforcement learning routing](reinforcement_learning.md)
- [Deep learning guide](deep_learning.md)
- [v1 benchmark artifacts](results/v1/README.md)
- [Terminal-Bench guided benchmark](results/terminal_bench_guided/README.md)

## Common Commands

```bash
agentprop optimize benchmarks/workflows/planner_coder_tester_reviewer.json --budget 2
agentprop simulate chain --seeds node_0 --model zero-forcing
agentprop simulate chain --seeds node_0 --model quality-cascade
agentprop prune planner_coder_tester_reviewer --target-token-reduction 0.3
agentprop benchmark planner_coder_tester_reviewer --budget 2 --trials 50
agentprop report planner_coder_tester_reviewer --out reports/demo.html --format html
agentprop agent-instructions planner_coder_tester_reviewer --target codex --out reports/codex_agent_brief.md
agentprop control-demo --demo terminal --out-dir reports/control-demo
agentprop viz planner_coder_tester_reviewer --out reports/workflow.dot
```

Reproduce the verifier-placement and seeding scaling studies:

```bash
python experiments/verifier_placement_evidence.py
python experiments/rzf_scaling_study.py
```

## What AgentProp Proves

The alpha research goal is to test whether agent workflows can be treated as
directed weighted graphs and optimized with:

- metric-dimension-style verifier placement
- quality-cascade propagation
- randomized-zero-forcing-style scaling studies
- runtime stop/retry/verify control
- optional ML/DL/RL policies
- reproducible benchmark artifacts
