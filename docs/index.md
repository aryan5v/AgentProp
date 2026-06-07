# AgentProp Documentation

AgentProp is a graph-control framework for AI-agent workflows. It models
agents, tools, context, verifiers, and execution attempts as directed weighted
graphs, then studies how information quality, failures, and cost propagate.

The harness emits one `ExecutionEvent` per step; the controller returns
continue, verify, switch strategy, or finalize. See the
[overview](overview.md) for the research framing and capability list.

## Guides

Onboarding and day-to-day use:

- [Overview](overview.md) — core ideas, performance, capabilities
- [Beta quickstart for coding agents](beta_quickstart.md)
- [Tutorial](tutorial.md)
- [Environment setup](environment.md)
- [Architecture](ARCHITECTURE.md)
- [Control layer quickstart](control_layer_quickstart.md)
- [Coding agent integration](coding_agents.md)
- [Plugin distribution](plugin_distribution.md)
- [Framework integrations](framework_integrations.md)
- [Trace ingestion](trace_ingestion.md)

## Reference

Contracts, algorithms, and repo structure:

- [Workflow JSON schema](workflow_schema.md)
- [Verifier semantics](verifier_semantics.md)
- [Quality scoring](quality_scoring.md)
- [Quality-aware routing](routing_quality.md)
- [Visualization](visualization.md)
- [Repository layout](repository_layout.md)
- [Examples catalog](../examples/README.md)
- [Experiments catalog](../experiments/README.md)

## Research and evaluation

Advanced topics and reproducible studies:

- [Learned propagation](learned_propagation.md)
- [Routing baseline evaluation](routing_baseline_evaluation.md)
- [Reinforcement learning routing](reinforcement_learning.md)
- [Deep learning guide](deep_learning.md)
- [Terminal-Bench 2.1 preparation](terminal_bench_21.md)

### Public artifacts

Sanitized benchmark outputs — see [ARTIFACTS.md](results/ARTIFACTS.md):

- [v1 benchmark](results/v1/README.md)
- [GAIA-style benchmark](results/gaia_benchmark/REPORT.md)
- [Real routing case study](results/real_routing_case_study/REPORT.md)
- [Terminal-Bench guided](results/terminal_bench_guided/README.md)
- [Failure localization](results/failure_localization/README.md)
- [Quality cascade vs IC](results/quality_cascade_vs_ic/README.md)
- [Scale / quality evidence](results/scale_quality_evidence/README.md)
- [Terminal-Bench multi-task protocol](results/terminal_bench_multi/README.md)

## Project

- [Project meta docs](project/README.md)
- [Contributing](project/CONTRIBUTING.md)
- [Security policy](../SECURITY.md)
- [Changelog](project/CHANGELOG.md)
- [Agent guide for coding agents](project/AGENTS.md)

Working notes belong in gitignored `docs/local/`, not in the public tree.

## Common commands

```bash
agentprop doctor --tier graph
agentprop workflows list
agentprop optimize planner_coder_tester_reviewer --budget 2
agentprop control-demo --demo terminal --out-dir reports/control-demo
agentprop readiness --json
```

Reproduce key studies (no API keys):

```bash
python experiments/failure_localization_study.py
python experiments/quality_cascade_vs_ic.py
agentprop run-evidence --out-dir docs/results/scale_quality_evidence
```

Full script list: [experiments/README.md](../experiments/README.md).
