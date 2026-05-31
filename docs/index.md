# AgentProp Documentation

AgentProp is a graph optimization framework for multi-agent LLM workflows.

## Start Here

- [Tutorial walkthrough](tutorial.md)
- [Workflow JSON schema](workflow_schema.md)
- [Trace ingestion](trace_ingestion.md)
- [Visualization](visualization.md)
- [Reinforcement learning routing](reinforcement_learning.md)

## Product and Research

- [Product requirements](PRD.md)
- [Literature review](research/literature_review.md)
- [v1 benchmark artifacts](results/v1/README.md)
- [v1 release checklist](release_checklist.md)
- [Deep learning roadmap](deep_learning.md)

## Common Commands

```bash
agentprop optimize benchmarks/workflows/planner_coder_tester_reviewer.json --budget 2
agentprop benchmark planner_coder_tester_reviewer --budget 2 --trials 50
agentprop report planner_coder_tester_reviewer --out reports/demo.md
agentprop viz planner_coder_tester_reviewer --out reports/workflow.dot
```

## What AgentProp Proves

The v1 goal is to show that multi-agent workflows can be treated as directed weighted graphs and optimized with:

- training-free graph algorithms
- propagation models
- verifier-placement and observability metrics
- optional ML/DL/RL policies
- reproducible benchmark artifacts
