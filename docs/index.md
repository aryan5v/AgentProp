# AgentProp Documentation

AgentProp is a graph optimization framework for multi-agent LLM workflows.

## Start Here

- [Tutorial walkthrough](tutorial.md)
- [Workflow JSON schema](workflow_schema.md)
- [Trace ingestion](trace_ingestion.md)
- [Learned propagation](learned_propagation.md)
- [Visualization](visualization.md)
- [Reinforcement learning routing](reinforcement_learning.md)
- [Routing baseline evaluation](routing_baseline_evaluation.md)
- [Verifier semantics](verifier_semantics.md)
- [Quality scoring](quality_scoring.md)
- [Framework integrations](framework_integrations.md)
- [Coding agent integration](coding_agents.md)

## Product and Research

- [Product requirements](PRD.md)
- [Remaining PRD archive](remaining_prd_archive.md)
- [Literature review](research/literature_review.md)
- [Real LLM case-study protocol](research/case_study_protocol.md)
- [Paper outline](research/paper_outline.md)
- [v1 benchmark artifacts](results/v1/README.md)
- [v1 release checklist](release_checklist.md)
- [Deep learning roadmap](deep_learning.md)
- [Public release decision](public_release_decision.md)

## Common Commands

```bash
agentprop optimize benchmarks/workflows/planner_coder_tester_reviewer.json --budget 2
agentprop simulate chain --seeds node_0 --model zero-forcing
agentprop prune planner_coder_tester_reviewer --target-token-reduction 0.3
agentprop benchmark planner_coder_tester_reviewer --budget 2 --trials 50
agentprop report planner_coder_tester_reviewer --out reports/demo.html --format html
agentprop agent-instructions planner_coder_tester_reviewer --target codex --out reports/codex_agent_brief.md
agentprop viz planner_coder_tester_reviewer --out reports/workflow.dot
```

## What AgentProp Proves

The v1 goal is to show that multi-agent workflows can be treated as directed weighted graphs and optimized with:

- training-free graph algorithms
- propagation models
- verifier-placement and observability metrics
- optional ML/DL/RL policies
- reproducible benchmark artifacts
