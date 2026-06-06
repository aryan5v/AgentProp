# Scale / Quality Evidence (synthetic matrix)

Generated: `2026-06-06T18:05:59.370695+00:00`

**Label:** directional benchmark result on built-in workflow templates.
Coverage and savings are propagation-simulation metrics, not live LLM task success.

## Configuration

- Workflows: planner_coder_tester_reviewer, fan_out_parallel, feedback_loop, shared_memory, dynamic_conditional, hub_and_spoke_supervisor
- Arms: broadcast, greedy, rzf-centrality, quality-aware-greedy, imm, degree
- Tasks per arm: 30
- Repeats: 3
- Seed budget: 3
- Trials: 50

## Summaries (mean coverage ± 95% CI half-width)

| Workflow | Arm | Mean coverage | Cov CI | Mean savings | Sav CI | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| planner_coder_tester_reviewer | broadcast | 1.000 | 0.000 | 0.000 | 0.000 | 90 |
| planner_coder_tester_reviewer | greedy | 1.000 | 0.000 | 0.083 | 0.000 | 90 |
| planner_coder_tester_reviewer | rzf-centrality | 1.000 | 0.000 | 0.083 | 0.000 | 90 |
| planner_coder_tester_reviewer | quality-aware-greedy | 1.000 | 0.000 | 0.083 | 0.000 | 90 |
| planner_coder_tester_reviewer | imm | 1.000 | 0.000 | 0.079 | 0.000 | 90 |
| planner_coder_tester_reviewer | degree | 0.800 | 0.000 | 0.338 | 0.000 | 90 |
| fan_out_parallel | broadcast | 1.000 | 0.000 | 0.000 | 0.000 | 90 |
| fan_out_parallel | greedy | 1.000 | 0.000 | 0.257 | 0.000 | 90 |
| fan_out_parallel | rzf-centrality | 1.000 | 0.000 | 0.220 | 0.000 | 90 |
| fan_out_parallel | quality-aware-greedy | 1.000 | 0.000 | 0.256 | 0.000 | 90 |
| fan_out_parallel | imm | 1.000 | 0.000 | 0.210 | 0.000 | 90 |
| fan_out_parallel | degree | 1.000 | 0.000 | 0.257 | 0.000 | 90 |
| feedback_loop | broadcast | 1.000 | 0.000 | 0.000 | 0.000 | 90 |
| feedback_loop | greedy | 1.000 | 0.000 | 0.203 | 0.000 | 90 |
| feedback_loop | rzf-centrality | 1.000 | 0.000 | 0.203 | 0.000 | 90 |
| feedback_loop | quality-aware-greedy | 1.000 | 0.000 | 0.203 | 0.000 | 90 |
| feedback_loop | imm | 1.000 | 0.000 | 0.203 | 0.000 | 90 |
| feedback_loop | degree | 0.800 | 0.000 | 0.391 | 0.000 | 90 |
| shared_memory | broadcast | 1.000 | 0.000 | 0.000 | 0.000 | 90 |
| shared_memory | greedy | 1.000 | 0.000 | 0.224 | 0.000 | 90 |
| shared_memory | rzf-centrality | 1.000 | 0.000 | 0.194 | 0.000 | 90 |
| shared_memory | quality-aware-greedy | 1.000 | 0.000 | 0.224 | 0.000 | 90 |
| shared_memory | imm | 1.000 | 0.000 | 0.187 | 0.000 | 90 |
| shared_memory | degree | 1.000 | 0.000 | 0.206 | 0.000 | 90 |
| dynamic_conditional | broadcast | 1.000 | 0.000 | 0.000 | 0.000 | 90 |
| dynamic_conditional | greedy | 1.000 | 0.000 | 0.248 | 0.001 | 90 |
| dynamic_conditional | rzf-centrality | 1.000 | 0.000 | 0.202 | 0.000 | 90 |
| dynamic_conditional | quality-aware-greedy | 1.000 | 0.000 | 0.248 | 0.001 | 90 |
| dynamic_conditional | imm | 1.000 | 0.000 | 0.202 | 0.000 | 90 |
| dynamic_conditional | degree | 1.000 | 0.000 | 0.250 | 0.000 | 90 |
| hub_and_spoke_supervisor | broadcast | 1.000 | 0.000 | 0.000 | 0.000 | 90 |
| hub_and_spoke_supervisor | greedy | 1.000 | 0.000 | 0.170 | 0.000 | 90 |
| hub_and_spoke_supervisor | rzf-centrality | 1.000 | 0.000 | 0.127 | 0.000 | 90 |
| hub_and_spoke_supervisor | quality-aware-greedy | 1.000 | 0.000 | 0.170 | 0.000 | 90 |
| hub_and_spoke_supervisor | imm | 1.000 | 0.000 | 0.170 | 0.000 | 90 |
| hub_and_spoke_supervisor | degree | 1.000 | 0.000 | 0.134 | 0.000 | 90 |

## Reproduce

```bash
agentprop run-evidence --tasks-per-arm 30 --repeats 3
```
