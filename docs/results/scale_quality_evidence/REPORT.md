# Scale / Quality Evidence (synthetic matrix)

Generated: `2026-06-06T18:01:57.623923+00:00`

**Label:** directional benchmark result on built-in workflow templates.
Coverage and savings are propagation-simulation metrics, not live LLM task success.

## Configuration

- Workflows: planner_coder_tester_reviewer, fan_out_parallel, feedback_loop, shared_memory, hub_and_spoke_supervisor
- Arms: broadcast, greedy, rzf-centrality, quality-aware-greedy, imm, degree
- Tasks per arm: 5
- Repeats: 2
- Seed budget: 3
- Trials: 20

## Summaries (mean coverage ± 95% CI half-width)

| Workflow | Arm | Mean coverage | CI half-width | Mean savings | Runs |
| --- | --- | ---: | ---: | ---: | ---: |
| planner_coder_tester_reviewer | broadcast | 1.000 | 0.000 | 0.000 | 10 |
| planner_coder_tester_reviewer | greedy | 1.000 | 0.000 | 0.083 | 10 |
| planner_coder_tester_reviewer | rzf-centrality | 1.000 | 0.000 | 0.083 | 10 |
| planner_coder_tester_reviewer | quality-aware-greedy | 1.000 | 0.000 | 0.083 | 10 |
| planner_coder_tester_reviewer | imm | 1.000 | 0.000 | 0.079 | 10 |
| planner_coder_tester_reviewer | degree | 1.000 | 0.000 | 0.083 | 10 |
| fan_out_parallel | broadcast | 1.000 | 0.000 | 0.000 | 10 |
| fan_out_parallel | greedy | 1.000 | 0.000 | 0.257 | 10 |
| fan_out_parallel | rzf-centrality | 1.000 | 0.000 | 0.220 | 10 |
| fan_out_parallel | quality-aware-greedy | 1.000 | 0.000 | 0.257 | 10 |
| fan_out_parallel | imm | 1.000 | 0.000 | 0.210 | 10 |
| fan_out_parallel | degree | 1.000 | 0.000 | 0.220 | 10 |
| feedback_loop | broadcast | 1.000 | 0.000 | 0.000 | 10 |
| feedback_loop | greedy | 1.000 | 0.000 | 0.203 | 10 |
| feedback_loop | rzf-centrality | 1.000 | 0.000 | 0.203 | 10 |
| feedback_loop | quality-aware-greedy | 1.000 | 0.000 | 0.203 | 10 |
| feedback_loop | imm | 1.000 | 0.000 | 0.203 | 10 |
| feedback_loop | degree | 1.000 | 0.000 | 0.203 | 10 |
| shared_memory | broadcast | 1.000 | 0.000 | 0.000 | 10 |
| shared_memory | greedy | 1.000 | 0.000 | 0.224 | 10 |
| shared_memory | rzf-centrality | 1.000 | 0.000 | 0.172 | 10 |
| shared_memory | quality-aware-greedy | 1.000 | 0.000 | 0.224 | 10 |
| shared_memory | imm | 1.000 | 0.000 | 0.187 | 10 |
| shared_memory | degree | 1.000 | 0.000 | 0.172 | 10 |
| hub_and_spoke_supervisor | broadcast | 1.000 | 0.000 | 0.000 | 10 |
| hub_and_spoke_supervisor | greedy | 1.000 | 0.000 | 0.170 | 10 |
| hub_and_spoke_supervisor | rzf-centrality | 1.000 | 0.000 | 0.127 | 10 |
| hub_and_spoke_supervisor | quality-aware-greedy | 1.000 | 0.000 | 0.170 | 10 |
| hub_and_spoke_supervisor | imm | 1.000 | 0.000 | 0.170 | 10 |
| hub_and_spoke_supervisor | degree | 1.000 | 0.000 | 0.127 | 10 |

## Reproduce

```bash
PYTHONPATH=src:. python experiments/run_evidence_harness.py --tasks-per-arm 5 --repeats 2
```
