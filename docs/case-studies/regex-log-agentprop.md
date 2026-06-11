# Regex Log Case Study

This case study uses the saved Terminal-Bench `regex-log` run as the first
developer-facing Track A adoption artifact.

## Workflow

- Baseline: raw Codex solving the task end to end.
- AgentProp: analyze the workflow graph, place verifier/control checkpoints,
  and run the same task through the control loop.

## Results

| Arm | Result | Tokens | Cost |
| --- | --- | ---: | ---: |
| A0 raw Codex | pass | 123,731 | $0.33 |
| A2 AgentProp control | pass | 81,949 | $0.20 |

AgentProp preserved pass status while reducing tokens by 33.8% and cost by
39.4% on this single task.

## Failure Attribution

The useful debugging signal was not only lower cost. The controller trace
separated untrusted local signals from independent verifier results, so a
developer can see whether a proposed final answer was confirmed or merely
self-reported by the agent. That verifier signature is the practical surface of
AgentProp's resolving-set analysis: a small set of checkpoints makes failures
localizable without broadcasting every intermediate state to every node.

## Reproduce The Flow

```bash
agentprop analyze planner_coder_tester_reviewer --out reports/langgraph-analysis.md
python examples/minimal_control_loop.py
agentprop trace-replay reports/control-demo/trace.jsonl --no-control
```

This is directional evidence, not a benchmark claim. The next proof point should
repeat the same before/after shape on a live LangGraph example workflow using
`agentprop.wrap(...)`.
