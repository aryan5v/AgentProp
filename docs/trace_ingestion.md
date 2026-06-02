# Trace Ingestion

AgentProp can turn message-level workflow traces into an `AgentGraph`.

## Trace Shape

```json
{
  "events": [
    {
      "source": "planner",
      "target": "coder",
      "source_type": "PLANNER",
      "target_type": "EXECUTOR",
      "token_cost": 600,
      "latency": 0.8,
      "success": true
    }
  ]
}
```

`messages` may be used instead of `events`.

## CLI

```bash
agentprop trace trace.json --out workflow.json
agentprop optimize workflow.json --budget 2
```

## Aggregation

Trace ingestion aggregates:

- node token cost
- node average latency
- node error rate
- edge message count
- edge token/message cost
- edge average latency
- edge reliability
- edge activation probability proxy

The result is a normal AgentProp workflow JSON file.

## Calibrating an Existing Graph

Template graph values are priors. When trace evidence exists, calibrate a copy
of the graph instead of relying on hard-coded `activation_probability`,
reliability, token cost, and latency values:

```python
from agentprop.integrations import calibrate_graph_from_trace_dict
from agentprop.workflows import planner_coder_tester_reviewer

graph = planner_coder_tester_reviewer()
calibrated = calibrate_graph_from_trace_dict(graph, trace).graph
```

Observed edges receive measured activation probability, reliability, message
cost, and latency. Unobserved outgoing edges from a source that appears in the
trace receive a small smoothed activation probability instead of keeping the
default `1.0`, which prevents synthetic propagation from assuming every
possible handoff always fires.
