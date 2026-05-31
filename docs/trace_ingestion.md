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
