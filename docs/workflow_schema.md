# AgentProp Workflow JSON Schema

AgentProp workflows are directed weighted graphs serialized as JSON.

Machine-readable schema: [dev/configs/schemas/workflow.json](../dev/configs/schemas/workflow.json)

## Top-Level Shape

```json
{
  "nodes": [],
  "edges": []
}
```

Both `nodes` and `edges` are required lists.

## Node Fields

Required:

- `id`: unique non-empty string

Common optional fields:

- `type`: one of `AGENT`, `TOOL`, `MEMORY`, `DOCUMENT`, `VERIFIER`, `PLANNER`, `EXECUTOR`, `REVIEWER`, `OUTPUT`, `CUSTOM`
- `name`: display name
- `role`: natural-language role
- `token_cost`: non-negative number
- `latency`: non-negative number
- `reliability`: number between 0 and 1
- `error_rate`: number between 0 and 1
- `context_capacity`: integer or null
- `tool_access`: list of strings
- `importance_score`: number or null
- `metadata`: object

## Edge Fields

Required:

- `source`: source node id
- `target`: target node id

Common optional fields:

- `message_cost`: non-negative number
- `latency`: non-negative number
- `relevance`: number between 0 and 1
- `reliability`: number between 0 and 1
- `activation_probability`: number between 0 and 1
- `dependency_strength`: number between 0 and 1
- `weight`: non-negative number
- `metadata`: object

## Validation Rules

AgentProp validates workflow JSON before loading:

- Node ids must be unique.
- Edge endpoints must reference existing nodes.
- Self-loops are rejected for v1.
- Cost, latency, and weight fields must be non-negative.
- Probability-like fields must be between 0 and 1.
- Node `type` must be a supported `NodeType`.

## Minimal Example

```json
{
  "nodes": [
    {"id": "planner", "type": "PLANNER", "token_cost": 1000},
    {"id": "coder", "type": "EXECUTOR", "token_cost": 1500}
  ],
  "edges": [
    {"source": "planner", "target": "coder", "message_cost": 500, "weight": 0.9}
  ]
}
```
