# Framework Integrations

AgentProp includes dependency-light adapters for common agent orchestration
frameworks. These adapters use plain dictionaries, so they do not require the
framework packages at core install time.

Supported export/import targets:

- `langgraph`
- `autogen`
- `crewai`
- `openai-agents`
- `llamaindex`

Example:

```python
from agentprop.integrations import graph_from_framework_dict, to_framework_dict
from agentprop.workflows import planner_coder_tester_reviewer

graph = planner_coder_tester_reviewer()
spec = to_framework_dict(graph, "langgraph")
round_tripped = graph_from_framework_dict(spec, "langgraph")
```

The adapter layer preserves:

- node ids, names, roles, types, tool access, cost, latency, reliability, and
  error metadata
- directed communication edges
- edge cost, latency, relevance, reliability, activation probability, and weight
- entrypoint/output concepts where the target framework shape has an analogue

These are interchange specs, not runtime launchers. Real framework execution can
be added later as optional extras that consume the same dictionary contracts.
