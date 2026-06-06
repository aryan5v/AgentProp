# Framework Integrations

AgentProp includes dependency-light adapters for common agent orchestration
frameworks. These adapters use plain dictionaries, so they do not require the
framework packages at core install time.

## Integration status

| Framework | Dict interchange | Native builder | Round-trip tested | Notes |
| --- | --- | --- | --- | --- |
| `langgraph` | Yes | Yes (placeholder nodes) | Yes | Best E2E path; see `examples/langgraph_e2e.py` |
| `crewai` | Yes | Yes when installed | Partial | Requires `crewai` package |
| `openai-agents` | Yes | Yes when installed | Partial | Requires OpenAI Agents SDK |
| `autogen` | Yes | No | Dict only | Native builder raises `NativeFrameworkUnavailable` |
| `llamaindex` | Yes | No | Dict only | Native builder raises `NativeFrameworkUnavailable` |

Dict adapters are **interchange specs**, not runtime launchers. Use
`ControlSession` to wrap execution regardless of framework.

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

The dictionary adapters are interchange specs, not runtime launchers. Native
execution hooks can be layered on top as optional extras that consume the same
dictionary contracts.

## Optional Native Builders

AgentProp also exposes best-effort native builders for frameworks where a
workflow object can be constructed without model credentials or user-defined step
functions:

```python
from agentprop.integrations import native_framework_status, to_native_framework
from agentprop.workflows import planner_coder_tester_reviewer

print([status.to_dict() for status in native_framework_status()])

graph = planner_coder_tester_reviewer()
langgraph_builder = to_native_framework(graph, "langgraph")
```

Current native-builder status:

- `langgraph`: builds a `StateGraph` with placeholder pass-through node
  functions, entrypoint, finish-point, and graph edges.
- `crewai`: builds `Agent`, `Task`, and `Crew` objects when `crewai` is
  installed.
- `openai-agents`: builds a list of `Agent` objects when the OpenAI Agents SDK
  package is installed.
- `autogen`: still requires configured model clients and runtime wiring, so the
  native builder raises `NativeFrameworkUnavailable` with the required next step.
- `llamaindex`: still requires user-defined workflow step functions, so the
  native builder raises `NativeFrameworkUnavailable` with the required next step.

These native builders are implementation hooks, not validation evidence. They
make it easier to plug AgentProp graphs into real framework projects while
keeping the base package free of heavy orchestration dependencies.
