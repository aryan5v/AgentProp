# Framework Builders

Use this when the user is building or improving a real multi-agent framework
workflow.

## Supported Shapes

AgentProp can import or sketch:

- LangGraph-style dictionaries
- CrewAI-style dictionaries
- AutoGen-style dictionaries
- OpenAI Agents-style dictionaries
- LlamaIndex-style dictionaries
- custom AgentProp workflow JSON

## Integration Pattern

1. Convert or sketch the framework workflow as an AgentGraph.
2. Analyze it with the CLI.
3. Use seed/verifier/pruning guidance while editing the workflow.
4. Emit runtime events from each agent/tool/verifier step if using
   `ControlSession`.
5. Save reports and traces.

## CLI First

```bash
agentprop analyze <workflow> --json
agentprop report <workflow> --out reports/framework_agentprop_report.md
```

## Python Import Pattern

```python
from agentprop.integrations.framework_adapters import graph_from_framework_dict
from agentprop.runtime import ControlSession

graph = graph_from_framework_dict(workflow_dict, "langgraph")
session = ControlSession.start(
    graph,
    task_id="framework-task",
    category="research-pipeline",
)
```

## Builder Guidance

- Treat verifier nodes as first-class workflow nodes, not comments in prompts.
- Route more context to high-sensitivity roles.
- Couple verifier placement to compressed or pruned upstream context.
- Log enough per-node events to explain why a failure propagated.
- Do not claim a workflow is cheaper unless saved artifacts compare tokens,
  elapsed time, and pass/fail outcome.
