# Examples

Small, runnable integrations. Install first: `pip install -e ".[dev]"`.

## Learning path

| Order | File | What you learn |
| --- | --- | --- |
| 1 | [quickstart.py](quickstart.py) | Analyze and optimize a built-in workflow from Python |
| 2 | [minimal_control_loop.py](minimal_control_loop.py) | Wrap a harness with `ControlSession` and act on decisions |
| 3 | [langgraph_e2e.py](langgraph_e2e.py) | Round-trip LangGraph adapter + control session |
| 4 | [custom_workflow.json](custom_workflow.json) | Workflow JSON consumed by CLI and SDK |

Expected CLI outputs for quickstart are documented in
[expected_outputs.md](expected_outputs.md).

## Related docs

- [Control layer quickstart](../docs/control_layer_quickstart.md)
- [Framework integrations](../docs/framework_integrations.md)
- [Workflow JSON schema](../docs/workflow_schema.md)
- [AGENTS.md](../docs/project/AGENTS.md) — coding-agent map
