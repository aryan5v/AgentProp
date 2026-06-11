# Visualization

## Interactive HTML view

`agentprop view` writes a single self-contained HTML file — interactive
force-directed graph with verifier/seed/bottleneck overlays from `analyze()`,
node detail on click, and an optional control-decision timeline. No external
assets, no server; open it in any browser or attach it to CI artifacts.

```bash
agentprop view planner_coder_tester_reviewer --out reports/view.html
agentprop view my_workflow.json --trace reports/control-demo/trace.jsonl
```

From Python:

```python
from agentprop.visualization import write_workflow_view

write_workflow_view(graph, "reports/view.html", title="my workflow")
```

## Graphviz DOT export

AgentProp can also export workflow graphs as Graphviz DOT files.

```bash
agentprop viz planner_coder_tester_reviewer --out reports/workflow.dot
```

Render with Graphviz:

```bash
dot -Tpng reports/workflow.dot -o reports/workflow.png
```

The DOT export includes:

- node id
- node type
- edge weight
- edge message cost

This keeps visualization dependency-free in the base package while still making graph artifacts easy to render in papers, READMEs, and reports.
