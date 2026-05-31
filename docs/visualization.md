# Visualization

AgentProp can export workflow graphs as Graphviz DOT files.

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
