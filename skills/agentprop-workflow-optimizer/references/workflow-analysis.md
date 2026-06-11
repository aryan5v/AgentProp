# Workflow Analysis With AgentProp

## Pick A Workflow

Use a built-in workflow when it matches the task:

- `planner_coder_tester_reviewer`
- `research_writer_verifier`
- `rag_pipeline`
- `tool_use_pipeline`
- `chain`
- `tree`
- `star`
- `dense_graph`
- `small_world_graph`
- `generic_dag`

Otherwise use a workflow JSON path.

## Analyze Structure

```bash
agentprop analyze <workflow> --json
```

Use this to identify bottlenecks, pruning candidates, and verifier candidates.

## Optimize Context Routing

```bash
agentprop optimize <workflow> \
  --budget 2 \
  --algorithm greedy \
  --model rzf \
  --trials 50 \
  --json
```

Use the selected seeds as first full-context recipients. For coding workflows,
do not starve high-sensitivity roles such as coder, tester, verifier, planner,
or domain specialist just because topology-only centrality ranked another node.

## Check Pruning Risk

```bash
agentprop prune <workflow> \
  --target-token-reduction 0.3 \
  --model rzf \
  --trials 50 \
  --json
```

Do not prune edges that carry user constraints, tool outputs, test results, or
verifier feedback unless the report shows acceptable risk and the user approves
the trade-off.

## Write A Durable Report

```bash
agentprop report <workflow> \
  --budget 2 \
  --algorithm greedy \
  --model rzf \
  --trials 50 \
  --out reports/agentprop_report.html \
  --format html
```

Save the report path in the final response.

## Interactive View

```bash
agentprop view <workflow> --out reports/view.html
agentprop view <workflow> --trace reports/control-demo/trace.jsonl --out reports/view.html
```

Writes a single self-contained HTML file: force-directed graph with verifier
(green), seed (blue), and bottleneck (orange) overlays, node detail on click,
and a decision timeline when a control trace is supplied. No server or
external assets — safe to attach to CI artifacts or send to the user.
