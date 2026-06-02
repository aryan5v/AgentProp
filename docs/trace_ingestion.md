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

## Empirical Training Rows

Traces can also become empirical ML/DL/RL training rows when they include, or
can be joined to, task outcome fields such as `verification_passed`,
`quality_score`, or `quality_passed`:

```python
from agentprop.integrations import empirical_rows_from_trace_dicts

result = empirical_rows_from_trace_dicts(traces, outcome_rows=case_study_rows)
rows = result.rows
```

The generated rows include `selected_seeds`, `context_allocations`, outcome
labels, cost fields, and latency fields. They can be passed to:

```bash
PYTHONPATH=src:. python experiments/build_empirical_rows.py \
  --trace results/case_study/traces.jsonl \
  --outcome-results results/case_study/results.json \
  --out rows.json

PYTHONPATH=src:. python experiments/train_seed_scorer.py --empirical-results rows.json
PYTHONPATH=src:. python experiments/train_torch_gnn.py --empirical-results rows.json
PYTHONPATH=src:. python experiments/run_rl_routing.py --reward-calibration-rows rows.json
```

Traces without any task outcome are skipped instead of becoming training labels.
This keeps trace-calibrated learning grounded in pass/fail or quality evidence,
not just topology or message frequency.

The row builder accepts JSON or JSONL traces, plus optional outcome artifacts
with `rows`, `tasks`, or `results` arrays. Its output records how many traces
were converted and how many were skipped for missing labels, so benchmark runs
cannot silently train on unlabeled topology-only data.
