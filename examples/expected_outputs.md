# Example Expected Outputs

## `examples/quickstart.py`

Command:

```bash
PYTHONPATH=src python examples/quickstart.py
```

Expected shape:

```text
seeds=['planner', 'tester']
coverage=1.000
savings=0.266
```

## CLI Optimization

Command:

```bash
agentprop optimize benchmarks/workflows/planner_coder_tester_reviewer.json --budget 2 --trials 20
```

Expected shape:

```text
AgentProp Optimization Report
Recommended seeds: planner, tester
Coverage: 100.0%
Estimated savings: 26.6%
```
