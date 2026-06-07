# Example Expected Outputs

## `dev/examples/quickstart.py`

Command:

```bash
PYTHONPATH=src python dev/examples/quickstart.py
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
agentprop optimize dev/benchmarks/workflows/planner_coder_tester_reviewer.json --budget 2 --trials 20
```

Expected shape:

```text
AgentProp Optimization Report
Recommended seeds: planner, tester
Coverage: 100.0%
Estimated savings: 26.6%
```

## `dev/examples/coding_agent_full_suite.py`

Command:

```bash
python dev/examples/coding_agent_full_suite.py
```

Expected shape:

```text
AgentProp full-suite coding-agent artifacts:
- context_advice: reports/beta-coding-agent-full-suite/context_advice.json
- control_report: reports/beta-coding-agent-full-suite/control_session/report.md
- control_summary: reports/beta-coding-agent-full-suite/control_session/summary.json
- control_trace: reports/beta-coding-agent-full-suite/control_session/trace.jsonl
- decisions: reports/beta-coding-agent-full-suite/decisions.json
- host_agent_prompt: reports/beta-coding-agent-full-suite/host_agent_prompt.md
- routing_report: reports/beta-coding-agent-full-suite/routing_report.md
- routing_summary: reports/beta-coding-agent-full-suite/routing_summary.json
```
