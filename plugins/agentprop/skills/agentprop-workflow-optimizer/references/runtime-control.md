# Runtime Control

Use runtime control when AgentProp should wrap execution, not just generate an
offline graph report.

## Key-Free Demo

```bash
agentprop control-demo --demo terminal --out-dir reports/control-demo
agentprop control-demo --demo multi-agent --out-dir reports/control-demo
agentprop control-demo --demo framework --out-dir reports/control-demo
```

Each demo writes:

- `trace.jsonl`
- `summary.json`
- `report.md`

## Python ControlSession

```python
from agentprop.runtime import ControlSession, ExecutionEvent

session = ControlSession.start(
    "planner_coder_tester_reviewer",
    task_id="task-123",
    category="implementation",
    token_budget=120_000,
    baseline_tokens=180_000,
)

decision = session.observe(
    ExecutionEvent(
        step=1,
        command="pytest -q",
        verifier_run=True,
        verifier_passed=False,
        error_signature="AssertionError:test_edge_case",
        tokens_used=18_000,
        elapsed_s=42.0,
    )
)

session.record_outcome(passed=True, quality_score=1.0)
session.write_artifacts("reports/task-123")
```

## Decision Meanings

- `CONTINUE`: stay within the current execution plan.
- `FORCE_VERIFY`: run or strengthen an independent verifier.
- `SWITCH_STRATEGY`: avoid repeating the same failing probe.
- `FINALIZE`: stop because a trusted verifier passed or a budget/stop rule was
  reached.

## Integration Rule

AgentProp observes and decides. The host agent or runtime executes shell
commands, model calls, tools, and verifiers.

## Escalation And Calibration Add-ons

- `CascadeRiskAdvisor` wraps controller decisions and upgrades borderline
  CONTINUE to FORCE_VERIFY when forward simulation says a failure at the
  active node would cascade widely. See
  `references/statistics-and-learning.md` for usage.
- `ConformalRiskGate` turns any risk score into a calibrated FORCE_VERIFY
  threshold with a guaranteed miss rate.
- Every `record_outcome` reward row carries graph-position features
  (schema v2, `docs/reward_record_schema.md`) so routing policies can learn
  from history.
