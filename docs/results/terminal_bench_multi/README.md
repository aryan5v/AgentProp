# Terminal-Bench Multi-Task Control Comparison

Protocol for comparing **A0 raw agent** vs **A2 AgentProp control** across
multiple Terminal-Bench tasks (target: ≥5 tasks with pass-preserving token
reduction).

## Arms

| Arm | Description |
| --- | --- |
| A0 | Raw coding agent (Harbor default) |
| A2 | AgentProp `StoppingController` wrapping the same agent via extra instructions |

## Prepare bundles (local only — do not commit)

```bash
agentprop terminal-bench prepare \
  --dataset terminal-bench/terminal-bench-2-1 \
  --agent codex \
  --model gpt-5.5 \
  --task-count 5 \
  --out-dir benchmark-results/terminal-bench-multi
```

Run A0 and A2 separately with Harbor, saving `result.json` per task under
distinct output roots.

## Summarize

```bash
agentprop terminal-bench summarize \
  --results-root benchmark-results/terminal-bench-multi/a0 \
  --out-dir benchmark-results/terminal-bench-multi/a0-report

agentprop terminal-bench summarize \
  --results-root benchmark-results/terminal-bench-multi/a2 \
  --out-dir benchmark-results/terminal-bench-multi/a2-report
```

## Early signal (single task)

On `regex-log` with Codex + gpt-5.5, A2 preserved pass while reducing tokens
33.8% vs A0. See [README early signal section](../../../README.md). Multi-task
replication is required before treating this as a benchmark claim.

## Public artifacts

After sanitization, copy aggregate `summary.json` and `report.md` here per
[ARTIFACTS.md](../ARTIFACTS.md). Do not commit Harbor launch bundles or raw
prompts.
