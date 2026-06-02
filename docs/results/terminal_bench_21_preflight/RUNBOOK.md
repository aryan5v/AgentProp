# Terminal-Bench 2.1 + Terminus-2 Launch Runbook

This bundle prepares the next AgentProp benchmark run. It does not execute the
benchmark by itself.

## Launch Contract

- Dataset: `terminal-bench/terminal-bench-2-1`
- Agent: `terminus-2`
- Model: `google/gemini-3.1-pro-preview`
- Environment: `modal`
- Manifest: `docs/results/terminal_bench_21_preflight/manifest.json`
- AgentProp extra instructions: `docs/results/terminal_bench_21_preflight/agentprop-extra-instructions.md`

## Preflight

1. Confirm Docker, Harbor, Modal, and model-provider credentials are configured.
2. Confirm the command below uses the intended model and environment.
3. Confirm the output root is empty or intentionally reused:
   `benchmark-results/terminal-bench-2.1/terminus-2-agentprop`.
4. Keep the watchdog enabled so hung external IO cannot burn budget silently.

## Prepared Command

```bash
python experiments/run_with_watchdog.py --timeout 21600 --idle-timeout 1800 --poll-interval 5 --log benchmark-results/terminal-bench-2.1/terminus-2-agentprop/launcher.log --status-json benchmark-results/terminal-bench-2.1/terminus-2-agentprop/watchdog-status.json -- harbor run -d terminal-bench/terminal-bench-2-1 -a terminus-2 -m google/gemini-3.1-pro-preview --env modal
```

## After The Run

```bash
python experiments/summarize_harbor_results.py \
  --results-root benchmark-results/terminal-bench-2.1/terminus-2-agentprop \
  --out-dir benchmark-results/terminal-bench-2.1/terminus-2-agentprop/report
```

The summary step writes `summary.json`, `task_results.csv`, and `report.md`.
