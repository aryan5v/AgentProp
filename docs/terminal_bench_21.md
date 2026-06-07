# Terminal-Bench 2.1 Preparation

AgentProp includes a dry-run preparation path for a future Terminal-Bench 2.1
run with Terminus-2. The preparation command writes the run manifest, runbook,
AgentProp guidance, and watchdog shell script without executing Harbor.

> **Local output only.** Launch bundles are written to your chosen output
> directory (for example `benchmark-results/`). Do not commit manifests, runbooks,
> or watchdog scripts to the public repository. See
> [results/ARTIFACTS.md](results/ARTIFACTS.md).

## Prepare The Launch Bundle

```bash
agentprop terminal-bench prepare \
  --dataset terminal-bench/terminal-bench-2-1 \
  --agent terminus-2 \
  --model google/gemini-3.1-pro-preview \
  --environment modal \
  --out-dir benchmark-results/terminal-bench-2.1 \
  --output-root benchmark-results/terminal-bench-2.1/terminus-2-agentprop \
  --registry-root results/benchmark_registry
```

This writes:

- `manifest.json`: the exact dataset, agent, model, environment, command, and
  watchdog configuration.
- `RUNBOOK.md`: a launch checklist and prepared command.
- `agentprop-extra-instructions.md`: benchmark guidance for task category
  routing, verifier use, numerical preflight, and budget-aware stopping.
- `run_with_watchdog.sh`: the command wrapper to use when the benchmark run is
  intentionally started.

## Watchdog Wrapper

Long external benchmark runs should be launched through the watchdog wrapper:

```bash
python dev/experiments/run_with_watchdog.py \
  --timeout 21600 \
  --idle-timeout 1800 \
  --log benchmark-results/terminal-bench-2.1/terminus-2-agentprop/launcher.log \
  --status-json benchmark-results/terminal-bench-2.1/terminus-2-agentprop/watchdog-status.json \
  -- harbor run -d terminal-bench/terminal-bench-2-1 -a terminus-2 -m google/gemini-3.1-pro-preview --env modal
```

The wrapper records launcher output and writes a status JSON even if the run
times out or goes idle.

## Summarize Saved Harbor Results

After a completed run, summarize saved Harbor `result.json` artifacts:

```bash
agentprop terminal-bench summarize \
  --results-root benchmark-results/terminal-bench-2.1/terminus-2-agentprop \
  --out-dir benchmark-results/terminal-bench-2.1/terminus-2-agentprop/report \
  --registry-root results/benchmark_registry
```

The summary command writes:

- `summary.json`: aggregate pass rate, exceptions, token totals, reported cost,
  timeout rate, and cost-adjusted success.
- `task_results.csv`: task-level reward, pass/fail, exception, token, and cost
  rows.
- `report.md`: a public-facing Markdown summary suitable for review before
  copying into release docs.
