# Benchmarks And Evidence

Use this when the user wants to run a benchmark, support a public claim, or
publish results.

## Evidence Bar

Do not overstate. Prefer:

- "early demo"
- "directional result"
- "pass-preserving token reduction on this task"
- "designed to"

Avoid broad claims like "AgentProp improves all multi-agent workflows" unless a
saved benchmark artifact supports it.

## Required Artifacts

For each result, save:

- task id
- arm or strategy name
- pass/fail reward
- token count
- cost if available
- elapsed time
- verifier output
- AgentProp trace/report path

## Terminal-Bench Preparation

Prepare launch bundles before running expensive benchmarks. Write them to a local
output directory (`benchmark-results/`); do not commit manifests, runbooks, or
watchdog scripts to the public repository. Only sanitized summaries belong under
`docs/results/` (see `docs/results/ARTIFACTS.md`).

```bash
agentprop terminal-bench prepare \
  --dataset terminal-bench/terminal-bench-2-1 \
  --agent terminus-2 \
  --model <model> \
  --environment <environment> \
  --out-dir benchmark-results/terminal-bench-2.1
```

Summarize saved Harbor results:

```bash
agentprop terminal-bench summarize \
  --results-root <results-root> \
  --out-dir <summary-out-dir>
```

## Interpreting A2/A3-Style Runs

- A0 is raw agent execution.
- A1 is static instruction guidance.
- A2 is frozen controller guidance.
- A3 is learning or category-conditioned control when enough training tasks
  exist.

For small samples, report task-level deltas instead of headline benchmark claims.
If A3 has no real train split, do not describe it as learned.
