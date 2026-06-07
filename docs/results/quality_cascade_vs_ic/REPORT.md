# Quality Cascade vs IC (synthetic routing)

**Label:** benchmark result on built-in workflow templates. Propagation-simulation
metrics only — not live LLM task success.

## What this measures

For each workflow, AgentProp compares three routing arms at seed budget 2 with
30 trials:

- **Quality cascade + quality-aware seeds** — propagates output quality along edges.
- **IC + PageRank** — classical influence maximization baseline.
- **IC + greedy** — greedy influence baseline.

Reported fields: coverage, estimated token savings vs broadcast, mean output
quality (QC arm only), optimized vs broadcast cost.

## Key finding

Quality-cascade arms preserve **non-zero mean output quality** on workflows where
quality decay matters (e.g. `planner_coder_tester_reviewer`, `rag_pipeline`).
IC arms report zero output quality by construction but can still match coverage
on simple chains.

On `planner_coder_tester_reviewer`, quality-aware + QC achieves higher estimated
savings than IC PageRank at comparable coverage — the main routing Pareto claim
for this study.

## Limitations

- Simulation costs from graph edge weights, not measured LLM tokens.
- Six built-in templates; generalization to custom graphs unverified.
- Single seed budget (2); sweep `agentprop benchmark` for broader Pareto curves.

## Reproduce

```bash
python dev/experiments/quality_cascade_vs_ic.py
```

See [README.md](README.md) and [quality scoring](../../quality_scoring.md).
