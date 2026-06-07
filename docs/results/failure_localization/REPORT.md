# Failure Localization (synthetic)

**Label:** benchmark result on built-in workflow templates. No live LLM calls.

## What this measures

For each workflow, the script places `k` verifiers using three heuristics and
injects a synthetic fault at every node. It records whether the observable
verifier-distance signature is unique (**collision rate**) and the graph's
**resolving coverage** for that placement.

Lower collision rate and higher resolving coverage mean failures are easier to
localize — the core metric-dimension contribution.

## Key finding (k = 3, all templates)

| Method | Mean collision rate | Mean resolving coverage |
| --- | ---: | ---: |
| `metric_dim` | lowest | highest |
| `risk_aware` | moderate | high |
| `betweenness` | highest | moderate |

On simple topologies (e.g. `chain`), metric-dimension placement reaches zero
collisions at small `k`. On denser graphs, betweenness placement collides more
often at low `k` before converging as `k` increases.

## Limitations

- Synthetic fault injection only; no real agent errors or verifier commands.
- Undirected shortest-path distances; runtime verifier semantics may differ.
- Built-in templates only — custom workflows need a fresh run.

## Reproduce

```bash
python dev/experiments/failure_localization_study.py
```

See [README.md](README.md) and [verifier semantics](../../verifier_semantics.md).
