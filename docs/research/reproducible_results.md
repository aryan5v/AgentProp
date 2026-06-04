# Reproducible Results

Every number on this page is produced by a deterministic script in
`experiments/`. Run the script and compare against the tables below; if your
numbers match, you have reproduced the published figures. All graph experiments
are CPU-only and require no model credentials.

## Verifier placement (core contribution)

```bash
PYTHONPATH=src:. python experiments/verifier_placement_evidence.py
```

**Resolving coverage vs budget `k`** — fraction of failure-mode pairs uniquely
localizable by the verifier set, mean over 14 workflow templates. Higher is
better; `min-k@1.0` is the average budget needed to reach a full resolving set.

| Method        | k=1   | k=2   | k=3   | k=4   | k=5   | min-k@1.0 |
|---------------|-------|-------|-------|-------|-------|-----------|
| `metric_dim`  | 0.790 | 0.904 | 0.956 | 0.980 | 0.992 | **3.50**  |
| `risk_aware`  | 0.688 | 0.851 | 0.930 | 0.966 | 0.983 | 4.14      |
| `betweenness` | 0.669 | 0.814 | 0.911 | 0.954 | 0.976 | 4.50      |

Metric-dimension placement dominates at every budget and reaches a full
resolving set on 11/14 workflows by `k=5` vs 8/14 for betweenness.

**Fault tolerance at `k=5`** — worst-case coverage after *any* single verifier
fails:

| Method               | ft_coverage | fully-robust workflows |
|----------------------|-------------|------------------------|
| `metric_dim` (plain) | 0.833       | 0/14                   |
| `metric_dim_ft`      | **0.972**   | **9/14**               |

Plain placement is robust to a verifier failure on zero workflows; the
fault-tolerant variant on nine. The remaining five are budget-limited
(fault-tolerant metric dimension can exceed `k=5`).

**Localization demo** — on `planner_coder_tester_reviewer`, the resolving set
gives every node a unique distance signature. Drop to a single verifier and
`planner` and `tester` collide on signature `(1,)`: two distinct failures become
indistinguishable.

## RZF seeding — scope characterization

```bash
PYTHONPATH=src:. python experiments/rzf_scaling_study.py
```

This is a **scoped, secondary** result, reported honestly.

**Large graphs (30–122 nodes, IC, k=3&5 pooled):** RZF leads and never strands
the goal node.

| Algorithm        | constrained savings | critical coverage |
|------------------|---------------------|-------------------|
| `rzf-centrality` | **0.377**           | **1.000**         |
| `random`         | 0.350               | 0.938             |
| `betweenness`    | 0.284               | 0.875             |
| `greedy`         | 0.243               | 1.000             |
| `pagerank`       | 0.193               | 0.875             |

RZF maintains 100% critical-node coverage while betweenness/pagerank strand the
output node ~12.5% of the time. RZF beats betweenness 5–3 head-to-head; wins are
large (`chain_30` +0.795, `dense_20` +0.538), losses confined to small
single-bottleneck DAGs.

**Small graphs (14 templates, under ~15 nodes):** no single winner — classical
centrality (betweenness) is competitive or better. The takeaway is a scope
claim, not a universal one: *process-based centrality matters when graphs are
large enough that static centrality misjudges reachability.*

## Goal-aware metric (methodology note)

Plain `estimated_savings` is confounded with coverage: random seeding can "win"
by reaching almost nothing. We report `constrained_savings`, which credits
savings only when all critical (OUTPUT/VERIFIER) nodes are reached, closing the
"do-nothing wins" loophole. Under the constrained metric, random's spurious lead
collapses because it reaches the goal only ~93% of the time.

## Early live signal (directional)

A single Terminal-Bench 2.1 task (`regex-log`, Harbor `codex` agent, `gpt-5.5`,
same model on both arms): A2 control preserved the pass while cutting 33.8% of
tokens and 41.0% of cost versus raw Codex. This is a single-task early signal,
not a benchmark claim — a larger repeated study is the real test.
