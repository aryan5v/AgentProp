"""RZF scaling experiment: does RZF dominate on large / latency-heavy graphs?

This script is deterministic (fixed seeds, trials=100). It tests the scoped claim
that process-based RZF centrality helps on LARGE graphs where static centrality
misjudges reachability. Expected output (large graphs, IC, k=3&5 pooled):

    TABLE S1 — constrained_savings on LARGE graphs
      rzf-centrality   constr_sav=0.377  crit_cov=1.000
      random           constr_sav=0.350  crit_cov=0.938
      betweenness      constr_sav=0.284  crit_cov=0.875
      greedy           constr_sav=0.243  crit_cov=1.000
      pagerank         constr_sav=0.193  crit_cov=0.875

    Key finding: RZF maintains 100% critical-node (goal) coverage while
    betweenness/pagerank strand the output node ~12.5% of the time.

    TABLE S3 — RZF vs betweenness per large graph (constrained_savings)
      RZF wins 5, ties 0, losses 3. Wins are large (chain_30 +0.795,
      dense_20 +0.538); losses are on small single-bottleneck DAGs where
      betweenness picks the one bottleneck directly.

Contrast with small graphs (under ~15 nodes), where classical centrality is
competitive or better — see dev/experiments/run_benchmark.py on the 14 templates.
"""

import statistics as st
from collections import defaultdict

from agentprop.evaluation import run_benchmark
from agentprop.workflows import (
    chain_workflow,
    dense_workflow,
    generic_dag_workflow,
    inject_quality_decay,
    layered_pipeline_workflow,
    random_directed_workflow,
    small_world_workflow,
    tree_workflow,
)

# Large / latency-heavy graphs — RZF's theoretical home (coverage / propagation time).
SCALED = {
    "chain_30": lambda: chain_workflow(length=30),
    "tree_b3_d4": lambda: tree_workflow(branching=3, depth=4),
    "dense_20": lambda: dense_workflow(size=20),
    "small_world_30": lambda: small_world_workflow(size=30, neighborhood=3),
    "random_40": lambda: random_directed_workflow(size=40, edge_probability=0.15, seed=0),
    "random_60": lambda: random_directed_workflow(size=60, edge_probability=0.10, seed=1),
    "generic_dag_6x6": lambda: generic_dag_workflow(layers=6, width=6),
    "layered": layered_pipeline_workflow,
}

ALGOS = ["rzf-centrality", "greedy", "betweenness", "pagerank", "random"]
MODELS = ["independent-cascade", "quality-cascade"]

rows = []
for name, fn in SCALED.items():
    g = inject_quality_decay(fn(), seed=0)
    for k in (3, 5):
        rows.extend(
            run_benchmark(
                g,
                workflow_name=name,
                algorithms=ALGOS,
                models=MODELS,
                budget=k,
                trials=100,
            )
        )

rd = [r.to_dict() for r in rows]
ic = [r for r in rd if r["propagation_model"] == "independent-cascade"]
qc = [r for r in rd if r["propagation_model"] == "quality-cascade"]


def agg(sub, vf):
    d = defaultdict(list)
    for r in sub:
        d[r["algorithm"]].append(vf(r))
    return {a: st.mean(v) for a, v in d.items()}


sizes = {name: (fn().node_count, fn().edge_count) for name, fn in SCALED.items()}
print("Graph sizes (nodes, edges):")
for n, (nn, ee) in sizes.items():
    print(f"  {n:18s} {nn:3d} nodes  {ee:3d} edges")

print("\n" + "=" * 70)
print("TABLE S1 — constrained_savings on LARGE graphs (IC, k=3&5 pooled)")
print("=" * 70)
cs = agg(ic, lambda r: r["constrained_savings"])
cc = agg(ic, lambda r: r["critical_coverage"])
ept = agg(ic, lambda r: r["expected_propagation_time"])
for a in sorted(cs, key=lambda a: -cs[a]):
    print(f"{a:16s} constr_sav={cs[a]:7.3f}  crit_cov={cc[a]:.3f}  exp_prop_time={ept[a]:6.2f}")

print("\n" + "=" * 70)
print("TABLE S2 — mean_output_quality on LARGE graphs (QC, k=3&5 pooled)")
print("=" * 70)
mq = agg(qc, lambda r: r.get("mean_output_quality", 0.0))
for a in sorted(mq, key=lambda a: -mq[a]):
    print(f"{a:16s} mean_output_quality={mq[a]:.3f}")

print("\n" + "=" * 70)
print("TABLE S3 — RZF vs betweenness per large graph (IC k=5, constrained_savings)")
print("=" * 70)
sub = [r for r in ic if r["budget"] == 5]
byw = defaultdict(dict)
for r in sub:
    byw[r["workflow"]][r["algorithm"]] = (
        r["constrained_savings"],
        r["expected_propagation_time"],
    )
w = t = loss = 0
print(
    f"{'workflow':18s} {'rzf_cs':>8s} {'betw_cs':>8s} {'delta':>8s}"
    f" {'rzf_time':>9s} {'betw_time':>9s}"
)
for wf in sorted(byw):
    rz = byw[wf].get("rzf-centrality")
    bt = byw[wf].get("betweenness")
    if not rz or not bt:
        continue
    d = rz[0] - bt[0]
    if d > 1e-9:
        w += 1
    elif d < -1e-9:
        loss += 1
    else:
        t += 1
    print(f"{wf:18s} {rz[0]:8.3f} {bt[0]:8.3f} {d:+8.3f} {rz[1]:9.2f} {bt[1]:9.2f}")
print(f"RZF vs betweenness (constrained_savings): wins={w} ties={t} losses={loss}")
