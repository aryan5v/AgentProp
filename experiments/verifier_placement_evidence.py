"""Verifier-placement evidence: resolving_coverage vs k + failure localization."""

import statistics as st
from collections import defaultdict

import networkx as nx

from agentprop.algorithms import (
    betweenness_verifier_placement,
    fault_tolerant_resolving_coverage,
    metric_dimension_verifier_placement,
    resolving_coverage,
    risk_aware_verifier_placement,
)
from agentprop.workflows import WORKFLOW_TEMPLATES

KS = [1, 2, 3, 4, 5]
METHODS = {
    "metric_dim": lambda g, k: metric_dimension_verifier_placement(g, k),
    "metric_dim_ft": lambda g, k: metric_dimension_verifier_placement(g, k, fault_tolerant=True),
    "risk_aware": lambda g, k: risk_aware_verifier_placement(g, k),
    "betweenness": lambda g, k: betweenness_verifier_placement(g, k),
}

print("=" * 78)
print("TABLE V1 — resolving_coverage vs k (mean over 14 workflows)")
print("  Fraction of failure-mode pairs uniquely localizable by the verifier set")
print("=" * 78)
cov = defaultdict(lambda: defaultdict(list))
firstperfect = defaultdict(list)  # method -> list of min-k achieving coverage 1.0
for _name, fn in WORKFLOW_TEMPLATES.items():
    g = fn()
    for method, placer in METHODS.items():
        reached = None
        for k in KS:
            verifiers = placer(g, k)
            c = resolving_coverage(g, verifiers)
            cov[method][k].append(c)
            if reached is None and c >= 1.0 - 1e-9:
                reached = k
        firstperfect[method].append(reached if reached is not None else max(KS) + 1)

header = "method          " + "".join(f"  k={k:<6d}" for k in KS) + "  min-k@1.0"
print(header)
for method in METHODS:
    row = f"{method:15s} "
    for k in KS:
        row += f" {st.mean(cov[method][k]):8.3f}"
    mk = st.mean(firstperfect[method])
    row += f"  {mk:9.2f}"
    print(row)

print()
print("=" * 78)
print("TABLE V2 — workflows reaching resolving_coverage==1.0 by budget k")
print("=" * 78)
print(f"{'method':15s}" + "".join(f"  k<={k}" for k in KS))
for method in METHODS:
    row = f"{method:15s}"
    for k in KS:
        n = sum(1 for c in cov[method][k] if c >= 1.0 - 1e-9)
        row += f"  {n:4d}"
    print(row)

print()
print("=" * 78)
print("TABLE V3 — fault-tolerant coverage: worst-case after ANY single verifier")
print("  fails. Compares plain metric-dim placement vs the fault-tolerant variant")
print("  at a fixed budget k=5 (mean over 14 workflows).")
print("=" * 78)
ft_plain = []
ft_ft = []
n_robust_plain = 0
n_robust_ft = 0
for _name, fn in WORKFLOW_TEMPLATES.items():
    g = fn()
    plain_set = metric_dimension_verifier_placement(g, 5)
    ft_set = metric_dimension_verifier_placement(g, 5, fault_tolerant=True)
    fp = fault_tolerant_resolving_coverage(g, plain_set)
    ff = fault_tolerant_resolving_coverage(g, ft_set)
    ft_plain.append(fp)
    ft_ft.append(ff)
    n_robust_plain += 1 if fp >= 1.0 - 1e-9 else 0
    n_robust_ft += 1 if ff >= 1.0 - 1e-9 else 0
print(f"  {'metric_dim (plain)':22s} ft_coverage={st.mean(ft_plain):.3f}  "
      f"fully-robust workflows={n_robust_plain}/14")
print(f"  {'metric_dim_ft':22s} ft_coverage={st.mean(ft_ft):.3f}  "
      f"fully-robust workflows={n_robust_ft}/14")

print()
print("=" * 78)
print("DEMO — Failure localization: resolving vs non-resolving verifier set")
print("=" * 78)


def signature(dist, node, verifiers):
    return tuple(dist.get(node, {}).get(v, -1) for v in verifiers)


g = WORKFLOW_TEMPLATES["planner_coder_tester_reviewer"]()
nodes = [n.id for n in g.nodes()]

# A resolving set vs a single (non-resolving) verifier.
resolving = metric_dimension_verifier_placement(g, 5)
single = [resolving[0]]

# Distances are static for g, so compute the shortest-path map once.
dist = dict(nx.all_pairs_shortest_path_length(g.to_networkx().to_undirected()))

for label, vset in [("RESOLVING set", resolving), ("SINGLE verifier (non-resolving)", single)]:
    sigs = {n: signature(dist, n, vset) for n in nodes}
    seen = defaultdict(list)
    for n, s in sigs.items():
        seen[s].append(n)
    collisions = {s: ns for s, ns in seen.items() if len(ns) > 1}
    cov_val = resolving_coverage(g, vset)
    print(f"\n  {label}: verifiers={vset}  resolving_coverage={cov_val:.3f}")
    print(f"    distinct signatures: {len(seen)} / {len(nodes)} nodes")
    if collisions:
        for s, ns in collisions.items():
            print(f"    COLLISION (indistinguishable failures): {ns}  signature={s}")
    else:
        print("    Every node has a unique signature -> any single failure is uniquely localizable")
