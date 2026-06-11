# Verifier Semantics

AgentProp treats verifier placement as a graph-design decision, not only a node
ranking problem.

## What A Verifier Can Observe

A verifier may observe:

- upstream nodes that influenced its input
- downstream nodes that receive its correction
- selected paths around the verifier
- full trace context in a broadcast-style workflow

The current v1 metrics assume graph-local observability: ancestors,
descendants, and paths adjacent to the verifier.

## What A Verifier Can Correct

A verifier may correct:

- a node output
- a message between nodes
- a routing decision
- the final answer

## Placement Methods

### Metric-dimension placement (recommended)

`metric_dimension_verifier_placement(graph, budget, *, fault_tolerant=False)`
selects verifiers so that every pair of workflow nodes has a distinct
distance-vector signature to the verifier set. When `resolving_coverage` reaches
1.0, any single-node failure is uniquely identifiable from the joint verifier
reports alone — no additional logging or side-channel information is needed.

The `fault_tolerant=True` variant ensures this guarantee holds even if one
verifier itself fails. This requires a larger budget on some workflows because
the fault-tolerant metric dimension can exceed the standard metric dimension.

Measure placement quality with two metrics:

- `resolving_coverage(graph, verifiers)`: fraction of node pairs that are
  distinguishable. Returns 1.0 when the set is a full resolving set.
- `fault_tolerant_resolving_coverage(graph, verifiers)`: worst-case resolving
  coverage after removing any single verifier. Returns 1.0 when fault-tolerant
  resolving holds at full coverage, 0.0 when only one verifier is present.

```python
from agentprop.algorithms import (
    metric_dimension_verifier_placement,
    resolving_coverage,
    fault_tolerant_resolving_coverage,
)
from agentprop.workflows import planner_coder_tester_reviewer

graph = planner_coder_tester_reviewer()

# Standard resolving set at budget k=3
verifiers = metric_dimension_verifier_placement(graph, budget=3)
print(f"coverage: {resolving_coverage(graph, verifiers):.3f}")

# Fault-tolerant placement — guarantee holds after any single verifier fails
ft_verifiers = metric_dimension_verifier_placement(graph, budget=5, fault_tolerant=True)
print(f"ft coverage: {fault_tolerant_resolving_coverage(graph, ft_verifiers):.3f}")
```

### Heuristic baselines

- `risk_aware_verifier_placement`: weighted combination of error rate,
  reliability, centrality, and out-degree. Useful as a training-free baseline
  when a full distance computation is too expensive.
- `betweenness_verifier_placement`: selects communication bridge points.
- `pagerank_verifier_placement`: selects context aggregation points.
- `error_propagation_verifier_placement`: nodes whose local errors are most
  likely to propagate downstream.
- `greedy_correction_coverage_placement`: maximizes observable ancestor and
  descendant coverage.

## Verifier Trust and Independent Confirmation

The `ExecutionEvent.trusted` field distinguishes verifier results by source:

- `trusted=True` (default): result comes from an independent, external verifier.
  The runtime controller treats this as confirmed evidence.
- `trusted=False`: result is self-reported by the agent under evaluation (for
  example, a locally-run evaluation script). The controller treats this as
  unconfirmed.

When `StoppingControllerConfig.require_independent_verification=True` (the
default), a self-reported pass triggers a `FORCE_VERIFY` control action rather
than immediately finalizing. Finalization only occurs after an independent
verifier confirms the result.

```python
from agentprop.runtime.control_loop import ExecutionEvent, StoppingControllerConfig

# Mark a self-reported eval result as untrusted
self_eval_event = ExecutionEvent(
    step=4,
    verifier_run=True,
    verifier_passed=True,
    trusted=False,  # agent's own eval.py — not independently confirmed
)

# Independent harness result is trusted by default
harness_event = ExecutionEvent(
    step=5,
    verifier_run=True,
    verifier_passed=True,
    trusted=True,
)
```

Set `require_independent_verification=False` only when your harness does not
support separate verification and you are willing to accept self-reported results.

## Submodular surrogate with a greedy guarantee

Exact resolving coverage is monotone but **not** submodular, so greedy
placement on it carries no approximation bound. The probabilistic
pairwise-coverage surrogate restores one:

F(A) = Σ over node pairs (u,v) of P(at least one verifier in A separates u,v)
     = Σ_{u<v} [ 1 − Π_{a∈A} (1 − s_a(u,v)) ]

with s_a(u,v) ∈ [0,1] the per-verifier separation probability (an indicator
under deterministic distances; a total-variation distance between the
verifier's fault-conditional observation distributions under noise).

*Why F is monotone submodular:* for each pair, 1 − Π_{a∈A}(1 − s_a) is the
standard probabilistic-coverage form — adding a verifier multiplies the
uncovered probability by a factor in [0,1], so per-pair gains are nonnegative
and shrink as A grows. F is a nonnegative weighted sum of such terms, and
monotone submodularity is closed under nonnegative sums. Lazy greedy
(`agentprop.algorithms.submodular_verifier_placement`) therefore achieves the
(1 − 1/e) guarantee of Nemhauser–Wolsey–Fisher (1978). A property-based test
(`tests/test_submodular_placement.py`) checks diminishing returns empirically
on random graphs for both separation models.

Under deterministic distances, F(A) equals the count of resolved pairs, so
F(A)/pairs = resolving coverage: the surrogate's greedy optimizes the same
quantity the exact tracker reports, now with a worst-case bound.
