# Verifier Semantics

AgentProp treats verifier placement as a graph-design decision, not only a node
ranking problem.

## What A Verifier Can Observe

A verifier may observe:

- upstream nodes that influenced its input
- downstream nodes that receive its correction
- selected paths around the verifier
- full trace context in a broadcast-style workflow

The current v1-alpha metrics assume graph-local observability: ancestors,
descendants, and paths adjacent to the verifier.

## What A Verifier Can Correct

A verifier may correct:

- a node output
- a message between nodes
- a routing decision
- the final answer

The current implementation scores verifier candidates by risk, centrality,
observability coverage, and likely error propagation.

## Placement Methods

- `risk_aware_verifier_placement`: risk plus structural centrality.
- `betweenness_verifier_placement`: communication bridge points.
- `pagerank_verifier_placement`: context aggregation points.
- `error_propagation_verifier_placement`: nodes whose local errors can spread.
- `greedy_correction_coverage_placement`: maximum observable ancestor/descendant
  coverage.

These are training-free baselines for future learned verifier-placement models.
