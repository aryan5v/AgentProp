"""Submodular probabilistic-coverage verifier placement with a greedy guarantee.

Exact resolving coverage is monotone but **not** submodular, so the greedy in
`metric_dimension_verifier_placement` carries no approximation bound. This
module optimizes the probabilistic pairwise-coverage surrogate

    F(A) = sum over node pairs (u, v) of P(at least one a in A separates u, v)
         = sum_{u<v} [ 1 - prod_{a in A} (1 - s_a(u, v)) ]

where ``s_a(u, v)`` is the probability that verifier ``a`` distinguishes the
pair. F is a nonnegative weighted coverage function, hence monotone
submodular, so lazy greedy achieves the classical (1 - 1/e) guarantee
(Nemhauser, Wolsey, Fisher 1978).

Two separation models:

- ``deterministic`` — s_a(u, v) = 1 if d(u, a) != d(v, a) else 0. F then
  counts resolved pairs exactly, and greedy-on-F equals greedy resolving
  coverage *with* a guarantee attached.
- ``noisy`` — s_a(u, v) in [0, 1] derived from how distinguishable the
  verifier's firing probabilities are for faults at u vs v (total-variation
  distance of the per-verifier Bernoulli observation), matching the
  fault-injection observation model.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Literal

from agentprop.core import AgentGraph
from agentprop.evaluation.fault_injection import detection_probability

SeparationModel = Literal["deterministic", "noisy"]


@dataclass(frozen=True, slots=True)
class SubmodularPlacement:
    """Greedy solution with its objective trajectory."""

    verifiers: tuple[str, ...]
    objective: float
    objective_fraction: float
    """F(A) divided by the number of pairs (1.0 = every pair fully covered)."""
    marginal_gains: tuple[float, ...]
    model: SeparationModel

    def to_dict(self) -> dict[str, object]:
        return {
            "verifiers": list(self.verifiers),
            "objective": self.objective,
            "objective_fraction": self.objective_fraction,
            "marginal_gains": list(self.marginal_gains),
            "model": self.model,
        }


def pair_separation_scores(
    graph: AgentGraph,
    *,
    model: SeparationModel = "deterministic",
    noise: float = 0.05,
    decay: float = 0.6,
) -> dict[str, dict[tuple[str, str], float]]:
    """Per-candidate-verifier separation score for every node pair."""

    node_ids, distances = graph.get_undirected_distances()
    pairs = [
        (node_ids[i], node_ids[j])
        for i in range(len(node_ids))
        for j in range(i + 1, len(node_ids))
    ]
    scores: dict[str, dict[tuple[str, str], float]] = {}
    for candidate in node_ids:
        row: dict[tuple[str, str], float] = {}
        for u, v in pairs:
            du = distances[u].get(candidate)
            dv = distances[v].get(candidate)
            if model == "deterministic":
                row[(u, v)] = 1.0 if du != dv else 0.0
            else:
                pu = detection_probability(du, noise=noise, decay=decay)
                pv = detection_probability(dv, noise=noise, decay=decay)
                row[(u, v)] = abs(pu - pv)  # TV distance between Bernoullis
        scores[candidate] = row
    return scores


def submodular_verifier_placement(
    graph: AgentGraph,
    k: int,
    *,
    model: SeparationModel = "deterministic",
    noise: float = 0.05,
    decay: float = 0.6,
) -> SubmodularPlacement:
    """Lazy-greedy (CELF) maximization of the pairwise-coverage surrogate.

    Monotone submodularity of F makes stale priority-queue entries safe: a
    candidate's recomputed gain can only shrink, so the first fresh entry at
    the top of the heap is the true argmax.
    """

    if k <= 0 or graph.node_count == 0:
        return SubmodularPlacement((), 0.0, 0.0, (), model)
    scores = pair_separation_scores(graph, model=model, noise=noise, decay=decay)
    node_ids = list(scores)
    pairs = next(iter(scores.values())).keys() if scores else []
    # uncovered probability per pair given the current selection
    residual: dict[tuple[str, str], float] = dict.fromkeys(pairs, 1.0)

    def gain(candidate: str) -> float:
        row = scores[candidate]
        return sum(residual[pair] * row[pair] for pair in residual)

    # CELF: max-heap of (-gain, selection_count_when_computed, candidate).
    # An entry is fresh iff it was computed at the current selection size.
    heap: list[tuple[float, int, str]] = [(-gain(c), 0, c) for c in node_ids]
    heapq.heapify(heap)
    chosen: list[str] = []
    gains: list[float] = []
    while heap and len(chosen) < min(k, len(node_ids)):
        negative_gain, computed_at, candidate = heapq.heappop(heap)
        if candidate in chosen:
            continue
        if computed_at < len(chosen):
            heapq.heappush(heap, (-gain(candidate), len(chosen), candidate))
            continue
        chosen.append(candidate)
        gains.append(-negative_gain)
        row = scores[candidate]
        for pair in residual:
            residual[pair] *= 1.0 - row[pair]
    total_pairs = max(len(residual), 1)
    objective = float(total_pairs - sum(residual.values()))
    return SubmodularPlacement(
        verifiers=tuple(chosen),
        objective=objective,
        objective_fraction=objective / total_pairs,
        marginal_gains=tuple(gains),
        model=model,
    )


def coverage_objective(
    graph: AgentGraph,
    verifiers: list[str],
    *,
    model: SeparationModel = "deterministic",
    noise: float = 0.05,
    decay: float = 0.6,
) -> float:
    """Evaluate F(A) for an arbitrary verifier set (for baseline comparisons)."""

    scores = pair_separation_scores(graph, model=model, noise=noise, decay=decay)
    if not scores:
        return 0.0
    pairs = next(iter(scores.values())).keys()
    objective = 0.0
    for pair in pairs:
        uncovered = 1.0
        for verifier in verifiers:
            uncovered *= 1.0 - scores[verifier][pair]
        objective += 1.0 - uncovered
    return objective
