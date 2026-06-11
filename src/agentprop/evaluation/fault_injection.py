"""Fault-injection validation of placement observation models (GO/NO-GO gate).

Compares two verifier-placement strategies on simulated single-fault
localization under noisy observations:

(a) **classical** — the existing undirected metric-dimension placement
    (`algorithms.verifier_placement.metric_dimension_verifier_placement`),
(b) **directed/noisy** — greedy placement maximizing the minimum pairwise
    Jensen-Shannon divergence between the observation distributions that
    different faults induce, computed on *directed* distances.

A fault at node f makes each verifier a fire with probability that decays in
the directed distance d(f -> a); observation noise flips readings. The
localizer is maximum a-posteriori over fault hypotheses. If (b) materially
beats (a) under realistic noise, the directed/noisy theory is worth the
investment (GO); otherwise the classical story stands (NO-GO).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean

from agentprop.algorithms.verifier_placement import metric_dimension_verifier_placement
from agentprop.core import AgentGraph

# ---------------------------------------------------------------------------
# Graph families


def layered_dag(layers: int, width: int, *, seed: int = 0) -> AgentGraph:
    """Layered workflow DAG: every node feeds 1-3 nodes in the next layer."""

    rng = random.Random(seed)
    graph = AgentGraph()
    grid = [[f"l{i}n{j}" for j in range(width)] for i in range(layers)]
    for row in grid:
        for node_id in row:
            graph.add_node(node_id)
    for i in range(layers - 1):
        for source in grid[i]:
            targets = rng.sample(grid[i + 1], k=min(width, rng.randint(1, 3)))
            for target in targets:
                graph.add_edge(source, target)
        # Guarantee every next-layer node is reachable.
        for target in grid[i + 1]:
            if not any(e.target == target for e in graph.edges()):
                graph.add_edge(rng.choice(grid[i]), target)
    return graph


def chain(n: int) -> AgentGraph:
    graph = AgentGraph()
    for i in range(n):
        graph.add_node(f"n{i}")
    for i in range(n - 1):
        graph.add_edge(f"n{i}", f"n{i + 1}")
    return graph


def spider(legs: int, leg_length: int) -> AgentGraph:
    graph = AgentGraph()
    graph.add_node("hub")
    for leg in range(legs):
        previous = "hub"
        for i in range(leg_length):
            node_id = f"leg{leg}_{i}"
            graph.add_node(node_id)
            graph.add_edge(previous, node_id)
            previous = node_id
    return graph


def random_arborescence(n: int, *, seed: int = 0) -> AgentGraph:
    rng = random.Random(seed)
    graph = AgentGraph()
    graph.add_node("n0")
    for i in range(1, n):
        graph.add_node(f"n{i}")
        graph.add_edge(f"n{rng.randint(0, i - 1)}", f"n{i}")
    return graph


GRAPH_FAMILIES: dict[str, Callable[[int], AgentGraph]] = {
    "layered": lambda seed: layered_dag(4, 4, seed=seed),
    "chain": lambda seed: chain(12),
    "spider": lambda seed: spider(4, 3),
    "arborescence": lambda seed: random_arborescence(14, seed=seed),
}

# ---------------------------------------------------------------------------
# Observation model


def directed_distances(graph: AgentGraph) -> dict[str, dict[str, int]]:
    """BFS distances along edge direction (fault -> downstream verifier)."""

    nx_graph = graph.to_networkx()
    distances: dict[str, dict[str, int]] = {}
    for source in nx_graph.nodes:
        level = {str(source): 0}
        frontier = [str(source)]
        depth = 0
        while frontier:
            depth += 1
            next_frontier: list[str] = []
            for node in frontier:
                for successor in nx_graph.successors(node):
                    name = str(successor)
                    if name not in level:
                        level[name] = depth
                        next_frontier.append(name)
            frontier = next_frontier
        distances[str(source)] = level
    return distances


def detection_probability(
    distance: int | None,
    *,
    noise: float,
    decay: float = 0.6,
) -> float:
    """P(verifier fires | fault at directed distance d). Unreachable -> noise."""

    if distance is None:
        return noise
    signal = decay**distance
    return min(1.0 - 1e-6, max(1e-6, noise + (1.0 - 2.0 * noise) * signal))


def fault_observation_table(
    graph: AgentGraph,
    verifiers: list[str],
    *,
    noise: float,
    decay: float = 0.6,
) -> dict[str, list[float]]:
    """Per-fault vector of firing probabilities, one entry per verifier."""

    distances = directed_distances(graph)
    table: dict[str, list[float]] = {}
    for node in graph.nodes():
        row = [
            detection_probability(distances[node.id].get(v), noise=noise, decay=decay)
            for v in verifiers
        ]
        table[node.id] = row
    return table

# ---------------------------------------------------------------------------
# Placements


def classical_placement(graph: AgentGraph, k: int) -> list[str]:
    """The existing undirected metric-dimension placement."""

    placement = metric_dimension_verifier_placement(graph, k)
    if len(placement) < k:
        remaining = [n.id for n in graph.nodes() if n.id not in placement]
        placement = list(placement) + remaining[: k - len(placement)]
    return list(placement)


def directed_js_placement(
    graph: AgentGraph,
    k: int,
    *,
    noise: float,
    decay: float = 0.6,
) -> list[str]:
    """Greedy placement maximizing minimum pairwise JS separation of faults."""

    node_ids = [n.id for n in graph.nodes()]
    distances = directed_distances(graph)
    pairs = [
        (node_ids[i], node_ids[j])
        for i in range(len(node_ids))
        for j in range(i + 1, len(node_ids))
    ]

    def probability_row(verifier: str) -> dict[str, float]:
        return {
            f: detection_probability(distances[f].get(verifier), noise=noise, decay=decay)
            for f in node_ids
        }

    # Saturating coverage objective: each pair contributes its accumulated JS
    # separation up to a cap, so the greedy spreads separation across many
    # pairs instead of piling divergence onto already-distinguished ones.
    # (Truncated sums of nonnegative gains keep the objective submodular.)
    cap = 0.5
    pair_separation = dict.fromkeys(pairs, 0.0)
    chosen: list[str] = []
    for _ in range(min(k, len(node_ids))):
        best_node, best_gain, best_row = None, -1.0, None
        for candidate in node_ids:
            if candidate in chosen:
                continue
            row = probability_row(candidate)
            gain = sum(
                min(cap, pair_separation[(u, v)] + _bernoulli_js(row[u], row[v]))
                - pair_separation[(u, v)]
                for u, v in pairs
            )
            if gain > best_gain:
                best_node, best_gain, best_row = candidate, gain, row
        if best_node is None or best_row is None:
            break
        chosen.append(best_node)
        for u, v in pairs:
            pair_separation[(u, v)] = min(
                cap, pair_separation[(u, v)] + _bernoulli_js(best_row[u], best_row[v])
            )
    return chosen

# ---------------------------------------------------------------------------
# Simulation


@dataclass(frozen=True, slots=True)
class GateCondition:
    """One experimental cell."""

    family: str
    noise: float
    budget: int
    classical_accuracy: float
    directed_accuracy: float
    trials: int

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "noise": self.noise,
            "budget": self.budget,
            "classical_accuracy": self.classical_accuracy,
            "directed_accuracy": self.directed_accuracy,
            "trials": self.trials,
        }


@dataclass(frozen=True, slots=True)
class GateReport:
    """GO/NO-GO summary across all conditions."""

    conditions: tuple[GateCondition, ...]
    mean_classical: float
    mean_directed: float
    mean_advantage: float
    go: bool
    go_threshold: float

    def to_dict(self) -> dict[str, object]:
        return {
            "conditions": [c.to_dict() for c in self.conditions],
            "mean_classical": self.mean_classical,
            "mean_directed": self.mean_directed,
            "mean_advantage": self.mean_advantage,
            "go": self.go,
            "go_threshold": self.go_threshold,
        }


def localization_accuracy(
    graph: AgentGraph,
    verifiers: list[str],
    *,
    noise: float,
    trials: int,
    seed: int,
    decay: float = 0.6,
) -> float:
    """MAP localization accuracy under the noisy observation model."""

    rng = random.Random(seed)
    table = fault_observation_table(graph, verifiers, noise=noise, decay=decay)
    node_ids = list(table)
    hits = 0
    for _ in range(trials):
        fault = rng.choice(node_ids)
        observed = [1 if rng.random() < p else 0 for p in table[fault]]
        best_node, best_loglik = None, -math.inf
        for candidate in node_ids:
            loglik = sum(
                math.log(p if bit else 1.0 - p)
                for p, bit in zip(table[candidate], observed, strict=True)
            )
            if loglik > best_loglik:
                best_node, best_loglik = candidate, loglik
        if best_node == fault:
            hits += 1
    return hits / trials


def run_gate(
    *,
    noises: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2),
    budgets: tuple[int, ...] = (2, 3, 4),
    trials: int = 400,
    seeds: int = 5,
    go_threshold: float = 0.05,
    decay: float = 0.6,
) -> GateReport:
    """Run the full GO/NO-GO sweep and return the decision report."""

    conditions: list[GateCondition] = []
    for family, build in GRAPH_FAMILIES.items():
        for noise in noises:
            for budget in budgets:
                classical_scores: list[float] = []
                directed_scores: list[float] = []
                for seed in range(seeds):
                    graph = build(seed)
                    classical = classical_placement(graph, budget)
                    directed = directed_js_placement(
                        graph, budget, noise=noise, decay=decay
                    )
                    classical_scores.append(
                        localization_accuracy(
                            graph, classical, noise=noise, trials=trials,
                            seed=1000 + seed, decay=decay,
                        )
                    )
                    directed_scores.append(
                        localization_accuracy(
                            graph, directed, noise=noise, trials=trials,
                            seed=1000 + seed, decay=decay,
                        )
                    )
                conditions.append(
                    GateCondition(
                        family=family,
                        noise=noise,
                        budget=budget,
                        classical_accuracy=fmean(classical_scores),
                        directed_accuracy=fmean(directed_scores),
                        trials=trials * seeds,
                    )
                )
    mean_classical = fmean(c.classical_accuracy for c in conditions)
    mean_directed = fmean(c.directed_accuracy for c in conditions)
    advantage = mean_directed - mean_classical
    return GateReport(
        conditions=tuple(conditions),
        mean_classical=mean_classical,
        mean_directed=mean_directed,
        mean_advantage=advantage,
        go=advantage >= go_threshold,
        go_threshold=go_threshold,
    )


def _bernoulli_js(p: float, q: float) -> float:
    m = (p + q) / 2.0
    return (_bernoulli_kl(p, m) + _bernoulli_kl(q, m)) / 2.0


def _bernoulli_kl(p: float, q: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, p))
    q = min(1.0 - 1e-9, max(1e-9, q))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
