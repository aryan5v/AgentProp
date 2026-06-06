"""Randomized zero forcing on directed weighted graphs."""

from __future__ import annotations

import random

from agentprop.core import AgentGraph
from agentprop.propagation.base import PropagationResult
from agentprop.propagation.fast_kernel import (
    aggregate_trial_results,
    get_propagation_index,
    rzf_simulate_once_indexed,
    simulate_batch_rzf,
)


class RandomizedZeroForcing:
    """Activate a node with probability equal to active incoming weight share."""

    name = "randomized-zero-forcing"

    def __init__(self, *, seed: int | None = None, max_rounds: int | None = None) -> None:
        self._random = random.Random(seed)
        self.max_rounds = max_rounds

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 100,
    ) -> PropagationResult:
        """Run Monte Carlo randomized zero forcing."""

        if trials < 1:
            raise ValueError("trials must be at least 1")

        index = get_propagation_index(graph)
        trial_rounds: list[dict[str, int]] = []
        trial_coverages: list[list[float]] = []
        max_rounds = self.max_rounds or max(1, graph.node_count * 4)
        for _ in range(trials):
            activation_rounds, coverage_by_round = rzf_simulate_once_indexed(
                index,
                seeds,
                self._random,
                max_rounds=max_rounds,
            )
            trial_rounds.append(activation_rounds)
            trial_coverages.append(coverage_by_round)
        return aggregate_trial_results(
            graph,
            trial_rounds,
            trial_coverages,
            trials=trials,
        )

    def simulate_batch(
        self,
        graph: AgentGraph,
        seed_sets: list[list[str]],
        *,
        trials: int = 100,
    ) -> list[PropagationResult]:
        """Run RZF for multiple seed sets sharing one graph index."""

        return simulate_batch_rzf(
            graph,
            seed_sets,
            trials=trials,
            max_rounds=self.max_rounds,
        )
