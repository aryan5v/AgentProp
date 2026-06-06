"""Independent Cascade propagation model."""

from __future__ import annotations

import random

from agentprop.core import AgentGraph
from agentprop.propagation.base import PropagationResult
from agentprop.propagation.fast_kernel import (
    aggregate_trial_results,
    get_propagation_index,
    ic_simulate_once_indexed,
    simulate_batch_ic,
)


class IndependentCascade:
    """Probabilistic diffusion where active nodes get one chance to activate neighbors."""

    name = "independent-cascade"

    def __init__(self, *, seed: int | None = None) -> None:
        self._seed = seed
        self._random = random.Random(seed)

    def simulate(
        self,
        graph: AgentGraph,
        seeds: list[str],
        *,
        trials: int = 100,
    ) -> PropagationResult:
        """Run Monte Carlo Independent Cascade simulation."""

        if trials < 1:
            raise ValueError("trials must be at least 1")

        index = get_propagation_index(graph)
        trial_rounds: list[dict[str, int]] = []
        trial_coverages: list[list[float]] = []
        for _ in range(trials):
            activation_rounds, coverage_by_round = ic_simulate_once_indexed(
                index,
                seeds,
                self._random,
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
        """Run IC for multiple seed sets sharing one graph index."""

        return simulate_batch_ic(
            graph,
            seed_sets,
            trials=trials,
            seed=self._seed,
        )
