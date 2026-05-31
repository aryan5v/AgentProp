"""Benchmark runner for comparing routing strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from agentprop.algorithms import (
    betweenness_seed_selection,
    celf_seed_selection,
    closeness_seed_selection,
    cost_aware_greedy_seed_selection,
    degree_seed_selection,
    greedy_seed_selection,
    k_core_seed_selection,
    pagerank_seed_selection,
    random_seed_selection,
)
from agentprop.core import AgentGraph
from agentprop.evaluation.metrics import compare_routing
from agentprop.propagation import (
    BootstrapPercolation,
    IndependentCascade,
    LearnedPropagation,
    LinearThreshold,
    PropagationModel,
    RandomizedZeroForcing,
    ZeroForcing,
)


@dataclass(slots=True)
class BenchmarkRow:
    """One benchmark result row."""

    workflow: str
    algorithm: str
    propagation_model: str
    budget: int
    seeds: list[str]
    coverage: float
    expected_propagation_time: float
    full_activation_probability: float
    broadcast_cost: float
    optimized_cost: float
    estimated_savings: float

    def to_dict(self) -> dict[str, object]:
        """Serialize row for JSON output."""

        return asdict(self)


def run_benchmark(
    graph: AgentGraph,
    *,
    workflow_name: str,
    algorithms: list[str],
    models: list[str],
    budget: int,
    trials: int = 100,
) -> list[BenchmarkRow]:
    """Run seed-selection algorithms across propagation models."""

    rows: list[BenchmarkRow] = []
    for model_name in models:
        model = make_propagation_model(model_name)
        for algorithm in algorithms:
            seeds = select_seeds(graph, algorithm, budget, model, trials)
            propagation = model.simulate(graph, seeds, trials=trials)
            report = compare_routing(graph, seeds, model.name, propagation)
            rows.append(
                BenchmarkRow(
                    workflow=workflow_name,
                    algorithm=algorithm,
                    propagation_model=model.name,
                    budget=budget,
                    seeds=seeds,
                    coverage=propagation.coverage,
                    expected_propagation_time=propagation.expected_propagation_time
                    or float(propagation.propagation_time),
                    full_activation_probability=propagation.full_activation_probability or 0.0,
                    broadcast_cost=report.broadcast_cost.total_cost,
                    optimized_cost=report.optimized_cost.total_cost,
                    estimated_savings=report.estimated_savings,
                )
            )
    return rows


def make_propagation_model(name: str) -> PropagationModel:
    """Create a propagation model by CLI/API name."""

    if name in {"independent-cascade", "ic"}:
        return IndependentCascade(seed=0)
    if name in {"linear-threshold", "lt"}:
        return LinearThreshold()
    if name in {"bootstrap", "bootstrap-percolation"}:
        return BootstrapPercolation()
    if name in {"rzf", "randomized-zero-forcing"}:
        return RandomizedZeroForcing(seed=0)
    if name in {"zero-forcing", "zf"}:
        return ZeroForcing()
    if name in {"learned", "trace-learned"}:
        return LearnedPropagation(seed=0)
    raise ValueError(f"Unknown propagation model: {name}")


def select_seeds(
    graph: AgentGraph,
    algorithm: str,
    budget: int,
    model: PropagationModel,
    trials: int,
) -> list[str]:
    """Select seeds by algorithm name."""

    if algorithm == "random":
        return random_seed_selection(graph, budget, seed=0)
    if algorithm == "degree":
        return degree_seed_selection(graph, budget)
    if algorithm == "in-degree":
        return degree_seed_selection(graph, budget, direction="in")
    if algorithm == "out-degree":
        return degree_seed_selection(graph, budget, direction="out")
    if algorithm == "pagerank":
        return pagerank_seed_selection(graph, budget)
    if algorithm == "betweenness":
        return betweenness_seed_selection(graph, budget)
    if algorithm == "closeness":
        return closeness_seed_selection(graph, budget)
    if algorithm == "k-core":
        return k_core_seed_selection(graph, budget)
    if algorithm == "greedy":
        return greedy_seed_selection(graph, budget, propagation_model=model, trials=trials)
    if algorithm == "celf":
        return celf_seed_selection(graph, budget, propagation_model=model, trials=trials)
    if algorithm == "cost-aware-greedy":
        return cost_aware_greedy_seed_selection(
            graph,
            budget,
            propagation_model=model,
            trials=trials,
        )
    raise ValueError(f"Unknown seed algorithm: {algorithm}")
