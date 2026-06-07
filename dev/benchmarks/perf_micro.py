"""Library microbenchmarks for AgentProp hot paths (CI trend gate)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agentprop.algorithms import (
    greedy_seed_selection,
    metric_dimension_verifier_placement,
    risk_aware_verifier_placement,
)
from agentprop.propagation import IndependentCascade, RandomizedZeroForcing
from agentprop.runtime.control_loop import ExecutionEvent, ExecutionStateTracker
from agentprop.workflows import chain_workflow


@dataclass(slots=True)
class BenchResult:
    name: str
    seconds: float
    node_count: int

    def to_dict(self) -> dict[str, float | str | int]:
        return {"name": self.name, "seconds": self.seconds, "node_count": self.node_count}


def _timed(name: str, node_count: int, fn) -> BenchResult:
    start = time.perf_counter()
    fn()
    return BenchResult(name=name, seconds=time.perf_counter() - start, node_count=node_count)


def run_microbenchmarks() -> list[BenchResult]:
    """Run a small perf suite used by CI to catch regressions."""

    results: list[BenchResult] = []

    for size in (10, 30, 60):
        graph = chain_workflow(length=size)

        results.append(
            _timed(
                f"verifier_placement_n{size}",
                size,
                lambda g=graph: metric_dimension_verifier_placement(g, min(3, g.node_count)),
            )
        )
        results.append(
            _timed(
                f"risk_verifier_placement_n{size}",
                size,
                lambda g=graph: risk_aware_verifier_placement(g, min(3, g.node_count)),
            )
        )
        results.append(
            _timed(
                f"greedy_seeds_n{size}",
                size,
                lambda g=graph: greedy_seed_selection(
                    g,
                    min(3, g.node_count),
                    propagation_model=IndependentCascade(seed=0),
                    trials=20,
                ),
            )
        )
        results.append(
            _timed(
                f"rzf_trials_n{size}",
                size,
                lambda g=graph, s=size: RandomizedZeroForcing(seed=0).simulate(
                    g,
                    [f"node_{s - 2}"],
                    trials=50,
                ),
            )
        )

    events = [
        ExecutionEvent(
            step=index,
            command=f"cmd-{index}",
            exit_code=0 if index % 5 else 1,
            verifier_run=index % 7 == 0,
            verifier_passed=True if index % 7 == 0 else None,
            progress_made=index % 3 != 0,
            tokens_used=120 + index,
            elapsed_s=0.5 * index,
            error_signature=None if index % 5 else "E_FAIL",
        )
        for index in range(200)
    ]

    def _tracker_loop() -> None:
        local = ExecutionStateTracker()
        for event in events:
            local.observe(event)
            local.features()

    results.append(_timed("tracker_long_trace", 200, _tracker_loop))
    return results


def assert_budgets(results: list[BenchResult]) -> None:
    """Fail when hot paths exceed generous interactive budgets."""

    budgets = {
        "verifier_placement_n60": 5.0,
        "risk_verifier_placement_n60": 5.0,
        "greedy_seeds_n60": 8.0,
        "rzf_trials_n60": 3.0,
        "tracker_long_trace": 0.25,
    }
    for result in results:
        limit = budgets.get(result.name)
        if limit is not None and result.seconds > limit:
            raise AssertionError(
                f"{result.name} took {result.seconds:.3f}s (budget {limit:.3f}s)"
            )


if __name__ == "__main__":
    bench_results = run_microbenchmarks()
    for row in bench_results:
        print(f"{row.name}: {row.seconds:.4f}s (n={row.node_count})")
    assert_budgets(bench_results)
    print("perf microbenchmarks: OK")
