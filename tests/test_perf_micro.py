"""Perf microbenchmark smoke tests."""

from __future__ import annotations

from benchmarks.perf_micro import assert_budgets, run_microbenchmarks


def test_microbenchmarks_complete_under_budget() -> None:
    results = run_microbenchmarks()
    assert len(results) >= 10
    assert_budgets(results)
