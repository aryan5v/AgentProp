"""Tests for the scale/quality evidence harness."""

from __future__ import annotations

from agentprop.evaluation.evidence_harness import (
    EvidenceHarnessConfig,
    run_evidence_harness,
    write_evidence_artifacts,
)


def test_evidence_harness_smoke_matrix() -> None:
    rows, summaries = run_evidence_harness(
        EvidenceHarnessConfig(
            workflows=("planner_coder_tester_reviewer", "dynamic_conditional"),
            arms=("broadcast", "degree", "imm"),
            tasks_per_arm=2,
            repeats=2,
            trials=10,
        )
    )
    assert len(rows) == 2 * 3 * 2 * 2
    assert len(summaries) == 2 * 3
    degree_rows = [row for row in rows if row.arm == "degree"]
    assert degree_rows
    assert all(row.trial_seed >= 0 for row in rows)


def test_write_evidence_artifacts(tmp_path) -> None:
    config = EvidenceHarnessConfig(
        workflows=("chain",),
        arms=("greedy",),
        tasks_per_arm=1,
        repeats=1,
        trials=5,
    )
    results_path = write_evidence_artifacts(config, tmp_path)
    assert results_path.exists()
    assert (tmp_path / "outputs.jsonl").exists()
    assert (tmp_path / "REPORT.md").exists()
