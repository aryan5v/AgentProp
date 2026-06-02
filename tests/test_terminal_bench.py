import json
import sys
from pathlib import Path

from experiments.run_with_watchdog import main as watchdog_main

from agentprop.cli import main
from agentprop.evaluation.failure_taxonomy import classify_benchmark_failure
from agentprop.evaluation.terminal_bench import (
    TerminalBenchLaunchConfig,
    collect_harbor_trial_results,
    summarize_terminal_bench_results,
    write_terminal_bench_launch_bundle,
    write_terminal_bench_summary_report,
)
from agentprop.evaluation.watchdog import run_command_with_watchdog


def test_terminal_bench_prepare_writes_dry_run_bundle(tmp_path: Path) -> None:
    paths = write_terminal_bench_launch_bundle(
        tmp_path,
        TerminalBenchLaunchConfig(task_count=3, output_root=str(tmp_path / "run")),
        registry_root=tmp_path / "registry",
    )

    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["dataset"] == "terminal-bench/terminal-bench-2-1"
    assert manifest["agent"] == "terminus-2"
    assert manifest["harbor_command"][:4] == ["harbor", "run", "-d", manifest["dataset"]]
    assert "-n" in manifest["harbor_command"]
    assert paths["run_script"].read_text().startswith("#!/usr/bin/env bash")
    instructions = paths["extra_instructions"].read_text()
    assert "enumerate the full answer set" in instructions
    assert "Limit candidate sweeps to a small fixed budget" in instructions
    assert "Budget-Aware Stop Conditions" in instructions
    assert "cancellation, cleanup, and max-concurrency invariants" in instructions
    assert "test release mode before" in instructions
    assert "primer Tm gaps" in instructions
    assert manifest["budget_policies"][0]["category"] == "direct-answer"
    assert (tmp_path / "registry" / "registry.json").exists()


def test_terminal_bench_summary_reads_harbor_task_results(tmp_path: Path) -> None:
    _write_result(tmp_path / "task-a" / "result.json", "terminal-bench/task-a", 1.0, 100, 20, 0.03)
    _write_result(
        tmp_path / "task-b" / "result.json",
        "terminal-bench/task-b",
        0.0,
        200,
        30,
        0.05,
        exception_name="AgentTimeoutError",
        elapsed_time_s=1800.0,
        command_count=12,
        model_call_count=5,
    )
    (tmp_path / "result.json").write_text(json.dumps({"aggregate": True}))

    rows = collect_harbor_trial_results(tmp_path)
    summary = summarize_terminal_bench_results(rows)

    assert len(rows) == 2
    assert summary.pass_count == 1
    assert summary.fail_count == 1
    assert summary.timeout_rate == 0.5
    assert summary.input_tokens == 300
    assert summary.output_tokens == 50
    assert summary.elapsed_time_s == 1800.0
    assert summary.command_count == 12
    assert summary.model_call_count == 5
    assert summary.retry_recommended_count == 1
    assert summary.failure_counts == {"passed": 1, "timeout_or_overexploration": 1}


def test_terminal_bench_summary_skips_partial_json(tmp_path: Path) -> None:
    _write_result(tmp_path / "task-a" / "result.json", "terminal-bench/task-a", 1.0)
    partial = tmp_path / "task-b" / "result.json"
    partial.parent.mkdir(parents=True)
    partial.write_text("{", encoding="utf-8")

    rows = collect_harbor_trial_results(tmp_path)

    assert len(rows) == 1
    assert rows[0].task_name == "terminal-bench/task-a"


def test_terminal_bench_summary_report_writes_artifacts(tmp_path: Path) -> None:
    _write_result(tmp_path / "run" / "task-a" / "result.json", "terminal-bench/task-a", 1.0)

    paths = write_terminal_bench_summary_report(tmp_path / "run", tmp_path / "report")

    assert json.loads(paths["summary"].read_text())["summary"]["pass_count"] == 1
    assert "Commands" in paths["report"].read_text()
    assert "Failure Taxonomy" in paths["report"].read_text()
    assert "Task Results" in paths["report"].read_text()
    assert "task_name" in paths["csv"].read_text()
    assert "failure_category" in paths["csv"].read_text()


def test_terminal_bench_classifies_known_failure_signatures(tmp_path: Path) -> None:
    _write_result(
        tmp_path / "break-filter-js-from-html" / "result.json",
        "terminal-bench/break-filter-js-from-html",
        0.0,
        exception_name="SessionNotCreatedException",
    )
    _write_result(
        tmp_path / "chess-best-move" / "result.json",
        "terminal-bench/chess-best-move",
        0.0,
    )
    _write_result(tmp_path / "dna-insert" / "result.json", "terminal-bench/dna-insert", 0.0)
    _write_result(
        tmp_path / "cancel-async-tasks" / "result.json",
        "terminal-bench/cancel-async-tasks",
        0.0,
    )
    _write_result(
        tmp_path / "custom-memory-heap-crash" / "result.json",
        "terminal-bench/custom-memory-heap-crash",
        0.0,
    )

    rows = {row.task_name: row for row in collect_harbor_trial_results(tmp_path)}

    assert rows["terminal-bench/break-filter-js-from-html"].failure_category == (
        "harness_infra_failure"
    )
    assert rows["terminal-bench/break-filter-js-from-html"].retry_recommended is True
    assert rows["terminal-bench/chess-best-move"].failure_category == "incomplete_output"
    assert rows["terminal-bench/dna-insert"].failure_category == "domain_constraint_miss"
    assert rows["terminal-bench/cancel-async-tasks"].failure_category == "async_lifecycle_miss"
    assert rows["terminal-bench/custom-memory-heap-crash"].failure_category == (
        "build_or_mode_mismatch"
    )


def test_failure_taxonomy_marks_unknown_exceptions_retryable() -> None:
    classification = classify_benchmark_failure(
        "terminal-bench/crack-7z-hash",
        passed=None,
        exception_name="UnexpectedTmuxError",
    )

    assert classification.category == "harness_infra_failure"
    assert classification.retry_recommended is True


def test_cli_terminal_bench_prepare_emits_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(
        [
            "terminal-bench",
            "prepare",
            "--out-dir",
            str(tmp_path),
            "--task-count",
            "2",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["manifest"]).exists()
    assert Path(payload["runbook"]).exists()


def test_cli_terminal_bench_requires_subcommand() -> None:
    try:
        main(["terminal-bench"])
    except SystemExit as exc:
        assert "requires a subcommand" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_watchdog_writes_status_and_log(tmp_path: Path) -> None:
    result = run_command_with_watchdog(
        [sys.executable, "-c", "print('ok')"],
        log_path=tmp_path / "run.log",
        status_path=tmp_path / "status.json",
        timeout_s=5,
    )

    assert result.status == "completed"
    assert result.exit_code == 0
    assert "ok" in (tmp_path / "run.log").read_text()
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "completed"


def test_watchdog_script_propagates_failed_exit_code(tmp_path: Path) -> None:
    exit_code = watchdog_main(
        [
            "--timeout",
            "5",
            "--log",
            str(tmp_path / "failed.log"),
            "--status-json",
            str(tmp_path / "failed-status.json"),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ]
    )

    assert exit_code == 7


def test_watchdog_handles_partial_line_idle_timeout(tmp_path: Path) -> None:
    result = run_command_with_watchdog(
        [
            sys.executable,
            "-c",
            "import sys, time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(5)",
        ],
        log_path=tmp_path / "partial.log",
        status_path=tmp_path / "partial-status.json",
        timeout_s=5,
        idle_timeout_s=0.2,
        poll_interval_s=0.05,
        terminate_grace_s=1,
    )

    assert result.status == "idle-timeout"
    assert result.idle_timed_out is True


def _write_result(
    path: Path,
    task_name: str,
    reward: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    *,
    exception_name: str | None = None,
    elapsed_time_s: float = 0.0,
    command_count: int = 0,
    model_call_count: int = 0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task_name": task_name,
                "trial_name": task_name.rsplit("/", 1)[-1],
                "agent_result": {
                    "n_input_tokens": input_tokens,
                    "n_cache_tokens": input_tokens // 2,
                    "n_output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                    "duration_s": elapsed_time_s,
                    "n_commands": command_count,
                    "n_model_calls": model_call_count,
                },
                "verifier_result": {"rewards": {"reward": reward}},
                "exception_info": (
                    {"type": exception_name} if exception_name is not None else None
                ),
            }
        )
    )
