"""Terminal-Bench launch preparation and Harbor artifact analysis."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentprop.evaluation.artifacts import register_artifact


@dataclass(frozen=True, slots=True)
class HarborWatchdogConfig:
    """Timeout policy for a long-running Harbor benchmark launch."""

    timeout_s: int = 21_600
    idle_timeout_s: int = 1_800
    poll_interval_s: float = 5.0


@dataclass(frozen=True, slots=True)
class TerminalBenchLaunchConfig:
    """Prepared Terminal-Bench launch configuration.

    This object intentionally describes the launch without executing it.
    """

    dataset: str = "terminal-bench/terminal-bench-2-1"
    agent: str = "terminus-2"
    model: str = "google/gemini-3.1-pro-preview"
    environment: str = "modal"
    run_name: str = "agentprop-tbench-21-terminus2"
    task_count: int | None = None
    output_root: str = "benchmark-results/terminal-bench-2.1/terminus-2-agentprop"
    extra_instructions_path: str = "agentprop-extra-instructions.md"
    watchdog: HarborWatchdogConfig = field(default_factory=HarborWatchdogConfig)

    def harbor_command(self) -> list[str]:
        command = [
            "harbor",
            "run",
            "-d",
            self.dataset,
            "-a",
            self.agent,
            "-m",
            self.model,
            "--env",
            self.environment,
        ]
        if self.task_count is not None:
            command.extend(["-n", str(self.task_count)])
        return command


@dataclass(frozen=True, slots=True)
class HarborTrialSummary:
    """Normalized task-level result extracted from a Harbor result.json."""

    task_name: str
    trial_name: str
    reward: float | None
    passed: bool | None
    exception_name: str | None
    input_tokens: int
    cache_tokens: int
    output_tokens: int
    cost_usd: float
    result_path: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "task_name": self.task_name,
            "trial_name": self.trial_name,
            "reward": self.reward,
            "passed": self.passed,
            "exception_name": self.exception_name,
            "input_tokens": self.input_tokens,
            "cache_tokens": self.cache_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "result_path": self.result_path,
        }


@dataclass(frozen=True, slots=True)
class TerminalBenchSummary:
    """Aggregate metrics for normalized Harbor task results."""

    task_count: int
    observed_score_count: int
    pass_count: int
    fail_count: int
    exception_count: int
    pass_rate: float
    input_tokens: int
    cache_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    timeout_rate: float
    cost_adjusted_success: float

    def to_dict(self) -> dict[str, object]:
        return {
            "task_count": self.task_count,
            "observed_score_count": self.observed_score_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "exception_count": self.exception_count,
            "pass_rate": self.pass_rate,
            "input_tokens": self.input_tokens,
            "cache_tokens": self.cache_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "timeout_rate": self.timeout_rate,
            "cost_adjusted_success": self.cost_adjusted_success,
        }


def write_terminal_bench_launch_bundle(
    out_dir: str | Path,
    config: TerminalBenchLaunchConfig,
    *,
    registry_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write a dry-run launch bundle for the next Terminal-Bench run."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_instructions = output_dir / config.extra_instructions_path
    manifest = output_dir / "manifest.json"
    runbook = output_dir / "RUNBOOK.md"
    run_script = output_dir / "run_with_watchdog.sh"

    extra_instructions.write_text(_render_extra_instructions())
    manifest_payload = {
        "schema_version": 1,
        "purpose": "Prepared Terminal-Bench 2.1 + Terminus-2 launch; not executed.",
        "dataset": config.dataset,
        "agent": config.agent,
        "model": config.model,
        "environment": config.environment,
        "run_name": config.run_name,
        "task_count": config.task_count,
        "output_root": config.output_root,
        "extra_instructions_path": str(extra_instructions),
        "harbor_command": config.harbor_command(),
        "watchdog": {
            "timeout_s": config.watchdog.timeout_s,
            "idle_timeout_s": config.watchdog.idle_timeout_s,
            "poll_interval_s": config.watchdog.poll_interval_s,
        },
        "report_after_run": [
            "python",
            "experiments/summarize_harbor_results.py",
            "--results-root",
            config.output_root,
            "--out-dir",
            f"{config.output_root}/report",
        ],
    }
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")
    runbook.write_text(_render_runbook(config, manifest, extra_instructions))
    run_script.write_text(_render_run_script(config))
    run_script.chmod(0o755)

    if registry_root is not None:
        register_artifact(
            registry_root,
            artifact_id=f"{config.run_name}-manifest",
            kind="benchmark-manifest",
            path=manifest,
            source="terminal-bench-launch-prep",
            tags=("terminal-bench", "terminus-2", "dry-run"),
            metadata={"dataset": config.dataset, "model": config.model},
        )
        register_artifact(
            registry_root,
            artifact_id=f"{config.run_name}-runbook",
            kind="benchmark-runbook",
            path=runbook,
            source="terminal-bench-launch-prep",
            tags=("terminal-bench", "terminus-2", "dry-run"),
            metadata={"dataset": config.dataset, "model": config.model},
        )

    return {
        "manifest": manifest,
        "runbook": runbook,
        "run_script": run_script,
        "extra_instructions": extra_instructions,
    }


def load_harbor_trial_result(path: str | Path) -> HarborTrialSummary | None:
    """Load one Harbor task result, returning None for aggregate job results."""

    result_path = Path(path)
    payload = json.loads(result_path.read_text())
    task_name = payload.get("task_name")
    trial_name = payload.get("trial_name")
    if not isinstance(task_name, str) or not isinstance(trial_name, str):
        return None

    agent_result = _mapping(payload.get("agent_result"))
    verifier_result = _mapping(payload.get("verifier_result"))
    rewards = _mapping(verifier_result.get("rewards"))
    reward = _optional_float(rewards.get("reward"))
    exception_info = _mapping(payload.get("exception_info"))
    exception_name = _optional_string(exception_info.get("type") or exception_info.get("name"))
    if exception_name is None and exception_info:
        exception_name = _optional_string(exception_info.get("exception_type"))

    return HarborTrialSummary(
        task_name=task_name,
        trial_name=trial_name,
        reward=reward,
        passed=(reward > 0.0) if reward is not None else None,
        exception_name=exception_name,
        input_tokens=_int(agent_result.get("n_input_tokens")),
        cache_tokens=_int(agent_result.get("n_cache_tokens")),
        output_tokens=_int(agent_result.get("n_output_tokens")),
        cost_usd=_float(agent_result.get("cost_usd")),
        result_path=str(result_path),
    )


def collect_harbor_trial_results(results_root: str | Path) -> list[HarborTrialSummary]:
    """Collect all task-level Harbor result.json files under a root."""

    root = Path(results_root)
    rows = []
    for path in sorted(root.rglob("result.json")):
        trial = load_harbor_trial_result(path)
        if trial is not None:
            rows.append(trial)
    return rows


def summarize_terminal_bench_results(
    rows: list[HarborTrialSummary],
    *,
    cost_weight: float = 0.01,
    timeout_weight: float = 0.25,
) -> TerminalBenchSummary:
    """Aggregate result rows into launch-report metrics."""

    observed = [row for row in rows if row.passed is not None]
    pass_count = sum(1 for row in observed if row.passed)
    fail_count = sum(1 for row in observed if row.passed is False)
    exception_count = sum(1 for row in rows if row.exception_name)
    pass_rate = pass_count / len(observed) if observed else 0.0
    timeout_count = sum(
        1 for row in rows if row.exception_name and "timeout" in row.exception_name.lower()
    )
    timeout_rate = timeout_count / len(rows) if rows else 0.0
    input_tokens = sum(row.input_tokens for row in rows)
    cache_tokens = sum(row.cache_tokens for row in rows)
    output_tokens = sum(row.output_tokens for row in rows)
    cost_usd = sum(row.cost_usd for row in rows)
    cost_adjusted_success = pass_rate - cost_weight * cost_usd - timeout_weight * timeout_rate
    return TerminalBenchSummary(
        task_count=len(rows),
        observed_score_count=len(observed),
        pass_count=pass_count,
        fail_count=fail_count,
        exception_count=exception_count,
        pass_rate=pass_rate,
        input_tokens=input_tokens,
        cache_tokens=cache_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost_usd,
        timeout_rate=timeout_rate,
        cost_adjusted_success=cost_adjusted_success,
    )


def write_terminal_bench_summary_report(
    results_root: str | Path,
    out_dir: str | Path,
    *,
    title: str = "AgentProp Terminal-Bench Result Summary",
    registry_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write JSON, CSV, and Markdown summaries from saved Harbor artifacts."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_harbor_trial_results(results_root)
    summary = summarize_terminal_bench_results(rows)
    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "task_results.csv"
    report_path = output_dir / "report.md"

    summary_payload = {
        "schema_version": 1,
        "results_root": str(results_root),
        "summary": summary.to_dict(),
        "tasks": [row.to_dict() for row in rows],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    _write_trial_csv(rows_path, rows)
    report_path.write_text(_render_summary_markdown(title, summary, rows))

    if registry_root is not None:
        register_artifact(
            registry_root,
            artifact_id=f"{safe_name(title)}-summary",
            kind="metrics",
            path=summary_path,
            source="terminal-bench-summary",
            tags=("terminal-bench",),
            metadata=summary.to_dict(),
        )
        register_artifact(
            registry_root,
            artifact_id=f"{safe_name(title)}-report",
            kind="report",
            path=report_path,
            source="terminal-bench-summary",
            metrics_path=summary_path,
            tags=("terminal-bench",),
            metadata={"task_count": summary.task_count},
        )

    return {"summary": summary_path, "csv": rows_path, "report": report_path}


def safe_name(value: str) -> str:
    return "-".join(part for part in value.lower().replace("_", "-").split() if part)


def _render_runbook(
    config: TerminalBenchLaunchConfig,
    manifest: Path,
    extra_instructions: Path,
) -> str:
    command = " ".join(config.harbor_command())
    watchdog_command = (
        "python experiments/run_with_watchdog.py "
        f"--timeout {config.watchdog.timeout_s} "
        f"--idle-timeout {config.watchdog.idle_timeout_s} "
        f"--poll-interval {config.watchdog.poll_interval_s:g} "
        f"--log {config.output_root}/launcher.log "
        f"--status-json {config.output_root}/watchdog-status.json "
        f"-- {command}"
    )
    return f"""# Terminal-Bench 2.1 + Terminus-2 Launch Runbook

This bundle prepares the next AgentProp benchmark run. It does not execute the
benchmark by itself.

## Launch Contract

- Dataset: `{config.dataset}`
- Agent: `{config.agent}`
- Model: `{config.model}`
- Environment: `{config.environment}`
- Manifest: `{manifest}`
- AgentProp extra instructions: `{extra_instructions}`

## Preflight

1. Confirm Docker, Harbor, Modal, and model-provider credentials are configured.
2. Confirm the command below uses the intended model and environment.
3. Confirm the output root is empty or intentionally reused:
   `{config.output_root}`.
4. Keep the watchdog enabled so hung external IO cannot burn budget silently.

## Prepared Command

```bash
{watchdog_command}
```

## After The Run

```bash
python experiments/summarize_harbor_results.py \\
  --results-root {config.output_root} \\
  --out-dir {config.output_root}/report
```

The summary step writes `summary.json`, `task_results.csv`, and `report.md`.
"""


def _render_run_script(config: TerminalBenchLaunchConfig) -> str:
    command = " ".join(config.harbor_command())
    return f"""#!/usr/bin/env bash
set -euo pipefail

mkdir -p "{config.output_root}"
python experiments/run_with_watchdog.py \\
  --timeout {config.watchdog.timeout_s} \\
  --idle-timeout {config.watchdog.idle_timeout_s} \\
  --poll-interval {config.watchdog.poll_interval_s:g} \\
  --log "{config.output_root}/launcher.log" \\
  --status-json "{config.output_root}/watchdog-status.json" \\
  -- {command}
"""


def _render_extra_instructions() -> str:
    return """# AgentProp Benchmark Guidance

Use AgentProp routing discipline, but keep it budget-aware.

- Classify the task before acting: setup/build, code repair, numerical/scientific,
  reverse engineering, repo hygiene, or direct-answer.
- Spend full context on implementation-sensitive and verifier-sensitive phases.
- Prefer executable checks over long speculative reasoning.
- Stop exploration when the next action is not expected to change the verifier
  outcome; write a concise final answer instead.
- For numerical or scientific tasks, confirm units, coordinate systems, schema,
  and expected output ranges before fitting or optimizing.
- If a task is direct-answer or perception-heavy, avoid heavyweight process loops.
- Preserve evidence: commands run, files changed, verification output, and any
  unresolved risk.
"""


def _render_summary_markdown(
    title: str,
    summary: TerminalBenchSummary,
    rows: list[HarborTrialSummary],
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Aggregate",
        "",
        f"- Tasks: {summary.task_count}",
        f"- Observed scores: {summary.observed_score_count}",
        f"- Passes: {summary.pass_count}",
        f"- Failures: {summary.fail_count}",
        f"- Exceptions: {summary.exception_count}",
        f"- Pass rate: {summary.pass_rate:.1%}",
        f"- Input tokens: {summary.input_tokens:,}",
        f"- Cached tokens: {summary.cache_tokens:,}",
        f"- Output tokens: {summary.output_tokens:,}",
        f"- Total input+output tokens: {summary.total_tokens:,}",
        f"- Reported cost: ${summary.cost_usd:.2f}",
        f"- Timeout rate: {summary.timeout_rate:.1%}",
        f"- Cost-adjusted success: {summary.cost_adjusted_success:.3f}",
        "",
        "## Task Results",
        "",
        "| Task | Reward | Passed | Exception | Tokens | Cost |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        reward = "" if row.reward is None else f"{row.reward:.3f}"
        passed = "" if row.passed is None else str(row.passed).lower()
        exception = row.exception_name or "-"
        lines.append(
            f"| `{row.task_name}` | {reward} | {passed} | {exception} | "
            f"{row.total_tokens} | ${row.cost_usd:.4f} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _write_trial_csv(path: Path, rows: list[HarborTrialSummary]) -> None:
    fieldnames = list(HarborTrialSummary("", "", None, None, None, 0, 0, 0, 0.0, "").to_dict())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _float(value: object) -> float:
    return _optional_float(value) or 0.0


def _int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0
