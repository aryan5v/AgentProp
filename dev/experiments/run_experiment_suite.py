"""Run config-defined AgentProp experiment suites."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SuiteRunResult:
    """Execution metadata for one configured experiment run."""

    run_id: str
    module: str
    argv: list[str]
    exit_code: int
    output: str | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a config-defined AgentProp experiment suite.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dev/configs/experiment_suites/ml_core.json"),
    )
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None, help="Optional run IDs to execute.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write a manifest without executing child experiments.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    suite = _load_suite(args.config)
    artifact_root = args.artifact_root or Path(str(suite["artifacts"]["root"]))
    manifest_path = args.manifest or artifact_root / "suite_manifest.json"
    selected = set(args.only or [])
    runs = [
        run
        for run in suite["runs"]
        if not selected or str(run["id"]) in selected
    ]
    if selected and len(runs) != len(selected):
        known = {str(run["id"]) for run in suite["runs"]}
        missing = sorted(selected - known)
        raise ValueError(f"Unknown suite run IDs: {', '.join(missing)}")

    artifact_root.mkdir(parents=True, exist_ok=True)
    results = []
    for run in runs:
        expanded_argv = _args_to_argv(
            run["args"],
            artifact_root=artifact_root,
        )
        output = _output_path_from_args(expanded_argv)
        if args.dry_run:
            exit_code = 0
        else:
            module = importlib.import_module(str(run["module"]))
            module_main = getattr(module, "main", None)
            if module_main is None:
                raise ValueError(f"{run['module']} does not expose main(argv)")
            exit_code = int(module_main(expanded_argv))
        results.append(
            SuiteRunResult(
                run_id=str(run["id"]),
                module=str(run["module"]),
                argv=expanded_argv,
                exit_code=exit_code,
                output=output,
            )
        )
        if exit_code != 0:
            break

    _write_manifest(
        suite,
        results,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        dry_run=args.dry_run,
    )
    print(f"Wrote {manifest_path}")
    return 0 if all(result.exit_code == 0 for result in results) else 1


def _load_suite(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("suite config must be a JSON object")
    if not isinstance(data.get("runs"), list):
        raise ValueError("suite config must contain a runs list")
    if not isinstance(data.get("artifacts"), dict):
        raise ValueError("suite config must contain artifacts")
    _check_required_env(data)
    return data


def _check_required_env(suite: dict[str, Any]) -> None:
    runtime = suite.get("runtime", {})
    env = runtime.get("env", {}) if isinstance(runtime, dict) else {}
    required = env.get("required", []) if isinstance(env, dict) else []
    missing = [name for name in required if isinstance(name, str) and name not in os.environ]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def _args_to_argv(args: dict[str, Any], *, artifact_root: Path) -> list[str]:
    argv: list[str] = []
    for key, value in args.items():
        flag = str(key)
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, _expand_template(value, artifact_root=artifact_root)])
    return argv


def _expand_template(value: Any, *, artifact_root: Path) -> str:
    return str(value).replace("{artifact_root}", str(artifact_root))


def _output_path_from_args(argv: list[str]) -> str | None:
    for index, item in enumerate(argv):
        if item == "--out" and index + 1 < len(argv):
            return argv[index + 1]
        if item == "--out-dir" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _write_manifest(
    suite: dict[str, Any],
    results: list[SuiteRunResult],
    *,
    artifact_root: Path,
    manifest_path: Path,
    dry_run: bool,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": suite.get("name"),
        "description": suite.get("description"),
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_root": str(artifact_root),
        "dry_run": dry_run,
        "runtime": suite.get("runtime", {}),
        "artifacts": suite.get("artifacts", {}),
        "runs": [
            {
                "id": result.run_id,
                "module": result.module,
                "argv": result.argv,
                "exit_code": result.exit_code,
                "output": result.output,
            }
            for result in results
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
