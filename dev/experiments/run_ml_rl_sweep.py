"""Run config-defined ML/RL sweeps with indexed artifacts."""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentprop.evaluation import register_artifact, safe_artifact_id


@dataclass(frozen=True, slots=True)
class SweepRunResult:
    """Execution metadata for one expanded sweep run."""

    run_id: str
    sweep_id: str
    module: str
    argv: list[str]
    params: dict[str, Any]
    exit_code: int
    output: str | None
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "sweep_id": self.sweep_id,
            "module": self.module,
            "argv": self.argv,
            "params": self.params,
            "exit_code": self.exit_code,
            "output": self.output,
            "metrics": self.metrics,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run config-defined AgentProp ML/RL sweeps.")
    parser.add_argument("--config", type=Path, default=Path("dev/configs/sweeps/ml_rl_smoke.json"))
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--only", nargs="+", default=None, help="Optional sweep IDs to execute.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    artifact_root = args.artifact_root or Path(str(config["artifacts"]["root"]))
    registry_root = args.registry_root or artifact_root / "model_registry"
    manifest_path = args.manifest or artifact_root / "sweep_manifest.json"
    selected = set(args.only or [])
    expanded_runs = _expand_sweeps(config, artifact_root=artifact_root, registry_root=registry_root)
    if selected:
        known = {run["sweep_id"] for run in expanded_runs}
        missing = sorted(selected - known)
        if missing:
            raise ValueError(f"Unknown sweep IDs: {', '.join(missing)}")
        expanded_runs = [run for run in expanded_runs if run["sweep_id"] in selected]

    artifact_root.mkdir(parents=True, exist_ok=True)
    results = []
    for run in expanded_runs:
        output = _output_path_from_args(run["argv"])
        if args.dry_run:
            exit_code = 0
        else:
            module = importlib.import_module(run["module"])
            module_main = getattr(module, "main", None)
            if module_main is None:
                raise ValueError(f"{run['module']} does not expose main(argv)")
            exit_code = int(module_main(run["argv"]))
        metrics = (
            _metrics_from_output(output)
            if output is not None and Path(output).exists()
            else {}
        )
        if output is not None and Path(output).exists():
            register_artifact(
                registry_root,
                artifact_id=f"{run['run_id']}-metrics",
                kind="metrics",
                path=output,
                source=str(run["module"]),
                tags=("sweep", str(run["sweep_id"])),
                metadata={"params": run["params"], "exit_code": exit_code},
            )
        result = SweepRunResult(
            run_id=str(run["run_id"]),
            sweep_id=str(run["sweep_id"]),
            module=str(run["module"]),
            argv=list(run["argv"]),
            params=dict(run["params"]),
            exit_code=exit_code,
            output=output,
            metrics=metrics,
        )
        results.append(result)
        if exit_code != 0:
            break

    _write_manifest(
        config,
        results,
        artifact_root=artifact_root,
        registry_root=registry_root,
        manifest_path=manifest_path,
        dry_run=args.dry_run,
    )
    print(f"Wrote {manifest_path}")
    return 0 if all(result.exit_code == 0 for result in results) else 1


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("sweep config must be a JSON object")
    if not isinstance(data.get("sweeps"), list):
        raise ValueError("sweep config must contain a sweeps list")
    if not isinstance(data.get("artifacts"), dict):
        raise ValueError("sweep config must contain artifacts")
    return data


def _expand_sweeps(
    config: dict[str, Any],
    *,
    artifact_root: Path,
    registry_root: Path,
) -> list[dict[str, Any]]:
    runs = []
    for sweep in config["sweeps"]:
        if not isinstance(sweep, dict):
            raise ValueError("each sweep must be an object")
        sweep_id = safe_artifact_id(str(sweep["id"]))
        base_args = _mapping(sweep.get("args", {}))
        for index, params in enumerate(_grid_params(_mapping(sweep.get("grid", {}))), start=1):
            run_id = _run_id(sweep_id, index, params)
            values = {
                "artifact_root": str(artifact_root),
                "registry_root": str(registry_root),
                "sweep_id": sweep_id,
                "run_id": run_id,
                "index": str(index),
            }
            merged_args = {**base_args, **params}
            argv = _args_to_argv(merged_args, values=values)
            runs.append(
                {
                    "run_id": run_id,
                    "sweep_id": sweep_id,
                    "module": str(sweep["module"]),
                    "argv": argv,
                    "params": params,
                }
            )
    return runs


def _grid_params(grid: dict[str, Any]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid)
    value_lists = []
    for key in keys:
        values = grid[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid parameter {key} must be a non-empty list")
        value_lists.append(values)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*value_lists)
    ]


def _args_to_argv(args: dict[str, Any], *, values: dict[str, str]) -> list[str]:
    argv: list[str] = []
    for key, value in args.items():
        flag = str(key)
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, _expand_template(value, values=values)])
    return argv


def _expand_template(value: Any, *, values: dict[str, str]) -> str:
    text = str(value)
    for key, replacement in values.items():
        text = text.replace("{" + key + "}", replacement)
    return text


def _run_id(sweep_id: str, index: int, params: dict[str, Any]) -> str:
    if not params:
        return f"{sweep_id}-{index}"
    suffix = "-".join(
        safe_artifact_id(f"{key}-{value}".lstrip("-"))
        for key, value in params.items()
    )
    return safe_artifact_id(f"{sweep_id}-{index}-{suffix}")


def _output_path_from_args(argv: list[str]) -> str | None:
    for index, item in enumerate(argv):
        if item == "--out" and index + 1 < len(argv):
            return argv[index + 1]
        if item == "--out-dir" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _metrics_from_output(path: str) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        return _metrics_from_mapping(data)
    if isinstance(data, list):
        summaries = [
            item.get("summary")
            for item in data
            if isinstance(item, dict) and isinstance(item.get("summary"), dict)
        ]
        return _mean_metrics([_metrics_from_mapping(summary) for summary in summaries])
    return {}


def _metrics_from_mapping(data: dict[str, Any]) -> dict[str, float]:
    metric_keys = {
        "mean_top_k_recall",
        "efficiency_score",
        "cost_adjusted_success",
        "final_coverage",
        "proxy_success_rate",
        "mean_efficiency_score",
    }
    return {
        key: float(value)
        for key, value in data.items()
        if key in metric_keys and isinstance(value, int | float)
    } | _evaluation_metrics(data)


def _evaluation_metrics(data: dict[str, Any]) -> dict[str, float]:
    evaluations = data.get("evaluations")
    if not isinstance(evaluations, list):
        return {}
    top_scores = []
    for evaluation in evaluations:
        if not isinstance(evaluation, dict):
            continue
        scores = evaluation.get("scores")
        if isinstance(scores, dict) and scores:
            numeric_scores = [
                float(value)
                for value in scores.values()
                if isinstance(value, int | float)
            ]
            if numeric_scores:
                top_scores.append(max(numeric_scores))
            continue
        ranked_edges = evaluation.get("ranked_edges")
        if isinstance(ranked_edges, list) and ranked_edges:
            first = ranked_edges[0]
            if isinstance(first, dict) and isinstance(first.get("score"), int | float):
                top_scores.append(float(first["score"]))
    if not top_scores:
        return {}
    return {"mean_top_score": sum(top_scores) / len(top_scores)}


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: sum(row.get(key, 0.0) for row in rows) / len(rows)
        for key in keys
        if rows
    }


def _write_manifest(
    config: dict[str, Any],
    results: list[SweepRunResult],
    *,
    artifact_root: Path,
    registry_root: Path,
    manifest_path: Path,
    dry_run: bool,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sweep": config.get("name"),
        "description": config.get("description"),
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_root": str(artifact_root),
        "registry_root": str(registry_root),
        "dry_run": dry_run,
        "run_count": len(results),
        "success_count": sum(1 for result in results if result.exit_code == 0),
        "best_runs": _best_runs(results),
        "runs": [result.to_dict() for result in results],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _best_runs(results: list[SweepRunResult]) -> dict[str, str]:
    scores = []
    for result in results:
        score = _primary_score(result.metrics)
        if score is not None:
            scores.append((score, result.run_id))
    if not scores:
        return {}
    best_score, run_id = max(scores)
    return {"primary_score": f"{best_score:.6f}", "run_id": run_id}


def _primary_score(metrics: dict[str, float]) -> float | None:
    for key in (
        "mean_top_k_recall",
        "mean_top_score",
        "mean_efficiency_score",
        "efficiency_score",
        "cost_adjusted_success",
        "final_coverage",
    ):
        if key in metrics:
            return metrics[key]
    return None


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    raise ValueError("sweep args/grid fields must be objects")


if __name__ == "__main__":
    raise SystemExit(main())
