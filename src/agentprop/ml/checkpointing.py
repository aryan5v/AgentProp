"""JSON checkpoints for dependency-light ML scorers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

from agentprop.ml.models import (
    LinearEdgeScorer,
    LinearNodeRegressor,
    LinearNodeScorer,
    MessagePassingNodeScorer,
    MLPNodeScorer,
    PairwiseNodeRanker,
)

MLCheckpointModel: TypeAlias = (
    LinearNodeScorer
    | MLPNodeScorer
    | PairwiseNodeRanker
    | LinearNodeRegressor
    | LinearEdgeScorer
    | MessagePassingNodeScorer
)


@dataclass(frozen=True, slots=True)
class MLModelCheckpoint:
    """Loaded dependency-light ML checkpoint."""

    model: MLCheckpointModel
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: int = 1


def save_ml_model(
    model: MLCheckpointModel,
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write a dependency-light ML model checkpoint."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": _model_kind(model),
        "model": _model_payload(model),
        "metadata": dict(metadata or {}),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def load_ml_model(path: str | Path) -> MLModelCheckpoint:
    """Load a dependency-light ML model checkpoint."""

    payload = _mapping(json.loads(Path(path).read_text()))
    schema_version = _int_value(payload.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"Unsupported ML checkpoint schema_version: {schema_version}")
    kind = _string(payload.get("kind"))
    model_payload = _mapping(payload.get("model"))
    return MLModelCheckpoint(
        model=_model_from_payload(kind, model_payload),
        metadata=dict(_mapping(payload.get("metadata", {}))),
        schema_version=schema_version,
    )


def _model_kind(model: MLCheckpointModel) -> str:
    if isinstance(model, LinearNodeScorer):
        return "linear_node_scorer"
    if isinstance(model, MLPNodeScorer):
        return "mlp_node_scorer"
    if isinstance(model, PairwiseNodeRanker):
        return "pairwise_node_ranker"
    if isinstance(model, LinearNodeRegressor):
        return "linear_node_regressor"
    if isinstance(model, LinearEdgeScorer):
        return "linear_edge_scorer"
    if isinstance(model, MessagePassingNodeScorer):
        return "message_passing_node_scorer"
    raise TypeError(f"Unsupported ML model type: {type(model).__name__}")


def _model_payload(model: MLCheckpointModel) -> dict[str, object]:
    if isinstance(model, LinearNodeScorer | LinearNodeRegressor | LinearEdgeScorer):
        return {"weights": model.weights, "bias": model.bias}
    if isinstance(model, MLPNodeScorer):
        return {
            "input_weights": model.input_weights,
            "output_weights": model.output_weights,
            "hidden_bias": model.hidden_bias,
            "output_bias": model.output_bias,
        }
    if isinstance(model, PairwiseNodeRanker):
        return {"weights": model.weights}
    if isinstance(model, MessagePassingNodeScorer):
        return {
            "base_scorer": _model_payload(model.base_scorer),
            "neighbor_weight": model.neighbor_weight,
        }
    raise TypeError(f"Unsupported ML model type: {type(model).__name__}")


def _model_from_payload(kind: str, payload: Mapping[str, object]) -> MLCheckpointModel:
    if kind == "linear_node_scorer":
        return LinearNodeScorer(
            weights=_float_list(payload.get("weights")),
            bias=_float_value(payload.get("bias")),
        )
    if kind == "mlp_node_scorer":
        return MLPNodeScorer(
            input_weights=_float_matrix(payload.get("input_weights")),
            output_weights=_float_list(payload.get("output_weights")),
            hidden_bias=_float_list(payload.get("hidden_bias")),
            output_bias=_float_value(payload.get("output_bias")),
        )
    if kind == "pairwise_node_ranker":
        return PairwiseNodeRanker(weights=_float_list(payload.get("weights")))
    if kind == "linear_node_regressor":
        return LinearNodeRegressor(
            weights=_float_list(payload.get("weights")),
            bias=_float_value(payload.get("bias")),
        )
    if kind == "linear_edge_scorer":
        return LinearEdgeScorer(
            weights=_float_list(payload.get("weights")),
            bias=_float_value(payload.get("bias")),
        )
    if kind == "message_passing_node_scorer":
        return MessagePassingNodeScorer(
            base_scorer=LinearNodeScorer(
                weights=_float_list(_mapping(payload.get("base_scorer")).get("weights")),
                bias=_float_value(_mapping(payload.get("base_scorer")).get("bias")),
            ),
            neighbor_weight=_float_value(payload.get("neighbor_weight")),
        )
    raise ValueError(f"Unsupported ML checkpoint kind: {kind}")


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise ValueError("checkpoint field must be an object")


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("checkpoint field must be a list")
    return [_float_value(item) for item in value]


def _float_matrix(value: object) -> list[list[float]]:
    if not isinstance(value, list):
        raise ValueError("checkpoint field must be a list")
    return [_float_list(row) for row in value]


def _float_value(value: object) -> float:
    if isinstance(value, int | float | str) and not isinstance(value, bool):
        return float(value)
    if value is None:
        return 0.0
    raise ValueError("checkpoint field must be numeric")


def _int_value(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError("checkpoint field must be an integer")


def _string(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError("checkpoint kind must be a non-empty string")
