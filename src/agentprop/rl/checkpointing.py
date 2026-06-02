"""JSON checkpoints for dependency-light RL routing policies."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

from agentprop.rl.feature_policy import GraphFeaturePolicy
from agentprop.rl.ppo import PPOPolicy
from agentprop.rl.q_learning import TabularQPolicy
from agentprop.rl.reinforce import ReinforcePolicy

RLCheckpointPolicy: TypeAlias = (
    TabularQPolicy | ReinforcePolicy | PPOPolicy | GraphFeaturePolicy
)


@dataclass(frozen=True, slots=True)
class RLPolicyCheckpoint:
    """Loaded dependency-light RL policy checkpoint."""

    policy: RLCheckpointPolicy
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: int = 1


def save_rl_policy(
    policy: RLCheckpointPolicy,
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write a dependency-light RL policy checkpoint."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": _policy_kind(policy),
        "policy": _policy_payload(policy),
        "metadata": dict(metadata or {}),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def load_rl_policy(path: str | Path) -> RLPolicyCheckpoint:
    """Load a dependency-light RL policy checkpoint."""

    payload = _mapping(json.loads(Path(path).read_text()))
    schema_version = _int_value(payload.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"Unsupported RL checkpoint schema_version: {schema_version}")
    kind = _string(payload.get("kind"))
    policy_payload = _mapping(payload.get("policy"))
    return RLPolicyCheckpoint(
        policy=_policy_from_payload(kind, policy_payload),
        metadata=dict(_mapping(payload.get("metadata", {}))),
        schema_version=schema_version,
    )


def _policy_kind(policy: RLCheckpointPolicy) -> str:
    if isinstance(policy, TabularQPolicy):
        return "tabular_q_policy"
    if isinstance(policy, ReinforcePolicy):
        return "reinforce_policy"
    if isinstance(policy, PPOPolicy):
        return "ppo_policy"
    if isinstance(policy, GraphFeaturePolicy):
        return "graph_feature_policy"
    raise TypeError(f"Unsupported RL policy type: {type(policy).__name__}")


def _policy_payload(policy: RLCheckpointPolicy) -> dict[str, object]:
    if isinstance(policy, TabularQPolicy):
        return {
            "q_values": _tuple_table_entries(policy.q_values),
            "expanded_actions": policy.expanded_actions,
        }
    if isinstance(policy, ReinforcePolicy):
        return {
            "preferences": _tuple_table_entries(policy.preferences),
            "expanded_actions": policy.expanded_actions,
        }
    if isinstance(policy, PPOPolicy):
        return {
            "preferences": _tuple_table_entries(policy.preferences),
            "values": dict(sorted(policy.values.items())),
            "expanded_actions": policy.expanded_actions,
        }
    if isinstance(policy, GraphFeaturePolicy):
        return {
            "weights": list(policy.weights),
            "feature_names": list(policy.feature_names),
        }
    raise TypeError(f"Unsupported RL policy type: {type(policy).__name__}")


def _policy_from_payload(kind: str, payload: Mapping[str, object]) -> RLCheckpointPolicy:
    if kind == "tabular_q_policy":
        return TabularQPolicy(
            q_values=_tuple_table(payload.get("q_values")),
            expanded_actions=_bool_value(payload.get("expanded_actions")),
        )
    if kind == "reinforce_policy":
        return ReinforcePolicy(
            preferences=_tuple_table(payload.get("preferences")),
            expanded_actions=_bool_value(payload.get("expanded_actions")),
        )
    if kind == "ppo_policy":
        return PPOPolicy(
            preferences=_tuple_table(payload.get("preferences")),
            values=_value_table(payload.get("values")),
            expanded_actions=_bool_value(payload.get("expanded_actions")),
        )
    if kind == "graph_feature_policy":
        return GraphFeaturePolicy(
            weights=_float_list(payload.get("weights")),
            feature_names=_string_list(payload.get("feature_names")),
        )
    raise ValueError(f"Unsupported RL checkpoint kind: {kind}")


def _tuple_table_entries(table: Mapping[tuple[str, str], float]) -> list[dict[str, object]]:
    return [
        {"state_key": state_key, "action": action, "value": value}
        for (state_key, action), value in sorted(table.items())
    ]


def _tuple_table(value: object) -> dict[tuple[str, str], float]:
    if not isinstance(value, list):
        raise ValueError("policy table must be a list")
    table: dict[tuple[str, str], float] = {}
    for row in value:
        item = _mapping(row)
        table[(_string(item.get("state_key")), _string(item.get("action")))] = _float_value(
            item.get("value")
        )
    return table


def _value_table(value: object) -> dict[str, float]:
    if isinstance(value, Mapping):
        return {str(key): _float_value(item) for key, item in value.items()}
    raise ValueError("value table must be an object")


def _float_list(value: object) -> list[float]:
    if isinstance(value, list):
        return [_float_value(item) for item in value]
    raise ValueError("checkpoint field must be a numeric list")


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [_string(item) for item in value]
    raise ValueError("checkpoint field must be a string list")


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise ValueError("checkpoint field must be an object")


def _float_value(value: object) -> float:
    if isinstance(value, int | float | str) and not isinstance(value, bool):
        return float(value)
    raise ValueError("checkpoint field must be numeric")


def _bool_value(value: object) -> bool:
    return bool(value) if isinstance(value, bool) else False


def _int_value(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError("checkpoint field must be an integer")


def _string(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    raise ValueError("checkpoint field must be a non-empty string")
