"""Artifact registry helpers for reproducible ML/RL experiment outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

ArtifactKind = Literal[
    "ml-model",
    "rl-policy",
    "metrics",
    "trace",
    "report",
    "benchmark-manifest",
    "benchmark-runbook",
]
_ARTIFACT_KINDS: dict[str, ArtifactKind] = {
    "ml-model": "ml-model",
    "rl-policy": "rl-policy",
    "metrics": "metrics",
    "trace": "trace",
    "report": "report",
    "benchmark-manifest": "benchmark-manifest",
    "benchmark-runbook": "benchmark-runbook",
}


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One indexed experiment artifact."""

    id: str
    kind: ArtifactKind
    path: str
    created_at: str
    source: str
    metrics_path: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "path": self.path,
            "created_at": self.created_at,
            "source": self.source,
            "metrics_path": self.metrics_path,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


def register_artifact(
    registry_root: str | Path,
    *,
    artifact_id: str,
    kind: ArtifactKind,
    path: str | Path,
    source: str,
    metrics_path: str | Path | None = None,
    tags: tuple[str, ...] | list[str] = (),
    metadata: Mapping[str, object] | None = None,
) -> ArtifactRecord:
    """Register or replace one artifact in a registry root."""

    root = Path(registry_root)
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / "registry.json"
    record = ArtifactRecord(
        id=safe_artifact_id(artifact_id),
        kind=kind,
        path=_display_path(path),
        created_at=datetime.now(UTC).isoformat(),
        source=source,
        metrics_path=_display_path(metrics_path) if metrics_path is not None else None,
        tags=tuple(str(tag) for tag in tags),
        metadata=dict(metadata or {}),
    )
    records = [
        existing
        for existing in load_artifact_registry(registry_path)
        if existing.id != record.id
    ]
    records.append(record)
    write_artifact_registry(registry_path, records)
    return record


def load_artifact_registry(path: str | Path) -> list[ArtifactRecord]:
    """Load a registry file if it exists."""

    registry_path = Path(path)
    if not registry_path.exists():
        return []
    payload = json.loads(registry_path.read_text())
    raw_records = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(raw_records, list):
        raise ValueError("artifact registry must contain an artifacts list")
    return [_record_from_dict(item) for item in raw_records if isinstance(item, dict)]


def write_artifact_registry(path: str | Path, records: list[ArtifactRecord]) -> Path:
    """Write an artifact registry file."""

    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(UTC).isoformat(),
        "artifacts": [
            record.to_dict()
            for record in sorted(records, key=lambda item: (item.kind, item.id))
        ],
    }
    registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return registry_path


def safe_artifact_id(value: str) -> str:
    """Normalize an artifact id for stable file names and registry keys."""

    normalized = []
    for character in value.lower():
        if character.isalnum():
            normalized.append(character)
        elif character in {"-", "_", ".", "/"}:
            normalized.append("-" if character == "/" else character)
        else:
            normalized.append("-")
    artifact_id = "".join(normalized).strip("-._")
    return artifact_id or "artifact"


def _record_from_dict(data: Mapping[str, object]) -> ArtifactRecord:
    return ArtifactRecord(
        id=_string(data.get("id")),
        kind=_kind(data.get("kind")),
        path=_string(data.get("path")),
        created_at=_string(data.get("created_at")),
        source=_string(data.get("source")),
        metrics_path=_optional_string(data.get("metrics_path")),
        tags=tuple(_string(tag) for tag in _list(data.get("tags"))),
        metadata=dict(_mapping(data.get("metadata", {}))),
    )


def _display_path(path: str | Path) -> str:
    return str(path)


def _kind(value: object) -> ArtifactKind:
    if isinstance(value, str) and value in _ARTIFACT_KINDS:
        return _ARTIFACT_KINDS[value]
    raise ValueError(f"Unsupported artifact kind: {value}")


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise ValueError("artifact metadata must be an object")


def _list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise ValueError("artifact tags must be a list")


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return _string(value)


def _string(value: object) -> str:
    if isinstance(value, str):
        return value
    raise ValueError("artifact field must be a string")
