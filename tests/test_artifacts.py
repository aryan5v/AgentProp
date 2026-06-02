from pathlib import Path

from agentprop.evaluation import load_artifact_registry, register_artifact, safe_artifact_id


def test_register_artifact_upserts_registry_records(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")

    first = register_artifact(
        tmp_path,
        artifact_id="Demo/Policy",
        kind="rl-policy",
        path=checkpoint,
        source="test",
        tags=("ppo", "demo"),
        metadata={"score": 0.8},
    )
    second = register_artifact(
        tmp_path,
        artifact_id="Demo/Policy",
        kind="rl-policy",
        path=checkpoint,
        source="test",
        metadata={"score": 0.9},
    )
    records = load_artifact_registry(tmp_path / "registry.json")

    assert first.id == "demo-policy"
    assert second.id == first.id
    assert len(records) == 1
    assert records[0].metadata["score"] == 0.9


def test_safe_artifact_id_has_stable_fallback() -> None:
    assert safe_artifact_id("MLP Seed Scorer!") == "mlp-seed-scorer"
    assert safe_artifact_id("///") == "artifact"
