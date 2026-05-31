"""Export built-in workflow templates as JSON fixtures."""

from __future__ import annotations

from pathlib import Path

from agentprop.workflows.templates import WORKFLOW_TEMPLATES


def export_builtin_workflows(output_dir: str | Path) -> list[Path]:
    """Write all built-in workflow templates to JSON files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, builder in sorted(WORKFLOW_TEMPLATES.items()):
        path = output_path / f"{name}.json"
        builder().to_json(path)
        written.append(path)
    return written
