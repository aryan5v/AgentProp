"""Guard against version drift between the package, README, and plugin bundles."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _package_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return str(data["project"]["version"])


def _base_release(version: str) -> str:
    match = re.match(r"\d+\.\d+\.\d+", version)
    assert match, f"unparseable version: {version}"
    return match.group(0)


def test_readme_badge_matches_package_version() -> None:
    readme = (ROOT / "README.md").read_text()
    version = _package_version()
    assert f"version-{version}" in readme, (
        f"README version badge does not mention package version {version}"
    )


def test_plugin_manifests_match_base_release() -> None:
    base = _base_release(_package_version())
    for manifest in [
        ROOT / "plugins/agentprop/.claude-plugin/plugin.json",
        ROOT / "plugins/agentprop/.codex-plugin/plugin.json",
    ]:
        data = json.loads(manifest.read_text())
        assert _base_release(str(data["version"])) == base, (
            f"{manifest} version {data['version']} drifted from package base {base}"
        )
