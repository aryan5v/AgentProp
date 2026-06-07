from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_codex_plugin_manifest_points_to_packaged_skill_and_mcp() -> None:
    plugin_root = REPO_ROOT / "distribution" / "plugins" / "agentprop"
    manifest = json.loads(
        (plugin_root / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    mcp_config = json.loads((plugin_root / ".mcp.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "agentprop"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert manifest["interface"]["displayName"] == "AgentProp"
    assert (
        plugin_root / "skills" / "agentprop-workflow-optimizer" / "SKILL.md"
    ).exists()
    assert mcp_config["mcpServers"]["agentprop"]["command"] == "agentprop-mcp"


def test_claude_plugin_manifest_points_to_same_bundle() -> None:
    plugin_root = REPO_ROOT / "distribution" / "plugins" / "agentprop"
    manifest = json.loads(
        (plugin_root / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    mcp_config = json.loads((plugin_root / ".mcp.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "agentprop"
    assert manifest["description"].startswith("Graph control")
    assert manifest["license"] == "Apache-2.0"
    assert (
        plugin_root / "skills" / "agentprop-workflow-optimizer" / "SKILL.md"
    ).exists()
    assert mcp_config["mcpServers"]["agentprop"]["command"] == "agentprop-mcp"


def test_claude_marketplace_exposes_agentprop_plugin() -> None:
    marketplace = json.loads(
        (REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text(
            encoding="utf-8"
        )
    )

    entries = {entry["name"]: entry for entry in marketplace["plugins"]}
    assert entries["agentprop"]["source"] == "./distribution/plugins/agentprop"


def test_repo_marketplace_exposes_agentprop_plugin() -> None:
    marketplace = json.loads(
        (REPO_ROOT / ".agents" / "plugins" / "marketplace.json").read_text(
            encoding="utf-8"
        )
    )

    entries = {entry["name"]: entry for entry in marketplace["plugins"]}
    assert entries["agentprop"]["source"]["path"] == "./distribution/plugins/agentprop"
    assert entries["agentprop"]["policy"]["installation"] == "AVAILABLE"
