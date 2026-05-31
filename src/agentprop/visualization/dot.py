"""Graphviz DOT export for AgentProp graphs."""

from __future__ import annotations

from pathlib import Path

from agentprop.core import AgentGraph


def graph_to_dot(graph: AgentGraph, *, name: str = "AgentPropWorkflow") -> str:
    """Render an AgentGraph as Graphviz DOT."""

    lines = [f"digraph {name} {{", "  rankdir=LR;"]
    for node in graph.nodes():
        label = node.name or node.id
        lines.append(
            f'  "{_escape(node.id)}" [label="{_escape(label)}\\n{node.type.value}"];'
        )
    for edge in graph.edges():
        label = f"w={edge.weight:.2f}, cost={edge.message_cost:.0f}"
        lines.append(
            f'  "{_escape(edge.source)}" -> "{_escape(edge.target)}" '
            f'[label="{_escape(label)}"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_dot(graph: AgentGraph, path: str | Path, *, name: str = "AgentPropWorkflow") -> Path:
    """Write Graphviz DOT to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(graph_to_dot(graph, name=name))
    return output_path


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
