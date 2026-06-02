"""Typed metadata models for agent workflow graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentprop.core.types import NodeType


@dataclass(slots=True)
class AgentNode:
    """A node in a multi-agent workflow graph."""

    id: str
    type: NodeType = NodeType.AGENT
    name: str | None = None
    role: str | None = None
    token_cost: float = 0.0
    latency: float = 0.0
    reliability: float = 1.0
    error_rate: float = 0.0
    context_capacity: int | None = None
    tool_access: tuple[str, ...] = ()
    importance_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize node metadata to JSON-compatible data."""

        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "role": self.role,
            "token_cost": self.token_cost,
            "latency": self.latency,
            "reliability": self.reliability,
            "error_rate": self.error_rate,
            "context_capacity": self.context_capacity,
            "tool_access": list(self.tool_access),
            "importance_score": self.importance_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentNode:
        """Build a node from JSON-compatible data."""

        payload = dict(data)
        payload["type"] = NodeType(payload.get("type", NodeType.AGENT.value))
        payload["tool_access"] = tuple(payload.get("tool_access", ()))
        payload.setdefault("metadata", {})
        return cls(**payload)


@dataclass(slots=True)
class AgentEdge:
    """A directed communication, dependency, retrieval, or tool-call edge."""

    source: str
    target: str
    message_cost: float = 0.0
    latency: float = 0.0
    relevance: float = 1.0
    reliability: float = 1.0
    activation_probability: float = 0.85
    dependency_strength: float = 1.0
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize edge metadata to JSON-compatible data."""

        return {
            "source": self.source,
            "target": self.target,
            "message_cost": self.message_cost,
            "latency": self.latency,
            "relevance": self.relevance,
            "reliability": self.reliability,
            "activation_probability": self.activation_probability,
            "dependency_strength": self.dependency_strength,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentEdge:
        """Build an edge from JSON-compatible data."""

        payload = dict(data)
        payload.setdefault("metadata", {})
        return cls(**payload)
