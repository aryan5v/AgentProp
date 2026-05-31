"""Optional graph neural network encoder contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class TorchBackendUnavailable(ImportError):
    """Raised when a torch-backed model is requested without torch installed."""


@dataclass(slots=True)
class GraphEncoderConfig:
    """Configuration for future torch-backed GNN encoders."""

    input_dim: int
    hidden_dim: int = 64
    output_dim: int = 1
    layers: int = 2
    dropout: float = 0.0
    architecture: str = "graphsage"


def require_torch() -> Any:
    """Import torch or raise a clear optional-dependency error."""

    try:
        import torch
    except ImportError as exc:
        raise TorchBackendUnavailable(
            "Install AgentProp with the optional deep-learning extra once it is enabled: "
            "`pip install agentprop[dl]`. The core package stays dependency-light by design."
        ) from exc
    return torch

