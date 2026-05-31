"""Optional graph neural network encoder contracts."""

from __future__ import annotations

from dataclasses import dataclass


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


def require_torch() -> object:
    """Import torch or raise a clear optional-dependency error."""

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise TorchBackendUnavailable(
            "Install AgentProp with the optional deep-learning extra once it is enabled: "
            "`pip install agentprop[dl]`. The core package stays dependency-light by design."
        ) from exc
    return torch


class TorchGNNSeedScorer:
    """Placeholder wrapper for torch-backed GNN seed scorers.

    The class intentionally gates torch import so core users can import
    `agentprop.dl` without installing a deep-learning stack.
    """

    def __init__(self, config: GraphEncoderConfig) -> None:
        self.config = config
        self.torch = require_torch()

    def score_nodes(self) -> None:
        """Reserve the public method name for the torch implementation."""

        raise NotImplementedError("Torch-backed GNN scoring is not implemented yet.")
