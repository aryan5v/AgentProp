"""Optional deep-learning interfaces for AgentProp."""

from agentprop.dl.encoders import GraphEncoderConfig, TorchBackendUnavailable
from agentprop.dl.torch_gnn import (
    TorchGNNSeedScorer,
    TorchTrainingResult,
    train_torch_seed_scorer,
)

__all__ = [
    "GraphEncoderConfig",
    "TorchBackendUnavailable",
    "TorchGNNSeedScorer",
    "TorchTrainingResult",
    "train_torch_seed_scorer",
]
