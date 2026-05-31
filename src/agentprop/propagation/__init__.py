"""Propagation model interfaces and implementations."""

from agentprop.propagation.base import PropagationModel, PropagationResult
from agentprop.propagation.bootstrap_percolation import BootstrapPercolation
from agentprop.propagation.independent_cascade import IndependentCascade
from agentprop.propagation.linear_threshold import LinearThreshold
from agentprop.propagation.randomized_zero_forcing import RandomizedZeroForcing
from agentprop.propagation.zero_forcing import ZeroForcing

__all__ = [
    "BootstrapPercolation",
    "IndependentCascade",
    "LinearThreshold",
    "PropagationModel",
    "PropagationResult",
    "RandomizedZeroForcing",
    "ZeroForcing",
]
