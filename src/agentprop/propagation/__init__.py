"""Propagation model interfaces and implementations."""

from agentprop.propagation.base import PropagationModel, PropagationResult
from agentprop.propagation.bootstrap_percolation import BootstrapPercolation
from agentprop.propagation.independent_cascade import IndependentCascade
from agentprop.propagation.learned import (
    LearnedPropagation,
    LearnedPropagationFit,
    fit_learned_propagation_from_graph,
    fit_learned_propagation_from_trace_dicts,
)
from agentprop.propagation.linear_threshold import LinearThreshold
from agentprop.propagation.randomized_zero_forcing import RandomizedZeroForcing
from agentprop.propagation.zero_forcing import ZeroForcing

__all__ = [
    "BootstrapPercolation",
    "IndependentCascade",
    "LearnedPropagation",
    "LearnedPropagationFit",
    "LinearThreshold",
    "PropagationModel",
    "PropagationResult",
    "RandomizedZeroForcing",
    "ZeroForcing",
    "fit_learned_propagation_from_graph",
    "fit_learned_propagation_from_trace_dicts",
]
