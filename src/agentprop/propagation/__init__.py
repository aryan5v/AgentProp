"""Propagation model interfaces and implementations."""

from agentprop.propagation.base import PropagationModel, PropagationResult
from agentprop.propagation.bootstrap_percolation import BootstrapPercolation
from agentprop.propagation.feature_calibrated import (
    FeatureCalibratedPropagation,
    observations_from_trace_dicts,
)
from agentprop.propagation.independent_cascade import IndependentCascade
from agentprop.propagation.learned import (
    LearnedPropagation,
    LearnedPropagationFit,
    fit_learned_propagation_from_graph,
    fit_learned_propagation_from_trace_dicts,
)
from agentprop.propagation.linear_threshold import LinearThreshold
from agentprop.propagation.plugins import get_plugin, list_plugins, load_plugins, register_plugin
from agentprop.propagation.quality_cascade import QualityCascade, QualityCascadeResult
from agentprop.propagation.randomized_zero_forcing import RandomizedZeroForcing
from agentprop.propagation.zero_forcing import ZeroForcing

__all__ = [
    "BootstrapPercolation",
    "FeatureCalibratedPropagation",
    "IndependentCascade",
    "LearnedPropagation",
    "LearnedPropagationFit",
    "LinearThreshold",
    "PropagationModel",
    "PropagationResult",
    "QualityCascade",
    "QualityCascadeResult",
    "RandomizedZeroForcing",
    "ZeroForcing",
    "fit_learned_propagation_from_graph",
    "fit_learned_propagation_from_trace_dicts",
    "get_plugin",
    "list_plugins",
    "load_plugins",
    "observations_from_trace_dicts",
    "register_plugin",
]
