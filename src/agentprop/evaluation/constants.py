"""Shared CLI and API choice lists for AgentProp evaluation."""

from __future__ import annotations

# Canonical seed-selection algorithm names exposed on the CLI.
SEED_ALGORITHM_CHOICES: tuple[str, ...] = (
    "random",
    "degree",
    "in-degree",
    "out-degree",
    "pagerank",
    "betweenness",
    "closeness",
    "k-core",
    "pure-greedy",
    "greedy",
    "celf",
    "cost-aware-greedy",
    "quality-aware-greedy",
    "rzf-centrality",
)

# Canonical propagation model names exposed on the CLI.
PROPAGATION_MODEL_CHOICES: tuple[str, ...] = (
    "independent-cascade",
    "linear-threshold",
    "bootstrap",
    "rzf",
    "randomized-zero-forcing",
    "zero-forcing",
    "learned",
    "quality-cascade",
)

# Short aliases accepted at runtime but not listed as primary CLI choices.
_PROPAGATION_MODEL_ALIASES: dict[str, str] = {
    "ic": "independent-cascade",
    "lt": "linear-threshold",
    "bootstrap-percolation": "bootstrap",
    "zf": "zero-forcing",
    "qc": "quality-cascade",
    "trace-learned": "learned",
}


def normalize_propagation_model(name: str) -> str:
    """Normalize a propagation model CLI name to its canonical form."""

    lowered = name.strip().lower()
    return _PROPAGATION_MODEL_ALIASES.get(lowered, lowered)


def is_valid_propagation_model(name: str) -> bool:
    """Return whether *name* is a supported propagation model."""

    normalized = normalize_propagation_model(name)
    return normalized in PROPAGATION_MODEL_CHOICES
