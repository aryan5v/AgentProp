"""Entry-point plugin system for custom propagation models.

Third-party packages can ship custom propagation models by declaring an entry
point under the ``agentprop.propagation`` group in their ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."agentprop.propagation"]
    my-model = "my_package.propagation:MyModel"

The entry-point value must be an importable object that satisfies the
:class:`~agentprop.propagation.base.PropagationModel` protocol (i.e. has a
``name: str`` attribute and a ``simulate`` method). Call
:func:`load_plugins` once at startup to discover and register all installed
plugins so they become available to :func:`agentprop.evaluation.runner.make_propagation_model`.
"""

from __future__ import annotations

import importlib.metadata
import logging

from agentprop.propagation.base import PropagationModel

logger = logging.getLogger(__name__)

_PLUGIN_REGISTRY: dict[str, PropagationModel] = {}

_ENTRY_POINT_GROUP = "agentprop.propagation"

_BUILTIN_MODELS: frozenset[str] = frozenset(
    {
        "independent-cascade",
        "linear-threshold",
        "bootstrap",
        "rzf",
        "randomized-zero-forcing",
        "zero-forcing",
        "learned",
        "quality-cascade",
    }
)


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def register_plugin(name: str, model: PropagationModel) -> None:
    """Register a propagation model under *name* for use by :func:`make_propagation_model`.

    This is the programmatic alternative to entry-point discovery — useful for
    tests and notebooks where you want to register a model without installing a
    package.

    Args:
        name: The key that callers pass to ``--model`` or
            :func:`make_propagation_model`. Must be non-empty, must not shadow
            a built-in model name, and must not already be registered.
        model: Any object satisfying the :class:`PropagationModel` protocol.

    Raises:
        ValueError: If *name* is empty, conflicts with a built-in, or is already registered.
    """

    if not name:
        raise ValueError("Plugin name must be a non-empty string.")
    normalized = _normalize(name)
    if normalized in _BUILTIN_MODELS:
        raise ValueError(
            f"Plugin name {name!r} conflicts with built-in model {normalized!r}. "
            "Choose a different name to avoid shadowing built-in behaviour."
        )
    if name in _PLUGIN_REGISTRY:
        raise ValueError(
            f"A propagation plugin named {name!r} is already registered. "
            "Call load_plugins() before registering to avoid duplicate entries."
        )
    _PLUGIN_REGISTRY[name] = model
    logger.debug("Registered propagation plugin: %s → %s", name, type(model).__name__)


def load_plugins() -> dict[str, PropagationModel]:
    """Discover and register all installed ``agentprop.propagation`` entry-point plugins.

    Safe to call multiple times — already-registered names are skipped with a
    debug-level log rather than raising. Returns the full registry (built-ins
    are not included; only plugins registered via this function or
    :func:`register_plugin`).

    Example::

        from agentprop.propagation import load_plugins, make_propagation_model
        load_plugins()                          # discover third-party models
        model = make_propagation_model("my-model")  # uses the plugin

    Returns:
        Mapping of plugin name → model instance for every successfully loaded plugin.
    """

    eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
    for ep in eps:
        normalized = _normalize(ep.name)
        if normalized in _BUILTIN_MODELS:
            logger.warning(
                "Skipping propagation plugin %r: conflicts with built-in model %r.",
                ep.name,
                normalized,
            )
            continue
        if ep.name in _PLUGIN_REGISTRY:
            logger.debug("Skipping already-registered propagation plugin: %s", ep.name)
            continue
        try:
            loaded = ep.load()
            model = loaded() if isinstance(loaded, type) else loaded
            _PLUGIN_REGISTRY[ep.name] = model
            logger.debug(
                "Loaded propagation plugin: %s → %s", ep.name, type(model).__name__
            )
        except Exception:
            logger.warning(
                "Failed to load propagation plugin %r from %r",
                ep.name,
                ep.value,
                exc_info=True,
            )
    return dict(_PLUGIN_REGISTRY)


def get_plugin(name: str) -> PropagationModel | None:
    """Return the plugin registered under *name*, or ``None`` if absent."""

    return _PLUGIN_REGISTRY.get(name)


def list_plugins() -> list[str]:
    """Return the names of all currently registered plugins."""

    return list(_PLUGIN_REGISTRY)
