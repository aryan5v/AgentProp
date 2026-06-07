"""Tests for the propagation plugin registry (AGE-41)."""

from __future__ import annotations

import pytest

from agentprop.propagation import get_plugin, list_plugins, load_plugins, register_plugin
from agentprop.propagation.base import PropagationResult
from agentprop.propagation.plugins import _PLUGIN_REGISTRY


class _ConstantModel:
    """Stub model that always returns an empty activation."""

    name = "test-constant"

    def simulate(self, graph, seeds, *, trials=1) -> PropagationResult:
        return PropagationResult(
            activated_nodes=set(seeds),
            propagation_time=1,
            coverage=len(seeds) / max(graph.node_count, 1),
            activation_rounds={s: 0 for s in seeds},
            trials=trials,
        )


def _clean_registry() -> None:
    """Remove all test-registered plugins between tests."""
    for key in list(_PLUGIN_REGISTRY):
        del _PLUGIN_REGISTRY[key]


def test_register_plugin_and_get_plugin() -> None:
    _clean_registry()
    model = _ConstantModel()
    register_plugin("test-constant", model)
    assert get_plugin("test-constant") is model
    _clean_registry()


def test_list_plugins_reflects_registry() -> None:
    _clean_registry()
    register_plugin("test-a", _ConstantModel())
    register_plugin("test-b", _ConstantModel())
    plugins = list_plugins()
    assert "test-a" in plugins
    assert "test-b" in plugins
    _clean_registry()


def test_duplicate_register_raises() -> None:
    _clean_registry()
    register_plugin("test-dup", _ConstantModel())
    with pytest.raises(ValueError, match="already registered"):
        register_plugin("test-dup", _ConstantModel())
    _clean_registry()


def test_empty_name_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        register_plugin("", _ConstantModel())


def test_builtin_name_raises() -> None:
    _clean_registry()
    with pytest.raises(ValueError, match="conflicts with built-in"):
        register_plugin("independent-cascade", _ConstantModel())
    with pytest.raises(ValueError, match="conflicts with built-in"):
        register_plugin("Independent_Cascade", _ConstantModel())


def test_get_plugin_returns_none_for_unknown() -> None:
    _clean_registry()
    assert get_plugin("does-not-exist") is None


def test_load_plugins_returns_dict() -> None:
    _clean_registry()
    result = load_plugins()
    assert isinstance(result, dict)


def test_load_plugins_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_registry()
    import importlib.metadata

    ep_mock = type(
        "EP", (), {"name": "idempotent-model", "value": "x", "load": lambda self: _ConstantModel()}
    )()

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [ep_mock])
    load_plugins()
    assert "idempotent-model" in _PLUGIN_REGISTRY
    load_plugins()
    assert list(_PLUGIN_REGISTRY).count("idempotent-model") == 1
    _clean_registry()


def test_load_plugins_auto_instantiates_class(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_registry()
    import importlib.metadata

    ep_mock = type(
        "EP", (), {"name": "class-model", "value": "x", "load": lambda self: _ConstantModel}
    )()

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [ep_mock])
    load_plugins()
    assert "class-model" in _PLUGIN_REGISTRY
    assert isinstance(_PLUGIN_REGISTRY["class-model"], _ConstantModel)
    _clean_registry()


def test_load_plugins_skips_builtin_names(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_registry()
    import importlib.metadata

    ep_mock = type(
        "EP",
        (),
        {"name": "independent-cascade", "value": "x", "load": lambda self: _ConstantModel()},
    )()

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [ep_mock])
    load_plugins()
    assert "independent-cascade" not in _PLUGIN_REGISTRY
    _clean_registry()


def test_registered_plugin_usable_via_make_propagation_model() -> None:
    _clean_registry()
    model = _ConstantModel()
    register_plugin("test-constant", model)

    from agentprop.evaluation.runner import make_propagation_model
    from agentprop.workflows import scaffold_workflow

    resolved = make_propagation_model("test-constant")
    assert resolved is model

    graph = scaffold_workflow(["a", "b", "c"])
    result = resolved.simulate(graph, ["a"])
    assert "a" in result.activated_nodes
    _clean_registry()


def test_unknown_model_error_mentions_load_plugins() -> None:
    _clean_registry()
    from agentprop.evaluation.runner import make_propagation_model

    with pytest.raises(ValueError, match="load_plugins"):
        make_propagation_model("totally-unknown-model-xyz")
