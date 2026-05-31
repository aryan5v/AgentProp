import pytest

from agentprop.dl import GraphEncoderConfig, TorchBackendUnavailable
from agentprop.dl.encoders import require_torch


def test_graph_encoder_config_has_research_defaults() -> None:
    config = GraphEncoderConfig(input_dim=9)

    assert config.architecture == "graphsage"
    assert config.hidden_dim == 64


def test_require_torch_raises_clear_error_without_torch() -> None:
    try:
        require_torch()
    except TorchBackendUnavailable as exc:
        assert "dependency-light" in str(exc)
    except ImportError:
        pytest.fail("torch import failures should use TorchBackendUnavailable")
