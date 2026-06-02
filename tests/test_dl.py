import pytest

from agentprop.core import NodeType
from agentprop.dl import GraphEncoderConfig, TorchBackendUnavailable, TorchGNNSeedScorer
from agentprop.dl.encoders import require_torch
from agentprop.dl.torch_gnn import _graph_to_tensors, train_torch_seed_scorer
from agentprop.ml import build_seed_selection_example
from agentprop.workflows import planner_coder_tester_reviewer


def test_graph_encoder_config_has_research_defaults() -> None:
    config = GraphEncoderConfig(input_dim=9)

    assert config.architecture == "graphsage"
    assert config.hidden_dim == 64
    assert config.task == "seed"
    assert config.edge_feature_dim == 9


def test_seed_examples_preserve_edge_features_for_gnn_training() -> None:
    graph = planner_coder_tester_reviewer()
    example = build_seed_selection_example(graph, budget=2, trials=3)

    assert example.edge_features is not None
    assert "message_cost_norm" in example.edge_features.feature_names
    assert "dependency_strength" in example.edge_features.feature_names


def test_require_torch_raises_clear_error_without_torch() -> None:
    try:
        require_torch()
    except TorchBackendUnavailable as exc:
        assert "dependency-light" in str(exc)
    except ImportError:
        pytest.fail("torch import failures should use TorchBackendUnavailable")


def test_torch_gnn_scorer_uses_optional_backend_boundary() -> None:
    config = GraphEncoderConfig(input_dim=9, architecture="graph_transformer")

    try:
        scorer = TorchGNNSeedScorer(config)
    except TorchBackendUnavailable as exc:
        assert "agentprop[dl]" in str(exc)
    else:
        assert scorer.config.architecture == "graph_transformer"


def test_torch_gnn_training_loop_when_torch_is_installed() -> None:
    pytest.importorskip("torch")
    graph = planner_coder_tester_reviewer()
    example = build_seed_selection_example(graph, budget=2, trials=3)

    scorer, result = train_torch_seed_scorer(
        [example],
        config=GraphEncoderConfig(input_dim=9, hidden_dim=8, architecture="graphsage"),
        epochs=2,
    )
    scores = scorer.score_nodes(graph)

    assert result.epochs == 2
    assert set(scores) == {node.id for node in graph.nodes() if node.type != NodeType.OUTPUT}


def test_torch_graph_tensors_include_rich_edge_features_when_torch_is_installed() -> None:
    torch = pytest.importorskip("torch")
    graph = planner_coder_tester_reviewer()

    batch = _graph_to_tensors(torch, graph, edge_feature_dim=9)
    planner = batch["node_ids"].index("planner")
    coder = batch["node_ids"].index("coder")

    assert batch["edge_features"].shape[-1] == 9
    assert batch["edge_features"][planner, coder, 0] > 0
    assert batch["edge_features"][planner, coder, 5] == 1.0


@pytest.mark.parametrize(
    "architecture",
    ["graph_transformer", "heterogeneous", "edge_conditioned"],
)
def test_advanced_torch_architectures_score_when_torch_is_installed(
    architecture: str,
) -> None:
    pytest.importorskip("torch")
    graph = planner_coder_tester_reviewer()
    config = GraphEncoderConfig(input_dim=9, hidden_dim=8, architecture=architecture)

    scorer = TorchGNNSeedScorer(config)
    scores = scorer.score_nodes(graph)

    assert set(scores) == {node.id for node in graph.nodes() if node.type != NodeType.OUTPUT}
