from agentprop.ml import (
    LinearEdgeScorer,
    LinearNodeRegressor,
    LinearNodeScorer,
    MLPNodeScorer,
    PairwiseNodeRanker,
    build_edge_pruning_example,
    build_empirical_edge_pruning_example,
    build_empirical_routing_example,
    build_seed_ranking_example,
    build_seed_selection_example,
    build_verifier_placement_example,
    extract_edge_features,
    extract_graph_features,
    load_ml_model,
    save_ml_model,
)
from agentprop.workflows import planner_coder_tester_reviewer


def test_extract_graph_features_returns_node_matrix() -> None:
    graph = planner_coder_tester_reviewer()

    features = extract_graph_features(graph)

    assert "planner" in features.node_features
    assert len(features.node_features["planner"]) == len(features.feature_names)


def test_linear_node_scorer_trains_on_seed_example() -> None:
    graph = planner_coder_tester_reviewer()
    example = build_seed_selection_example(graph, budget=2, trials=5)
    scorer = LinearNodeScorer.initialize(len(example.features.feature_names))

    scorer.train([example], epochs=5, learning_rate=0.05)
    scores = scorer.score_nodes(example.features)

    assert set(example.positive_seeds)
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_empirical_routing_example_uses_task_outcome_labels() -> None:
    graph = planner_coder_tester_reviewer()
    success = build_empirical_routing_example(
        graph,
        {
            "task_id": "roman-to-int",
            "policy": "quality_aware_greedy",
            "selected_seeds": ["coder"],
            "context_allocations": {"coder": 1.0, "tester": 0.85},
            "verification_passed": True,
            "cost_adjusted_success": 0.91,
        },
        default_budget=2,
    )
    failure = build_empirical_routing_example(
        graph,
        {
            "task_id": "roman-to-int",
            "policy": "greedy",
            "selected_seeds": ["planner"],
            "context_allocations": {"planner": 1.0, "coder": 0.25},
            "verification_passed": False,
        },
        default_budget=2,
    )
    retryable = build_empirical_routing_example(
        graph,
        {
            "task_id": "browser-flake",
            "selected_seeds": ["planner"],
            "verification_passed": False,
            "retry_recommended": True,
        },
    )

    assert success is not None
    assert success.labels["coder"] == 1.0
    assert success.labels["tester"] == 1.0
    assert success.sample_weight > 1.0
    assert failure is not None
    assert failure.labels["planner"] == 0.0
    assert retryable is None


def test_mlp_node_scorer_trains_on_seed_example() -> None:
    graph = planner_coder_tester_reviewer()
    example = build_seed_selection_example(graph, budget=2, trials=5)
    scorer = MLPNodeScorer.initialize(len(example.features.feature_names), hidden_dim=4)
    initial_input_weights = [row.copy() for row in scorer.input_weights]
    initial_hidden_bias = scorer.hidden_bias.copy()

    scorer.train([example], epochs=10, learning_rate=0.05, l2_penalty=0.001)
    scores = scorer.score_nodes(example.features)

    assert all(0.0 <= score <= 1.0 for score in scores.values())
    assert scorer.input_weights != initial_input_weights
    assert scorer.hidden_bias != initial_hidden_bias


def test_seed_ranking_example_builds_preferences_and_targets() -> None:
    graph = planner_coder_tester_reviewer()
    example = build_seed_ranking_example(graph, budget=2, trials=5)

    assert example.seed_candidates
    assert "final" not in example.seed_candidates
    assert set(example.seed_candidates).issubset(example.utility_targets)
    assert all(winner != loser for winner, loser in example.preference_pairs)


def test_pairwise_ranker_trains_on_seed_preferences() -> None:
    graph = planner_coder_tester_reviewer()
    example = build_seed_ranking_example(graph, budget=2, trials=5)
    scorer = PairwiseNodeRanker.initialize(len(example.features.feature_names))

    scorer.train([example], epochs=5, learning_rate=0.05)
    scores = scorer.score_nodes(example.features)

    assert set(scores) == set(example.features.node_features)
    assert any(score != 0.0 for score in scores.values())


def test_linear_node_regressor_trains_on_marginal_gain_targets() -> None:
    graph = planner_coder_tester_reviewer()
    example = build_seed_ranking_example(graph, budget=2, trials=5)
    scorer = LinearNodeRegressor.initialize(len(example.features.feature_names))

    scorer.train([example], epochs=5, learning_rate=0.05)
    scores = scorer.score_nodes(example.features)

    assert set(scores) == set(example.features.node_features)
    assert any(score != 0.0 for score in scores.values())


def test_edge_features_and_scorer_train_on_pruning_example() -> None:
    graph = planner_coder_tester_reviewer()
    features = extract_edge_features(graph)
    example = build_edge_pruning_example(graph, fraction=0.3)
    scorer = LinearEdgeScorer.initialize(len(features.feature_names))

    scorer.train([example], epochs=5, learning_rate=0.05)
    scores = scorer.score_edges(features)

    assert example.positive_edges
    assert example.features.feature_names == features.feature_names
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_empirical_edge_pruning_example_uses_task_outcome_labels() -> None:
    graph = planner_coder_tester_reviewer()
    success = build_empirical_edge_pruning_example(
        graph,
        {
            "task_id": "roman-to-int",
            "policy": "rl_ppo",
            "pruned_edges": [["planner", "reviewer"]],
            "verification_passed": True,
            "cost_adjusted_success": 0.8,
        },
    )
    failure = build_empirical_edge_pruning_example(
        graph,
        {
            "task_id": "roman-to-int",
            "policy": "rl_ppo",
            "pruned_edges": [{"source": "coder", "target": "tester"}],
            "verification_passed": False,
        },
    )
    retryable = build_empirical_edge_pruning_example(
        graph,
        {
            "task_id": "browser-flake",
            "pruned_edges": [["planner", "reviewer"]],
            "verification_passed": False,
            "retry_recommended": True,
        },
    )

    assert success is not None
    assert success.labels[("planner", "reviewer")] == 1.0
    assert success.sample_weight > 1.0
    assert failure is not None
    assert failure.labels[("coder", "tester")] == 0.0
    assert retryable is None


def test_verifier_placement_example_labels_nodes() -> None:
    graph = planner_coder_tester_reviewer()
    example = build_verifier_placement_example(graph, budget=2)

    assert len(example.positive_verifiers) == 2
    assert set(example.labels) == {node.id for node in graph.nodes()}


def test_ml_model_checkpoint_round_trips_scores(tmp_path) -> None:  # type: ignore[no-untyped-def]
    graph = planner_coder_tester_reviewer()
    example = build_seed_selection_example(graph, budget=2, trials=5)
    scorer = MLPNodeScorer.initialize(len(example.features.feature_names), hidden_dim=4)
    scorer.train([example], epochs=5, learning_rate=0.05)
    before = scorer.score_nodes(example.features)

    path = save_ml_model(
        scorer,
        tmp_path / "mlp_checkpoint.json",
        metadata={"workflow": "planner_coder_tester_reviewer"},
    )
    loaded = load_ml_model(path)

    assert loaded.metadata["workflow"] == "planner_coder_tester_reviewer"
    assert isinstance(loaded.model, MLPNodeScorer)
    assert loaded.model.score_nodes(example.features) == before
