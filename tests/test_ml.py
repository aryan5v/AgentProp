from agentprop.ml import (
    LinearEdgeScorer,
    LinearNodeRegressor,
    LinearNodeScorer,
    MLPNodeScorer,
    PairwiseNodeRanker,
    build_edge_pruning_example,
    build_seed_ranking_example,
    build_seed_selection_example,
    build_verifier_placement_example,
    extract_edge_features,
    extract_graph_features,
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


def test_mlp_node_scorer_trains_on_seed_example() -> None:
    graph = planner_coder_tester_reviewer()
    example = build_seed_selection_example(graph, budget=2, trials=5)
    scorer = MLPNodeScorer.initialize(len(example.features.feature_names), hidden_dim=4)

    scorer.train([example], epochs=5, learning_rate=0.05)
    scores = scorer.score_nodes(example.features)

    assert all(0.0 <= score <= 1.0 for score in scores.values())


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
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_verifier_placement_example_labels_nodes() -> None:
    graph = planner_coder_tester_reviewer()
    example = build_verifier_placement_example(graph, budget=2)

    assert len(example.positive_verifiers) == 2
    assert set(example.labels) == {node.id for node in graph.nodes()}
