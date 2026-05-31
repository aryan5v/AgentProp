from agentprop.ml import LinearNodeScorer, build_seed_selection_example, extract_graph_features
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
