from agentprop.integrations import graph_from_trace_dict
from agentprop.propagation import (
    LearnedPropagation,
    fit_learned_propagation_from_graph,
    fit_learned_propagation_from_trace_dicts,
)


def test_learned_propagation_fits_trace_probabilities() -> None:
    trace = {
        "events": [
            {"source": "planner", "target": "coder", "success": True, "token_cost": 10},
            {"source": "planner", "target": "coder", "success": True, "token_cost": 10},
            {"source": "planner", "target": "reviewer", "success": False, "token_cost": 10},
            {"source": "coder", "target": "tester", "success": True, "token_cost": 10},
        ]
    }

    fit = fit_learned_propagation_from_trace_dicts([trace], smoothing=0.5)

    assert fit.edge_probabilities[("planner", "coder")] > fit.edge_probabilities[
        ("planner", "reviewer")
    ]


def test_learned_propagation_simulates_from_trace_graph() -> None:
    trace = {
        "events": [
            {"source": "planner", "target": "coder", "success": True, "token_cost": 10},
            {"source": "coder", "target": "tester", "success": True, "token_cost": 10},
        ]
    }
    graph = graph_from_trace_dict(trace).graph
    model = LearnedPropagation.fit_from_trace_dicts([trace], seed=0)

    result = model.simulate(graph, ["planner"], trials=5)

    assert result.coverage >= 2 / 3
    assert result.expected_propagation_time is not None


def test_learned_propagation_fits_graph_trace_metadata() -> None:
    trace = {
        "events": [
            {"source": "planner", "target": "coder", "success": True, "token_cost": 10},
            {"source": "planner", "target": "coder", "success": False, "token_cost": 10},
        ]
    }
    graph = graph_from_trace_dict(trace).graph

    fit = fit_learned_propagation_from_graph(graph)

    assert ("planner", "coder") in fit.edge_probabilities
