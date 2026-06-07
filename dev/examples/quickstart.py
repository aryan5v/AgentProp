"""AgentProp quickstart example with expected output shape."""

from __future__ import annotations

from agentprop.algorithms import greedy_seed_selection
from agentprop.evaluation import compare_routing
from agentprop.propagation import IndependentCascade
from agentprop.workflows import planner_coder_tester_reviewer


def main() -> None:
    graph = planner_coder_tester_reviewer()
    model = IndependentCascade(seed=0)
    seeds = greedy_seed_selection(graph, 2, propagation_model=model, trials=20)
    result = model.simulate(graph, seeds, trials=20)
    report = compare_routing(graph, seeds, model.name, result)

    print(f"seeds={seeds}")
    print(f"coverage={report.propagation.coverage:.3f}")
    print(f"savings={report.estimated_savings:.3f}")


if __name__ == "__main__":
    main()
