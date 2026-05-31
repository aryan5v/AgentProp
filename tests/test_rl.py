from agentprop.rl import AgentRoutingEnv, GreedyCoveragePolicy, RoutingAction
from agentprop.workflows import planner_coder_tester_reviewer


def test_agent_routing_env_selects_seed_and_stops_at_budget() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=1, trials=5)

    state, reward, done, info = env.step("planner")

    assert done
    assert state.coverage > 0
    assert reward != 0
    assert info["selected_seeds"] == ("planner",)


def test_greedy_coverage_policy_returns_valid_action() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=5)
    action = GreedyCoveragePolicy().act(env)

    assert action in env.action_space
    assert action != RoutingAction.STOP.value
