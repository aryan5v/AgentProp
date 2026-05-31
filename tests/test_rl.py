from agentprop.rl import (
    AgentRoutingEnv,
    GreedyCoveragePolicy,
    QLearningConfig,
    RoutingAction,
    TabularQPolicy,
    train_q_policy,
)
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


def test_q_learning_trains_state_action_values() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=3)

    policy, result = train_q_policy(
        env,
        config=QLearningConfig(episodes=10, epsilon=0.4, seed=1),
    )
    env.reset()
    action = policy.act(env)

    assert isinstance(policy, TabularQPolicy)
    assert result.episodes == 10
    assert result.q_value_count > 0
    assert action in env.action_space
    assert action != RoutingAction.STOP.value
