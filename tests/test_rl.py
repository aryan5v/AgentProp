from agentprop.rl import (
    AgentRoutingEnv,
    GreedyCoveragePolicy,
    QLearningConfig,
    ReinforceConfig,
    ReinforcePolicy,
    RoutingAction,
    TabularQPolicy,
    format_routing_action,
    parse_routing_action,
    train_q_policy,
    train_reinforce_policy,
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


def test_agent_routing_env_supports_expanded_control_actions() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=3)

    verifier_action = format_routing_action(
        RoutingAction.ACTIVATE_VERIFIER,
        node_id="tester",
    )
    prune_action = format_routing_action(
        RoutingAction.PRUNE_EDGE,
        edge=("planner", "reviewer"),
    )
    state, _, done, info = env.step(verifier_action)
    next_state, _, _, _ = env.step(prune_action)

    assert not done
    assert state.activated_verifiers == ("tester",)
    assert next_state.pruned_edges == (("planner", "reviewer"),)
    assert info["action_type"] == RoutingAction.ACTIVATE_VERIFIER.value


def test_agent_routing_env_has_gymnasium_style_surface() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=1, trials=3)

    observation, info = env.reset_gymnasium(seed=7)
    next_observation, reward, terminated, truncated, step_info = env.step_gymnasium("planner")

    assert observation["remaining_budget"] == 1
    assert next_observation["selected_seeds"] == ["planner"]
    assert reward != 0
    assert terminated
    assert not truncated
    assert info["seed"] == 7
    assert step_info["selected_seeds"] == ("planner",)


def test_parse_routing_action_supports_edge_payloads() -> None:
    decision = parse_routing_action("PRUNE_EDGE:planner->reviewer")

    assert decision.action_type == RoutingAction.PRUNE_EDGE
    assert decision.edge == ("planner", "reviewer")


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


def test_q_learning_can_train_with_expanded_actions() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=2)

    policy, result = train_q_policy(
        env,
        config=QLearningConfig(episodes=3, epsilon=0.5, seed=2, expanded_actions=True),
    )

    assert policy.expanded_actions
    assert result.q_value_count > 0


def test_reinforce_trains_state_action_preferences() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=3)

    policy, result = train_reinforce_policy(
        env,
        config=ReinforceConfig(episodes=8, learning_rate=0.1, seed=1),
    )
    env.reset()
    action = policy.act(env)

    assert isinstance(policy, ReinforcePolicy)
    assert result.episodes == 8
    assert result.preference_count > 0
    assert action in env.action_space
    assert action != RoutingAction.STOP.value


def test_reinforce_can_train_with_expanded_actions() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=2)

    policy, result = train_reinforce_policy(
        env,
        config=ReinforceConfig(
            episodes=3,
            learning_rate=0.05,
            seed=2,
            expanded_actions=True,
            max_steps=5,
        ),
    )

    assert policy.expanded_actions
    assert result.preference_count > 0
