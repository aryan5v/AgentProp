from agentprop.evaluation import ExpectedSuccessProfile
from agentprop.rl import (
    AgentRoutingEnv,
    CategoryBanditRoutingPolicy,
    FeaturePolicyConfig,
    GraphFeaturePolicy,
    GreedyCoveragePolicy,
    PPOConfig,
    PPOPolicy,
    QLearningConfig,
    ReinforceConfig,
    ReinforcePolicy,
    RoutingAction,
    RoutingRewardProfile,
    TabularQPolicy,
    actions_from_exported_trajectory,
    calibrate_routing_reward_profile,
    format_routing_action,
    load_rl_policy,
    parse_routing_action,
    replay_actions,
    save_rl_policy,
    train_feature_policy,
    train_ppo_policy,
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


def test_routing_env_uses_calibrated_reward_profile() -> None:
    graph = planner_coder_tester_reviewer()
    default_env = AgentRoutingEnv(graph, budget=1, trials=3)
    calibrated_env = AgentRoutingEnv(
        graph,
        budget=1,
        trials=3,
        reward_profile=RoutingRewardProfile(token_cost_weight=0.01, source="test"),
    )

    _, default_reward, _, _ = default_env.step("planner")
    _, calibrated_reward, _, _ = calibrated_env.step("planner")

    assert calibrated_reward < default_reward
    assert calibrated_env.reward_profile.source == "test"


def test_routing_env_can_reward_empirical_expected_success() -> None:
    graph = planner_coder_tester_reviewer()
    profile = ExpectedSuccessProfile(
        default_success=0.95,
        node_context_penalties={"coder": 0.8},
        high_context_threshold=0.95,
        example_count=10,
    )
    env = AgentRoutingEnv(graph, budget=1, trials=3, success_profile=profile)

    state, reward, done, info = env.step("planner")

    assert done
    assert state.expected_success is not None
    assert state.expected_success < profile.default_success
    assert info["reward_target"] == "expected_success"
    assert info["reward_quality"] == state.expected_success
    assert env.observation()["expected_success"] == state.expected_success
    assert reward == info["propagation_reward"] + info["control_reward"]["total"]


def test_calibrate_routing_reward_profile_from_empirical_rows() -> None:
    profile = calibrate_routing_reward_profile(
        [
            {
                "verification_passed": True,
                "token_cost": 1000,
                "message_cost": 200,
                "latency": 10,
            },
            {
                "verification_passed": False,
                "token_cost": 3000,
                "message_cost": 400,
                "latency": 30,
            },
            {
                "verification_passed": False,
                "token_cost": 100,
                "retry_recommended": True,
            },
        ]
    )

    assert profile.source == "empirical"
    assert profile.example_count == 2
    assert profile.token_cost_weight > 0
    assert profile.time_weight > 0


def test_category_bandit_updates_routing_policy_by_task_type() -> None:
    policy = CategoryBanditRoutingPolicy(epsilon=0.0)

    policy.update("edge-case-heavy", "quality-aware-greedy", passed=True, token_savings=0.10)
    policy.update("edge-case-heavy", "cost-aware-greedy", passed=False, token_savings=0.35)

    assert policy.choose("edge-case-heavy") == "quality-aware-greedy"
    assert policy.values("edge-case-heavy")["quality-aware-greedy"] > 0


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


def test_expanded_control_actions_report_reward_terms() -> None:
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

    _, verifier_reward, _, verifier_info = env.step(verifier_action)
    _, prune_reward, _, prune_info = env.step(prune_action)

    verifier_terms = verifier_info["control_reward"]
    prune_terms = prune_info["control_reward"]
    assert isinstance(verifier_terms, dict)
    assert isinstance(prune_terms, dict)
    assert verifier_terms["verifier_bonus"] > 0
    assert prune_terms["safe_pruning_bonus"] > 0
    assert prune_terms["risky_pruning_penalty"] > 0
    assert verifier_reward == verifier_info["propagation_reward"] + verifier_terms["total"]
    assert prune_reward == prune_info["propagation_reward"] + prune_terms["total"]


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


def test_ppo_trains_preferences_and_values() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=3)

    policy, result = train_ppo_policy(
        env,
        config=PPOConfig(episodes=8, learning_rate=0.05, seed=1, max_steps=5),
    )
    env.reset()
    action = policy.act(env)

    assert isinstance(policy, PPOPolicy)
    assert result.episodes == 8
    assert result.preference_count > 0
    assert result.value_count > 0
    assert action in env.action_space
    assert action != RoutingAction.STOP.value


def test_ppo_can_train_with_expanded_actions() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=2)

    policy, result = train_ppo_policy(
        env,
        config=PPOConfig(
            episodes=3,
            learning_rate=0.05,
            seed=2,
            expanded_actions=True,
            max_steps=5,
        ),
    )

    assert policy.expanded_actions
    assert result.preference_count > 0
    assert result.value_count > 0


def test_graph_feature_policy_trains_transferable_action_scores() -> None:
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=3)

    policy, result = train_feature_policy(
        env,
        config=FeaturePolicyConfig(episodes=8, learning_rate=0.05, epsilon=0.3, seed=1),
    )
    env.reset()
    action = policy.act(env)

    assert isinstance(policy, GraphFeaturePolicy)
    assert result.episodes == 8
    assert result.feature_count == len(policy.feature_names)
    assert result.feature_count > 0
    assert any(weight != 0.0 for weight in policy.weights)
    assert "current_coverage" in policy.feature_names
    assert action in env.action_space
    assert action != RoutingAction.STOP.value


def test_rl_policy_checkpoint_round_trips_action(tmp_path) -> None:  # type: ignore[no-untyped-def]
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=3)
    policy, _ = train_ppo_policy(
        env,
        config=PPOConfig(episodes=8, learning_rate=0.05, seed=1, max_steps=5),
    )
    env.reset()
    before = policy.act(env)

    path = save_rl_policy(
        policy,
        tmp_path / "ppo_policy.json",
        metadata={"workflow": "planner_coder_tester_reviewer"},
    )
    loaded = load_rl_policy(path)

    assert loaded.metadata["workflow"] == "planner_coder_tester_reviewer"
    assert isinstance(loaded.policy, PPOPolicy)
    assert loaded.policy.values == policy.values
    assert loaded.policy.act(env) == before


def test_graph_feature_policy_checkpoint_round_trips_action(tmp_path) -> None:  # type: ignore[no-untyped-def]
    graph = planner_coder_tester_reviewer()
    env = AgentRoutingEnv(graph, budget=2, trials=3)
    policy, _ = train_feature_policy(
        env,
        config=FeaturePolicyConfig(episodes=8, learning_rate=0.05, epsilon=0.3, seed=1),
    )
    env.reset()
    before = policy.act(env)

    path = save_rl_policy(
        policy,
        tmp_path / "feature_policy.json",
        metadata={"workflow": "planner_coder_tester_reviewer"},
    )
    loaded = load_rl_policy(path)

    assert loaded.metadata["workflow"] == "planner_coder_tester_reviewer"
    assert isinstance(loaded.policy, GraphFeaturePolicy)
    assert loaded.policy.feature_names == policy.feature_names
    assert loaded.policy.weights == policy.weights
    assert loaded.policy.act(env) == before


def test_actions_from_exported_trajectory_extracts_actions() -> None:
    actions = actions_from_exported_trajectory(
        [
            {"action": "planner", "coverage": 0.5},
            {"action": "coder", "coverage": 0.75},
        ]
    )

    assert actions == ["planner", "coder"]


def test_replay_actions_recreates_expanded_routing_state() -> None:
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

    replay = replay_actions(env, [verifier_action, prune_action, "planner"])

    assert replay.final_state.activated_verifiers == ("tester",)
    assert replay.final_state.pruned_edges == (("planner", "reviewer"),)
    assert replay.final_state.selected_seeds == ("planner",)
    assert replay.total_reward != 0.0
    assert not replay.truncated
