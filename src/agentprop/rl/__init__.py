"""Reinforcement-learning foundations for sequential routing."""

from agentprop.rl.bandit import BanditArmStats, CategoryBanditRoutingPolicy
from agentprop.rl.checkpointing import (
    RLCheckpointPolicy,
    RLPolicyCheckpoint,
    load_rl_policy,
    save_rl_policy,
)
from agentprop.rl.contextual_thompson import (
    ContextualThompsonSamplingPolicy,
    rloo_advantages,
    shaped_reward,
)
from agentprop.rl.env import (
    AgentRoutingEnv,
    RoutingAction,
    RoutingDecision,
    RoutingState,
    format_routing_action,
    parse_routing_action,
)
from agentprop.rl.feature_policy import (
    FeaturePolicyConfig,
    FeaturePolicyTrainingResult,
    GraphFeaturePolicy,
    train_feature_policy,
)
from agentprop.rl.graph_features import (
    REWARD_RECORD_SCHEMA_VERSION,
    node_position_features,
    reward_record_graph_features,
    workflow_embedding,
)
from agentprop.rl.ope import (
    LoggedDecision,
    OPEResult,
    doubly_robust,
    load_logged_decisions,
    weighted_importance_sampling,
)
from agentprop.rl.policies import GreedyCoveragePolicy, NodeScorerRoutingPolicy
from agentprop.rl.ppo import (
    PPOConfig,
    PPOPolicy,
    PPOTrainingResult,
    train_ppo_policy,
)
from agentprop.rl.q_learning import (
    QLearningConfig,
    QLearningTrainingResult,
    TabularQPolicy,
    train_q_policy,
)
from agentprop.rl.reinforce import (
    ReinforceConfig,
    ReinforcePolicy,
    ReinforceTrainingResult,
    train_reinforce_policy,
)
from agentprop.rl.rewards import (
    RoutingRewardProfile,
    WorkflowControlReward,
    calibrate_routing_reward_profile,
    propagation_reward,
    workflow_control_reward,
)
from agentprop.rl.thompson import (
    GaussianArmPosterior,
    ThompsonSamplingRoutingPolicy,
)
from agentprop.rl.trajectory import (
    RoutingReplayResult,
    RoutingReplayStep,
    actions_from_exported_trajectory,
    replay_actions,
)

__all__ = [
    "AgentRoutingEnv",
    "BanditArmStats",
    "CategoryBanditRoutingPolicy",
    "ContextualThompsonSamplingPolicy",
    "FeaturePolicyConfig",
    "GaussianArmPosterior",
    "ThompsonSamplingRoutingPolicy",
    "FeaturePolicyTrainingResult",
    "GraphFeaturePolicy",
    "GreedyCoveragePolicy",
    "LoggedDecision",
    "NodeScorerRoutingPolicy",
    "OPEResult",
    "PPOConfig",
    "PPOPolicy",
    "PPOTrainingResult",
    "QLearningConfig",
    "REWARD_RECORD_SCHEMA_VERSION",
    "QLearningTrainingResult",
    "ReinforceConfig",
    "ReinforcePolicy",
    "ReinforceTrainingResult",
    "RLCheckpointPolicy",
    "RLPolicyCheckpoint",
    "RoutingReplayResult",
    "RoutingReplayStep",
    "RoutingAction",
    "RoutingDecision",
    "RoutingState",
    "RoutingRewardProfile",
    "actions_from_exported_trajectory",
    "format_routing_action",
    "load_rl_policy",
    "node_position_features",
    "parse_routing_action",
    "reward_record_graph_features",
    "TabularQPolicy",
    "WorkflowControlReward",
    "calibrate_routing_reward_profile",
    "doubly_robust",
    "load_logged_decisions",
    "propagation_reward",
    "replay_actions",
    "rloo_advantages",
    "save_rl_policy",
    "shaped_reward",
    "train_feature_policy",
    "train_ppo_policy",
    "train_q_policy",
    "train_reinforce_policy",
    "weighted_importance_sampling",
    "workflow_control_reward",
    "workflow_embedding",
]
