"""Baseline policies for AgentRoutingEnv."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agentprop.rl.env import AgentRoutingEnv, RoutingAction, RoutingState


@dataclass(slots=True)
class NodeScorerRoutingPolicy:
    """Route by selecting the highest-scoring available node from a learned scorer."""

    node_scores: Mapping[str, float]

    def act(self, env: AgentRoutingEnv) -> str:
        """Return the best available seed action according to node scores."""

        actions = env.available_seed_actions()
        if not actions:
            return RoutingAction.STOP.value
        return max(
            sorted(actions),
            key=lambda node_id: float(self.node_scores.get(node_id, -float("inf"))),
        )


class GreedyCoveragePolicy:
    """Choose the action with best immediate reward."""

    def act(self, env: AgentRoutingEnv) -> str:
        """Return the best next action by one-step lookahead."""

        candidate_actions = env.available_seed_actions()
        if not candidate_actions:
            return RoutingAction.STOP.value

        best_action = RoutingAction.STOP.value
        best_reward = float("-inf")
        snapshot = env._selected.copy()

        for action in candidate_actions:
            env._selected = snapshot.copy()
            state, reward, _, _ = env.step(action)
            if reward > best_reward and isinstance(state, RoutingState):
                best_reward = reward
                best_action = action

        env._selected = snapshot
        env._state = env._evaluate(done=False)
        return best_action
