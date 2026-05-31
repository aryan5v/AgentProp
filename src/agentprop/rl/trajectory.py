"""Trajectory import and replay helpers for routing experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentprop.rl.env import AgentRoutingEnv, RoutingState


@dataclass(frozen=True, slots=True)
class RoutingReplayStep:
    """Single replayed action and resulting state metrics."""

    action: str
    reward: float
    coverage: float
    token_cost: float
    message_cost: float
    propagation_time: float
    done: bool
    info: dict[str, object]


@dataclass(frozen=True, slots=True)
class RoutingReplayResult:
    """Summary of a replayed routing trajectory."""

    actions: list[str]
    steps: list[RoutingReplayStep]
    final_state: RoutingState
    total_reward: float
    truncated: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize replay metrics to JSON-compatible data."""

        return {
            "actions": self.actions,
            "steps": [
                {
                    "action": step.action,
                    "reward": step.reward,
                    "coverage": step.coverage,
                    "token_cost": step.token_cost,
                    "message_cost": step.message_cost,
                    "propagation_time": step.propagation_time,
                    "done": step.done,
                    "info": step.info,
                }
                for step in self.steps
            ],
            "final_state": {
                "selected_seeds": list(self.final_state.selected_seeds),
                "activated_verifiers": list(self.final_state.activated_verifiers),
                "used_edges": [list(edge) for edge in self.final_state.used_edges],
                "pruned_edges": [list(edge) for edge in self.final_state.pruned_edges],
                "called_tools": list(self.final_state.called_tools),
                "summary_nodes": list(self.final_state.summary_nodes),
                "remaining_budget": self.final_state.remaining_budget,
                "coverage": self.final_state.coverage,
                "token_cost": self.final_state.token_cost,
                "message_cost": self.final_state.message_cost,
                "propagation_time": self.final_state.propagation_time,
                "done": self.final_state.done,
            },
            "total_reward": self.total_reward,
            "truncated": self.truncated,
        }


def actions_from_exported_trajectory(
    trajectory: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Extract action strings from a JSON trajectory exported by experiments."""

    actions = []
    for index, step in enumerate(trajectory):
        action = step.get("action")
        if not isinstance(action, str):
            raise ValueError(f"Trajectory step {index} is missing a string action")
        actions.append(action)
    return actions


def replay_actions(
    env: AgentRoutingEnv,
    actions: Sequence[str],
    *,
    stop_on_done: bool = True,
) -> RoutingReplayResult:
    """Replay routing actions against an environment."""

    env.reset()
    steps: list[RoutingReplayStep] = []
    total_reward = 0.0
    truncated = False

    for action in actions:
        if env.state.done and stop_on_done:
            truncated = True
            break
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps.append(
            RoutingReplayStep(
                action=action,
                reward=reward,
                coverage=state.coverage,
                token_cost=state.token_cost,
                message_cost=state.message_cost,
                propagation_time=state.propagation_time,
                done=done,
                info=dict(info),
            )
        )

    return RoutingReplayResult(
        actions=list(actions),
        steps=steps,
        final_state=env.state,
        total_reward=total_reward,
        truncated=truncated,
    )
