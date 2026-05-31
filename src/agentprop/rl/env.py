"""A small Gym-like routing environment without a hard Gymnasium dependency."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agentprop.core import AgentGraph, NodeType
from agentprop.evaluation.metrics import seeded_routing_cost
from agentprop.propagation import IndependentCascade, PropagationModel
from agentprop.rl.rewards import propagation_reward


class RoutingAction(StrEnum):
    """Supported V1 routing actions."""

    SELECT_NEXT_SEED_NODE = "SELECT_NEXT_SEED_NODE"
    STOP = "STOP"


@dataclass(slots=True)
class RoutingState:
    """Observable environment state."""

    selected_seeds: tuple[str, ...]
    remaining_budget: int
    coverage: float
    token_cost: float
    message_cost: float
    propagation_time: float
    done: bool


class AgentRoutingEnv:
    """Sequential seed-selection environment for RL experiments."""

    metadata = {"render_modes": ["text"]}

    def __init__(
        self,
        graph: AgentGraph,
        *,
        budget: int,
        propagation_model: PropagationModel | None = None,
        trials: int = 50,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be at least 1")
        self.graph = graph
        self.budget = budget
        self.propagation_model = propagation_model or IndependentCascade(seed=0)
        self.trials = trials
        self._eligible_nodes = [node.id for node in graph.nodes() if node.type != NodeType.OUTPUT]
        self._selected: list[str] = []
        self._state = self._evaluate(done=False)

    @property
    def action_space(self) -> list[str]:
        """Return available action labels for lightweight agents."""

        return [*self.available_seed_actions(), RoutingAction.STOP.value]

    def available_seed_actions(self) -> list[str]:
        """Return node ids that can still be selected."""

        if len(self._selected) >= self.budget:
            return []
        selected = set(self._selected)
        return [node_id for node_id in self._eligible_nodes if node_id not in selected]

    def reset(self) -> RoutingState:
        """Reset selected seeds and return initial state."""

        self._selected = []
        self._state = self._evaluate(done=False)
        return self._state

    def step(self, action: str) -> tuple[RoutingState, float, bool, dict[str, object]]:
        """Apply a node-selection action or STOP."""

        if self._state.done:
            return self._state, 0.0, True, {"reason": "already_done"}

        done = action == RoutingAction.STOP.value
        if not done:
            if action not in self.available_seed_actions():
                raise ValueError(f"Invalid routing action: {action}")
            self._selected.append(action)
            done = len(self._selected) >= self.budget

        self._state = self._evaluate(done=done)
        reward = propagation_reward(
            coverage=self._state.coverage,
            token_cost=self._state.token_cost,
            message_cost=self._state.message_cost,
            propagation_time=self._state.propagation_time,
        )
        return self._state, reward, done, {"selected_seeds": tuple(self._selected)}

    def render(self) -> str:
        """Render state as text."""

        return (
            f"seeds={list(self._state.selected_seeds)} "
            f"coverage={self._state.coverage:.2f} "
            f"remaining_budget={self._state.remaining_budget}"
        )

    def _evaluate(self, *, done: bool) -> RoutingState:
        if self._selected:
            result = self.propagation_model.simulate(
                self.graph,
                self._selected,
                trials=self.trials,
            )
            cost = seeded_routing_cost(self.graph, self._selected, result.activated_nodes)
            coverage = result.coverage
            propagation_time = result.expected_propagation_time or result.propagation_time
        else:
            cost = seeded_routing_cost(self.graph, [], set())
            coverage = 0.0
            propagation_time = 0.0

        return RoutingState(
            selected_seeds=tuple(self._selected),
            remaining_budget=max(self.budget - len(self._selected), 0),
            coverage=coverage,
            token_cost=cost.token_cost,
            message_cost=cost.message_cost,
            propagation_time=float(propagation_time),
            done=done,
        )
