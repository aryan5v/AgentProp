"""A small Gym-like routing environment without a hard Gymnasium dependency."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agentprop.core import AgentGraph, NodeType
from agentprop.evaluation import (
    ContextCompressionProfile,
    ExpectedSuccessProfile,
    estimate_expected_success,
    graded_context_allocations,
)
from agentprop.evaluation.metrics import seeded_routing_cost
from agentprop.propagation import IndependentCascade, PropagationModel
from agentprop.rl.rewards import (
    RoutingRewardProfile,
    WorkflowControlReward,
    propagation_reward,
    workflow_control_reward,
)


class RoutingAction(StrEnum):
    """Supported V1 routing actions."""

    SELECT_NEXT_SEED_NODE = "SELECT_NEXT_SEED_NODE"
    SEND_CONTEXT = "SEND_CONTEXT"
    ACTIVATE_VERIFIER = "ACTIVATE_VERIFIER"
    SEND_MESSAGE = "SEND_MESSAGE"
    PRUNE_EDGE = "PRUNE_EDGE"
    CALL_TOOL = "CALL_TOOL"
    REQUEST_SUMMARY = "REQUEST_SUMMARY"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Parsed routing action."""

    action_type: RoutingAction
    node_id: str | None = None
    edge: tuple[str, str] | None = None


@dataclass(slots=True)
class RoutingState:
    """Observable environment state."""

    selected_seeds: tuple[str, ...]
    activated_verifiers: tuple[str, ...]
    used_edges: tuple[tuple[str, str], ...]
    pruned_edges: tuple[tuple[str, str], ...]
    called_tools: tuple[str, ...]
    summary_nodes: tuple[str, ...]
    remaining_budget: int
    coverage: float
    token_cost: float
    message_cost: float
    propagation_time: float
    expected_success: float | None
    context_ratios: dict[str, float]
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
        reward_profile: RoutingRewardProfile | None = None,
        success_profile: ExpectedSuccessProfile | None = None,
        context_profile: ContextCompressionProfile | None = None,
        trials: int = 50,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be at least 1")
        self.graph = graph
        self.budget = budget
        self.propagation_model = propagation_model or IndependentCascade(seed=0)
        self.reward_profile = reward_profile or RoutingRewardProfile()
        self.success_profile = success_profile
        self.context_profile = context_profile
        self.trials = trials
        self._eligible_nodes = [node.id for node in graph.nodes() if node.type != NodeType.OUTPUT]
        self._selected: list[str] = []
        self._activated_verifiers: list[str] = []
        self._used_edges: list[tuple[str, str]] = []
        self._pruned_edges: list[tuple[str, str]] = []
        self._called_tools: list[str] = []
        self._summary_nodes: list[str] = []
        self._state = self._evaluate(done=False)

    @property
    def action_space(self) -> list[str]:
        """Return available action labels for lightweight agents."""

        return [*self.available_seed_actions(), RoutingAction.STOP.value]

    @property
    def state(self) -> RoutingState:
        """Return the current environment state."""

        return self._state

    def available_seed_actions(self) -> list[str]:
        """Return node ids that can still be selected."""

        if len(self._selected) >= self.budget:
            return []
        selected = set(self._selected)
        return [node_id for node_id in self._eligible_nodes if node_id not in selected]

    def available_control_actions(self) -> list[str]:
        """Return expanded workflow-control actions."""

        actions = [
            format_routing_action(RoutingAction.SEND_CONTEXT, node_id=node_id)
            for node_id in self.available_seed_actions()
        ]
        actions.extend(
            format_routing_action(RoutingAction.ACTIVATE_VERIFIER, node_id=node.id)
            for node in self.graph.nodes()
            if node.type == NodeType.VERIFIER and node.id not in self._activated_verifiers
        )
        actions.extend(
            format_routing_action(RoutingAction.SEND_MESSAGE, edge=(edge.source, edge.target))
            for edge in self.graph.edges()
            if (edge.source, edge.target) not in self._used_edges
            and (edge.source, edge.target) not in self._pruned_edges
        )
        actions.extend(
            format_routing_action(RoutingAction.PRUNE_EDGE, edge=(edge.source, edge.target))
            for edge in self.graph.edges()
            if (edge.source, edge.target) not in self._pruned_edges
        )
        actions.extend(
            format_routing_action(RoutingAction.CALL_TOOL, node_id=node.id)
            for node in self.graph.nodes()
            if node.type == NodeType.TOOL and node.id not in self._called_tools
        )
        actions.extend(
            format_routing_action(RoutingAction.REQUEST_SUMMARY, node_id=node_id)
            for node_id in self._eligible_nodes
            if node_id not in self._summary_nodes
        )
        actions.append(RoutingAction.STOP.value)
        return actions

    def reset(self) -> RoutingState:
        """Reset selected seeds and return initial state."""

        self._selected = []
        self._activated_verifiers = []
        self._used_edges = []
        self._pruned_edges = []
        self._called_tools = []
        self._summary_nodes = []
        self._state = self._evaluate(done=False)
        return self._state

    def reset_gymnasium(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Gymnasium-style reset without requiring Gymnasium as a dependency."""

        state = self.reset()
        return self.observation(state), {"seed": seed, "options": options or {}}

    def step(self, action: str) -> tuple[RoutingState, float, bool, dict[str, object]]:
        """Apply a routing action."""

        if self._state.done:
            return self._state, 0.0, True, {"reason": "already_done"}

        decision = parse_routing_action(action, graph=self.graph)
        done = decision.action_type == RoutingAction.STOP
        if not done:
            self._apply_decision(decision)
            done = len(self._selected) >= self.budget

        self._state = self._evaluate(done=done)
        reward_quality = (
            self._state.expected_success
            if self._state.expected_success is not None
            else self._state.coverage
        )
        propagation_component = propagation_reward(
            coverage=reward_quality,
            token_cost=self._state.token_cost,
            message_cost=self._state.message_cost,
            propagation_time=self._state.propagation_time,
            coverage_weight=self.reward_profile.coverage_weight,
            token_cost_weight=self.reward_profile.token_cost_weight,
            message_cost_weight=self.reward_profile.message_cost_weight,
            time_weight=self.reward_profile.time_weight,
        )
        control_component = self._control_reward(decision)
        reward = propagation_component + control_component.total
        info = self._info(decision)
        info["propagation_reward"] = propagation_component
        info["reward_quality"] = reward_quality
        info["reward_target"] = "expected_success" if self.success_profile else "coverage"
        info["control_reward"] = control_component.to_dict()
        return self._state, reward, done, info

    def step_gymnasium(
        self,
        action: str,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        """Gymnasium-style step returning observation, reward, terminated, truncated, info."""

        state, reward, done, info = self.step(action)
        return self.observation(state), reward, done, False, info

    def observation(self, state: RoutingState | None = None) -> dict[str, object]:
        """Return a dictionary observation for Gymnasium-style agents."""

        current = state or self._state
        return {
            "selected_seeds": list(current.selected_seeds),
            "activated_verifiers": list(current.activated_verifiers),
            "used_edges": [list(edge) for edge in current.used_edges],
            "pruned_edges": [list(edge) for edge in current.pruned_edges],
            "called_tools": list(current.called_tools),
            "summary_nodes": list(current.summary_nodes),
            "remaining_budget": current.remaining_budget,
            "coverage": current.coverage,
            "token_cost": current.token_cost,
            "message_cost": current.message_cost,
            "propagation_time": current.propagation_time,
            "expected_success": current.expected_success,
            "context_ratios": dict(current.context_ratios),
            "done": current.done,
        }

    def render(self) -> str:
        """Render state as text."""

        return (
            f"seeds={list(self._state.selected_seeds)} "
            f"coverage={self._state.coverage:.2f} "
            f"remaining_budget={self._state.remaining_budget}"
        )

    def _evaluate(self, *, done: bool) -> RoutingState:
        active_seeds = [*self._selected, *self._activated_verifiers]
        effective_graph = _without_edges(self.graph, self._pruned_edges)
        activated_nodes: set[str] = set()
        if active_seeds:
            result = self.propagation_model.simulate(
                effective_graph,
                active_seeds,
                trials=self.trials,
            )
            activated_nodes = set(result.activated_nodes)
            cost = seeded_routing_cost(effective_graph, active_seeds, result.activated_nodes)
            coverage = result.coverage
            propagation_time = result.expected_propagation_time or result.propagation_time
        else:
            cost = seeded_routing_cost(effective_graph, [], set())
            coverage = 0.0
            propagation_time = 0.0
        context_ratios = graded_context_allocations(
            effective_graph,
            seeds=active_seeds,
            activated_nodes=activated_nodes,
            profile=self.context_profile,
        )
        expected_success = (
            estimate_expected_success(
                effective_graph,
                context_ratios=context_ratios,
                profile=self.success_profile,
            )
            if self.success_profile is not None
            else None
        )

        manual_edge_cost = sum(
            self.graph.edge(source, target).message_cost
            for source, target in self._used_edges
            if self.graph.to_networkx().has_edge(source, target)
        )
        tool_cost = sum(self.graph.node(node_id).token_cost for node_id in self._called_tools)
        summary_cost = 0.1 * sum(
            self.graph.node(node_id).token_cost for node_id in self._summary_nodes
        )

        return RoutingState(
            selected_seeds=tuple(self._selected),
            activated_verifiers=tuple(self._activated_verifiers),
            used_edges=tuple(self._used_edges),
            pruned_edges=tuple(self._pruned_edges),
            called_tools=tuple(self._called_tools),
            summary_nodes=tuple(self._summary_nodes),
            remaining_budget=max(self.budget - len(self._selected), 0),
            coverage=coverage,
            token_cost=cost.token_cost + tool_cost + summary_cost,
            message_cost=cost.message_cost + manual_edge_cost,
            propagation_time=float(propagation_time),
            expected_success=expected_success,
            context_ratios=context_ratios,
            done=done,
        )

    def _apply_decision(self, decision: RoutingDecision) -> None:
        context_actions = {RoutingAction.SELECT_NEXT_SEED_NODE, RoutingAction.SEND_CONTEXT}
        if decision.action_type in context_actions:
            if decision.node_id not in self.available_seed_actions():
                raise ValueError(f"Invalid routing action: {decision.node_id}")
            self._selected.append(str(decision.node_id))
            return
        if decision.action_type == RoutingAction.ACTIVATE_VERIFIER:
            node_id = _required_node(decision)
            if node_id not in self._activated_verifiers:
                self._activated_verifiers.append(node_id)
            return
        if decision.action_type == RoutingAction.SEND_MESSAGE:
            edge = _required_edge(decision)
            if not self.graph.to_networkx().has_edge(*edge):
                raise ValueError(f"Unknown edge: {edge[0]} -> {edge[1]}")
            if edge not in self._used_edges:
                self._used_edges.append(edge)
            return
        if decision.action_type == RoutingAction.PRUNE_EDGE:
            edge = _required_edge(decision)
            if not self.graph.to_networkx().has_edge(*edge):
                raise ValueError(f"Unknown edge: {edge[0]} -> {edge[1]}")
            if edge not in self._pruned_edges:
                self._pruned_edges.append(edge)
            return
        if decision.action_type == RoutingAction.CALL_TOOL:
            node_id = _required_node(decision)
            if self.graph.node(node_id).type != NodeType.TOOL:
                raise ValueError(f"Node is not a tool: {node_id}")
            if node_id not in self._called_tools:
                self._called_tools.append(node_id)
            return
        if decision.action_type == RoutingAction.REQUEST_SUMMARY:
            node_id = _required_node(decision)
            if node_id not in self._summary_nodes:
                self._summary_nodes.append(node_id)
            return
        raise ValueError(f"Unsupported routing action: {decision.action_type}")

    def _info(self, decision: RoutingDecision) -> dict[str, object]:
        return {
            "action_type": decision.action_type.value,
            "selected_seeds": tuple(self._selected),
            "activated_verifiers": tuple(self._activated_verifiers),
            "used_edges": tuple(self._used_edges),
            "pruned_edges": tuple(self._pruned_edges),
            "called_tools": tuple(self._called_tools),
            "summary_nodes": tuple(self._summary_nodes),
        }

    def _control_reward(self, decision: RoutingDecision) -> WorkflowControlReward:
        if decision.action_type == RoutingAction.ACTIVATE_VERIFIER:
            node = self.graph.node(_required_node(decision))
            return workflow_control_reward(
                activated_verifier_risk=(1.0 - node.reliability) + node.error_rate,
                verifier_weight=self.reward_profile.verifier_weight,
            )
        if decision.action_type == RoutingAction.PRUNE_EDGE:
            edge = self.graph.edge(*_required_edge(decision))
            safe_savings = edge.message_cost * max(1.0 - edge.relevance, 0.0)
            risky_exposure = edge.relevance * edge.dependency_strength * edge.reliability
            return workflow_control_reward(
                safe_pruning_savings=safe_savings,
                risky_pruning_exposure=risky_exposure,
                safe_pruning_weight=self.reward_profile.safe_pruning_weight,
                risky_pruning_weight=self.reward_profile.risky_pruning_weight,
            )
        if decision.action_type == RoutingAction.CALL_TOOL:
            node = self.graph.node(_required_node(decision))
            return workflow_control_reward(
                tool_reliability_gain=max(node.reliability - node.error_rate, 0.0),
                tool_weight=self.reward_profile.tool_weight,
            )
        if decision.action_type == RoutingAction.REQUEST_SUMMARY:
            node = self.graph.node(_required_node(decision))
            importance = node.importance_score or 0.0
            return workflow_control_reward(
                summary_token_savings=0.9 * node.token_cost * max(1.0 - importance, 0.0),
                summary_weight=self.reward_profile.summary_weight,
            )
        return WorkflowControlReward()

def format_routing_action(
    action_type: RoutingAction,
    *,
    node_id: str | None = None,
    edge: tuple[str, str] | None = None,
) -> str:
    """Format a structured routing action as a stable string."""

    if action_type == RoutingAction.STOP:
        return RoutingAction.STOP.value
    if edge is not None:
        return f"{action_type.value}:{edge[0]}->{edge[1]}"
    if node_id is not None:
        return f"{action_type.value}:{node_id}"
    raise ValueError("node_id or edge is required for non-STOP actions")


def parse_routing_action(action: str, *, graph: AgentGraph | None = None) -> RoutingDecision:
    """Parse raw seed-node actions and structured routing actions."""

    if action == RoutingAction.STOP.value:
        return RoutingDecision(action_type=RoutingAction.STOP)
    if ":" not in action:
        return RoutingDecision(action_type=RoutingAction.SELECT_NEXT_SEED_NODE, node_id=action)
    raw_type, payload = action.split(":", 1)
    action_type = RoutingAction(raw_type)
    if action_type in {RoutingAction.SEND_MESSAGE, RoutingAction.PRUNE_EDGE}:
        if "->" not in payload:
            raise ValueError(f"Edge action must use source->target payload: {action}")
        source, target = payload.split("->", 1)
        return RoutingDecision(action_type=action_type, edge=(source, target))
    if graph is not None and payload not in graph.to_networkx():
        raise ValueError(f"Unknown node: {payload}")
    return RoutingDecision(action_type=action_type, node_id=payload)


def _required_node(decision: RoutingDecision) -> str:
    if decision.node_id is None:
        raise ValueError(f"Action requires a node id: {decision.action_type}")
    return decision.node_id


def _required_edge(decision: RoutingDecision) -> tuple[str, str]:
    if decision.edge is None:
        raise ValueError(f"Action requires an edge: {decision.action_type}")
    return decision.edge


def _without_edges(graph: AgentGraph, edges_to_remove: list[tuple[str, str]]) -> AgentGraph:
    if not edges_to_remove:
        return graph
    data = graph.to_dict()
    blocked = set(edges_to_remove)
    data["edges"] = [
        edge
        for edge in data["edges"]
        if (str(edge["source"]), str(edge["target"])) not in blocked
    ]
    return AgentGraph.from_dict(data)
