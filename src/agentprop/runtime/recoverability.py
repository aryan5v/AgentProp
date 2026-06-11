"""Forward-simulation cascade-risk advisor for runtime control decisions.

Bridges the analysis layer and the controller: instead of reacting to error
counts alone, the advisor simulates how a failure at the current node would
propagate through the workflow graph and converts the expected downstream
impact into an advisory escalation. The :class:`StoppingController` thresholds
stay authoritative; the advisor upgrades a borderline CONTINUE into
FORCE_VERIFY (or recommends SWITCH_STRATEGY) when forward simulation says a
failure here would cascade widely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentprop.core import AgentGraph
from agentprop.evaluation.intervals import ConfidenceInterval, bootstrap_mean_interval
from agentprop.propagation.independent_cascade import IndependentCascade
from agentprop.runtime.control_loop import ControlDecision, ExecutionStateFeatures


@dataclass(frozen=True, slots=True)
class CascadeRiskEstimate:
    """Expected downstream impact of a failure at one node."""

    node_id: str
    impact: ConfidenceInterval
    """Fraction of the workflow expected to be reached by a cascade from the node."""
    downstream_nodes: int
    trials: int

    def to_dict(self) -> dict[str, object]:
        """Serialize for trace rows and reports."""

        return {
            "node_id": self.node_id,
            "impact": self.impact.to_dict(),
            "downstream_nodes": self.downstream_nodes,
            "trials": self.trials,
        }


def estimate_cascade_risk(
    graph: AgentGraph,
    node_id: str,
    *,
    trials: int = 50,
    seed: int = 0,
) -> CascadeRiskEstimate:
    """Simulate failure cascades from ``node_id`` and return expected impact.

    Uses Independent Cascade with the graph's edge activation probabilities as
    the failure-spread model: an error in a node's output propagates to a
    consumer with the probability that consumer actually activates on the
    producer's messages.
    """

    if not any(node.id == node_id for node in graph.nodes()):
        raise ValueError(f"unknown node: {node_id}")
    node_count = max(graph.node_count, 1)
    impacts = [
        len(
            IndependentCascade(seed=seed + trial)
            .simulate(graph, [node_id], trials=1)
            .activated_nodes
        )
        / node_count
        for trial in range(max(trials, 1))
    ]
    interval = bootstrap_mean_interval(impacts, seed=seed)
    reachable = _reachable_count(graph, node_id)
    return CascadeRiskEstimate(
        node_id=node_id,
        impact=interval,
        downstream_nodes=reachable,
        trials=max(trials, 1),
    )


@dataclass(slots=True)
class CascadeRiskAdvisor:
    """Escalate borderline controller decisions using forward simulation.

    ``advise`` only ever escalates (CONTINUE -> FORCE_VERIFY); it never
    downgrades a controller decision, so wrapping is always safe.
    """

    graph: AgentGraph
    impact_threshold: float = 0.5
    """Escalate when the lower CI bound of cascade impact exceeds this."""
    min_steps_since_verifier: int = 2
    """Do not escalate immediately after a verification."""
    trials: int = 50
    seed: int = 0
    _cache: dict[str, CascadeRiskEstimate] = field(default_factory=dict, repr=False)

    def estimate(self, node_id: str) -> CascadeRiskEstimate:
        """Cached per-node cascade risk (the graph is static within a session)."""

        if node_id not in self._cache:
            self._cache[node_id] = estimate_cascade_risk(
                self.graph, node_id, trials=self.trials, seed=self.seed
            )
        return self._cache[node_id]

    def advise(
        self,
        decision: ControlDecision,
        features: ExecutionStateFeatures,
        *,
        node_id: str | None,
    ) -> ControlDecision:
        """Return the original decision or an escalated FORCE_VERIFY."""

        if decision.action != "CONTINUE" or node_id is None:
            return decision
        if features.steps_since_verifier < self.min_steps_since_verifier:
            return decision
        if features.repeated_error_count == 0 and features.last_exit_code in (0, None):
            return decision
        estimate = self.estimate(node_id)
        if estimate.impact.lower >= self.impact_threshold:
            return ControlDecision(
                "FORCE_VERIFY",
                (
                    f"errors at {node_id} would cascade to "
                    f"{estimate.impact.mean:.0%} of the workflow "
                    f"(lower bound {estimate.impact.lower:.0%})"
                ),
                defer_command=False,
            )
        return decision


def _reachable_count(graph: AgentGraph, node_id: str) -> int:
    nx_graph = graph.to_networkx()
    seen = {node_id}
    stack = [node_id]
    while stack:
        current = stack.pop()
        for successor in nx_graph.successors(current):
            if successor not in seen:
                seen.add(successor)
                stack.append(successor)
    return len(seen) - 1
