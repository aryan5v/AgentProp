"""In-loop context expansion advisors for LangGraph and AutoGen integrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agentprop.core import AgentGraph, NodeType
from agentprop.runtime.critical_facts import CriticalFactStore
from agentprop.runtime.hierarchical_context import ContextTier, HierarchicalContextBundle


@dataclass(slots=True)
class ContextExpansionAdvice:
    """Recommendation returned to framework checkpoints / handoffs."""

    expand: bool
    target_ratio: float
    reason: str
    node_id: str
    fact_id: str | None = None
    tier: ContextTier = ContextTier.NODE

    def to_dict(self) -> dict[str, object]:
        return {
            "expand": self.expand,
            "target_ratio": self.target_ratio,
            "reason": self.reason,
            "node_id": self.node_id,
            "fact_id": self.fact_id,
            "tier": self.tier.value,
        }


@dataclass(slots=True)
class ContextExpansionAdvisor:
    """Answer 'should I expand context for X?' inside an agent loop."""

    graph: AgentGraph
    fact_store: CriticalFactStore | None = None
    min_ratio: float = 0.35
    verifier_floor: float = 0.85

    def should_expand(
        self,
        node_id: str,
        *,
        current_ratio: float,
        task: str = "",
        fact_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextExpansionAdvice:
        """Decide whether to expand context for a node or fact."""

        node = self.graph.node(node_id)
        if node.type == NodeType.VERIFIER and current_ratio < self.verifier_floor:
            return ContextExpansionAdvice(
                expand=True,
                target_ratio=1.0,
                reason="verifier requires full context",
                node_id=node_id,
                fact_id=fact_id,
                tier=ContextTier.FACT if fact_id else ContextTier.NODE,
            )

        high_score_facts = []
        if self.fact_store is not None:
            for fact in self.fact_store.facts_for(node_id):
                if fact.score >= 0.7:
                    high_score_facts.append(fact)

        if high_score_facts and current_ratio < 0.6:
            return ContextExpansionAdvice(
                expand=True,
                target_ratio=min(1.0, current_ratio + 0.35),
                reason="high-score critical facts not fully visible",
                node_id=node_id,
                fact_id=fact_id or "fact_0",
                tier=ContextTier.FACT,
            )

        if node.importance_score is not None and node.importance_score >= 0.8:
            if current_ratio < 0.75:
                return ContextExpansionAdvice(
                    expand=True,
                    target_ratio=0.85,
                    reason="high-importance node below safe ratio",
                    node_id=node_id,
                    fact_id=fact_id,
                    tier=ContextTier.NODE,
                )

        if current_ratio < self.min_ratio:
            return ContextExpansionAdvice(
                expand=True,
                target_ratio=self.min_ratio,
                reason="below configured minimum ratio",
                node_id=node_id,
                fact_id=fact_id,
                tier=ContextTier.NODE,
            )

        return ContextExpansionAdvice(
            expand=False,
            target_ratio=current_ratio,
            reason="current allocation sufficient",
            node_id=node_id,
            fact_id=fact_id,
            tier=ContextTier.NODE,
        )


def langgraph_checkpoint_advice(
    advisor: ContextExpansionAdvisor,
    *,
    node_id: str,
    state: Mapping[str, Any],
) -> dict[str, object]:
    """Hook for LangGraph checkpoints: emit expansion advice into state."""

    ratio = float(state.get("context_ratio", 0.35) or 0.35)
    advice = advisor.should_expand(
        node_id,
        current_ratio=ratio,
        task=str(state.get("task", "")),
        metadata=state,
    )
    return {
        "agentprop_context_advice": advice.to_dict(),
        "agentprop_expand_context": advice.expand,
        "agentprop_target_ratio": advice.target_ratio,
    }


def autogen_handoff_advice(
    advisor: ContextExpansionAdvisor,
    *,
    node_id: str,
    handoff_payload: Mapping[str, Any],
) -> dict[str, object]:
    """Hook for AutoGen handoffs: attach context expansion guidance."""

    ratio = float(handoff_payload.get("context_ratio", 0.35) or 0.35)
    advice = advisor.should_expand(
        node_id,
        current_ratio=ratio,
        task=str(handoff_payload.get("task", "")),
        metadata=handoff_payload,
    )
    return {
        "context_expansion": advice.to_dict(),
        "recommended_ratio": advice.target_ratio,
    }


def bundle_from_advice(
    *,
    shared_context: str,
    task: str,
    node_id: str,
    advice: ContextExpansionAdvice,
    fact_store: CriticalFactStore | None = None,
) -> HierarchicalContextBundle:
    """Build a hierarchical bundle after an expansion decision."""

    from agentprop.runtime.hierarchical_context import build_hierarchical_bundle

    return build_hierarchical_bundle(
        shared_context=shared_context,
        task=task,
        node_id=node_id,
        fact_store=fact_store,
        ratio=advice.target_ratio,
    )
