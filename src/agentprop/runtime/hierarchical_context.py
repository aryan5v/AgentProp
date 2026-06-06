"""Hierarchical typed context and fact-level verifier placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agentprop.core import AgentGraph, NodeType
from agentprop.runtime.critical_facts import CriticalFact, CriticalFactStore


class ContextTier(StrEnum):
    """Layers of context visibility in hierarchical routing."""

    GLOBAL = "global"
    TASK = "task"
    NODE = "node"
    FACT = "fact"


@dataclass(slots=True)
class TypedContextSlice:
    """One typed slice of context with provenance."""

    tier: ContextTier
    text: str
    node_id: str | None = None
    fact_id: str | None = None
    score: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "tier": self.tier.value,
            "text": self.text,
            "node_id": self.node_id,
            "fact_id": self.fact_id,
            "score": self.score,
        }


@dataclass(slots=True)
class HierarchicalContextBundle:
    """Ordered context layers for a single execution step."""

    global_context: str = ""
    task_context: str = ""
    node_slices: list[TypedContextSlice] = field(default_factory=list)
    fact_slices: list[TypedContextSlice] = field(default_factory=list)

    def render(self, *, ratio: float = 1.0) -> str:
        """Flatten layers into visible context respecting a ratio budget."""

        parts: list[str] = []
        if self.global_context:
            parts.append(self.global_context)
        if self.task_context:
            parts.append(self.task_context)
        ordered_facts = sorted(self.fact_slices, key=lambda s: -s.score)
        ordered_nodes = sorted(self.node_slices, key=lambda s: -s.score)

        if ratio >= 0.99:
            for slice_ in ordered_facts + ordered_nodes:
                parts.append(slice_.text)
            return "\n".join(parts)

        # Truncate at slice boundaries to avoid mid-word or mid-sentence cuts.
        total_len = sum(len(p) for p in parts) + sum(
            len(s.text) for s in ordered_facts + ordered_nodes
        )
        target_len = max(1, int(total_len * max(0.0, min(1.0, ratio))))
        current_len = sum(len(p) for p in parts)
        for slice_ in ordered_facts + ordered_nodes:
            if current_len + len(slice_.text) <= target_len or not parts:
                parts.append(slice_.text)
                current_len += len(slice_.text)
            else:
                break
        return "\n".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "global_context": self.global_context,
            "task_context": self.task_context,
            "node_slices": [s.to_dict() for s in self.node_slices],
            "fact_slices": [s.to_dict() for s in self.fact_slices],
        }


@dataclass(slots=True)
class FactLevelVerifier:
    """Maps one critical fact to a dedicated verifier signature."""

    fact_id: str
    fact_text: str
    verifier_node_id: str
    resolving_distance: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "fact_text": self.fact_text,
            "verifier_node_id": self.verifier_node_id,
            "resolving_distance": self.resolving_distance,
        }


def build_hierarchical_bundle(
    *,
    shared_context: str,
    task: str,
    node_id: str,
    fact_store: CriticalFactStore | None = None,
    ratio: float = 1.0,
) -> HierarchicalContextBundle:
    """Compose typed context layers for one node."""

    from agentprop.runtime.critical_facts import extract_critical_facts

    facts = extract_critical_facts(shared_context, task=task)
    if fact_store is not None:
        for stored in fact_store.facts_for(node_id):
            if stored.text not in {fact.text for fact in facts}:
                facts.append(stored)

    bundle = HierarchicalContextBundle(
        global_context=shared_context if ratio >= 0.95 else "",
        task_context=task,
    )
    for index, fact in enumerate(facts):
        bundle.fact_slices.append(
            TypedContextSlice(
                tier=ContextTier.FACT,
                text=fact.text,
                node_id=node_id,
                fact_id=f"fact_{index}",
                score=fact.score,
            )
        )
    bundle.node_slices.append(
        TypedContextSlice(
            tier=ContextTier.NODE,
            text=f"node={node_id}",
            node_id=node_id,
            score=0.5,
        )
    )
    return bundle


def place_fact_level_verifiers(
    graph: AgentGraph,
    facts: list[CriticalFact],
    *,
    budget: int,
) -> list[FactLevelVerifier]:
    """Assign verifiers to critical facts using metric-dimension candidates."""

    from agentprop.algorithms import metric_dimension_verifier_placement

    verifier_nodes = [
        node.id
        for node in graph.nodes()
        if node.type == NodeType.VERIFIER
    ]
    if not verifier_nodes:
        verifier_nodes = metric_dimension_verifier_placement(graph, min(budget, graph.node_count))
    if not verifier_nodes:
        return []

    placements: list[FactLevelVerifier] = []
    for index, fact in enumerate(facts[:budget]):
        verifier_id = verifier_nodes[index % len(verifier_nodes)]
        placements.append(
            FactLevelVerifier(
                fact_id=f"fact_{index}",
                fact_text=fact.text,
                verifier_node_id=verifier_id,
                resolving_distance=1.0 + 0.1 * index,
            )
        )
    return placements
