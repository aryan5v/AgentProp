"""Structured critical-fact extraction for context routing."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from agentprop.core import AgentGraph, AgentNode, NodeType

# Patterns that often encode convention-sensitive facts in coding workflows.
_CRITICAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(must|required|shall|never|always)\b"),
    re.compile(r"(?i)\b(convention|format|schema|api|signature|interface)\b"),
    re.compile(r"(?i)\b(error|exception|fail|timeout|retry)\b"),
    re.compile(r"(?i)\b(verif(?:y|ier)|test|assert|lint)\b"),
    re.compile(r"`[^`]+`"),
    re.compile(r"(?i)\bTODO\b.*"),
)


@dataclass(slots=True)
class CriticalFact:
    """One provenance-carrying fact slice."""

    text: str
    source_span: str
    score: float
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "source_span": self.source_span,
            "score": self.score,
            "tags": list(self.tags),
        }


@dataclass(slots=True)
class CriticalFactStore:
    """Per-node must-have facts learned from traces and verifiers."""

    node_facts: dict[str, list[CriticalFact]] = field(default_factory=dict)
    global_facts: list[CriticalFact] = field(default_factory=list)

    def facts_for(self, node_id: str) -> list[CriticalFact]:
        return list(self.global_facts) + list(self.node_facts.get(node_id, []))

    def register_node_facts(self, node_id: str, facts: Iterable[CriticalFact]) -> None:
        existing = self.node_facts.setdefault(node_id, [])
        seen = {fact.text for fact in existing}
        for fact in facts:
            if fact.text not in seen:
                existing.append(fact)
                seen.add(fact.text)

    def to_dict(self) -> dict[str, object]:
        return {
            "global_facts": [fact.to_dict() for fact in self.global_facts],
            "node_facts": {
                node_id: [fact.to_dict() for fact in facts]
                for node_id, facts in sorted(self.node_facts.items())
            },
        }


def extract_critical_facts(
    context: str,
    *,
    task: str = "",
    node: AgentNode | None = None,
    max_facts: int = 12,
) -> list[CriticalFact]:
    """Extract structured must-have facts from free-form context.

    Replaces blind ratio truncation for convention-sensitive workflows by
    preserving sentences and inline code that match critical patterns.
    """

    if not context.strip():
        return []

    sentences = _split_sentences(context)
    scored: list[CriticalFact] = []
    node_boost = _node_importance_boost(node)

    for sentence in sentences:
        score = _score_sentence(sentence, task=task) + node_boost
        if score <= 0.0:
            continue
        tags = _tags_for_sentence(sentence)
        scored.append(
            CriticalFact(
                text=sentence.strip(),
                source_span="sentence",
                score=score,
                tags=tags,
            )
        )

    # Always retain short convention blocks (e.g. bullet lists under 200 chars).
    for block in _convention_blocks(context):
        if any(fact.text == block for fact in scored):
            continue
        scored.append(
            CriticalFact(
                text=block,
                source_span="block",
                score=0.75 + node_boost,
                tags=("convention",),
            )
        )

    scored.sort(key=lambda fact: (-fact.score, len(fact.text)))
    return scored[:max_facts]


def build_context_slice(
    context: str,
    *,
    task: str,
    ratio: float,
    node: AgentNode | None = None,
    fact_store: CriticalFactStore | None = None,
) -> str:
    """Compose visible context from critical facts plus ratio-bounded filler."""

    bounded_ratio = max(0.0, min(1.0, ratio))
    if bounded_ratio >= 0.99:
        return context
    if bounded_ratio <= 0.0:
        return ""

    facts = list(extract_critical_facts(context, task=task, node=node))
    if fact_store is not None and node is not None:
        for stored in fact_store.facts_for(node.id):
            if stored.text not in {fact.text for fact in facts}:
                facts.append(stored)
    facts.sort(key=lambda fact: (-fact.score, len(fact.text)))

    target_chars = max(1, int(len(context) * bounded_ratio))
    parts: list[str] = []
    used = 0
    for fact in facts:
        addition = fact.text if not parts else f" {fact.text}"
        if used + len(addition) > target_chars and parts:
            break
        parts.append(fact.text if not parts else addition.strip())
        used += len(addition)
        if used >= target_chars:
            break

    if parts:
        return " ".join(parts)[:target_chars]

    # Fallback: head truncation when no critical facts matched.
    words = context.split()
    keep = max(1, int(len(words) * bounded_ratio))
    return " ".join(words[:keep])


def learn_facts_from_trace_row(
    store: CriticalFactStore,
    row: dict[str, object],
    *,
    graph: AgentGraph | None = None,
) -> None:
    """Ingest must-have facts from a routed trace row (node output / verifier)."""

    node_id = row.get("node_id")
    output = row.get("output") or row.get("visible_context") or row.get("context")
    if not isinstance(node_id, str) or not isinstance(output, str) or not output.strip():
        return
    node = graph.node(node_id) if graph is not None and graph.has_node(node_id) else None
    facts = extract_critical_facts(output, node=node)
    if node is not None and node.type in {NodeType.VERIFIER, NodeType.PLANNER, NodeType.DOCUMENT}:
        store.register_node_facts(node_id, facts)
    elif facts:
        store.register_node_facts(node_id, facts[:4])


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _convention_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "*", "•")) and 8 <= len(stripped) <= 200:
            blocks.append(stripped)
    return blocks


def _score_sentence(sentence: str, *, task: str) -> float:
    score = 0.0
    for pattern in _CRITICAL_PATTERNS:
        if pattern.search(sentence):
            score += 0.35
    if task and any(token in sentence.lower() for token in task.lower().split()[:6]):
        score += 0.2
    if "`" in sentence:
        score += 0.15
    return score


def _tags_for_sentence(sentence: str) -> tuple[str, ...]:
    tags: list[str] = []
    lowered = sentence.lower()
    if "convention" in lowered or "format" in lowered:
        tags.append("convention")
    if "verif" in lowered or "test" in lowered:
        tags.append("verification")
    if "error" in lowered or "fail" in lowered:
        tags.append("failure")
    return tuple(tags)


def _node_importance_boost(node: AgentNode | None) -> float:
    if node is None:
        return 0.0
    if node.type in {NodeType.VERIFIER, NodeType.PLANNER, NodeType.DOCUMENT}:
        return 0.25
    if node.importance_score is not None and node.importance_score >= 0.8:
        return 0.15
    return 0.0
