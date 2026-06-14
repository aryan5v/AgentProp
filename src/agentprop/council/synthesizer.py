"""Quality-weighted synthesis of verified sub-answers into a final answer.

Where Fusion's synthesizer reads all panel responses roughly equally, the
Council orders and annotates sub-answers by their propagated quality (from the
plan graph's quality cascade) and excludes quarantined ones, so the synthesizer
leans on the sub-answers most likely to be reliable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agentprop.council.claim_check import CheckedSubAnswer
from agentprop.council.model_pool import ModelPool, ModelResponse

_SYNTH_SYSTEM = """You are a synthesizer. Write a single, well-structured, fully \
cited final answer that integrates the verified sub-answers below. Prefer \
higher-quality sub-answers, resolve conflicts explicitly, do not introduce \
claims unsupported by the sub-answers, and preserve citations."""


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """The fused final answer plus the call's cost/latency."""

    text: str
    model: str
    cost_usd: float
    latency_s: float
    citations: tuple[str, ...]
    used_subanswers: int
    tokens_used: int = 0


@dataclass(slots=True)
class Synthesizer:
    """Fuse checked sub-answers with a synthesizer model, quality-weighted."""

    model: str
    system_prompt: str = _SYNTH_SYSTEM
    temperature: float = 0.2
    instruction: str = ""

    def synthesize(
        self,
        pool: ModelPool,
        task: str,
        checked: Sequence[CheckedSubAnswer],
        *,
        quality: dict[str, float] | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> SynthesisResult:
        kept = [c for c in checked if not c.quarantined and c.text.strip()]
        ordered = sorted(
            kept, key=lambda c: (quality or {}).get(c.subtask_id, 0.5), reverse=True
        )
        blocks = []
        all_citations: list[str] = []
        for c in ordered:
            q = (quality or {}).get(c.subtask_id, 0.5)
            cites = " ".join(c.citations)
            blocks.append(
                f"[sub-answer {c.subtask_id} | quality {q:.2f} | model {c.model}]\n"
                f"{c.text}\n{('sources: ' + cites) if cites else ''}".strip()
            )
            all_citations.extend(c.citations)
        user_prompt = (
            f"TASK:\n{task}\n\n"
            + (f"SYNTHESIS GUIDANCE:\n{self.instruction}\n\n" if self.instruction else "")
            + "VERIFIED SUB-ANSWERS (highest quality first):\n\n"
            + "\n\n".join(blocks)
        )
        response: ModelResponse = pool.call(
            self.model,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            extra_body=extra_body,
        )
        seen: dict[str, None] = {}
        for url in [*all_citations, *response.citations]:
            seen.setdefault(url, None)
        return SynthesisResult(
            text=response.text,
            model=response.model,
            cost_usd=response.cost_usd,
            latency_s=response.latency_s,
            citations=tuple(seen),
            used_subanswers=len(ordered),
            tokens_used=response.usage.total_tokens,
        )
