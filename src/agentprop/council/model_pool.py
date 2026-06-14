"""A pool of heterogeneous models with cost metadata and parallel fan-out.

``ModelSpec`` captures what the assignment policy needs to choose a model: its
price (so we can score cost), a capability tier (so we never route a hard
sub-task to a model that cannot do it), and free-form tags/capabilities. The
pool calls models through the OpenAI-compatible client, so anything reachable
via an OpenAI-style endpoint — and the 500+ models behind one OpenRouter
``base_url`` — is usable without per-provider code.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from agentprop.evaluation.llm_execution import LLMUsage, OpenAICompatibleChatClient


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model endpoint plus the metadata routing needs."""

    name: str
    """Provider model slug, e.g. ``google/gemini-3-flash-preview``."""
    input_price_per_mtok: float = 0.0
    output_price_per_mtok: float = 0.0
    capability_tier: int = 1
    """Higher = more capable. Sub-tasks declare a minimum required tier."""
    base_url: str | None = None
    """Overrides the pool default (e.g. a different provider for one model)."""
    tags: tuple[str, ...] = ()
    """Free-form capabilities, e.g. ``("search", "code", "long-context")``."""

    def cost_usd(self, usage: LLMUsage) -> float:
        """Dollar cost of one call from token usage and per-Mtok prices."""

        return (
            usage.prompt_tokens * self.input_price_per_mtok
            + usage.completion_tokens * self.output_price_per_mtok
        ) / 1_000_000.0

    def supports(self, *, min_tier: int = 0, required_tags: Sequence[str] = ()) -> bool:
        """Whether this model meets a sub-task's capability requirements."""

        if self.capability_tier < min_tier:
            return False
        return all(tag in self.tags for tag in required_tags)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Result of one model call, with cost resolved from the spec."""

    model: str
    text: str
    usage: LLMUsage
    cost_usd: float
    latency_s: float
    citations: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class ModelPool:
    """Heterogeneous model pool with cost tracking and parallel fan-out."""

    specs: tuple[ModelSpec, ...]
    api_key: str | None = None
    default_base_url: str = "https://openrouter.ai/api/v1"
    timeout_s: float = 600.0
    max_workers: int = 8
    _by_name: dict[str, ModelSpec] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("ModelPool requires at least one ModelSpec")
        self._by_name = {spec.name: spec for spec in self.specs}
        if len(self._by_name) != len(self.specs):
            raise ValueError("duplicate model names in pool")
        if self.api_key is None:
            self.api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
                "OPENAI_API_KEY"
            )

    def spec(self, name: str) -> ModelSpec:
        if name not in self._by_name:
            raise KeyError(f"unknown model: {name}")
        return self._by_name[name]

    def candidates(
        self,
        *,
        min_tier: int = 0,
        required_tags: Sequence[str] = (),
    ) -> list[ModelSpec]:
        """Models meeting capability requirements, cheapest first."""

        eligible = [
            spec
            for spec in self.specs
            if spec.supports(min_tier=min_tier, required_tags=required_tags)
        ]
        return sorted(eligible, key=lambda s: s.input_price_per_mtok + s.output_price_per_mtok)

    def call(
        self,
        model: str,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Call one model; never raises — failures are captured on the response."""

        try:
            spec = self.spec(model)
            client = self._client(spec)
            result = client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        except Exception as exc:  # noqa: BLE001 - pool isolates one model's failure
            return ModelResponse(
                model=model,
                text="",
                usage=LLMUsage(),
                cost_usd=0.0,
                latency_s=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ModelResponse(
            model=model,
            text=result.response,
            usage=result.usage,
            cost_usd=spec.cost_usd(result.usage),
            latency_s=result.latency_s,
            citations=result.citations,
        )

    def fan_out(
        self,
        models: Sequence[str],
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, ModelResponse]:
        """Call several models on the same prompt in parallel (the Fusion shape)."""

        unique = list(dict.fromkeys(models))  # dedupe, preserve order
        if not unique:
            return {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(unique))) as pool:
            futures = {
                model: pool.submit(
                    self.call,
                    model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body,
                )
                for model in unique
            }
            return {model: future.result() for model, future in futures.items()}

    def map_assignments(
        self,
        assignments: Sequence[
            tuple[str, str, str] | tuple[str, str, str, dict[str, Any] | None]
        ],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> list[ModelResponse]:
        """Run ``(model, system_prompt, user_prompt[, extra_body])`` calls in parallel.

        This is the decompose-and-assign shape: different models on different
        sub-tasks, unlike ``fan_out`` which runs many models on one prompt. A
        4th tuple element carries per-assignment request extensions (e.g. a
        sub-task's own retrieval plugin), merged over the shared ``extra_body``.
        """

        if not assignments:
            return []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(assignments))) as pool:
            futures = []
            for item in assignments:
                model, system_prompt, user_prompt = item[0], item[1], item[2]
                item_extra = item[3] if len(item) > 3 else None
                merged = (
                    {**(extra_body or {}), **(item_extra or {})}
                    if (extra_body or item_extra)
                    else None
                )
                futures.append(
                    pool.submit(
                        self.call,
                        model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_body=merged,
                    )
                )
            return [future.result() for future in futures]

    def _client(self, spec: ModelSpec) -> OpenAICompatibleChatClient:
        if not self.api_key:
            raise ValueError(
                "no API key: set OPENROUTER_API_KEY/OPENAI_API_KEY or pass api_key"
            )
        return OpenAICompatibleChatClient(
            api_key=self.api_key,
            model=spec.name,
            base_url=spec.base_url or self.default_base_url,
            timeout_s=self.timeout_s,
        )
