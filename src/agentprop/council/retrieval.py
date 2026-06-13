"""Pluggable retrieval tools for Council sub-tasks.

Retrieval is a hard dependency for deep research but optional in general — a
coding Council needs none. The ``RetrievalTool`` protocol keeps the orchestrator
tool-agnostic; ``OpenRouterWebSearch`` is the default (the same web plugin
Fusion uses, attached via the chat ``extra_body``), and a first-party retriever
can be dropped in later without touching ``council/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """How a sub-task should be augmented with retrieval, plus provenance."""

    extra_body: dict[str, Any]
    """Merged into the model call (e.g. OpenRouter ``plugins``)."""
    note: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.extra_body)


@runtime_checkable
class RetrievalTool(Protocol):
    """Augments a model call so the model can gather external evidence."""

    name: str

    def for_subtask(self, query: str) -> RetrievalResult:
        """Return request extensions enabling retrieval for this sub-task."""
        ...


@dataclass(frozen=True, slots=True)
class NullRetrieval:
    """No retrieval — for tasks (e.g. coding, closed-book QA) that need none."""

    name: str = "none"

    def for_subtask(self, query: str) -> RetrievalResult:
        return RetrievalResult(extra_body={})


@dataclass(frozen=True, slots=True)
class OpenRouterWebSearch:
    """Attach OpenRouter's web-search plugin to every augmented call.

    Citations come back on the response (``LLMExecutionResult.citations`` /
    ``ModelResponse.citations``), which DRACO's citation-quality axis needs.
    """

    name: str = "openrouter-web"
    max_results: int = 5
    engine: str | None = None

    def for_subtask(self, query: str) -> RetrievalResult:
        plugin: dict[str, Any] = {"id": "web", "max_results": self.max_results}
        if self.engine:
            plugin["engine"] = self.engine
        return RetrievalResult(
            extra_body={"plugins": [plugin]},
            note=f"web search (max_results={self.max_results})",
        )
