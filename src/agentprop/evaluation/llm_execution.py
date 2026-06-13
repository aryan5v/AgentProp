"""OpenAI-compatible LLM execution helpers for real case-study runs."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Token usage returned by an LLM provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMExecutionResult:
    """One real LLM execution result with retained prompt/response metadata."""

    model: str
    prompt: str
    response: str
    usage: LLMUsage
    latency_s: float
    raw_response: dict[str, Any] = field(default_factory=dict)
    citations: tuple[str, ...] = ()
    """Source URLs surfaced by a web/search tool, when the provider returns them."""


class OpenAICompatibleChatClient:
    """Minimal OpenAI-compatible chat client using only the standard library."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not model:
            raise ValueError("model is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> OpenAICompatibleChatClient:
        """Build a client from OpenAI-compatible environment variables."""

        api_key = os.environ.get("OPENAI_API_KEY")
        resolved_model = model or os.environ.get("OPENAI_MODEL")
        resolved_base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        if not api_key:
            raise ValueError("set OPENAI_API_KEY for LLM execution")
        if not resolved_model:
            raise ValueError("set --llm-model or OPENAI_MODEL")
        return cls(
            api_key=api_key,
            model=resolved_model,
            base_url=resolved_base_url,
            timeout_s=timeout_s,
        )

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> LLMExecutionResult:
        """Run one chat completion call.

        ``extra_body`` is merged into the request payload, carrying provider
        extensions such as OpenRouter ``plugins`` (web search) or OpenAI-style
        ``tools`` without coupling this client to any one provider.
        """

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            for key, value in extra_body.items():
                payload[key] = value
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM provider returned HTTP {exc.code}: {body}") from exc
        latency_s = time.perf_counter() - started
        response_text = _extract_message(raw_payload)
        usage = _extract_usage(raw_payload.get("usage", {}))
        return LLMExecutionResult(
            model=self.model,
            prompt=user_prompt,
            response=response_text,
            usage=usage,
            latency_s=latency_s,
            raw_response=raw_payload,
            citations=_extract_citations(raw_payload),
        )


def openai_compatible_env_status(
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Return credential/model readiness for OpenAI-compatible case-study runs."""

    api_key_env = _first_present_env("OPENAI_API_KEY")
    resolved_model = model or os.environ.get("OPENAI_MODEL")
    resolved_base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    missing = []
    if api_key_env is None:
        missing.append("OPENAI_API_KEY")
    if not resolved_model:
        missing.append("--llm-model or OPENAI_MODEL")
    return {
        "ready": not missing,
        "api_key_env": api_key_env,
        "model": resolved_model,
        "base_url": resolved_base_url,
        "missing": missing,
    }


def _extract_message(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message", {})
    if isinstance(message, dict):
        content = message.get("content", "")
        return str(content) if content is not None else ""
    text = first.get("text", "")
    return str(text) if text is not None else ""


def _extract_usage(raw_usage: Any) -> LLMUsage:
    if not isinstance(raw_usage, dict):
        return LLMUsage()
    prompt_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(raw_usage.get("completion_tokens", 0) or 0)
    total_tokens = int(raw_usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _extract_citations(payload: dict[str, Any]) -> tuple[str, ...]:
    """Pull source URLs from message annotations (OpenRouter/OpenAI web search)."""

    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ()
    first = choices[0]
    if not isinstance(first, dict):
        return ()
    message = first.get("message", {})
    if not isinstance(message, dict):
        return ()
    urls: list[str] = []
    for annotation in message.get("annotations", []) or []:
        if not isinstance(annotation, dict):
            continue
        citation = annotation.get("url_citation")
        if isinstance(citation, dict) and isinstance(citation.get("url"), str):
            urls.append(citation["url"])
        elif isinstance(annotation.get("url"), str):
            urls.append(annotation["url"])
    # De-duplicate, preserve order.
    seen: dict[str, None] = {}
    for url in urls:
        seen.setdefault(url, None)
    return tuple(seen)


def _first_present_env(*names: str) -> str | None:
    for name in names:
        if os.environ.get(name):
            return name
    return None
