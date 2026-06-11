"""PII scrubbing and optional OpenTelemetry export for execution traces."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

from agentprop.runtime.control_loop import ControlDecision, ExecutionEvent


class Scrubber(Protocol):
    """Callable redaction hook."""

    def __call__(self, text: str) -> str:
        """Return redacted text."""


DEFAULT_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"(?i)(api[_-]?key|token|secret)=([^\s&]+)"), r"\1=[REDACTED_SECRET]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
)


class RegexPIIScrubber:
    """Default regex scrubber for secrets, keys, and email addresses."""

    def __init__(
        self,
        patterns: tuple[tuple[re.Pattern[str], str], ...] = DEFAULT_REDACTION_PATTERNS,
    ) -> None:
        self.patterns = patterns

    def __call__(self, text: str) -> str:
        safe = text
        for pattern, replacement in self.patterns:
            safe = pattern.sub(replacement, safe)
        return safe


def scrub_event(event: ExecutionEvent, scrubber: Scrubber | None = None) -> ExecutionEvent:
    """Apply a redaction hook to textual event fields before persistence/export."""

    active = scrubber or RegexPIIScrubber()
    return replace(
        event,
        command=active(event.command) if event.command is not None else None,
        error_signature=(
            active(event.error_signature) if event.error_signature is not None else None
        ),
    )


class OTelTraceExporter:
    """Export ``ExecutionEvent`` rows to OpenTelemetry when the SDK is installed."""

    def __init__(
        self,
        *,
        service_name: str = "agentprop",
        scrubber: Scrubber | None = None,
        tracer_provider: object | None = None,
    ) -> None:
        self.service_name = service_name
        self.scrubber = scrubber or RegexPIIScrubber()
        self._tracer = _build_tracer(service_name, tracer_provider)

    @property
    def enabled(self) -> bool:
        """Return true when OpenTelemetry is importable and configured."""

        return self._tracer is not None

    def export_event(
        self,
        event: ExecutionEvent,
        *,
        run_id: str,
        decision: ControlDecision | None = None,
    ) -> None:
        """Export one event as a span; silently no-op when OTel is unavailable."""

        if self._tracer is None:
            return
        safe_event = scrub_event(event, self.scrubber)
        span_name = f"agentprop.step.{safe_event.step}"
        tracer: Any = self._tracer
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("service.name", self.service_name)
            span.set_attribute("trace_id", run_id)
            span.set_attribute(
                "span_id",
                str(uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{safe_event.step}")),
            )
            span.set_attribute("agentprop.step", safe_event.step)
            span.set_attribute("agentprop.tokens_used", safe_event.tokens_used)
            span.set_attribute("agentprop.elapsed_s", safe_event.elapsed_s)
            span.set_attribute("agentprop.trusted", safe_event.trusted)
            if safe_event.exit_code is not None:
                span.set_attribute("agentprop.exit_code", safe_event.exit_code)
            if safe_event.error_signature:
                span.set_attribute("error.type", safe_event.error_signature)
            if decision is not None:
                span.add_event(
                    "agentprop.controller.decision",
                    {
                        "agentprop.decision": decision.action,
                        "agentprop.decision.reason": decision.reason,
                        "agentprop.decision.defer_command": decision.defer_command,
                    },
                )


def _build_tracer(service_name: str, tracer_provider: object | None) -> Any:
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    if tracer_provider is not None:
        return trace.get_tracer(service_name, tracer_provider=tracer_provider)
    return trace.get_tracer(service_name)


TraceScrubber = Callable[[str], str]
