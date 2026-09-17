"""Small tracing seam used by agents without requiring Phoenix at runtime."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from src.config.settings import AppSettings, get_settings
from src.utils.helpers import setup_logging

log = setup_logging("agents.tracing")


class NoopTracer:
    """No-op tracer that keeps production routing independent of observability."""

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        state = {"name": name, "attributes": attributes or {}}
        yield state


class PhoenixTracer:
    """OpenTelemetry tracer configured for Phoenix when enabled."""

    def __init__(self) -> None:
        from opentelemetry import trace

        self._tracer = trace.get_tracer("medical-agentic-rag.agents")

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
        with self._tracer.start_as_current_span(name) as active_span:
            for key, value in (attributes or {}).items():
                if value is not None:
                    active_span.set_attribute(f"app.{key}", str(value))
            state: dict[str, Any] = {}
            try:
                yield state
            finally:
                for key, value in state.items():
                    if value is not None and isinstance(value, (str, int, float, bool)):
                        active_span.set_attribute(f"app.{key}", value)


def get_tracer(settings: AppSettings | None = None) -> Any:
    """Register Phoenix OTEL only when explicitly enabled; otherwise no-op."""
    app_settings = settings or get_settings()
    if app_settings.tracing.enabled:
        try:
            from phoenix.otel import register

            register(
                project_name=app_settings.tracing.project_name,
                endpoint=app_settings.tracing.collector_endpoint or None,
                api_key=app_settings.secrets.phoenix_api_key or None,
                auto_instrument=False,
            )
            return PhoenixTracer()
        except Exception as err:
            log.warning("Phoenix tracing unavailable; using no-op tracer: %s", err)
    return NoopTracer()
