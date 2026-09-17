"""Canonical Phoenix tracer entry point used by the application pipeline."""

from src.agents.tracing import NoopTracer, PhoenixTracer, get_tracer

__all__ = ["NoopTracer", "PhoenixTracer", "get_tracer"]

