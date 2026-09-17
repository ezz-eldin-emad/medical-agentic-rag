"""Backward-compatible LLM constants backed by centralized settings."""

from __future__ import annotations

from src.config import get_settings

_SETTINGS = get_settings()
DEFAULT_MODEL = _SETTINGS.llm.default_model
CLASSIFIER_MODEL = _SETTINGS.llm.classifier_model
REWRITER_MODEL = _SETTINGS.llm.rewriter_model
GENERATOR_MODEL = _SETTINGS.llm.generator_model
TEMPERATURE = _SETTINGS.llm.temperature
CLASSIFIER_TEMPERATURE = _SETTINGS.llm.classifier_temperature
REWRITER_TEMPERATURE = _SETTINGS.llm.rewriter_temperature
GENERATOR_TEMPERATURE = _SETTINGS.llm.generator_temperature
TIMEOUT = _SETTINGS.llm.timeout_seconds
RETRIES = _SETTINGS.llm.retries

_PROVIDER_PREFIXES = (
    "gemini/",
    "vertex_ai/",
    "google/",
    "groq/",
    "openrouter/",
    "openai/",
    "anthropic/",
)


def normalize_model(model: str) -> str:
    """Normalize a model id so Gemini uses Google AI Studio key auth."""
    model = model.strip() if model else DEFAULT_MODEL
    lower = model.lower()

    if lower.startswith(_PROVIDER_PREFIXES):
        return model
    if lower.startswith("gemini-"):
        return f"gemini/{model}"
    return model
