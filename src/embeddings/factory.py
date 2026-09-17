"""Factory for the Hugging Face Inference embedding backend."""

from __future__ import annotations

from src.config import AppSettings, get_settings
from src.embeddings.protocol import Embedder


def get_embedder(
    *,
    backend: str | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    use_fp16: bool | None = None,
    hf_token: str | None = None,
    settings: AppSettings | None = None,
) -> Embedder:
    """Return an ``Embedder`` for the configured backend.

    Behavioral defaults come from ``src.config.settings``; only deployment
    backend and endpoint overrides are read from the settings loader.
    """
    app_settings = settings or get_settings()
    chosen = (backend or app_settings.embedding.backend or "huggingface").strip().lower()

    if chosen in {"huggingface", "hf", "hf_inference"}:
        from src.embeddings.huggingface_api import HuggingFaceInferenceEmbedder
        return HuggingFaceInferenceEmbedder(
            settings=app_settings,
            model_id=model_id,
            revision=revision,
        )

    raise ValueError(f"Unsupported embedding backend {chosen!r}; use EMBEDDER_BACKEND=huggingface")
