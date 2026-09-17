"""Factory for local and optional remote embedding backends."""

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
    chosen = (backend or app_settings.embedding.backend or "local").strip().lower()

    if chosen in {"huggingface", "hf", "hf_inference"}:
        from src.embeddings.huggingface_api import HuggingFaceInferenceEmbedder
        return HuggingFaceInferenceEmbedder(
            settings=app_settings,
            model_id=model_id,
            revision=revision,
        )

    if chosen in {"modal", "modal_api", "remote"}:
        from src.embeddings.modal_api import BGEM3ModalAPIEmbedder

        return BGEM3ModalAPIEmbedder(
            model_id=model_id or app_settings.embedding.model_id,
            revision=revision if revision is not None else app_settings.embedding.revision,
            api_url=app_settings.embedding.modal_url,
            timeout=app_settings.embedding.request_timeout_seconds,
            max_retries=app_settings.embedding.max_retries,
        )

    if chosen in {"local", "flag", "flagembedding"}:
        from src.embeddings.local import BGEM3LocalEmbedder

        if use_fp16 is None:
            use_fp16 = app_settings.embedding.use_fp16

        return BGEM3LocalEmbedder(
            model_id=model_id or app_settings.embedding.model_id,
            revision=revision if revision is not None else app_settings.embedding.revision,
            use_fp16=use_fp16,
            hf_token=hf_token if hf_token is not None else app_settings.secrets.hf_token,
        )

    raise ValueError(
        f"Unknown EMBEDDER_BACKEND={chosen!r}. Supported values: 'local' or 'modal'."
    )
