"""Factory for embedder backends (Modal GPU API vs Local FlagEmbedding)."""

from __future__ import annotations

import os

from src.embeddings.protocol import Embedder


def get_embedder(
    *,
    backend: str | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    use_fp16: bool = True,
    hf_token: str | None = None,
) -> Embedder:
    """Return an ``Embedder`` for the configured backend.

    Env:
        ``EMBEDDER_BACKEND`` — ``modal`` (default) or ``local``
        ``BGE_MODEL_ID``  — HuggingFace model id
        ``BGE_MODEL_REVISION`` — optional commit SHA for weight pinning
    """
    chosen = (backend or os.environ.get("EMBEDDER_BACKEND") or "modal").strip().lower()

    if chosen in {"modal", "modal_api", "remote"}:
        from src.embeddings.modal_api import BGEM3ModalAPIEmbedder

        return BGEM3ModalAPIEmbedder(
            model_id=model_id,
            revision=revision,
        )

    if chosen in {"local", "flag", "flagembedding"}:
        from src.embeddings.local import BGEM3LocalEmbedder

        return BGEM3LocalEmbedder(
            model_id=model_id,
            revision=revision,
            use_fp16=use_fp16,
            hf_token=hf_token,
        )

    raise ValueError(
        f"Unknown EMBEDDER_BACKEND={chosen!r}. Supported values: 'modal' or 'local'."
    )
