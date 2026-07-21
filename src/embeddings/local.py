"""
Local BGE-M3 embedder via FlagEmbedding (dense + sparse).

Use for offline indexing on a laptop or Colab. Pins model weights with
``BGE_MODEL_ID`` + ``BGE_MODEL_REVISION``.
"""

from __future__ import annotations

import os
from typing import Any

from src.embeddings.protocol import EmbeddingResult


class BGEM3LocalEmbedder:
    """FlagEmbedding ``BGEM3FlagModel`` wrapper — dense + sparse lexical weights."""

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        *,
        use_fp16: bool = True,
        hf_token: str | None = None,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.model_id = model_id or os.environ.get("BGE_MODEL_ID") or os.environ.get(
            "BGE_MODEL_NAME", "BAAI/bge-m3"
        )
        self.revision = (
            revision
            if revision is not None
            else os.environ.get("BGE_MODEL_REVISION", "")
        ).strip()
        self.use_fp16 = use_fp16
        token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")

        model_path = self.model_id
        if self.revision:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=self.model_id,
                revision=self.revision,
                token=token or True,
            )

        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16)

    def encode(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        """Encode texts into dense and sparse (lexical) vectors."""
        output: dict[str, Any] = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=batch_size,
        )
        return {
            "dense_vecs": output["dense_vecs"],
            "lexical_weights": output["lexical_weights"],
        }


# Backwards-compatible alias
BGEM3Embedder = BGEM3LocalEmbedder
