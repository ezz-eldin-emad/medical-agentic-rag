"""
Local BGE-M3 embedder via FlagEmbedding (dense + sparse).

Use for offline indexing on a laptop or Colab. Model settings are centralized
in ``src.config.settings``.
"""

from __future__ import annotations

from typing import Any

from src.config import AppSettings, get_settings
from src.embeddings.protocol import EmbeddingResult


class BGEM3LocalEmbedder:
    """FlagEmbedding ``BGEM3FlagModel`` wrapper — dense + sparse lexical weights."""

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        *,
        use_fp16: bool = False,
        hf_token: str | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        app_settings = settings or get_settings()
        self.model_id = model_id or app_settings.embedding.model_id
        self.revision = (revision if revision is not None else app_settings.embedding.revision).strip()
        self.use_fp16 = use_fp16
        token = hf_token if hf_token is not None else app_settings.secrets.hf_token

        model_path = self.model_id
        if self.revision:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=self.model_id,
                revision=self.revision,
                token=token or None,
            )

        self.model = BGEM3FlagModel(model_path, use_fp16=use_fp16)

    def encode(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        """Encode texts into dense and sparse (lexical) vectors."""
        output: Any = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            batch_size=batch_size,
        )
        return {
            "dense_vecs": output["dense_vecs"],
            "lexical_weights": output["lexical_weights"],
            "colbert_vecs": None,
        }


# Backwards-compatible alias
BGEM3Embedder = BGEM3LocalEmbedder
