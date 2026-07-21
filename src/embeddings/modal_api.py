"""
Modal GPU FastAPI embedder for BGE-M3 (dense, sparse, and optional ColBERT).

Connects to the serverless Modal endpoint (e.g. app.py deployed on Modal GPU)
to generate high-performance embeddings without local GPU hardware.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import requests

from src.embeddings.protocol import EmbeddingResult

_DEFAULT_MODAL_URL = "https://eea29990--bge-m3-api-bgem3api-embed.modal.run"


class BGEM3ModalAPIEmbedder:
    """BGE-M3 Embedder via Modal GPU serverless endpoint."""

    def __init__(
        self,
        model_id: str | None = None,
        revision: str | None = None,
        *,
        api_url: str | None = None,
        timeout: float = 90.0,
        max_retries: int = 5,
        return_colbert: bool = False,
    ) -> None:
        self.model_id = model_id or os.environ.get("BGE_MODEL_ID") or os.environ.get(
            "BGE_MODEL_NAME", "BAAI/bge-m3"
        )
        self.revision = (
            revision
            if revision is not None
            else os.environ.get("BGE_MODEL_REVISION", "")
        ).strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.return_colbert = return_colbert

        url = (
            api_url
            or os.environ.get("MODAL_EMBED_URL")
            or os.environ.get("BGEM3_API_URL")
            or _DEFAULT_MODAL_URL
        )
        self.api_url = url.strip().rstrip("/")

        self._headers = {"Content-Type": "application/json"}

    def _post(self, texts: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "texts": texts,
            "return_dense": True,
            "return_sparse": True,
            "return_colbert": self.return_colbert,
        }
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self._headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(min(2**attempt, 20))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(2**attempt, 20))

        raise RuntimeError(
            f"Modal API embedding request failed for endpoint '{self.api_url}' after {self.max_retries} retries"
        ) from last_error

    def encode(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        """Encode a list of text strings into dense, sparse, and optional colbert vectors."""
        if not texts:
            empty_dense = np.zeros((0, 1024), dtype=np.float32)
            return {
                "dense_vecs": empty_dense,
                "lexical_weights": [],
                "colbert_vecs": [] if self.return_colbert else None,
            }

        dense_vecs_list: list[np.ndarray] = []
        lexical_weights_list: list[dict[str, float]] = []
        colbert_vecs_list: list[Any] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            data = self._post(batch_texts)

            if "dense" in data:
                dense_batch = np.array(data["dense"], dtype=np.float32)
                dense_vecs_list.append(dense_batch)

            if "sparse" in data:
                lexical_weights_list.extend(data["sparse"])

            if self.return_colbert and "colbert" in data:
                colbert_vecs_list.extend(data["colbert"])

        dense_vecs = (
            np.vstack(dense_vecs_list)
            if dense_vecs_list
            else np.zeros((0, 1024), dtype=np.float32)
        )

        return {
            "dense_vecs": dense_vecs,
            "lexical_weights": lexical_weights_list if lexical_weights_list else None,
            "colbert_vecs": colbert_vecs_list if self.return_colbert else None,
        }
