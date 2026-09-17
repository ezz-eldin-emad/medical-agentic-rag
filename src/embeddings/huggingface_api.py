"""Hugging Face Inference API embedding backend for serverless deployment."""
from __future__ import annotations

import time
import os
from typing import Any

import numpy as np
import requests

from src.config import AppSettings, get_settings
from src.embeddings.protocol import EmbeddingResult


class HuggingFaceInferenceEmbedder:
    """Call HF feature-extraction inference without loading a local model."""

    def __init__(self, *, settings: AppSettings | None = None, model_id: str | None = None, revision: str | None = None, timeout: float = 90.0, max_retries: int = 3) -> None:
        app_settings = settings or get_settings()
        # HF deployments may use a different model than the local/Modal default
        # (for example E5 instead of BGE-M3).  An explicit environment override
        # must win even when the shared factory passes the central model id.
        configured_model = os.environ.get("HF_EMBED_MODEL", "").strip()
        self.model_id = (configured_model or model_id or "intfloat/multilingual-e5-large").strip() or app_settings.embedding.model_id
        self.revision = revision or app_settings.embedding.revision or "default"
        self.timeout = timeout
        self.max_retries = max_retries
        self.token = app_settings.secrets.hf_token
        if not self.token:
            raise ValueError("HF_TOKEN is required when EMBEDDER_BACKEND=huggingface")
        self.url = f"https://router.huggingface.co/hf-inference/models/{self.model_id}"

    def _post(self, texts: list[str]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.url, headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}, json={"inputs": texts, "options": {"wait_for_model": True}}, timeout=self.timeout)
                if response.status_code in {401, 403}:
                    raise RuntimeError("Hugging Face embedding authorization failed")
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(min(2**attempt, 8))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if getattr(getattr(exc, "response", None), "status_code", None) in {401, 403}:
                    raise RuntimeError("Hugging Face embedding authorization failed") from exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Hugging Face embedding request failed for model {self.model_id}") from last_error

    def encode(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        if not texts:
            return {"dense_vecs": np.zeros((0, 1024), dtype=np.float32), "lexical_weights": None, "colbert_vecs": None}
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            data = np.asarray(self._post(texts[start : start + batch_size]), dtype=np.float32)
            if data.ndim == 3:
                data = data.mean(axis=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            vectors.append(data)
        return {"dense_vecs": np.vstack(vectors), "lexical_weights": None, "colbert_vecs": None}
