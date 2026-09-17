"""Shared embedding contracts for the API embedding backend."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class EmbeddingResult(TypedDict):
    """Dense vectors plus optional sparse and ColBERT representations."""

    dense_vecs: Any
    lexical_weights: Any | None
    colbert_vecs: Any | None



class Embedder(Protocol):
    """Common contract for indexing and query-time embedding."""

    model_id: str
    revision: str

    def encode(self, texts: list[str], batch_size: int = 32) -> EmbeddingResult:
        """Encode texts into dense (and optionally sparse) vectors."""
        ...
