"""Shared embedding contracts for local and API backends."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class EmbeddingResult(TypedDict, total=False):
    """Dense vectors always; sparse lexical weights and colbert token vectors when supported."""

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
