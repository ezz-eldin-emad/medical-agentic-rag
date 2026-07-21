"""
BGE-M3 embeddings package.

Backends:
  - modal — Modal GPU API (dense + sparse + colbert)
  - local — FlagEmbedding (dense + sparse)
"""

from src.embeddings.factory import get_embedder
from src.embeddings.local import BGEM3Embedder, BGEM3LocalEmbedder
from src.embeddings.modal_api import BGEM3ModalAPIEmbedder
from src.embeddings.protocol import Embedder, EmbeddingResult

__all__ = [
    "Embedder",
    "EmbeddingResult",
    "BGEM3LocalEmbedder",
    "BGEM3Embedder",
    "BGEM3ModalAPIEmbedder",
    "get_embedder",
]
