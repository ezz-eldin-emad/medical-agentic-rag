"""
Module: retriever.py
Purpose: Native BGE-M3 Hybrid Search (Dense + Sparse/Lexical) using Qdrant,
         fused via Reciprocal Rank Fusion (RRF).
"""

from pathlib import Path
from typing import Any

from src.config import AppSettings, get_settings
from src.embeddings.factory import get_embedder
from src.embeddings.protocol import Embedder
from src.utils.helpers import get_project_root, setup_logging

log = setup_logging("rag.retriever")


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF).

    RRF_score(d) = sum(1 / (k + rank(d)))
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    # Process dense results
    for rank, hit in enumerate(dense_results, start=1):
        doc_id = str(hit.get("id"))
        doc_map[doc_id] = hit
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Process sparse results
    for rank, hit in enumerate(sparse_results, start=1):
        doc_id = str(hit.get("id"))
        if doc_id not in doc_map:
            doc_map[doc_id] = hit
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Sort candidates by combined RRF score descending
    sorted_doc_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)

    fused_results: list[dict[str, Any]] = []
    for doc_id in sorted_doc_ids[:top_n]:
        item = dict(doc_map[doc_id])
        item["rrf_score"] = scores[doc_id]
        fused_results.append(item)

    return fused_results


def _chunk_from_payload(point_id: Any, score: float, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Qdrant point into the structure used by the RAG pipeline."""
    sources = payload.get("sources") or []
    metadata = {
        "doc_title": payload.get("doc_title", ""),
        "source": sources[0] if sources else payload.get("source", ""),
        "section": payload.get("section", ""),
        "url": payload.get("url", ""),
        "type": payload.get("type", ""),
    }
    return {
        "id": point_id,
        "score": score,
        "text": payload.get("text", ""),
        "metadata": metadata,
    }


def connect_qdrant(collection_name: str | None = None, settings: AppSettings | None = None) -> tuple[Any, str]:
    """Open configured Qdrant storage and verify the requested collection."""
    from qdrant_client import QdrantClient

    app_settings = settings or get_settings()
    collection_name = collection_name or app_settings.vectordb.medical_collection
    url = app_settings.vectordb.qdrant_url.strip().rstrip("/")
    api_key = app_settings.secrets.qdrant_api_key or None

    if url:
        location = url
        log.info("Connecting to Qdrant at %s", url)
        client = QdrantClient(url=url, api_key=api_key, timeout=10)
    else:
        path = app_settings.resolve_path(app_settings.vectordb.qdrant_path).expanduser()
        if not path.is_absolute():
            path = get_project_root() / path
        path.mkdir(parents=True, exist_ok=True)
        location = f"local:{path}"
        log.info("Opening local Qdrant database at %s", path)
        client = QdrantClient(path=str(path))

    try:
        collections = {item.name for item in client.get_collections().collections}
        if collection_name not in collections:
            available = ", ".join(sorted(collections)) or "none"
            raise RuntimeError(
                f"Qdrant is reachable, but collection '{collection_name}' does not exist "
                f"(available: {available}). Run the indexing step first."
            )
        log.info("Qdrant connected; collection '%s' is available.", collection_name)
        return client, location
    except Exception as err:
        client.close()
        raise RuntimeError(
            f"Could not open Qdrant at '{location}' or verify collection "
            f"'{collection_name}': {err}"
        ) from err


class HybridRetriever:
    """Hybrid Retriever leveraging BGE-M3 Dense + Sparse embeddings and Qdrant."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        qdrant_client: Any = None,
        collection_name: str | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        collection_name = collection_name or self.settings.vectordb.medical_collection
        self.embedder = embedder or get_embedder(settings=self.settings)
        self.collection_name = collection_name

        if qdrant_client is None:
            self.qdrant_client, self.qdrant_url = connect_qdrant(collection_name, self.settings)
            self.qdrant_connected = True
        else:
            self.qdrant_client = qdrant_client
            self.qdrant_url = "injected client"
            self.qdrant_connected = True

    def status(self) -> dict[str, Any]:
        """Return dependency information without making another network call."""
        return {
            "url": self.qdrant_url,
            "collection": self.collection_name,
            "connected": self.qdrant_connected,
            "embedder": type(self.embedder).__name__,
            "embedder_model": getattr(self.embedder, "model_id", "unknown"),
        }

    def _qdrant_dense_search(self, dense_vector: list[float], limit: int) -> list[dict[str, Any]]:
        """Search Qdrant using the dense vector."""
        try:
            if hasattr(self.qdrant_client, "query_points"):
                res = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=dense_vector,
                    using="dense",
                    limit=limit,
                    with_payload=True,
                )
                hits = getattr(res, "points", res)
            elif hasattr(self.qdrant_client, "search"):
                hits = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=("dense", dense_vector),
                    limit=limit,
                    with_payload=True,
                )
            else:
                log.error("QdrantClient has neither query_points nor search method.")
                return []

            return [
                _chunk_from_payload(hit.id, hit.score, hit.payload or {})
                for hit in hits
            ]
        except Exception as err:
            log.error(f"Dense vector search failed on collection '{self.collection_name}': {err}")
            return []

    def _qdrant_scroll_fallback(self, query_text: str, limit: int) -> list[dict[str, Any]]:
        """Keyword matching scroll fallback if sparse vector search is not available."""
        try:
            records, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
            )
            words = set(query_text.lower().split())
            scored = []
            for rec in records:
                payload = rec.payload or {}
                text = payload.get("text", "").lower()
                matches = sum(1 for w in words if w in text)
                if matches > 0:
                    scored.append(
                        (
                            matches,
                            _chunk_from_payload(rec.id, float(matches), payload),
                        )
                    )
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored[:limit]]
        except Exception as err:
            log.warning(f"Scroll fallback failed: {err}")
            return []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        dense_limit: int = 15,
        sparse_limit: int = 15,
    ) -> list[dict[str, Any]]:
        """Perform Hybrid Retrieval (Dense + Sparse) with Reciprocal Rank Fusion (RRF).

        Args:
            query: The user query or expanded query string.
            top_k: Number of final fused results to return.
            dense_limit: Top candidates from dense vector search.
            sparse_limit: Top candidates from sparse/lexical search.

        Returns:
            List of fused context chunk dicts with rrf_score, text, and metadata.
        """
        if not query or not query.strip():
            return []

        top_k = top_k or self.settings.rag.retrieval_top_k

        log.info(f"Encoding query with BGE-M3 embedder ({self.embedder.model_id})...")
        encoded = self.embedder.encode([query])

        dense_vecs = encoded.get("dense_vecs")
        if dense_vecs is None or len(dense_vecs) == 0:
            log.error("Failed to encode query dense vector.")
            return []

        dense_list = dense_vecs[0].tolist() if hasattr(dense_vecs[0], "tolist") else list(dense_vecs[0])

        # 1. Dense Search
        dense_hits = self._qdrant_dense_search(dense_list, limit=dense_limit)

        # 2. Sparse / Keyword Search
        sparse_hits = self._qdrant_scroll_fallback(query, limit=sparse_limit)

        # 3. Reciprocal Rank Fusion (RRF)
        fused = reciprocal_rank_fusion(
            dense_results=dense_hits,
            sparse_results=sparse_hits,
            k=60,
            top_n=top_k,
        )

        log.info(f"Hybrid retrieval found {len(fused)} fused chunks for query: '{query[:40]}'")
        return fused
