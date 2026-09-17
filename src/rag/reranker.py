"""
Module: reranker.py
Purpose: Re-rank and score retrieved candidates to maximize top-K precision.
"""

from typing import Any

from src.config import AppSettings, get_settings
from src.utils.helpers import setup_logging

log = setup_logging("rag.reranker")


class Reranker:
    """Re-ranking component for retrieved context chunks."""

    def __init__(self, model_name: str | None = None, use_local_model: bool = False, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.rag.reranker_model
        self.use_local_model = use_local_model
        self._cross_encoder: Any = None

    def _init_model(self) -> None:
        """Optionally load local FlagEmbedding / CrossEncoder if requested."""
        if self.use_local_model and self._cross_encoder is None:
            try:
                from FlagEmbedding import FlagReranker
                self._cross_encoder = FlagReranker(self.model_name, use_fp16=True)
                log.info("Loaded FlagReranker model: %s", self.model_name)
            except Exception as err:
                log.warning("Could not load local FlagReranker (%s). Using RRF score order fallback.", err)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Re-rank candidate chunks against the user query.

        Args:
            query: The original user query.
            candidates: List of chunk dicts from HybridRetriever.
            top_k: Number of candidates to return.

        Returns:
            Re-ranked list of chunk dicts with updated 'rerank_score'.
        """
        if not candidates:
            return []

        top_k = top_k or self.settings.rag.retrieval_top_k

        if self.use_local_model:
            self._init_model()

        if self._cross_encoder is not None:
            try:
                pairs = [(query, c.get("text", "")) for c in candidates]
                scores: Any = self._cross_encoder.compute_score(pairs)
                if isinstance(scores, (float, int)):
                    scores = [scores]

                scored_candidates = []
                for score, cand in zip(scores, candidates):
                    item = dict(cand)
                    item["rerank_score"] = float(score)
                    item["rerank_score_type"] = "cross_encoder"
                    scored_candidates.append(item)

                scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored_candidates[:top_k]
            except Exception as err:
                log.warning("FlagReranker computation failed: %s. Using RRF order fallback.", err)

        # RRF Score-based ranking
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("rrf_score", 0.0),
            reverse=True,
        )
        for idx, c in enumerate(sorted_candidates):
            c["rerank_score"] = c.get("rrf_score", 1.0 / (idx + 1))
            c["rerank_score_type"] = "rrf"

        return sorted_candidates[:top_k]
