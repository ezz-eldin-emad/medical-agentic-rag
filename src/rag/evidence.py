"""Evidence identity, relevance gating, and claim validation."""

from __future__ import annotations

import hashlib
from typing import Any

from src.config import AppSettings, get_settings

from .contracts import ClaimRecord, EvidenceRecord


def evidence_id_for_chunk(chunk: dict[str, Any]) -> str:
    """Return a stable ID even for legacy chunks without an explicit ID."""

    metadata = chunk.get("metadata") or {}
    raw_id = metadata.get("chunk_id") or chunk.get("id")
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id)

    fingerprint = "|".join(
        str(value)
        for value in (
            metadata.get("source", ""),
            metadata.get("doc_title", ""),
            metadata.get("section", ""),
            chunk.get("text", ""),
        )
    )
    return f"evidence_{hashlib.sha256(fingerprint.encode('utf-8')).hexdigest()[:20]}"


def evidence_record_from_chunk(
    chunk: dict[str, Any],
    *,
    relevance_status: str = "unknown",
) -> EvidenceRecord:
    metadata = chunk.get("metadata") or {}
    stable_id = evidence_id_for_chunk(chunk)
    return EvidenceRecord(
        evidence_id=stable_id,
        chunk_id=str(metadata.get("chunk_id") or chunk.get("id") or stable_id),
        document_id=str(metadata.get("document_id") or metadata.get("doc_title") or stable_id),
        source=str(metadata.get("source", "")),
        document_title=str(metadata.get("doc_title", "")),
        section=str(metadata.get("section", "")),
        url=str(metadata.get("url", "")),
        text=str(chunk.get("text", "")),
        retrieval_score=_number_or_none(chunk.get("rrf_score", chunk.get("score"))),
        rerank_score=_number_or_none(chunk.get("rerank_score")),
        relevance_status=relevance_status,
    )


def enrich_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy chunks and expose their stable evidence IDs to the generator."""

    enriched: list[dict[str, Any]] = []
    for chunk in chunks:
        item = dict(chunk)
        metadata = dict(item.get("metadata") or {})
        stable_id = evidence_id_for_chunk(item)
        metadata.setdefault("chunk_id", stable_id)
        metadata.setdefault("evidence_id", stable_id)
        item["metadata"] = metadata
        item["evidence_id"] = stable_id
        enriched.append(item)
    return enriched


class RelevanceGate:
    """Block generation when reranked evidence is below the safety threshold."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.threshold = self.settings.rag.relevance_gate_threshold

    def evaluate(
        self,
        chunks: list[dict[str, Any]],
        query: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        enriched = enrich_chunks(chunks)
        relevant: list[dict[str, Any]] = []
        for chunk in enriched:
            score = _number_or_none(chunk.get("rerank_score"))
            if score is None:
                score = _number_or_none(chunk.get("relevance_score"))
            if score is None:
                score = _number_or_none(chunk.get("rrf_score"))

            score_type = str(chunk.get("rerank_score_type", "cross_encoder"))
            lexical_overlap = _lexical_overlap(query, str(chunk.get("text", "")))
            if score_type == "rrf":
                is_relevant = (
                    score is not None
                    and score >= self.settings.rag.rrf_relevance_floor
                    and lexical_overlap > 0
                )
            else:
                is_relevant = score is not None and score >= self.threshold
            item = dict(chunk)
            item["relevance_status"] = "relevant" if is_relevant else "unrelated"
            item["relevance_score"] = score
            item["lexical_overlap"] = lexical_overlap
            item["relevance_score_type"] = score_type
            if is_relevant:
                relevant.append(item)

        relevant_ids = {item["evidence_id"] for item in relevant}
        details = {
            "threshold": self.threshold,
            "rrf_floor": self.settings.rag.rrf_relevance_floor,
            "candidate_count": len(enriched),
            "relevant_count": len(relevant),
            "selected_ids": [item["evidence_id"] for item in relevant],
            "rejected_ids": [item["evidence_id"] for item in enriched if item["evidence_id"] not in relevant_ids],
            "passed": bool(relevant),
        }
        return relevant, details


def validate_claims(
    claims: list[ClaimRecord],
    evidence: list[EvidenceRecord],
) -> tuple[list[ClaimRecord], dict[str, Any]]:
    """Keep only claims mapped to valid, relevant evidence IDs."""

    allowed = {
        item.evidence_id
        for item in evidence
        if item.relevance_status in {"relevant", "unknown"}
    }
    valid: list[ClaimRecord] = []
    invalid_claims: list[str] = []
    for claim in claims:
        valid_ids = [evidence_id for evidence_id in claim.evidence_ids if evidence_id in allowed]
        if not claim.text.strip() or not valid_ids:
            invalid_claims.append(claim.claim_id)
            continue
        valid.append(ClaimRecord(claim_id=claim.claim_id, text=claim.text.strip(), evidence_ids=valid_ids))

    return valid, {
        "input_claim_count": len(claims),
        "valid_claim_count": len(valid),
        "invalid_claim_ids": invalid_claims,
        "passed": not invalid_claims,
    }


def citations_from_evidence(evidence: list[EvidenceRecord]) -> list[dict[str, Any]]:
    """Create optional presentation citations from validated evidence only."""

    return [
        {
            "id": item.evidence_id,
            "doc_title": item.document_title,
            "source": item.source,
            "section": item.section,
            "url": item.url,
        }
        for item in evidence
        if item.relevance_status == "relevant"
    ]


def _number_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _lexical_overlap(query: str, text: str) -> int:
    """Small deterministic fallback signal when no cross-encoder is loaded."""

    def terms(value: str) -> set[str]:
        return {
            token.strip(".,!?؛،:()[]{}\"'").casefold()
            for token in value.split()
            if len(token.strip(".,!?؛،:()[]{}\"'")) > 2
        }

    return len(terms(query) & terms(text))
