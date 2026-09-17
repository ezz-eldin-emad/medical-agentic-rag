"""Structured response and provenance contracts for the RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ClaimRecord:
    """A factual statement and the evidence that is allowed to support it."""

    claim_id: str
    text: str
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceRecord:
    """Stable provenance metadata for one retrieved chunk."""

    evidence_id: str
    chunk_id: str
    document_id: str
    source: str = ""
    document_title: str = ""
    section: str = ""
    url: str = ""
    text: str = ""
    retrieval_score: float | None = None
    rerank_score: float | None = None
    relevance_status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResponseEnvelope:
    """Canonical internal result from every route and renderer."""

    request_id: str
    trace_id: str
    route: str
    status: str
    language: str
    user_response: str
    claims: list[ClaimRecord] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    safety_actions: list[str] = field(default_factory=list)
    internal_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "route": self.route,
            "status": self.status,
            "language": self.language,
            "user_response": self.user_response,
            "claims": [claim.to_dict() for claim in self.claims],
            "evidence": [item.to_dict() for item in self.evidence],
            "citations": list(self.citations),
            "safety_actions": list(self.safety_actions),
            "internal_metadata": dict(self.internal_metadata),
        }


def detect_language(text: str) -> str:
    """Return a small language label used by renderers and inquiry state."""

    arabic = sum("\u0600" <= char <= "\u06ff" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    return "ar" if arabic > latin else "en"

