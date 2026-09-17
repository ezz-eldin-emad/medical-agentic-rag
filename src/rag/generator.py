"""Structured medical generation with validated evidence provenance."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config import AppSettings, get_settings
from src.llm import config
from src.llm.client import LLMClient
from src.utils.helpers import setup_logging

from .contracts import ClaimRecord, detect_language
from .evidence import citations_from_evidence, enrich_chunks, evidence_record_from_chunk, validate_claims

log = setup_logging("rag.generator")

_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "rag" / "rag_system.txt"


class CitationGenerator:
    """Generate a structured answer; citations are derived by the application."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str = config.GENERATOR_MODEL,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or LLMClient(default_model=model, settings=self.settings)
        self.model = model
        self.system_prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        if _SYSTEM_PROMPT_PATH.is_file():
            return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        return (
            "Return only JSON with answer, status, claims, and safety_actions. "
            "Use only the evidence blocks provided. Never invent evidence IDs.\n\n"
            "PATIENT CONTEXT:\n{patient_context}\n\nCONTEXT:\n{context}\n\nUSER QUESTION:\n{query}"
        )

    def _format_context(self, context_chunks: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for chunk in enrich_chunks(context_chunks):
            meta = chunk.get("metadata", {})
            evidence_id = chunk["evidence_id"]
            header = (
                f"[EVIDENCE_ID: {evidence_id}] Source: {meta.get('source', 'Medical Reference')} | "
                f"Document: {meta.get('doc_title', 'Medical Doc')}"
            )
            if meta.get("section"):
                header += f" | Section: {meta['section']}"
            blocks.append(f"{header}\n{chunk.get('text', '').strip()}")
        return "\n\n".join(blocks)

    def generate(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        temperature: float | None = None,
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate and validate a structured response from retrieved evidence."""

        temperature = self.settings.llm.generator_temperature if temperature is None else temperature
        context_chunks = enrich_chunks(context_chunks)
        context_str = self._format_context(context_chunks) if context_chunks else "(No relevant evidence retrieved)"
        patient_context_str = json.dumps(patient_context or {}, ensure_ascii=False, sort_keys=True)
        prompt = (
            self.system_prompt_template
            .replace("{context}", context_str)
            .replace("{patient_context}", patient_context_str)
            .replace("{query}", query)
        )
        language = detect_language(query)

        if not context_chunks:
            return self._safe_fallback(prompt, context_str, language, "no_context")

        messages = [{"role": "user", "content": prompt}]
        raw_content = ""
        log.info("Generating structured answer using model: %s ...", self.model)
        try:
            try:
                response = self.llm_client.complete(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
            except Exception as format_error:
                log.warning("Structured response format unavailable: %s; retrying prompt-only JSON", format_error)
                response = self.llm_client.complete(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                )
            raw_content = str(response.choices[0].message.content or "").strip()
            payload = self._parse_payload(raw_content)
            claims = self._claims_from_payload(payload)
            evidence = [
                evidence_record_from_chunk(
                    chunk,
                    # The generator receives chunks after the pipeline gate.
                    # Direct callers therefore treat an unspecified status as
                    # accepted evidence; the pipeline marks rejected chunks
                    # explicitly as unrelated before this method is called.
                    relevance_status=str(chunk.get("relevance_status", "relevant")),
                )
                for chunk in context_chunks
            ]
            valid_claims, validation = validate_claims(claims, evidence)
            status = str(payload.get("status") or ("answered" if valid_claims else "insufficient_evidence"))
            if not valid_claims and status == "answered":
                status = "insufficient_evidence"
            answer = str(payload.get("answer") or "").strip()
            if status == "insufficient_evidence" and not answer:
                answer = self._insufficient_message(language)

            cited_ids = {evidence_id for claim in valid_claims for evidence_id in claim.evidence_ids}
            cited_evidence = [item for item in evidence if item.evidence_id in cited_ids]
            return {
                "answer": answer,
                "user_response": answer,
                "status": status,
                "language": language,
                "claims": [claim.to_dict() for claim in valid_claims],
                "evidence": [item.to_dict() for item in evidence],
                "evidence_ids": [item.evidence_id for item in cited_evidence],
                "citations": citations_from_evidence(cited_evidence),
                "source_documents": context_chunks,
                "model": self.model,
                "final_prompt": prompt,
                "formatted_context": context_str,
                "validation": validation,
                "raw_generation": raw_content,
            }
        except Exception as err:
            log.error("Structured generator failed (%s): %s", self.model, err)
            return self._safe_fallback(prompt, context_str, language, "generation_error", str(err))

    @staticmethod
    def _parse_payload(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            cleaned = fenced.group(1).strip()
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("LLM structured output must be a JSON object")
        return payload

    @staticmethod
    def _claims_from_payload(payload: dict[str, Any]) -> list[ClaimRecord]:
        claims: list[ClaimRecord] = []
        raw_claims = payload.get("claims") or []
        if not isinstance(raw_claims, list):
            return claims
        for index, raw in enumerate(raw_claims, start=1):
            if not isinstance(raw, dict):
                continue
            evidence_ids = raw.get("evidence_ids") or []
            if isinstance(evidence_ids, str):
                evidence_ids = [evidence_ids]
            claims.append(
                ClaimRecord(
                    claim_id=str(raw.get("claim_id") or f"claim_{index:03d}"),
                    text=str(raw.get("text") or ""),
                    evidence_ids=[str(item) for item in evidence_ids if str(item).strip()],
                )
            )
        return claims

    def _safe_fallback(
        self,
        prompt: str,
        context_str: str,
        language: str,
        reason: str,
        error: str = "",
    ) -> dict[str, Any]:
        answer = self._insufficient_message(language)
        return {
            "answer": answer,
            "user_response": answer,
            "status": "insufficient_evidence",
            "language": language,
            "claims": [],
            "evidence": [],
            "evidence_ids": [],
            "citations": [],
            "source_documents": [],
            "model": self.model,
            "final_prompt": prompt,
            "formatted_context": context_str,
            "validation": {"passed": False, "reason": reason, "error": error},
        }

    @staticmethod
    def _insufficient_message(language: str) -> str:
        if language == "ar":
            return "لا أملك معلومات طبية موثوقة كافية للإجابة عن هذا السؤال بأمان ضمن المصادر المتاحة حالياً."
        return "I do not have enough reliable medical information in the available sources to answer this safely."
