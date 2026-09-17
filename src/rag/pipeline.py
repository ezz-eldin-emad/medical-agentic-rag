"""End-to-end medical RAG workflow with provenance and observability."""

from __future__ import annotations

import time
import uuid
import hashlib
import json
from typing import Any

from src.config import AppSettings, get_settings
from src.guardrails import InputSanitizer, QueryClassifier
from src.observability.tracer import get_tracer
from src.rag.contracts import detect_language
from src.rag.evidence import RelevanceGate, evidence_record_from_chunk
from src.rag.generator import CitationGenerator
from src.rag.query_rewriter import QueryRewriter
from src.rag.reranker import Reranker
from src.rag.retriever import HybridRetriever
from src.utils.helpers import setup_logging

log = setup_logging("rag.pipeline")


class MedicalRAGPipeline:
    """Run guardrails, retrieval, validation, and structured generation."""

    def __init__(
        self,
        rewriter_model: str | None = None,
        generator_model: str | None = None,
        collection_name: str | None = None,
        embedder: Any | None = None,
        qdrant_client: Any | None = None,
        sanitizer: InputSanitizer | None = None,
        classifier: QueryClassifier | None = None,
        settings: AppSettings | None = None,
        tracer: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        rewriter_model = rewriter_model or self.settings.llm.rewriter_model
        generator_model = generator_model or self.settings.llm.generator_model
        collection_name = collection_name or self.settings.vectordb.medical_collection
        self.tracer = tracer or get_tracer(self.settings)
        self.rewriter = QueryRewriter(model=rewriter_model, settings=self.settings)
        self.retriever = HybridRetriever(
            embedder=embedder,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            settings=self.settings,
        )
        self.reranker = Reranker(settings=self.settings)
        self.relevance_gate = RelevanceGate(settings=self.settings)
        self.generator = CitationGenerator(model=generator_model, settings=self.settings)
        self.sanitizer = sanitizer or InputSanitizer()
        self.classifier = classifier or QueryClassifier()

    def run(
        self,
        query: str,
        top_k: int | None = None,
        enable_query_rewrite: bool = True,
        patient_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the workflow and return a patient/internal response envelope."""

        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        top_k = top_k or self.settings.rag.retrieval_top_k
        log.info("Starting Medical RAG pipeline for query: %r", query)

        with self.tracer.span("medical_request", {"request_id": request_id}) as root_span:
            with self.tracer.span("input_guardrails", {"request_id": request_id}) as guard_span:
                # The orchestrator may append structured context (including
                # numeric severity/age) to the already-sanitized user text.
                # Re-sanitize only the user question portion; scanning the
                # internal context can falsely classify it as a phone/card.
                user_query_for_guard = query.split("\nStructured patient context:", 1)[0]
                sanitization = self.sanitizer.sanitize(user_query_for_guard)
                guard_span["allowed"] = sanitization.allowed
                guard_span["pii_detected"] = sanitization.pii_detected
                guard_span["prompt_injection_detected"] = sanitization.prompt_injection_detected
            if not sanitization.allowed:
                log.warning("Input rejected by guardrails: %s", sanitization.findings)
                result = self._early_result(
                    request_id=request_id,
                    query=query,
                    safe_query=sanitization.sanitized_text,
                    status="blocked",
                    route="blocked",
                    answer="I cannot process this request because it contains unsafe or sensitive content. Please remove personal details and try again.",
                    findings=list(sanitization.findings),
                )
                root_span["status"] = "blocked"
                return self._finish(result, started)

            safe_query = sanitization.sanitized_text
            with self.tracer.span("query_classifier", {"query_length": len(safe_query)}) as classifier_span:
                classification = self.classifier.classify(safe_query)
                classifier_span["query_class"] = classification.query_class.value
                classifier_span["confidence"] = classification.confidence
            log.info("Input classified as %s", classification.query_class.value)

            common = {
                "query": query,
                "sanitized_query": safe_query,
                "input_allowed": True,
                "guardrail_findings": list(sanitization.findings),
                "query_class": classification.query_class.value,
                "route": classification.query_class.value,
                "classification": {
                    "intent": classification.intent,
                    "confidence": classification.confidence,
                    "reason": classification.reason,
                    "entities": classification.entities or {},
                    "source": classification.source,
                },
            }

            if classification.query_class.value == "emergency":
                result = self._early_result(
                    request_id=request_id,
                    query=query,
                    safe_query=safe_query,
                    status="emergency",
                    route="emergency",
                    answer="This may be an emergency. Call your local emergency number or go to the nearest emergency department now. Do not wait for an online response.",
                    common=common,
                )
                return self._finish(result, started)

            if classification.query_class.value == "clinic_query":
                result = self._early_result(
                    request_id=request_id,
                    query=query,
                    safe_query=safe_query,
                    status="clinic_query",
                    route="clinic_query",
                    answer="This is a clinic-related request. The clinic agent will handle appointments, pricing, and availability.",
                    common=common,
                )
                return self._finish(result, started)

            rewritten_query = {"expanded_query": safe_query, "medical_keywords": [], "hyde_passage": ""}
            search_query = safe_query
            if enable_query_rewrite:
                with self.tracer.span("query_rewrite", {"model": self.rewriter.model}) as rewrite_span:
                    rewritten_query = self.rewriter.rewrite(safe_query)
                    search_query = rewritten_query.get("expanded_query") or safe_query
                    rewrite_span["expanded_query_length"] = len(search_query)

            with self.tracer.span("retrieval", {"collection": self.retriever.collection_name, "top_k": top_k}) as retrieval_span:
                candidates = self.retriever.retrieve(query=search_query, top_k=top_k * 3)
                retrieval_span["candidate_count"] = len(candidates)
                retrieval_span["candidate_ids"] = ",".join(str(item.get("id")) for item in candidates)
                retrieval_span["candidate_scores"] = json.dumps(
                    {
                        str(item.get("id")): item.get("rrf_score", item.get("score"))
                        for item in candidates
                    },
                    ensure_ascii=False,
                )

            with self.tracer.span("reranker", {"model": self.reranker.model_name}) as rerank_span:
                top_chunks = self.reranker.rerank(query=safe_query, candidates=candidates, top_k=top_k)
                rerank_span["selected_count"] = len(top_chunks)
                rerank_span["selected_ids"] = ",".join(str(item.get("id")) for item in top_chunks)
                rerank_span["selected_scores"] = json.dumps(
                    {
                        str(item.get("id")): item.get("rerank_score")
                        for item in top_chunks
                    },
                    ensure_ascii=False,
                )

            with self.tracer.span("relevance_gate", {"threshold": self.relevance_gate.threshold}) as gate_span:
                relevant_chunks, gate_details = self.relevance_gate.evaluate(top_chunks, query=search_query)
                gate_span.update({key: value for key, value in gate_details.items() if isinstance(value, (str, int, float, bool))})
                gate_span["selected_ids"] = ",".join(gate_details["selected_ids"])
                gate_span["rejected_ids"] = ",".join(gate_details["rejected_ids"])

            if not gate_details["passed"]:
                generation = {
                    "answer": self._insufficient_message(detect_language(safe_query)),
                    "user_response": self._insufficient_message(detect_language(safe_query)),
                    "status": "insufficient_evidence",
                    "language": detect_language(safe_query),
                    "claims": [],
                    "evidence": [
                        evidence_record_from_chunk(item, relevance_status="unrelated").to_dict()
                        for item in top_chunks
                    ],
                    "evidence_ids": [],
                    "citations": [],
                    "source_documents": top_chunks,
                    "model": self.generator.model,
                    "final_prompt": "",
                    "formatted_context": "",
                    "validation": {"passed": False, "reason": "relevance_gate_failed"},
                }
            else:
                with self.tracer.span("generation", {"model": self.generator.model}) as generation_span:
                    generation = self.generator.generate(
                        query=safe_query,
                        context_chunks=relevant_chunks,
                        patient_context=patient_context,
                    )
                    generation_span["status"] = generation.get("status", "")
                    generation_span["claim_count"] = len(generation.get("claims", []))
                    prompt = str(generation.get("final_prompt", ""))
                    generation_span["prompt_hash"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    generation_span["prompt_length"] = len(prompt)
                    generation_span["context_ids"] = ",".join(
                        str(item.get("evidence_id") or item.get("id"))
                        for item in relevant_chunks
                    )
                    if self.settings.tracing.capture_prompts:
                        generation_span["prompt"] = prompt
                    if self.settings.tracing.capture_context:
                        generation_span["context"] = generation.get("formatted_context", "")

            with self.tracer.span("evidence_validation", {}) as validation_span:
                validation = generation.get("validation", {})
                validation_span["passed"] = bool(validation.get("passed", False))
                validation_span["invalid_claim_count"] = len(validation.get("invalid_claim_ids", []))

            with self.tracer.span("output_guardrails", {}) as output_span:
                if not str(generation.get("user_response") or generation.get("answer") or "").strip():
                    generation["status"] = "error"
                    generation["user_response"] = self._insufficient_message(detect_language(safe_query))
                    generation["answer"] = generation["user_response"]
                output_span["status"] = generation.get("status", "")
                output_span["response_length"] = len(str(generation.get("user_response", "")))

            elapsed = round(time.perf_counter() - started, 3)
            result = {
                **common,
                "request_id": request_id,
                "trace_id": request_id,
                "status": generation.get("status", "insufficient_evidence"),
                "user_response": generation.get("user_response", generation.get("answer", "")),
                "answer": generation.get("answer", ""),
                "language": generation.get("language", detect_language(safe_query)),
                "claims": generation.get("claims", []),
                "evidence": generation.get("evidence", []),
                "evidence_ids": generation.get("evidence_ids", []),
                "citations": generation.get("citations", []),
                "retrieved_chunks": candidates,
                "reranked_chunks": top_chunks,
                "relevant_chunks": relevant_chunks,
                "relevance_gate": gate_details,
                "final_prompt": generation.get("final_prompt", ""),
                "formatted_context": generation.get("formatted_context", ""),
                "generator_model": generation.get("model", self.generator.model),
                "execution_time_seconds": elapsed,
                "internal_metadata": {
                    "request_id": request_id,
                    "trace_id": request_id,
                    "validation": validation,
                    "patient_context_present": bool(patient_context),
                },
            }
            root_span["status"] = result["status"]
            root_span["execution_time_seconds"] = elapsed
            return result

    def _early_result(
        self,
        *,
        request_id: str,
        query: str,
        safe_query: str,
        status: str,
        route: str,
        answer: str,
        common: dict[str, Any] | None = None,
        findings: list[str] | None = None,
    ) -> dict[str, Any]:
        language = detect_language(query)
        return {
            **(common or {}),
            "query": query,
            "sanitized_query": safe_query,
            "request_id": request_id,
            "trace_id": request_id,
            "route": route,
            "status": status,
            "language": language,
            "user_response": answer,
            "answer": answer,
            "claims": [],
            "evidence": [],
            "evidence_ids": [],
            "citations": [],
            "guardrail_findings": findings or (common or {}).get("guardrail_findings", []),
        }

    @staticmethod
    def _finish(result: dict[str, Any], started: float) -> dict[str, Any]:
        result["execution_time_seconds"] = round(time.perf_counter() - started, 3)
        result.setdefault("internal_metadata", {})["execution_time_seconds"] = result["execution_time_seconds"]
        return result

    @staticmethod
    def _insufficient_message(language: str) -> str:
        return (
            "لا أملك معلومات طبية موثوقة كافية للإجابة عن هذا السؤال بأمان ضمن المصادر المتاحة حالياً."
            if language == "ar"
            else "I do not have enough reliable medical information in the available sources to answer this safely."
        )

    def status(self) -> dict[str, Any]:
        """Return configured dependency state without another network call."""
        return {
            "qdrant": self.retriever.status(),
            "embedder_backend": self.settings.embedding.backend,
            "rewriter_model": self.rewriter.model,
            "generator_model": self.generator.model,
            "relevance_gate_threshold": self.relevance_gate.threshold,
        }
