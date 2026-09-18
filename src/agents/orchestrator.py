"""Layer 5 routing coordinator."""

from __future__ import annotations

import uuid
from typing import Any

from src.config import AppSettings, get_settings
from src.guardrails.classifier import QueryClass, QueryClassification, QueryClassifier
from src.guardrails.input_guardrails import InputSanitizer
from src.memory import PatientContextManager
from src.rag.contracts import detect_language

from .clinic_agent import ClinicAgent
from .clarifying_agent import ClarifyingAgent
from .doc_agent import DocumentationAgent
from .inquiry_loop import InquiryLoop
from .safety_agent import SafetyAgent
from .tracing import get_tracer


class AgentOrchestrator:
    """Route one sanitized user request to exactly one specialized agent."""

    def __init__(
        self,
        *,
        classifier: QueryClassifier | None = None,
        clinic_agent: ClinicAgent | None = None,
        safety_agent: SafetyAgent | None = None,
        documentation_agent: DocumentationAgent | None = None,
        tracer: Any | None = None,
        clarifying_agent: ClarifyingAgent | None = None,
        inquiry_loop: InquiryLoop | None = None,
        context_manager: PatientContextManager | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        app_settings = settings or get_settings()
        self.settings = app_settings
        self.tracer = tracer or get_tracer(app_settings)
        self.classifier = classifier or QueryClassifier(
            model=app_settings.llm.classifier_model,
            use_llm=True,
            settings=app_settings,
        )
        self.clinic_agent = clinic_agent or ClinicAgent(settings=app_settings)
        self.safety_agent = safety_agent or SafetyAgent(settings=app_settings)
        self.context_manager = context_manager or PatientContextManager(settings=app_settings)
        self.clarifying_agent = clarifying_agent or ClarifyingAgent()
        self.inquiry_loop = inquiry_loop or InquiryLoop(
            self.context_manager,
            settings=app_settings,
            use_llm=True,
        )
        self._documentation_agent = documentation_agent
        self.sanitizer = InputSanitizer()

    @property
    def pipeline(self) -> Any:
        """Expose the medical pipeline for the Telegram status command."""
        return self.documentation_agent.pipeline

    @property
    def documentation_agent(self) -> DocumentationAgent:
        if self._documentation_agent is None:
            self._documentation_agent = DocumentationAgent(settings=self.settings, tracer=self.tracer)
        return self._documentation_agent

    def handle(self, query: str, *, user_ref: str = "anonymous", top_k: int | None = None) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        with self.tracer.span("agent_orchestrator", {"query_length": len(query)}) as span:
            sanitization = self.sanitizer.sanitize(query)
            if not sanitization.allowed:
                language = detect_language(query)
                answer = "لا أستطيع معالجة هذا الطلب. اكتب سؤالك الطبي بصياغة آمنة ومباشرة." if language == "ar" else "I cannot process that request. Please rephrase your medical question safely and directly."
                return {"request_id": request_id, "trace_id": request_id, "query": query, "route": "blocked", "agent": "guardrails", "status": "blocked", "user_response": answer, "answer": answer, "citations": [], "claims": [], "evidence": []}
            query = sanitization.sanitized_text
            context_key = self.context_manager.context_key(user_ref)
            existing_context = self.context_manager.get(context_key)
            if existing_context and (
                existing_context.status in {"ready", "completed"}
                or PatientContextManager.is_stale(
                    existing_context, self.settings.agents.session_timeout_minutes
                )
            ):
                self.context_manager.clear(context_key)
                existing_context = None

            classification = self.classifier.classify(query)
            # A short reply such as "6" or "منذ يومين" belongs to the active inquiry.
            # Explicit emergency phrases still win through the deterministic emergency gate.
            if (
                existing_context is not None
                and existing_context.status == "collecting"
                and not QueryClassifier.has_deterministic_emergency_signal(query)
            ):
                classification = QueryClassification(
                    query_class=QueryClass.MEDICAL,
                    confidence=1.0,
                    reason="Reply to an active medical inquiry; preserve inquiry routing.",
                    intent="medical_question",
                    source="active_inquiry",
                )
            span["classification"] = classification.query_class.value
            span["confidence"] = classification.confidence

            common = {
                "request_id": request_id,
                "trace_id": request_id,
                "query": query,
                "query_class": classification.query_class.value,
                "classification": {
                    "intent": classification.intent,
                    "confidence": classification.confidence,
                    "reason": classification.reason,
                    "entities": classification.entities or {},
                    "source": classification.source,
                },
            }
            entities = classification.entities or {}
            if (
                classification.query_class is not QueryClass.EMERGENCY
                and classification.confidence < self.settings.agents.classifier_confidence_threshold
            ):
                language = detect_language(query)
                answer = (
                    "لأساعدك بأمان، اذكر العرض أو المشكلة التي تريد مناقشتها، مع مكانها ومدة وجودها."
                    if language == "ar"
                    else "To help you safely, please describe the symptom or concern, where it is, and how long it has been present."
                )
                return {
                    **common,
                    "route": "clarification_needed",
                    "agent": "orchestrator",
                    "status": "clarification_needed",
                    "user_response": answer,
                    "answer": answer,
                    "citations": [],
                }
            if classification.query_class is QueryClass.EMERGENCY:
                result = self.safety_agent.handle(query=query)
                return {**common, **self._normalize_result(result, "emergency"), "agent": "safety_agent"}

            if classification.query_class is QueryClass.CLINIC:
                intent = classification.intent or "clinic_info"
                result = self.clinic_agent.handle(intent=intent, entities=entities, user_ref=user_ref, query=query)
                return {**common, **self._normalize_result(result, "clinic_query"), "agent": "clinic_agent"}

            with self.tracer.span("clarifying_agent", {"query_length": len(query)}) as clarifying_span:
                clarification = self.clarifying_agent.analyze(query, existing_context)
                clarifying_span["is_personal"] = clarification["is_personal"]
                clarifying_span["needs_inquiry"] = clarification["needs_inquiry"]

            patient_context = None
            medical_query = query
            if clarification["needs_inquiry"]:
                with self.tracer.span("inquiry_loop", {"context_key": context_key}) as inquiry_span:
                    inquiry = self.inquiry_loop.process(context_key, query, clarification)
                    inquiry_span["status"] = inquiry["status"]
                    inquiry_span["sufficiency_score"] = inquiry["context"].sufficiency_score
                if inquiry["status"] != "ready":
                    answer = inquiry.get("answer", "I need a little more information before continuing.")
                    return {
                        **common,
                        "route": "clarification_needed",
                        "agent": "clarifying_agent",
                        "status": inquiry["status"],
                        "user_response": answer,
                        "answer": answer,
                        "citations": [],
                        "claims": [],
                        "evidence": [],
                        "context_key": context_key,
                        "sufficiency_score": inquiry["context"].sufficiency_score,
                        "missing_fields": inquiry["context"].missing_fields,
                        "session_id": inquiry["context"].session_id,
                    }
                patient_context = inquiry["context"].to_dict()
                medical_query = inquiry["augmented_query"]

            result = self.documentation_agent.handle(
                medical_query,
                top_k=top_k,
                patient_context=patient_context,
            )
            return {**common, **result, "agent": "documentation_agent", "route": "medical_query"}

    def reset_session(self, *, user_ref: str) -> None:
        """Explicitly start a fresh conversation for one user."""

        self.context_manager.reset_for_user(user_ref)

    @staticmethod
    def _normalize_result(result: dict[str, Any], default_status: str) -> dict[str, Any]:
        """Give direct agents the same response contract as the RAG route."""

        answer = str(result.get("user_response") or result.get("answer") or "")
        normalized = dict(result)
        normalized.update(
            {
                "status": result.get("status", default_status),
                "user_response": answer,
                "answer": answer,
                "claims": result.get("claims", []),
                "evidence": result.get("evidence", []),
                "citations": result.get("citations", []),
            }
        )
        return normalized
