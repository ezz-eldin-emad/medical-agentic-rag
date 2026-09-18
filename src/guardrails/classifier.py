"""Fast, explainable query classification for Layer 1 routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
import logging
from typing import Any

from src.config import AppSettings, get_settings
from src.llm.client import LLMClient

log = logging.getLogger(__name__)


class QueryClass(str, Enum):
    EMERGENCY = "emergency"
    CLINIC = "clinic_query"
    MEDICAL = "medical_query"


@dataclass(frozen=True)
class QueryClassification:
    query_class: QueryClass
    confidence: float
    matched_terms: tuple[str, ...] = ()
    reason: str = ""
    intent: str = ""
    entities: dict[str, Any] | None = None
    source: str = "deterministic"


class QueryClassifier:
    """Classify queries with an optional LLM and a deterministic safety fallback."""

    _EMERGENCY = (
        "can't breathe", "cannot breathe", "difficulty breathing", "chest pain",
        "unconscious", "not responding", "severe bleeding", "stroke", "seizure",
        "suicide", "overdose", "anaphylaxis", "vomiting blood", "heart attack",
        "لا أستطيع التنفس", "صعوبة في التنفس", "ألم في الصدر", "نزيف شديد",
        "نزيف حاد", "فاقد الوعي", "فاقد للوعي", "سكتة دماغية", "جلطة", "جلطه",
        "تشنج", "جرعة زائدة", "حساسية شديدة", "أزمة قلبية", "ازمة قلبية",
        "توقف القلب", "مش قادر اتنفس", "مش عارف اتنفس", "إغماء", "اغماء",
    )
    _CLINIC = (
        "appointment", "book", "booking", "cancel", "reschedule", "clinic",
        "doctor availability", "opening hours", "hours", "price", "pricing",
        "cost", "insurance", "location", "address",
        "موعد", "مواعيد", "حجز", "احجز", "إلغاء", "الغاء", "إعادة الحجز",
        "عيادة", "العياده", "سعر", "أسعار", "اسعار", "تكلفة", "تأمين",
        "التأمين", "عنوان", "موقع", "ساعات العمل", "مواعيد الأطباء",
        "تخصصات", "تخصص", "دكاترة", "دكتور", "أطباء", "طبيب", "متاح", "المتاح",
    )

    _CLASSIFIER_PROMPT = """You are a safe routing classifier for a medical clinic assistant.
Classify the user's message, without answering it, into exactly one query_class:
"emergency", "clinic_query", or "medical_query".

Use intent values:
- emergency
- clinic_info
- availability
- booking
- cancellation
- confirmation
- medical_question

Return only valid JSON with this shape:
{"query_class":"...","intent":"...","confidence":0.0,"reason":"short reason", "entities":{"doctor":"", "specialty":"", "date":"YYYY-MM-DD", "weekday":"", "time":"HH:MM", "booking_id":""}}

Routing rules: questions about clinic doctors, specialties, services, prices,
working hours, appointments, or booking are clinic_query even when the user
also mentions a symptom. Questions asking for symptoms, causes, or medical
explanations without clinic logistics are medical_query. Short follow-ups such
as "غيرهم؟" inherit the previous route when conversation context is supplied.
Use null for unknown entities. Any possible emergency symptom must be classified as emergency.
"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str | None = None,
        *,
        use_llm: bool = False,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.llm.classifier_model
        self.use_llm = use_llm
        self.llm_client = llm_client or (
            LLMClient(default_model=self.model, settings=self.settings) if use_llm else None
        )

    @staticmethod
    def _matches(query: str, terms: tuple[str, ...]) -> tuple[str, ...]:
        lowered = query.casefold()
        return tuple(term for term in terms if term in lowered)

    @classmethod
    def has_deterministic_emergency_signal(cls, query: str) -> bool:
        """Check only explicit emergency phrases, never an LLM guess."""

        return bool(cls._matches(query, cls._EMERGENCY))

    def classify(self, query: str) -> QueryClassification:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        normalized = re.sub(r"\s+", " ", query).strip()
        emergency = self._matches(normalized, self._EMERGENCY)
        if emergency:
            return QueryClassification(
                QueryClass.EMERGENCY, 0.99, emergency,
                "Emergency terms detected; escalate immediately.",
                intent="emergency", source="deterministic_emergency_gate",
            )

        if self.use_llm and self.llm_client is not None:
            llm_result = self._classify_with_llm(normalized)
            if llm_result is not None:
                log.info("Routing decision: class=%s intent=%s confidence=%.2f source=%s", llm_result.query_class.value, llm_result.intent, llm_result.confidence, llm_result.source)
                # The LLM cannot downgrade a deterministic emergency finding;
                # this second check protects against Arabic paraphrase errors.
                if llm_result.query_class is QueryClass.EMERGENCY:
                    return llm_result
                return llm_result

        clinic = self._matches(normalized, self._CLINIC)
        if clinic:
            intent = self._normalize_clinic_intent(normalized, "clinic_info")
            entities = self._enrich_entities(normalized, {})
            return QueryClassification(
                QueryClass.CLINIC, 0.92, clinic,
                "Clinic terms detected; use clinic flow.",
                intent=intent,
                entities=entities,
                source="deterministic_clinic_gate",
            )

        return QueryClassification(
            QueryClass.MEDICAL, 0.70, (),
            "No emergency or clinic routing term detected; use medical flow.",
            intent="medical_question",
        )

    def _classify_with_llm(self, query: str) -> QueryClassification | None:
        """Classify natural-language variants through the configured small LLM."""
        try:
            response = self.llm_client.complete(
                messages=[{
                    "role": "user",
                    "content": f"{self._CLASSIFIER_PROMPT}\n\nUSER MESSAGE:\n{query}",
                }],
                model=self.model,
                temperature=self.settings.llm.classifier_temperature,
                response_format={"type": "json_object"},
                num_retries=1,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```", 2)[1].removeprefix("json").strip()
            parsed = json.loads(content)
            query_class = QueryClass(str(parsed.get("query_class", "")).lower())
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
            entities = parsed.get("entities")
            if not isinstance(entities, dict):
                entities = {}
            entities = self._enrich_entities(query, entities)
            intent = str(parsed.get("intent", ""))
            if query_class is QueryClass.CLINIC:
                intent = self._normalize_clinic_intent(query, intent)
            return QueryClassification(
                query_class=query_class,
                confidence=confidence,
                matched_terms=(),
                reason=str(parsed.get("reason", "LLM classification")),
                intent=intent,
                entities=entities,
                source="llm",
            )
        except Exception:
            # Routing must remain available when the provider is down or the
            # model emits malformed JSON.
            return None

    @staticmethod
    def _normalize_clinic_intent(query: str, intent: str) -> str:
        """Correct common under-specified LLM intents using explicit wording."""
        lowered = query.casefold()
        if any(term in lowered for term in ("cancel", "cancellation", "إلغاء", "الغاء")):
            return "cancellation"
        if any(term in lowered for term in ("confirm", "confirmation", "تأكيد", "تاكيد")):
            return "confirmation"
        if any(term in lowered for term in ("book", "booking", "حجز", "احجز")):
            return "booking"
        if any(term in lowered for term in ("appointment", "availability", "available", "موعد", "مواعيد", "متاحة", "متاح")):
            return "availability"
        return intent or "clinic_info"

    @staticmethod
    def _enrich_entities(query: str, entities: dict[str, Any]) -> dict[str, Any]:
        """Recover high-value specialty terms if the LLM omits them."""
        empty_markers = {"", "unknown", "none", "null", "yyyy-mm-dd", "hh:mm"}
        enriched = {
            key: (None if str(value).strip().casefold() in empty_markers else value)
            for key, value in entities.items()
        }
        if not enriched.get("specialty"):
            aliases = {
                "صدرية": "صدرية", "باطنة": "باطنة عامة", "أطفال": "أطفال", "اطفال": "أطفال",
                "جلدية": "جلدية وتناسلية", "عظام": "عظام", "جراحة": "جراحة عامة",
                "نساء": "نساء وتوليد", "توليد": "نساء وتوليد",
            }
            lowered = query.casefold()
            for term, specialty in aliases.items():
                if term in lowered:
                    enriched["specialty"] = specialty
                    break
        return enriched

    def __call__(self, query: str) -> QueryClassification:
        return self.classify(query)
