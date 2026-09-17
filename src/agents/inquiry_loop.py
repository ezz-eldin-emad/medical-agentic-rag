"""Layer 3: one-question-per-turn inquiry loop."""

from __future__ import annotations

import re
import uuid
from typing import Any

from src.config import AppSettings, get_settings
from src.llm.client import LLMClient
from src.memory import PatientContext, PatientContextManager


class InquiryLoop:
    """Collect enough structured context before invoking medical RAG."""

    def __init__(
        self,
        context_manager: PatientContextManager | None = None,
        settings: AppSettings | None = None,
        llm_client: LLMClient | None = None,
        use_llm: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.context_manager = context_manager or PatientContextManager(settings=self.settings)
        self.max_turns = self.settings.agents.max_inquiry_turns
        self.llm_client = llm_client or (
            LLMClient(default_model=self.settings.llm.clarifier_model, settings=self.settings)
            if use_llm else None
        )

    def process(self, context_key: str, query: str, analysis: dict[str, Any]) -> dict[str, Any]:
        current = self.context_manager.get(context_key) or PatientContext(
            context_key=context_key,
            session_id=uuid.uuid4().hex,
            language=str(analysis.get("language") or "en"),
        )
        updates = self._extract_updates(current, query)
        # Let the configured API model interpret natural-language answers
        # (Arabic date/duration expressions, severity, etc.). Deterministic
        # extraction remains the safety fallback when the model is unavailable
        # or returns invalid JSON.
        if self.llm_client is not None:
            llm_updates = self._extract_updates_with_llm(current, query)
            if llm_updates:
                updates = {**updates, **{k: v for k, v in llm_updates.items() if v}}
        turn_count = current.turn_count + 1
        score, missing = self._score(updates)
        status = "ready" if score >= 0.7 else "collecting"

        if status == "collecting" and turn_count >= self.max_turns:
            status = "limit_reached"

        next_question = "" if status in {"ready", "limit_reached"} else self._next_question(
            missing=missing,
            language=current.language,
            query=query,
            context=updates,
        )
        context = self.context_manager.create_or_update(
            context_key,
            {
                **updates,
                "language": current.language,
                "sufficiency_score": score,
                "missing_fields": missing,
                "next_question": next_question,
                "turn_count": turn_count,
                "status": status,
            },
        )

        if status == "ready":
            return {
                "status": "ready",
                "context": context,
                "augmented_query": self._augmented_query(query, context),
            }
        if status == "limit_reached":
            return {
                "status": "insufficient_context",
                "context": context,
                "answer": self._limit_message(context.language),
            }
        return {
            "status": "clarification_needed",
            "context": context,
            "answer": next_question,
            "next_question": next_question,
        }

    def _extract_updates_with_llm(self, current: PatientContext, query: str) -> dict[str, Any]:
        """Extract only known context fields as JSON using the clarification model."""
        import json
        prompt = (
            "Extract facts from the user's latest medical message. Return JSON only, no markdown. "
            "Allowed keys: symptoms, onset_or_duration, trajectory, severity, associated_symptoms, relevant_context. "
            "Use empty strings or [] when a value is not present. Do not infer or diagnose.\n"
            f"Existing context: {current.to_dict()}\nLatest message: {query}"
        )
        try:
            response = self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self.settings.llm.clarifier_model,
                temperature=0,
                num_retries=1,
            )
            raw = str(response.choices[0].message.content or "").strip()
            data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
            allowed = {"symptoms", "onset_or_duration", "trajectory", "severity", "associated_symptoms", "relevant_context"}
            return {k: data[k] for k in allowed if k in data and isinstance(data[k], (str, list))}
        except Exception:
            return {}

    def _next_question(
        self,
        *,
        missing: list[str],
        language: str,
        query: str,
        context: dict[str, Any],
    ) -> str:
        """Generate a natural question, with a deterministic safe fallback."""

        fallback = self._question_for(missing, language)
        if self.llm_client is None:
            return fallback
        field = missing[0] if missing else "relevant_context"
        language_name = "Arabic" if language == "ar" else "English"
        prompt = (
            "You are the clarification agent for a medical assistant. "
            "Ask exactly one concise question to collect the missing field. "
            "Do not diagnose, provide treatment, mention internal fields, or ask for names/phone numbers. "
            f"Reply entirely in {language_name}. Return only the question.\n\n"
            f"Missing field: {field}\nUser's latest message: {query}\n"
            f"Known context: {context}"
        )
        try:
            response = self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self.settings.llm.clarifier_model,
                temperature=self.settings.llm.classifier_temperature,
                num_retries=1,
            )
            generated = str(response.choices[0].message.content or "").strip()
            if generated and len(generated) <= 500:
                return generated
        except Exception:
            return fallback
        return fallback

    def _extract_updates(self, current: PatientContext, query: str) -> dict[str, Any]:
        normalized = self._normalize_digits(query.casefold())
        symptoms = list(current.symptoms)
        if not symptoms:
            cleaned = re.sub(r"^(i have|i'm having|i am having|عندي|لدي|أعاني|اعاني)\s*", "", query, flags=re.I).strip(" .؟?")
            if cleaned:
                symptoms = [cleaned]

        duration = current.onset_or_duration
        duration_match = re.search(
            r"(?:for|since)\s+[^,.!?]+|"
            r"(?:منذ|من|لمدة|بقال(?:ي|ه|ها|و)?)\s*[^،,.!?]+|"
            r"\d+\s*(?:minute|minutes|hour|hours|day|days|week|weeks|month|months|دقيقة|دقائق|ساعة|ساعات|يوم|يومين|يومان|أيام|ايام|أسبوع|أسابيع|اسابيع|شهر|شهور)|"
            r"\b(?:today|yesterday|اليوم|أمس|امس)\b",
            normalized,
            flags=re.I,
        )
        if duration_match:
            duration = duration_match.group(0).strip()

        trajectory = current.trajectory
        if re.search(
            r"(?:improving|improved|better|worse|worsening|getting worse|same|stable|unchanged|"
            r"يتحسن|تحسن|أفضل|افضل|يزداد سوءا|يزداد سوءا|بيزيد|يزيد|أسوأ|اسوا|ثابت|مستمر|كما هو)",
            normalized,
            flags=re.I,
        ):
            trajectory = query.strip()

        severity = current.severity
        if any(marker in normalized for marker in ("severe", "mild", "moderate", "شديد", "خفيف", "متوسط")):
            severity = query.strip()
        numeric_severity = re.search(r"\b(?:[1-9]|10)\s*(?:/\s*10)?\b", query)
        if numeric_severity:
            severity = numeric_severity.group(0)

        associated = list(current.associated_symptoms)
        if re.search(r"\bwith\b|\bother symptoms?\b|\bأعراض أخرى\b|\bمع\b", normalized):
            associated = [query.strip()]

        relevant = current.relevant_context
        if re.search(r"\bage\b|\bpregnan|\bmedication|\bhistory\b|\bمرض مزمن\b|\bحامل\b|\bأدوية\b|\bتاريخ مرضي\b", normalized):
            relevant = query.strip()

        return {
            "symptoms": symptoms,
            "onset_or_duration": duration,
            "trajectory": trajectory,
            "severity": severity,
            "associated_symptoms": associated,
            "relevant_context": relevant,
        }

    @staticmethod
    def _score(values: dict[str, Any]) -> tuple[float, list[str]]:
        checks = (
            ("symptoms", 0.25, bool(values.get("symptoms"))),
            ("onset_or_duration", 0.20, bool(values.get("onset_or_duration"))),
            ("severity", 0.20, bool(values.get("severity"))),
            ("associated_symptoms", 0.20, bool(values.get("associated_symptoms"))),
            ("relevant_context", 0.15, bool(values.get("relevant_context"))),
        )
        score = round(sum(weight for _, weight, present in checks if present), 2)
        missing = [name for name, _, present in checks if not present]
        return score, missing

    @staticmethod
    def _question_for(missing: list[str], language: str) -> str:
        field = missing[0] if missing else "relevant_context"
        questions = {
            "en": {
                "symptoms": "What symptom or concern should I focus on?",
                "onset_or_duration": "How long have you had this symptom, and is it getting better or worse?",
                "severity": "How severe is it, for example mild, moderate, severe, or on a scale from 1 to 10?",
                "associated_symptoms": "Do you have any other symptoms along with it?",
                "relevant_context": "Is there any relevant context, such as your age group, pregnancy, medicines, or a medical history?",
            },
            "ar": {
                "symptoms": "ما العرض أو المشكلة التي تريد مني التركيز عليها؟",
                "onset_or_duration": "منذ متى بدأ العرض؟ وهل يتحسن أم يزداد سوءاً؟",
                "severity": "ما شدة العرض: خفيف أم متوسط أم شديد، أو من 1 إلى 10؟",
                "associated_symptoms": "هل توجد أعراض أخرى مصاحبة؟",
                "relevant_context": "هل توجد معلومات مهمة مثل الفئة العمرية أو الحمل أو الأدوية أو تاريخ مرضي؟",
            },
        }
        return questions.get(language, questions["en"])[field]

    @staticmethod
    def _augmented_query(query: str, context: PatientContext) -> str:
        return (
            f"User question: {query}\n"
            f"Structured patient context: symptoms={context.symptoms}; "
            f"onset_or_duration={context.onset_or_duration}; trajectory={context.trajectory}; "
            f"severity={context.severity}; "
            f"associated_symptoms={context.associated_symptoms}; relevant_context={context.relevant_context}"
        )

    @staticmethod
    def _normalize_digits(value: str) -> str:
        """Normalize Arabic-Indic and Persian numerals for reliable matching."""

        return value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789"))

    @staticmethod
    def _limit_message(language: str) -> str:
        return (
            "I do not have enough information to assess this safely. Please provide more detail or consult a healthcare professional."
            if language != "ar"
            else "لا أملك معلومات كافية لتقييم الحالة بأمان. يرجى تقديم مزيد من التفاصيل أو استشارة طبيب مختص."
        )
