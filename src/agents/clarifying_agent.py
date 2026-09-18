"""Layer 2: identify personal symptom questions that need inquiry."""

from __future__ import annotations

from typing import Any

from src.memory import PatientContext
from src.rag.contracts import detect_language


class ClarifyingAgent:
    """Conservative, deterministic personal-vs-general medical classifier."""

    _personal_markers = {
        "i have", "i'm having", "i am having", "my ", "for me",
        "عندي", "لدي", "أعاني", "اعاني", "أشعر", "اشعر", "ابني", "ابنتي",
    }
    _symptom_markers = {
        "pain", "ache", "fever", "cough", "rash", "swelling", "dizzy", "vomit",
        "headache", "breath", "bleeding", "ألم", "وجع", "حمى", "حرارة", "سعال",
        "تورم", "دوخة", "قيء", "صداع", "تنفس", "نزيف", "عرض", "أعراض",
    }
    _general_markers = {
        "what is", "what are", "common", "causes of", "symptoms of", "how does",
        "how to", "treatment", "treat", "cure", "remedies", "prevention", "definition",
        "ما هو", "ما هي", "ما هى", "أسباب", "اسباب", "اعراض", "أعراض مرض", "كيف يحدث",
        "علاج", "طرق علاج", "كيفية علاج", "الوقاية", "تعريف", "معلومات عن", "شرح",
    }

    def analyze(self, query: str, context: PatientContext | None = None) -> dict[str, Any]:
        normalized = query.casefold().strip()
        if context is not None and context.status == "collecting":
            return {
                "is_personal": True,
                "needs_inquiry": True,
                "language": context.language,
                "confidence": 1.0,
                "source": "context",
            }

        has_symptom = any(marker in normalized for marker in self._symptom_markers)
        has_personal = any(marker in normalized for marker in self._personal_markers)
        is_general = any(marker in normalized for marker in self._general_markers)
        is_personal = has_symptom and (has_personal or not is_general)
        return {
            "is_personal": is_personal,
            "needs_inquiry": is_personal,
            "language": detect_language(query),
            "confidence": 0.85 if is_personal else 0.8,
            "source": "deterministic",
        }
