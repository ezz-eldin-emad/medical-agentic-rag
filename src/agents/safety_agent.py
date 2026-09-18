"""Emergency handling agent."""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import AppSettings, get_settings
class SafetyAgent:
    """Return immediate escalation guidance without invoking an LLM."""

    def __init__(self, clinic_path: Path | None = None, settings: AppSettings | None = None) -> None:
        app_settings = settings or get_settings()
        path = clinic_path or app_settings.resolve_path(app_settings.agents.clinic_data_path)
        if path.exists():
            self._clinic = json.loads(path.read_text(encoding="utf-8"))
        else:
            self._clinic = {"clinic_info": {}}

    def handle(self, *, query: str = "", **_: object) -> dict[str, object]:
        from src.rag.contracts import detect_language
        lang = detect_language(query)
        info = self._clinic.get("clinic_info", {})
        emergency_phone = info.get("emergency_phone", "")
        if lang == "ar":
            answer = "هذه قد تكون حالة طارئة. اتصل برقم الطوارئ المحلي أو توجه إلى أقرب قسم طوارئ فوراً. لا تنتظر رداً عبر الإنترنت."
            if emergency_phone:
                answer += f"\nرقم طوارئ العيادة: {emergency_phone}"
        else:
            answer = "This may be a medical emergency. Call your local emergency number or go to the nearest emergency department immediately. Do not wait for an online response."
            if emergency_phone:
                answer += f"\nClinic emergency phone: {emergency_phone}"
        return {"route": "emergency", "answer": answer, "citations": [], "query": query, "language": lang}
