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
        if clinic_path is None and app_settings.runtime.state_backend == "qdrant":
            self._clinic = {"clinic_info": {}}
            try:
                from src.rag.retriever import connect_qdrant
                client, _ = connect_qdrant(app_settings.vectordb.clinic_collection, app_settings)
                points, _ = client.scroll(collection_name=app_settings.vectordb.clinic_collection, limit=128, with_payload=True, with_vectors=False)
                text = "\n".join(str((point.payload or {}).get("text", "")) for point in points)
                match = re.search(r"Emergency:\s*([+\d ()-]{7,})", text, re.I)
                if match:
                    self._clinic["clinic_info"]["emergency_phone"] = match.group(1).strip()
            except Exception:
                pass
            return
        path = clinic_path or app_settings.resolve_path(app_settings.agents.clinic_data_path)
        self._clinic = json.loads(path.read_text(encoding="utf-8"))

    def handle(self, *, query: str = "", **_: object) -> dict[str, object]:
        info = self._clinic.get("clinic_info", {})
        emergency_phone = info.get("emergency_phone", "")
        answer = "هذه قد تكون حالة طارئة. اتصل برقم الطوارئ المحلي أو توجه إلى أقرب قسم طوارئ فوراً. لا تنتظر رداً عبر الإنترنت."
        if emergency_phone:
            answer += f"\nرقم طوارئ العيادة: {emergency_phone}"
        return {"route": "emergency", "answer": answer, "citations": [], "query": query}
