"""Final response safety checks before text reaches a patient-facing channel."""
from __future__ import annotations

from typing import Any


def evaluate(result: dict[str, Any]) -> dict[str, Any]:
    """Return a safe, user-facing result envelope without exposing internals."""
    out = dict(result)
    status = str(out.get("status", ""))
    if status == "answered":
        evidence = out.get("evidence") or out.get("citations") or []
        if not evidence:
            out["status"] = "insufficient_evidence"
            out["user_response"] = "I don't have enough reliable evidence to answer that safely. Please consult a qualified clinician."
            out["answer"] = out["user_response"]
            return out
        answer = str(out.get("user_response") or out.get("answer") or "").strip()
        if not answer:
            out["status"] = "insufficient_evidence"
            answer = "I don't have enough reliable evidence to answer that safely. Please consult a qualified clinician."
        language = str(out.get("language", "en"))
        disclaimer = "هذه المعلومات تعليمية وليست تشخيصًا طبيًا." if language == "ar" else "Educational information only; not a medical diagnosis."
        if disclaimer not in answer:
            answer = f"{answer}\n\n{disclaimer}"
        out["user_response"] = answer
        out["answer"] = answer
    return out
