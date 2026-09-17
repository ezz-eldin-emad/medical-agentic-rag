"""Render the canonical response envelope for different audiences."""

from __future__ import annotations

from typing import Any


def render_patient_response(result: dict[str, Any]) -> str:
    from src.guardrails.output_guardrails import evaluate
    result = evaluate(result)
    """Return only the safe response intended for a patient or chat user."""

    response = result.get("user_response") or result.get("answer")
    if response:
        return str(response).strip()
    return "I could not produce a safe response. Please try again or contact a healthcare professional."


def render_internal_response(result: dict[str, Any]) -> dict[str, Any]:
    """Return the complete structured record for internal use and evaluation."""

    return dict(result)
