"""Deterministic input sanitization for the first safety boundary.

This module intentionally does not claim to identify every form of PII or prompt
injection. It provides conservative, explainable checks before a query reaches
an LLM. Findings are returned to callers so they can be logged or escalated
without storing the original sensitive value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata


@dataclass(frozen=True)
class SanitizationResult:
    """The safe representation and findings for one user input."""

    original_length: int
    sanitized_text: str
    allowed: bool
    prompt_injection_detected: bool = False
    pii_detected: bool = False
    malicious_content_detected: bool = False
    findings: tuple[str, ...] = field(default_factory=tuple)


class InputSanitizer:
    """Sanitize untrusted text and detect common high-risk patterns.

    PII is masked rather than returned to downstream components. Injection and
    executable-content findings cause rejection because passing the text on can
    alter agent behavior or create unsafe rendering/execution paths.
    """

    _INJECTION_PATTERNS = (
        re.compile(r"\b(ignore|disregard|override)\b.{0,80}\b(previous| acima|system|developer|instructions?)\b", re.I | re.S),
        re.compile(r"\b(system|developer)\s*(prompt|message)\b", re.I),
        re.compile(r"\b(jailbreak| DAN mode|do anything now)\b", re.I),
        re.compile(r"\b(reveal|print|show|leak)\b.{0,50}\b(prompt|instructions?|secret|api key)\b", re.I | re.S),
        re.compile(r"<\s*(system|instruction|prompt)\b", re.I),
    )
    _PII_PATTERNS = (
        ("email", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.I)),
        ("phone", re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")),
        ("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
        ("card", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
    )
    _MALICIOUS_PATTERNS = (
        ("script", re.compile(r"<\s*/?\s*script\b|javascript\s*:", re.I)),
        ("event_handler", re.compile(r"\bon\w+\s*=", re.I)),
        ("control_character", re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")),
    )

    def sanitize(self, text: str) -> SanitizationResult:
        """Return a normalized, PII-masked query and safety findings."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        normalized = unicodedata.normalize("NFKC", text).strip()
        findings: list[str] = []
        injection = any(pattern.search(normalized) for pattern in self._INJECTION_PATTERNS)
        if injection:
            findings.append("prompt_injection")

        masked = normalized
        pii = False
        date_pattern = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$")
        for label, pattern in self._PII_PATTERNS:
            if label == "phone":
                found_phone = False
                def _replace_phone(m: re.Match[str]) -> str:
                    nonlocal found_phone
                    val = m.group(0).strip()
                    if date_pattern.match(val):
                        return m.group(0)
                    found_phone = True
                    return "[REDACTED_PHONE]"
                masked = pattern.sub(_replace_phone, masked)
                if found_phone:
                    pii = True
                    findings.append("pii:phone")
            else:
                if pattern.search(masked):
                    pii = True
                    findings.append(f"pii:{label}")
                    masked = pattern.sub(f"[REDACTED_{label.upper()}]", masked)

        malicious = False
        for label, pattern in self._MALICIOUS_PATTERNS:
            if pattern.search(masked):
                malicious = True
                findings.append(f"malicious:{label}")

        # Remove markup brackets from otherwise harmless text, while rejecting
        # executable/script constructs above.
        cleaned = re.sub(r"<[^>]*>", " ", masked)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return SanitizationResult(
            original_length=len(text),
            sanitized_text=cleaned,
            # PII is rejected at this boundary. The redacted value remains
            # available for safe downstream logging or a user-facing retry.
            allowed=bool(cleaned) and not injection and not pii and not malicious,
            prompt_injection_detected=injection,
            pii_detected=pii,
            malicious_content_detected=malicious,
            findings=tuple(findings),
        )

    def __call__(self, text: str) -> SanitizationResult:
        return self.sanitize(text)
