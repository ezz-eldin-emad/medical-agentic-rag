"""Small JSON-backed patient-context repository for the inquiry loop.

The repository stores sanitized structured fields only. It intentionally has a
small interface so the storage backend can later be replaced without changing
the agents or adapters.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import AppSettings, get_settings


@dataclass
class PatientContext:
    context_key: str
    session_id: str = ""
    language: str = "en"
    symptoms: list[str] = field(default_factory=list)
    onset_or_duration: str = ""
    trajectory: str = ""
    severity: str = ""
    associated_symptoms: list[str] = field(default_factory=list)
    relevant_context: str = ""
    sufficiency_score: float = 0.0
    missing_fields: list[str] = field(default_factory=list)
    next_question: str = ""
    turn_count: int = 0
    status: str = "collecting"
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PatientContext":
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        values = {key: value for key, value in payload.items() if key in allowed}
        return cls(**values)


class PatientContextManager:
    """Atomic JSON repository keyed by a non-reversible conversation key."""

    def __init__(self, path: Path | None = None, settings: AppSettings | None = None, store: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store
        if self.store is None and path is None and self.settings.runtime.state_backend == "qdrant":
            from src.state import QdrantStateStore

            self.store = QdrantStateStore.from_settings(self.settings)
        configured = path or self.settings.agents.patient_context_path
        self.path = self.settings.resolve_path(configured)

    @staticmethod
    def context_key(user_ref: str) -> str:
        """Hash adapter identifiers before storing them on disk."""

        return hashlib.sha256(user_ref.strip().encode("utf-8")).hexdigest()[:32]

    def get(self, context_key: str) -> PatientContext | None:
        if self.store is not None:
            payload = self.store.get("patient_context", context_key)
            return PatientContext.from_dict(payload) if payload else None
        records = self._read()
        payload = records.get(context_key)
        return PatientContext.from_dict(payload) if isinstance(payload, dict) else None

    def create_or_update(self, context_key: str, updates: dict[str, Any]) -> PatientContext:
        if self.store is not None:
            current = self.get(context_key) or PatientContext(context_key=context_key, session_id=uuid.uuid4().hex)
            for key, value in updates.items():
                if key in current.__dataclass_fields__ and key != "context_key":
                    setattr(current, key, value)
            current.updated_at = datetime.now(timezone.utc).isoformat()
            self.store.put("patient_context", context_key, current.to_dict())
            return current
        records = self._read()
        current = PatientContext.from_dict(records[context_key]) if context_key in records else PatientContext(
            context_key=context_key,
            session_id=uuid.uuid4().hex,
        )
        for key, value in updates.items():
            if key in current.__dataclass_fields__ and key != "context_key":
                setattr(current, key, value)
        current.updated_at = datetime.now(timezone.utc).isoformat()
        records[context_key] = current.to_dict()
        self._write(records)
        return current

    def reset_for_user(self, user_ref: str) -> None:
        """End the current conversation session for an adapter user."""

        self.clear(self.context_key(user_ref))

    @staticmethod
    def is_stale(context: PatientContext, timeout_minutes: int) -> bool:
        """Return whether an unfinished context exceeded the inactivity window."""

        if not context.updated_at or timeout_minutes <= 0:
            return False
        try:
            updated = datetime.fromisoformat(context.updated_at)
        except ValueError:
            return False
        return (datetime.now(timezone.utc) - updated).total_seconds() > timeout_minutes * 60

    def clear(self, context_key: str) -> None:
        if self.store is not None:
            self.store.delete("patient_context", context_key)
            return
        records = self._read()
        if context_key in records:
            del records[context_key]
            self._write(records)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write(self, records: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix="patient-context-", suffix=".json", dir=self.path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
