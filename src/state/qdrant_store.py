"""Small keyed state store backed by the configured Qdrant instance.

The medical knowledge collection remains vector-search data. This collection
stores only demo session/booking/update payloads, allowing the Vercel
filesystem to stay ephemeral without introducing another hosted database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from src.config import AppSettings, get_settings


class QdrantStateStore:
    """CRUD repository for namespaced JSON-like records in Qdrant."""

    _namespace = "medical-agentic-rag-state"

    def __init__(self, client: Any, collection_name: str, *, ensure_collection: bool = True) -> None:
        self.client = client
        self.collection_name = collection_name
        if ensure_collection:
            self.ensure_collection()

    @classmethod
    def from_settings(cls, settings: AppSettings | None = None) -> "QdrantStateStore":
        from src.rag.retriever import connect_qdrant

        app_settings = settings or get_settings()
        client, _ = connect_qdrant(app_settings.vectordb.medical_collection, app_settings)
        return cls(client, app_settings.vectordb.state_collection)

    @staticmethod
    def _point_id(namespace: str, key: str) -> str:
        return str(uuid.uuid5(uuid.UUID("a3f1b2c4-d5e6-7890-abcd-ef1234567890"), f"{namespace}:{key}"))

    def ensure_collection(self) -> None:
        from qdrant_client import models

        names = {item.name for item in self.client.get_collections().collections}
        if self.collection_name not in names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
            )

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        point_id = self._point_id(namespace, key)
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return None
        payload = points[0].payload or {}
        value = payload.get("value")
        return value if isinstance(value, dict) else None

    def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        from qdrant_client import models

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=self._point_id(namespace, key),
                    vector=[0.0],
                    payload={"namespace": namespace, "key": key, "value": value},
                )
            ],
        )

    def delete(self, namespace: str, key: str) -> None:
        from qdrant_client import models

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[self._point_id(namespace, key)]),
        )

    def list(self, namespace: str, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        from qdrant_client import models

        must = [models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace))]
        for key, value in (filters or {}).items():
            must.append(models.FieldCondition(key=f"value.{key}", match=models.MatchValue(value=value)))
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(must=must),
            limit=1024,
            with_payload=True,
        )
        values = []
        for point in records:
            payload = point.payload or {}
            value = payload.get("value")
            if isinstance(value, dict):
                values.append(value)
        return values

    def claim_update(self, update_id: str, *, lease_seconds: int = 300) -> bool:
        """Claim a Telegram update unless it is already completed or leased."""

        key = str(update_id)
        current = self.get("telegram_update", key)
        now = datetime.now(timezone.utc)
        if current:
            if current.get("status") == "completed":
                return False
            try:
                claimed_at = datetime.fromisoformat(str(current.get("claimed_at", "")))
            except ValueError:
                claimed_at = now
            if current.get("status") == "processing" and (now - claimed_at).total_seconds() < lease_seconds:
                return False
            attempts = int(current.get("attempts", 0)) + 1
        else:
            attempts = 1
        self.put(
            "telegram_update",
            key,
            {"status": "processing", "claimed_at": now.isoformat(), "attempts": attempts},
        )
        return True

    def complete_update(self, update_id: str) -> None:
        """Mark a claimed Telegram update as completed."""

        key = str(update_id)
        current = self.get("telegram_update", key) or {}
        self.put(
            "telegram_update",
            key,
            {**current, "status": "completed", "completed_at": datetime.now(timezone.utc).isoformat()},
        )

    def fail_update(self, update_id: str) -> None:
        """Release a failed update so a later delivery can retry it."""

        key = str(update_id)
        current = self.get("telegram_update", key) or {}
        self.put("telegram_update", key, {**current, "status": "failed"})
