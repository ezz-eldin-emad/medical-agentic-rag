from datetime import datetime, timedelta, timezone

from src.state.qdrant_store import QdrantStateStore


class MemoryStateStore(QdrantStateStore):
    def __init__(self):
        self.values = {}

    def get(self, namespace, key):
        return self.values.get((namespace, key))

    def put(self, namespace, key, value):
        self.values[(namespace, key)] = value


def test_update_claim_is_exclusive_until_completed():
    store = MemoryStateStore()

    assert store.claim_update("42") is True
    assert store.claim_update("42") is False

    store.complete_update("42")
    assert store.claim_update("42") is False
    assert store.values[("telegram_update", "42")]["status"] == "completed"


def test_failed_update_can_be_retried():
    store = MemoryStateStore()

    assert store.claim_update("42") is True
    store.fail_update("42")
    assert store.claim_update("42") is True
    assert store.values[("telegram_update", "42")]["attempts"] == 2


def test_expired_processing_lease_can_be_reclaimed():
    store = MemoryStateStore()
    old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    store.values[("telegram_update", "42")] = {
        "status": "processing",
        "claimed_at": old_time.isoformat(),
        "attempts": 1,
    }

    assert store.claim_update("42", lease_seconds=300) is True
    assert store.values[("telegram_update", "42")]["attempts"] == 2
