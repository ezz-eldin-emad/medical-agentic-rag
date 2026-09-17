from __future__ import annotations

from src.embeddings import local
from src.embeddings.factory import get_embedder
from src.config import get_settings
from src.llm.config import GENERATOR_MODEL, REWRITER_MODEL, normalize_model
from src.rag.retriever import reciprocal_rank_fusion


def test_local_embedder_is_the_default(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEmbedder:
        model_id = "fake"
        revision = ""

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("EMBEDDER_BACKEND", "local")
    get_settings.cache_clear()
    monkeypatch.setattr(local, "BGEM3LocalEmbedder", FakeEmbedder)

    get_embedder()

    assert captured["use_fp16"] is False


def test_provider_model_ids_are_not_rewritten():
    assert normalize_model("groq/llama-3.1-8b-instant") == "groq/llama-3.1-8b-instant"


def test_groq_defaults_use_current_model_family():
    assert REWRITER_MODEL == "groq/openai/gpt-oss-20b"
    assert GENERATOR_MODEL == "groq/openai/gpt-oss-120b"


def test_reciprocal_rank_fusion_keeps_shared_documents_first():
    dense = [{"id": "shared"}, {"id": "dense-only"}]
    sparse = [{"id": "shared"}, {"id": "sparse-only"}]

    results = reciprocal_rank_fusion(dense, sparse, top_n=3)

    assert results[0]["id"] == "shared"
    assert len(results) == 3
    assert all("rrf_score" in result for result in results)
