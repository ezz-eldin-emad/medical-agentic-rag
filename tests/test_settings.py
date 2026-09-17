from src.config import get_settings


def test_settings_keep_behavioral_defaults_centralized(monkeypatch):
    monkeypatch.setenv("LLM_GENERATOR_MODEL", "legacy/should-not-control-default")
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.llm.generator_model == "groq/openai/gpt-oss-120b"
    assert settings.llm.classifier_model == "groq/allam-2-7b"
    assert settings.rag.reranker_model == "BAAI/bge-reranker-base"


def test_deployment_overrides_are_typed_and_process_env_wins(monkeypatch):
    monkeypatch.setenv("EMBEDDER_BACKEND", "local")
    monkeypatch.setenv("QDRANT_URL", "http://example-qdrant:6333")
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.embedding.backend == "local"
    assert settings.vectordb.qdrant_url == "http://example-qdrant:6333"


def test_local_override_is_immutable_and_does_not_change_base_settings(monkeypatch):
    monkeypatch.setenv("EMBEDDER_BACKEND", "modal")
    monkeypatch.setenv("MODAL_EMBED_URL", "https://example.modal.run")
    get_settings.cache_clear()
    settings = get_settings()
    local_settings = settings.for_local()

    assert settings.embedding.backend == "modal"
    assert local_settings.embedding.backend == "local"
    assert local_settings.vectordb.qdrant_url == ""
    assert settings.embedding.modal_url == "https://example.modal.run"
