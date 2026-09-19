"""Typed, centralized application settings.

Python defaults define application behavior. Environment variables are limited
to credentials and deployment-specific endpoints/switches, with process
environment values taking precedence over the local ``.env`` file.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import os
from pathlib import Path

from src.utils.helpers import get_project_root, load_env


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _bool_env(name: str, default: bool) -> bool:
    value = _env(name)
    if not value:
        return default
    if value.casefold() in {"1", "true", "yes", "on"}:
        return True
    if value.casefold() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class SecretsSettings:
    telegram_bot_token: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    hf_token: str = ""
    qdrant_api_key: str = ""
    phoenix_api_key: str = ""
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    anthropic_api_key: str = ""
    telegram_webhook_secret: str = ""
    modal_embed_token: str = ""


@dataclass(frozen=True)
class LLMSettings:
    default_model: str = "groq/openai/gpt-oss-20b"
    classifier_model: str = "groq/allam-2-7b"
    rewriter_model: str = "groq/openai/gpt-oss-20b"
    generator_model: str = "groq/openai/gpt-oss-120b"
    temperature: float = 0.0
    classifier_temperature: float = 0.0
    rewriter_temperature: float = 0.1
    generator_temperature: float = 0.1
    clarifier_model: str = "groq/openai/gpt-oss-20b"
    timeout_seconds: int = 90
    retries: int = 3


@dataclass(frozen=True)
class EmbeddingSettings:
    backend: str = "huggingface"
    model_id: str = "BAAI/bge-m3"
    revision: str = ""
    use_fp16: bool = False
    modal_url: str = ""
    request_timeout_seconds: float = 90.0
    max_retries: int = 5


@dataclass(frozen=True)
class VectorDBSettings:
    qdrant_url: str = ""
    qdrant_path: Path = Path("data/qdrant")
    medical_collection: str = "medical_kb"
    clinic_collection: str = "clinic_kb"
    state_collection: str = "medical_app_state"


@dataclass(frozen=True)
class RuntimeSettings:
    """Transport and persistence choices that vary by deployment."""

    telegram_transport: str = "webhook"
    public_base_url: str = ""
    state_backend: str = "qdrant"


@dataclass(frozen=True)
class AgentSettings:
    classifier_confidence_threshold: float = 0.55
    clinic_data_path: Path = Path("data/clinic/clinic_info.json")
    bookings_path: Path = Path("data/clinic/bookings.json")
    patient_context_path: Path = Path("data/memory/patient_contexts.json")
    max_inquiry_turns: int = 5
    session_timeout_minutes: int = 30


@dataclass(frozen=True)
class RAGSettings:
    reranker_model: str = "BAAI/bge-reranker-base"
    retrieval_top_k: int = 5
    relevance_gate_threshold: float = 0.2
    rrf_relevance_floor: float = 0.03


@dataclass(frozen=True)
class TracingSettings:
    enabled: bool = False
    project_name: str = "medical-agentic-rag"
    collector_endpoint: str = ""
    capture_prompts: bool = False
    capture_context: bool = False


@dataclass(frozen=True)
class AppSettings:
    secrets: SecretsSettings
    llm: LLMSettings
    embedding: EmbeddingSettings
    vectordb: VectorDBSettings
    agents: AgentSettings
    rag: RAGSettings
    tracing: TracingSettings
    runtime: RuntimeSettings
    project_root: Path

    def resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.project_root / path

    def for_local(self) -> "AppSettings":
        """Return CLI-local overrides without mutating process environment."""
        return replace(
            self,
            embedding=replace(self.embedding, backend="local", use_fp16=False),
            vectordb=replace(self.vectordb, qdrant_url=""),
            runtime=replace(self.runtime, telegram_transport="polling", state_backend="local"),
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Build and cache validated settings from defaults plus allowed overrides."""
    load_env()
    root = get_project_root()
    backend = _env("EMBEDDER_BACKEND", "huggingface").casefold()
    if backend not in {"local", "modal", "modal_api", "remote", "huggingface", "hf", "hf_inference", "flag", "flagembedding"}:
        raise ValueError(f"Unsupported EMBEDDER_BACKEND={backend!r}")
    return AppSettings(
        project_root=root,
        secrets=SecretsSettings(
            telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
            groq_api_key=_env("GROQ_API_KEY"),
            gemini_api_key=_env("GEMINI_API_KEY"),
            hf_token=_env("HF_TOKEN"),
            qdrant_api_key=_env("QDRANT_API_KEY"),
            phoenix_api_key=_env("PHOENIX_API_KEY"),
            openai_api_key=_env("OPENAI_API_KEY"),
            openrouter_api_key=_env("OPENROUTER_API_KEY"),
            anthropic_api_key=_env("ANTHROPIC_API_KEY"),
            telegram_webhook_secret=_env("TELEGRAM_WEBHOOK_SECRET"),
            modal_embed_token=_env("MODAL_EMBED_TOKEN"),
        ),
        llm=LLMSettings(),
        embedding=EmbeddingSettings(
            backend=backend,
            modal_url=_env("MODAL_EMBED_URL"),
        ),
        vectordb=VectorDBSettings(
            qdrant_url=_env("QDRANT_URL"),
            qdrant_path=Path(_env("QDRANT_PATH", "data/qdrant")),
            medical_collection=_env(
                "QDRANT_STORAGE_COLLECTION",
                _env("QDRANT_MEDICAL_COLLECTION", "medical_kb"),
            ),
            clinic_collection=_env("QDRANT_CLINIC_COLLECTION", "clinic_kb"),
            state_collection=_env("QDRANT_STATE_COLLECTION", "medical_app_state"),
        ),
        agents=AgentSettings(
            clinic_data_path=Path("data/clinic/clinic_info.json"),
            bookings_path=Path("data/clinic/bookings.json"),
        ),
        rag=RAGSettings(),
        tracing=TracingSettings(
            enabled=_bool_env("PHOENIX_ENABLED", False),
            project_name=_env("PHOENIX_PROJECT_NAME", "medical-agentic-rag"),
            collector_endpoint=_env("PHOENIX_COLLECTOR_ENDPOINT"),
        ),
        runtime=RuntimeSettings(
        telegram_transport=_env("TELEGRAM_TRANSPORT", "webhook").casefold(),
            public_base_url=_env("PUBLIC_BASE_URL"),
        state_backend=_env("STATE_BACKEND", "qdrant").casefold(),
        ),
    )
