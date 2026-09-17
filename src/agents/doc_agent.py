"""Medical documentation agent backed by the existing cited RAG pipeline."""

from __future__ import annotations

from typing import Any

from src.config import AppSettings
from src.rag.pipeline import MedicalRAGPipeline


class DocumentationAgent:
    def __init__(
        self,
        pipeline: MedicalRAGPipeline | None = None,
        settings: AppSettings | None = None,
        tracer: Any | None = None,
    ) -> None:
        self.pipeline = pipeline or MedicalRAGPipeline(settings=settings, tracer=tracer)

    def handle(self, query: str, **kwargs: Any) -> dict[str, Any]:
        result = self.pipeline.run(
            query=query,
            top_k=kwargs.get("top_k", 5),
            enable_query_rewrite=True,
            patient_context=kwargs.get("patient_context"),
        )
        result["route"] = "medical_query"
        result["agent"] = "documentation_agent"
        return result
