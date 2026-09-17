"""
Module: query_rewriter.py
Purpose: Fast medical query expansion and HyDE (Hypothetical Document Embeddings) formulation.
"""

import json
from pathlib import Path
from typing import Any

from src.llm import config
from src.llm.client import LLMClient
from src.config import AppSettings, get_settings
from src.utils.helpers import setup_logging

log = setup_logging("rag.query_rewriter")

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "rag" / "query_rewrite.txt"


class QueryRewriter:
    """Fast medical query rewriter & HyDE generator."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        model: str = config.REWRITER_MODEL,
        settings: AppSettings | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient(default_model=model)
        self.model = model
        self.settings = settings or get_settings()
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        if _PROMPT_PATH.is_file():
            return _PROMPT_PATH.read_text(encoding="utf-8")
        return (
            "Rewrite and expand the following user medical query for retrieval. "
            "Return JSON with keys 'expanded_query', 'medical_keywords', and 'hyde_passage':\n\n{query}"
        )

    def rewrite(self, query: str) -> dict[str, Any]:
        """Rewrite and expand a raw medical query into enriched terms and HyDE paragraph.

        Returns:
            Dict containing 'expanded_query', 'medical_keywords', 'hyde_passage'.
        """
        if not query or not query.strip():
            return {
                "expanded_query": "",
                "medical_keywords": [],
                "hyde_passage": "",
            }

        prompt = f"{self.prompt_template}\n\nUSER QUERY:\n{query}" if "{query}" not in self.prompt_template else self.prompt_template.replace("{query}", query)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm_client.complete(
                messages,
                model=self.model,
                temperature=self.settings.llm.rewriter_temperature,
                response_format={"type": "json_object"} if "gemini" in self.model or "groq" in self.model else None,
            )

            content = response.choices[0].message.content.strip()
            # Clean possible markdown json wrapper
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.startswith("json"):
                    content = content[4:].strip()

            parsed = json.loads(content)
            return {
                "expanded_query": parsed.get("expanded_query", query),
                "medical_keywords": parsed.get("medical_keywords", []),
                "hyde_passage": parsed.get("hyde_passage", ""),
                "prompt_sent": prompt,
            }

        except Exception as err:
            log.warning(f"QueryRewriter LLM call failed: {err}. Using raw query fallback.")
            return {
                "expanded_query": query,
                "medical_keywords": [],
                "hyde_passage": "",
                "prompt_sent": prompt,
            }
