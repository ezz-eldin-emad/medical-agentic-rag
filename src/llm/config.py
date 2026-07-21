import os

# ── LLM Defaults (chat/completions only — RAG embeddings are in src.embeddings) ──
DEFAULT_MODEL = os.environ.get("LLM_DEFAULT_MODEL", "gemini/gemini-3.5-flash")
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "90"))  # seconds


def normalize_model(model: str) -> str:
    """Normalize a LiteLLM model id for Google AI Studio (API key auth).

    Bare names like ``gemini-3.5-flash`` are routed by LiteLLM to Vertex AI,
    which requires Application Default Credentials — not ``GEMINI_API_KEY``.
    The ``gemini/`` prefix forces Google AI Studio.
    """
    model = model.strip()
    if not model:
        return normalize_model(DEFAULT_MODEL)

    lower = model.lower()
    if lower.startswith(("gemini/", "vertex_ai/", "google/")):
        return model
    if lower.startswith("gemini-"):
        return f"gemini/{model}"
    return model
