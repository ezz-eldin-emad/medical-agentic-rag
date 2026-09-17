"""Thin wrapper around LiteLLM for chat completions."""

from typing import Any

from src.config import AppSettings, get_settings

from . import config


def _resolve_api_keys(api_keys: list[str] | str | None) -> list[str]:
    """Return a deduplicated list of non-empty API key strings."""
    if isinstance(api_keys, str):
        raw = [api_keys]
    elif isinstance(api_keys, list):
        raw = api_keys
    else:
        return []

    seen: set[str] = set()
    keys: list[str] = []
    for key in raw:
        key = key.strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


class LLMClient:
    """Thin wrapper around LiteLLM for chat/completions only.

    Gemini models are normalized to the ``gemini/`` prefix so requests use
    Google AI Studio with ``GEMINI_API_KEY``, not Vertex AI ADC.

    Supports multiple API keys via ``litellm.Router`` (rotation on 429).
    """

    def __init__(
        self,
        default_model: str = config.DEFAULT_MODEL,
        temperature: float = config.TEMPERATURE,
        timeout: int = config.TIMEOUT,
        api_keys: list[str] | str | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.default_model = config.normalize_model(default_model)
        self.settings = settings or get_settings()
        self.temperature = temperature if temperature != config.TEMPERATURE else self.settings.llm.temperature
        self.timeout = timeout if timeout != config.TIMEOUT else self.settings.llm.timeout_seconds
        self.router = None
        self._api_key: str | None = None

        if api_keys is None:
            api_keys = self._default_api_key(self.default_model)
        keys = _resolve_api_keys(api_keys)
        if len(keys) > 1:
            import litellm

            model_list = [
                {
                    "model_name": self.default_model,
                    "litellm_params": {
                        "model": self.default_model,
                        "api_key": key,
                        "timeout": self.timeout,
                    },
                }
                for key in keys
            ]
            router_class = getattr(litellm, "Router", None)
            if router_class is None:
                raise RuntimeError("This LiteLLM version does not provide Router support.")
            self.router = router_class(
                model_list=model_list,
                routing_strategy="simple-shuffle",
            )
        elif len(keys) == 1:
            self._api_key = keys[0]

    def _default_api_key(self, model: str) -> str:
        lower = model.casefold()
        if lower.startswith("groq/"):
            return self.settings.secrets.groq_api_key
        if lower.startswith(("gemini/", "google/", "vertex_ai/")):
            return self.settings.secrets.gemini_api_key
        if lower.startswith("openrouter/"):
            return self.settings.secrets.openrouter_api_key
        if lower.startswith("anthropic/"):
            return self.settings.secrets.anthropic_api_key
        if lower.startswith("openai/"):
            return self.settings.secrets.openai_api_key
        return ""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: Any | None = None,
        stream: bool = False,
        num_retries: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate a chat completion via LiteLLM."""
        import litellm

        call_kwargs = dict(kwargs)
        if response_format is not None:
            call_kwargs["response_format"] = response_format

        target_model = config.normalize_model(model or self.default_model)
        temp = self.temperature if temperature is None else temperature
        retries = self.settings.llm.retries if num_retries is None else num_retries

        if self.router and target_model == self.default_model:
            return self.router.completion(
                model=target_model,
                messages=messages,
                temperature=temp,
                stream=stream,
                num_retries=retries,
                **call_kwargs,
            )

        if self._api_key is not None:
            call_kwargs.setdefault("api_key", self._api_key)

        return litellm.completion(
            model=target_model,
            messages=messages,
            temperature=temp,
            timeout=self.timeout,
            stream=stream,
            num_retries=retries,
            **call_kwargs,
        )
