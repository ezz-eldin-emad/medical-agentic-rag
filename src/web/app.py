"""FastAPI entrypoint for the no-card Vercel deployment."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from src.utils.helpers import setup_logging

log = setup_logging("web.app")

from src.config import AppSettings, get_settings


class WebRuntime:
    def __init__(self, settings: AppSettings) -> None:
        # Keep lightweight Vercel routes importable without initializing the
        # Telegram/RAG dependency graph. Heavy integrations load on webhook.
        from src.adapters.telegram_bot import TelegramBotAdapter
        from src.state import QdrantStateStore

        self.settings = settings
        if settings.runtime.telegram_transport != "webhook":
            raise RuntimeError("Vercel HTTP runtime requires TELEGRAM_TRANSPORT=webhook")
        if not settings.runtime.public_base_url:
            raise RuntimeError("PUBLIC_BASE_URL is required for webhook deployment")
        if not settings.secrets.telegram_webhook_secret:
            raise RuntimeError("TELEGRAM_WEBHOOK_SECRET is required for webhook deployment")
        if settings.runtime.state_backend != "qdrant":
            raise RuntimeError("Vercel HTTP runtime requires STATE_BACKEND=qdrant")
        self.state_store = QdrantStateStore.from_settings(settings)
        self.bot = TelegramBotAdapter(settings=settings, update_store=self.state_store)

    async def startup(self) -> None:
        await self.bot.app.initialize()
        await self.bot.app.start()

    async def shutdown(self) -> None:
        await self.bot.app.stop()
        await self.bot.app.shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Serverless platforms may invoke lightweight routes before secrets and
    # external services are needed. Defer Telegram/Qdrant startup until the
    # webhook is actually called so `/` and `/healthz` remain useful probes.
    app.state.runtime = None
    yield


app = FastAPI(title="Medical Agentic RAG", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "medical-agentic-rag", "mode": "educational-telegram-demo"}


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        settings = get_settings()
        return {
            "status": "ok",
            "transport": settings.runtime.telegram_transport,
            "state_backend": settings.runtime.state_backend,
            "configured": bool(settings.runtime.public_base_url and settings.secrets.telegram_webhook_secret),
            "checks": {
                "public_base_url": bool(settings.runtime.public_base_url),
                "webhook_secret": bool(settings.secrets.telegram_webhook_secret),
                "telegram_token": bool(settings.secrets.telegram_bot_token),
                "qdrant_url": bool(settings.vectordb.qdrant_url),
                "qdrant_api_key": bool(settings.secrets.qdrant_api_key),
                "modal_url": bool(settings.embedding.modal_url),
                "modal_token": bool(settings.secrets.modal_embed_token),
                "hf_token": bool(settings.secrets.hf_token),
                "hf_embed_model": __import__("os").environ.get("HF_EMBED_MODEL", settings.embedding.model_id),
                "groq_api_key": bool(settings.secrets.groq_api_key),
            },
        }
    return {"status": "ok", "transport": "webhook", "qdrant_state_collection": runtime.state_store.collection_name}


@app.get("/healthz/dependencies")
async def dependency_healthz(
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Authenticated, non-secret diagnostics for deployment troubleshooting."""
    settings = get_settings()
    expected = settings.secrets.telegram_webhook_secret
    if not expected or not x_telegram_bot_api_secret_token or not hmac.compare_digest(x_telegram_bot_api_secret_token, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    checks: dict[str, Any] = {}
    for name in ("telegram", "qdrant_client", "numpy", "requests"):
        try:
            __import__(name)
            checks[name] = True
        except Exception as exc:
            checks[name] = type(exc).__name__
    try:
        from src.rag.retriever import connect_qdrant
        client, _ = connect_qdrant(settings.vectordb.medical_collection, settings)
        collections = {item.name for item in client.get_collections().collections}
        checks["medical_collection"] = {
            "name": settings.vectordb.medical_collection,
            "available": settings.vectordb.medical_collection in collections,
        }
        checks["clinic_collection"] = {
            "name": settings.vectordb.clinic_collection,
            "available": settings.vectordb.clinic_collection in collections,
        }
        checks["state_collection"] = {
            "name": settings.vectordb.state_collection,
            "available": settings.vectordb.state_collection in collections,
        }
    except Exception as exc:
        checks["qdrant"] = type(exc).__name__
    return {"status": "ok", "checks": checks}


@app.get("/healthz/runtime")
async def runtime_healthz(
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, Any]:
    """Authenticated diagnostics for Telegram runtime initialization stages."""
    settings = get_settings()
    expected = settings.secrets.telegram_webhook_secret
    if not expected or not x_telegram_bot_api_secret_token or not hmac.compare_digest(x_telegram_bot_api_secret_token, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        runtime = WebRuntime(settings)
    except Exception as exc:
        log.exception("Runtime constructor failed (%s)", type(exc).__name__)
        return {"status": "error", "stage": "constructor", "error_type": type(exc).__name__}
    try:
        await runtime.startup()
    except Exception as exc:
        log.exception("Runtime startup failed (%s)", type(exc).__name__)
        return {"status": "error", "stage": "startup", "error_type": type(exc).__name__}
    try:
        await runtime.shutdown()
    except Exception as exc:
        log.exception("Runtime shutdown failed (%s)", type(exc).__name__)
        return {"status": "error", "stage": "shutdown", "error_type": type(exc).__name__}
    return {"status": "ok"}


@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, str]:
    runtime: WebRuntime | None = getattr(request.app.state, "runtime", None)
    try:
        if runtime is None:
            runtime = WebRuntime(get_settings())
            await runtime.startup()
            request.app.state.runtime = runtime
        expected = runtime.settings.secrets.telegram_webhook_secret
        if not expected or not x_telegram_bot_api_secret_token or not hmac.compare_digest(
            x_telegram_bot_api_secret_token, expected
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")

        from telegram import Update

        payload = await request.json()
        update = Update.de_json(payload, runtime.bot.app.bot)
        if update is None:
            raise HTTPException(status_code=400, detail="Invalid Telegram update")
        await runtime.bot.process_webhook_update(update)
        return {"status": "accepted"}
    except HTTPException:
        raise
    except Exception as exc:
        # Keep user/Telegram response generic while preserving a type-only
        # diagnostic in Vercel logs; never log secrets or request payloads.
        log.exception("Webhook invocation failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=500, detail="Webhook processing failed") from exc
