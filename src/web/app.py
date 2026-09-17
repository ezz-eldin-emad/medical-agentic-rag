"""FastAPI entrypoint for the no-card Vercel deployment."""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request

from src.adapters.telegram_bot import TelegramBotAdapter
from src.config import AppSettings, get_settings
from src.state import QdrantStateStore


class WebRuntime:
    def __init__(self, settings: AppSettings) -> None:
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
    settings = get_settings()
    runtime = WebRuntime(settings)
    app.state.runtime = runtime
    await runtime.startup()
    try:
        yield
    finally:
        await runtime.shutdown()


app = FastAPI(title="Medical Agentic RAG", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "medical-agentic-rag", "mode": "educational-telegram-demo"}


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        return {"status": "starting"}
    return {"status": "ok", "transport": "webhook", "qdrant_state_collection": runtime.state_store.collection_name}


@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, str]:
    runtime: WebRuntime = request.app.state.runtime
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
