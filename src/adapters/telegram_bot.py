"""
Module: telegram_bot.py
Purpose: Production-grade Telegram Bot Adapter for Medical Agentic RAG.

Usage:
    # 1. Add TELEGRAM_BOT_TOKEN to your .env file
    # 2. Run:
    python -m src.adapters.telegram_bot
"""

from __future__ import annotations

import sys
import asyncio
from typing import Any

from src.config import AppSettings, get_settings
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.agents.doc_agent import DocumentationAgent
from src.agents.orchestrator import AgentOrchestrator
from src.rag.pipeline import MedicalRAGPipeline
from src.response import render_patient_response
from src.utils.helpers import setup_logging

log = setup_logging("adapters.telegram_bot")


class TelegramBotAdapter:
    """Telegram Bot Adapter wrapping MedicalRAGPipeline with async polling."""

    _telegram_message_limit = 4096

    def __init__(
        self,
        token: str | None = None,
        pipeline: MedicalRAGPipeline | None = None,
        orchestrator: AgentOrchestrator | None = None,
        settings: AppSettings | None = None,
        update_store: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.token = token or self.settings.secrets.telegram_bot_token

        if not self.token or self.token == "your_telegram_bot_token_here":
            raise ValueError(
                "❌ TELEGRAM_BOT_TOKEN is missing or not set in .env file! "
                "Please get a token from @BotFather on Telegram and set TELEGRAM_BOT_TOKEN in .env"
            )

        self.orchestrator = orchestrator or AgentOrchestrator(
            documentation_agent=None if pipeline is None else DocumentationAgent(pipeline, settings=self.settings),
            settings=self.settings,
        )
        self.pipeline = pipeline or self.orchestrator.pipeline
        self.app: Application = ApplicationBuilder().token(self.token).build()
        self._processed_update_ids: set[int] = set()
        self._processed_update_limit = 2048
        self._session_generations: dict[str, int] = {}
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self.update_store = update_store
        self._webhook_seen_ids: set[int] = set()
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register Telegram command and message handlers."""
        self.app.add_handler(CommandHandler("start", self._start_command))
        self.app.add_handler(CommandHandler("help", self._help_command))
        self.app.add_handler(CommandHandler("status", self._status_command))
        self.app.add_handler(CommandHandler("new", self._new_session_command))
        self.app.add_handler(CommandHandler("reset", self._new_session_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

    # ── Command Handlers ─────────────────────────────────────────────
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        welcome_text = (
            "🏥 *مرحباً بك في المساعد الطبي الذكي* (Medical Agentic RAG)\n\n"
            "أنا مساعدك الطبي المدعوم بنظام البحث الهجين BGE-M3 وقواعد البيانات الطبية الموثوقة (NHS, Mayo Clinic, MedlinePlus).\n\n"
            "💡 *كيف تستخدم البوت؟*\n"
            "• اكتب أي استفسار طبي بالعربية أو الإنجليزية.\n"
            "• سيقوم النظام بتحليل استفسارك، البحث عن أدق المصادر الطبية، وتوليد إجابة مدعومة بمراجع موثقة.\n\n"
            "⚙️ *الأوامر المتاحة:*\n"
            "/help — تعليمات الاستخدام\n"
            "/new أو /reset — بدء محادثة طبية جديدة\n"
            "/status — حالة النظام وتتبع الأداء (Qdrant)\n\n"
            "⚠️ *تنويه هام:* هذا البوت مخصص للتوعية والمعلومات الطبية العامة فقط، ولا يغني عن استشارة الطبيب المختص."
        )
        if update.message:
            await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = (
            "📖 *دليل الاستخدام:*\n\n"
            "1️⃣ *الأسئلة الطبية العامة:*\n"
            "مثال: \"ما هي أعراض وأسباب صداع الشقيقة؟\"\n"
            "مثال: \"What are the main causes of back pain?\"\n\n"
            "2️⃣ *الاستفسار عن الأعراض:*\n"
            "أدخل الأعراض بشرح واضح وسيتم تحليليها واستخراج المصادر الموثوقة.\n\n"
            "🔍 *المصادر المعتمدة في النظام:*\n"
            "• National Health Service (NHS UK)\n"
            "• Mayo Clinic\n"
            "• MedlinePlus National Library of Medicine\n"
            "• PubMed Central\n\n"
            "لبدء حالة جديدة استخدم /new أو /reset."
        )
        if update.message:
            await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command using the actual runtime state."""
        status = self.pipeline.status()
        qdrant = status["qdrant"]
        qdrant_ok = qdrant["connected"]

        status_text = (
            "📊 حالة النظام (System Status):\n\n"
            f"{'🟢' if qdrant_ok else '🔴'} Qdrant: {qdrant['url']}\n"
            f"   Collection: {qdrant['collection']}\n"
            f"{'🟢' if qdrant_ok else '🔴'} Embedder: {status['embedder_backend']}\n"
            f"🟢 Rewriter: {status['rewriter_model']}\n"
            f"🟢 Generator: {status['generator_model']}"
        )
        if update.message:
            await update.message.reply_text(status_text)

    async def _new_session_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """End the current inquiry and start a clean conversation."""

        if not update.message:
            return
        user = update.message.from_user
        user_id = user.id if user is not None else "unknown"
        chat_id = update.message.chat_id
        self._session_generations[str(user_id)] = self._session_generations.get(str(user_id), 0) + 1
        self.orchestrator.reset_session(user_ref=f"telegram:{chat_id}")
        await update.message.reply_text(
            "بدأت محادثة جديدة. ما العرض أو السؤال الذي تريد مناقشته؟"
        )

    # ── Text Message Handler ─────────────────────────────────────────
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process user text messages through MedicalRAGPipeline."""
        if not update.message or not update.message.text:
            return

        update_id = getattr(update, "update_id", None)
        if update_id is not None:
            if update_id in self._processed_update_ids:
                log.warning("[Telegram] Ignoring duplicate update %s", update_id)
                return
            self._processed_update_ids.add(update_id)
            if len(self._processed_update_ids) > self._processed_update_limit:
                self._processed_update_ids = set(sorted(self._processed_update_ids)[-1024:])

        query = update.message.text.strip()
        user = update.message.from_user
        user_id = user.id if user is not None else "unknown"
        chat_id = update.message.chat_id
        conversation_ref = f"telegram:{chat_id}"
        log.info("[Telegram] Received message for chat %s (length=%s)", chat_id, len(query))

        # Send typing action indicator while processing
        await update.message.reply_chat_action(ChatAction.TYPING)

        lock = self._conversation_locks.setdefault(str(chat_id), asyncio.Lock())
        async with lock:
            await self._process_message(update, query, conversation_ref)

    async def _process_message(self, update: Update, query: str, conversation_ref: str) -> None:
        try:
            context_key = self.orchestrator.context_manager.context_key(conversation_ref)
            context_before = self.orchestrator.context_manager.get(context_key)
            session_before = context_before.session_id if context_before else None
            version_before = context_before.session_version if context_before else None

            # Run RAG Pipeline in thread pool to prevent blocking asyncio loop
            result = await asyncio.to_thread(
                self.orchestrator.handle,
                query=query,
                user_ref=conversation_ref,
            )

            # /new or /reset may have been received while this request was
            # running. Never deliver a response belonging to the old session.
            context_after = self.orchestrator.context_manager.get(context_key)
            result_session = result.get("session_id")
            if session_before and context_after is None:
                log.info("[Telegram] Discarding stale response after session reset for chat %s", conversation_ref)
                return
            if result_session and context_after and context_after.session_id != result_session:
                log.info("[Telegram] Discarding response from an older session for chat %s", conversation_ref)
                return
            result_version = result.get("session_version")
            if (
                result_version is not None
                and version_before is not None
                and context_after is not None
                and context_after.session_version != result_version
            ):
                log.info("[Telegram] Discarding stale response for chat %s", conversation_ref)
                return

            with self.orchestrator.tracer.span("response_render", {"channel": "telegram"}) as render_span:
                formatted_reply = render_patient_response(result)
                render_span["response_length"] = len(formatted_reply)

            # Keep internal evidence, prompts, scores, trace data, and latency
            # out of the patient-facing Telegram message.
            await self._send_response(update.message, formatted_reply)

        except Exception:
            log.exception("[Telegram] Error executing pipeline")
            error_msg = (
                "⚠️ *حدث خطأ أثناء معالجة طلبك*\n\n"
                "يرجى المحاولة مرة أخرى أو التأكد من إعدادات الاتصال بالشبكة."
            )
            await update.message.reply_text(error_msg, parse_mode=ParseMode.MARKDOWN)

    async def _send_response(self, message: Any, text: str) -> None:
        """Send a patient response in Telegram-safe chunks with one fallback."""

        chunks = [
            text[index : index + self._telegram_message_limit]
            for index in range(0, len(text), self._telegram_message_limit)
        ] or [""]
        for chunk in chunks:
            try:
                await message.reply_text(chunk)
            except TelegramError:
                # Formatting is intentionally not enabled for generated answers;
                # a second attempt is useful for transient Telegram failures.
                await asyncio.sleep(0.2)
                await message.reply_text(chunk)

    # ── Start Polling ────────────────────────────────────────────────
    def run_polling(self) -> None:
        """Start long-polling server for Telegram bot."""
        log.info("🚀 Starting Telegram Bot Adapter in Long Polling mode...")
        self.app.run_polling(drop_pending_updates=True)

    async def process_webhook_update(self, update: Update) -> None:
        """Process one verified Telegram update and persist idempotency state."""
        update_id = getattr(update, "update_id", None)
        if update_id is not None:
            key = str(update_id)
            if self.update_store is not None:
                if hasattr(self.update_store, "claim_update"):
                    claimed = self.update_store.claim_update(key)
                else:
                    existing = self.update_store.get("telegram_update", key)
                    claimed = not (existing and existing.get("status") in {"received", "completed"})
                    if claimed:
                        self.update_store.put("telegram_update", key, {"status": "received"})
                if not claimed:
                    log.info("[Telegram] Ignoring duplicate webhook update %s", update_id)
                    return
            elif update_id in self._webhook_seen_ids:
                return
            else:
                self._webhook_seen_ids.add(update_id)
        try:
            await self.app.process_update(update)
            if update_id is not None and self.update_store is not None:
                if hasattr(self.update_store, "complete_update"):
                    self.update_store.complete_update(str(update_id))
                else:
                    self.update_store.put("telegram_update", str(update_id), {"status": "completed"})
        except Exception:
            if update_id is not None and self.update_store is not None:
                if hasattr(self.update_store, "fail_update"):
                    self.update_store.fail_update(str(update_id))
                else:
                    self.update_store.put("telegram_update", str(update_id), {"status": "failed"})
            raise


def main(settings: AppSettings | None = None) -> None:
    """CLI entry point for running Telegram Bot."""
    try:
        bot = TelegramBotAdapter(settings=settings)
        bot.run_polling()
    except ValueError as val_err:
        print(f"\n{val_err}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Telegram Bot stopped by user.")
    except Exception as err:
        log.critical(f"Failed to start Telegram Bot: {err}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
