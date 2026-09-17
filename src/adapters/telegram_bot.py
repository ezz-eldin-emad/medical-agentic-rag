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
        self._session_generations[str(user_id)] = self._session_generations.get(str(user_id), 0) + 1
        self.orchestrator.reset_session(user_ref=f"telegram:{user_id}")
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
        log.info(f"[Telegram] Received query from user {user_id}: '{query[:50]}'")

        # Send typing action indicator while processing
        await update.message.reply_chat_action(ChatAction.TYPING)

        try:
            context_key = self.orchestrator.context_manager.context_key(f"telegram:{user_id}")
            context_before = self.orchestrator.context_manager.get(context_key)
            session_before = context_before.session_id if context_before else None
            generation_before = self._session_generations.get(str(user_id), 0)

            # Run RAG Pipeline in thread pool to prevent blocking asyncio loop
            import asyncio
            result = await asyncio.to_thread(
                self.orchestrator.handle,
                query=query,
                user_ref=f"telegram:{user_id}",
            )

            # /new or /reset may have been received while this request was
            # running. Never deliver a response belonging to the old session.
            context_after = self.orchestrator.context_manager.get(context_key)
            result_session = result.get("session_id")
            if self._session_generations.get(str(user_id), 0) != generation_before:
                log.info("[Telegram] Discarding response after a new session was started for user %s", user_id)
                return
            if session_before and context_after is None:
                log.info("[Telegram] Discarding stale response after session reset for user %s", user_id)
                return
            if result_session and context_after and context_after.session_id != result_session:
                log.info("[Telegram] Discarding response from an older session for user %s", user_id)
                return

            with self.orchestrator.tracer.span("response_render", {"channel": "telegram"}) as render_span:
                formatted_reply = render_patient_response(result)
                render_span["response_length"] = len(formatted_reply)

            # Keep internal evidence, prompts, scores, trace data, and latency
            # out of the patient-facing Telegram message.
            try:
                await update.message.reply_text(formatted_reply)
            except TelegramError:
                await update.message.reply_text(formatted_reply)

        except Exception:
            log.exception("[Telegram] Error executing pipeline")
            error_msg = (
                "⚠️ *حدث خطأ أثناء معالجة طلبك*\n\n"
                "يرجى المحاولة مرة أخرى أو التأكد من إعدادات الاتصال بالشبكة."
            )
            await update.message.reply_text(error_msg, parse_mode=ParseMode.MARKDOWN)

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
                existing = self.update_store.get("telegram_update", key)
                if existing and existing.get("status") in {"received", "completed"}:
                    log.info("[Telegram] Ignoring duplicate webhook update %s", update_id)
                    return
                self.update_store.put("telegram_update", key, {"status": "received"})
            elif update_id in self._webhook_seen_ids:
                return
            else:
                self._webhook_seen_ids.add(update_id)
        try:
            await self.app.process_update(update)
            if update_id is not None and self.update_store is not None:
                self.update_store.put("telegram_update", str(update_id), {"status": "completed"})
        except Exception:
            if update_id is not None and self.update_store is not None:
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
