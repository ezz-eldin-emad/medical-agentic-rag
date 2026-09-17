"""
Main Entry Point for Medical Agentic RAG System.

Usage:
    python main.py                         # Launch Telegram Bot
    python main.py --bot                   # Same as above
    python main.py --check                 # Validate Qdrant/config without polling
    python main.py --bot --local           # Force local Qdrant and BGE-M3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_PROJECT_PYTHON = (_PROJECT_ROOT / ".venv" / "bin" / "python").resolve()

from src.config import AppSettings, get_settings


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Medical Agentic RAG Telegram bot.")
    parser.add_argument("--bot", action="store_true", help="Run the Telegram bot (the default).")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run startup checks for configuration, Qdrant, and the indexed collection, then exit.",
    )
    return parser.parse_args(argv)


def _run_check(settings: AppSettings) -> None:
    """Validate Qdrant and configuration without loading runtime models."""
    from src.rag.retriever import connect_qdrant

    client, location = connect_qdrant(
        collection_name=settings.vectordb.medical_collection,
        settings=settings,
    )
    try:
        print("✅ Startup check passed")
        print(f"   Qdrant: {location}")
        print(f"   Collection: {settings.vectordb.medical_collection}")
        print(f"   Embedder backend: {settings.embedding.backend}")
        print(f"   Rewriter: {settings.llm.rewriter_model}")
        print(f"   Generator: {settings.llm.generator_model}")
        print("ℹ️  This check does not call the LLM or embedder.")
    finally:
        client.close()


def _ensure_project_interpreter() -> None:
    """Fail early when the bot is launched with system Python."""
    running_python = Path(sys.executable).resolve()
    if running_python != _PROJECT_PYTHON:
        raise RuntimeError(
            "This project must run with its local virtual environment. "
            f"Current interpreter: {running_python}. "
            f"From the project directory, use: .venv/bin/python main.py "
            "or activate it with: source .venv/bin/activate"
        )


def _validate_telegram_dependency() -> None:
    """Give an actionable error for a missing or shadowed Telegram package."""
    try:
        import telegram
        from telegram import Update
        from telegram.ext import Application
    except (ImportError, ModuleNotFoundError) as err:
        raise RuntimeError(
            "Telegram dependency is unavailable or shadowed. Install "
            "python-telegram-bot (not the unrelated 'telegram' package) "
            f"inside {_PROJECT_ROOT / '.venv'} and run this script with "
            f"{_PROJECT_PYTHON}. Original error: {err}"
        ) from err

    if not hasattr(telegram, "__version__") or Update is None or Application is None:
        raise RuntimeError(
            "The imported 'telegram' module is not python-telegram-bot. "
            "Remove the unrelated 'telegram' package and install "
            "python-telegram-bot in the project virtual environment."
        )


def main(argv: list[str] | None = None) -> None:
    """Run the production Telegram Bot adapter or its startup check."""
    _ensure_project_interpreter()
    settings = get_settings()
    args = _parse_args(argv)

    if args.check:
        try:
            _run_check(settings)
        except Exception as err:
            print(f"❌ Startup check failed: {err}", file=sys.stderr)
            raise SystemExit(1) from err
        return

    # Import after loading .env and applying CLI overrides.
    try:
        _validate_telegram_dependency()
        from src.adapters.telegram_bot import main as run_bot
    except RuntimeError as err:
        print(f"❌ Startup dependency check failed: {err}", file=sys.stderr)
        raise SystemExit(1) from err

    print("🏥 Starting Medical Agentic RAG System...")
    run_bot(settings=settings)


if __name__ == "__main__":
    main()
