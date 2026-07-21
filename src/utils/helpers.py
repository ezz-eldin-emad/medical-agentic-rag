"""
Module: helpers.py
Purpose: Shared utilities for all pipeline modules.

Provides:
    - load_env()           — Load .env from project root
    - get_project_root()   — Resolve the project root directory
    - setup_logging(name)  — Consistent logging configuration
    - get_git_commit()     — Current short git commit hash
    - deterministic_uuid() — Reproducible UUID5 from text content
"""

import logging
import os
import subprocess
import uuid
from pathlib import Path

# ── Namespace for deterministic UUIDs ────────────────────────────────
_UUID_NAMESPACE = uuid.UUID("a3f1b2c4-d5e6-7890-abcd-ef1234567890")


# ── Project Root ─────────────────────────────────────────────────────
def get_project_root() -> Path:
    """Return the project root directory.

    Resolution order:
        1. ``PROJECT_ROOT`` environment variable (if set).
        2. Three levels up from this file (``src/utils/helpers.py`` → project root).
    """
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    return Path(__file__).resolve().parent.parent.parent


# ── Environment Loading ──────────────────────────────────────────────
def load_env(env_path: Path | None = None) -> None:
    """Load variables from a ``.env`` file into ``os.environ``.

    Tries ``python-dotenv`` first; falls back to a simple manual parser
    so the script works without extra dependencies.

    Args:
        env_path: Explicit path to ``.env``. Defaults to
                  ``<project_root>/.env``.
    """
    if env_path is None:
        env_path = get_project_root() / ".env"

    # Try python-dotenv
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    # Manual fallback
    if not env_path.is_file():
        return
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            # Don't overwrite existing env vars
            os.environ.setdefault(key, value)


# ── Logging ──────────────────────────────────────────────────────────
def setup_logging(
    name: str,
    *,
    level: int = logging.INFO,
    fmt: str = "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure and return a logger with a consistent format.

    Args:
        name: Logger name (typically ``__name__`` of the calling script).
        level: Logging level. Defaults to ``INFO``.
        fmt: Log message format string.
        datefmt: Date/time format string.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


# ── Git Commit ───────────────────────────────────────────────────────
def get_git_commit() -> str:
    """Return the short git commit hash, or ``'unknown'`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(get_project_root()),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


# ── Deterministic UUID ───────────────────────────────────────────────
def deterministic_uuid(text: str) -> str:
    """Generate a reproducible UUID5 string from *text*.

    The same input text always produces the same UUID, making upserts
    idempotent — re-indexing identical chunks won't create duplicates.

    Args:
        text: The chunk text to hash.

    Returns:
        UUID string (e.g. ``'a1b2c3d4-...'``).
    """
    return str(uuid.uuid5(_UUID_NAMESPACE, text))
