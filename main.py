"""
Application entry point.

Reserved for the Medical Q&A app runtime (bots / agents / API).
KB pipeline stages are run separately as modules, e.g.:

    python -m src.ingestion.loader
    python -m src.ingestion.cleaner
    python -m src.ingestion.llm_cleaner
    python -m src.chunking.chunker
    python -m src.vectordb.vector_store
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> None:
    """App entry point — not yet implemented."""
    pass


if __name__ == "__main__":
    main()
