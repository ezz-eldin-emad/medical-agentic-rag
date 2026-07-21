"""
Module: loader.py
Purpose: Orchestrate scrapers in-process to gather raw medical data.

Usage:
    python -m src.ingestion.loader
    python -m src.ingestion.loader --scrapers nhs_scraper nice_scraper
    python -m src.ingestion.loader --list
"""

import argparse
import importlib
import sys

from src.utils.helpers import setup_logging

log = setup_logging("loader")

# ── Default scrapers (module names under src.ingestion.scrapers) ──────
_DEFAULT_SCRAPERS = [
    "nhs_scraper",
    "mayo_scraper",
    "medline_scraper",
    "nice_scraper",
]

_SCRAPER_PACKAGE = "src.ingestion.scrapers"


def _normalize_scraper_name(name: str) -> str:
    """Accept ``nhs_scraper`` or ``nhs_scraper.py``."""
    return name.removesuffix(".py")


def run_scraper(module_name: str, argv: list[str] | None = None) -> bool:
    """Import and run a scraper's ``main()`` in-process.

    Args:
        module_name: Scraper module name (e.g. ``'nhs_scraper'``).
        argv: Optional argv forwarded to the scraper's ``main``.

    Returns:
        ``True`` if the scraper succeeded, ``False`` otherwise.
    """
    module_name = _normalize_scraper_name(module_name)
    full_name = f"{_SCRAPER_PACKAGE}.{module_name}"

    log.info("-" * 50)
    log.info("Running scraper: %s", module_name)
    log.info("-" * 50)

    try:
        module = importlib.import_module(full_name)
    except ModuleNotFoundError:
        log.error("Scraper module not found: %s", full_name)
        return False

    if not hasattr(module, "main"):
        log.error("Scraper %s has no main() entry point", module_name)
        return False

    try:
        module.main(argv)
        log.info("Finished: %s", module_name)
        return True
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
        if code != 0:
            log.error("Scraper %s exited with code %s", module_name, code)
            return False
        log.info("Finished: %s", module_name)
        return True
    except Exception as e:
        log.error("Scraper %s failed: %s", module_name, e)
        return False


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Orchestrate medical data scrapers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scrapers",
        nargs="+",
        default=_DEFAULT_SCRAPERS,
        help="Scraper module names to run (with or without .py).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        dest="list_scrapers",
        help="List available scrapers and exit.",
    )
    return p.parse_args(argv)


# ── Main ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    """Entry point — run selected scrapers in-process."""
    args = parse_args(argv)

    if args.list_scrapers:
        log.info("Available scrapers:")
        for name in _DEFAULT_SCRAPERS:
            log.info("  - %s", name)
        return

    success_count = 0
    for scraper in args.scrapers:
        if run_scraper(scraper):
            success_count += 1

    total = len(args.scrapers)
    log.info("Done! Successfully ran %d/%d scrapers.", success_count, total)
    if success_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
