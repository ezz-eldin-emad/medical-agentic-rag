"""
Module: cleaner.py
Purpose: Apply regex-based text cleaning to raw scraped medical articles.
         Removes boilerplate, URLs, emails, and source-specific noise.

Usage:
    python -m src.ingestion.cleaner
    python -m src.ingestion.cleaner --sources nhs mayo
    python -m src.ingestion.cleaner --raw-dir data/raw --output-dir data/processed
"""

import argparse
import json
import re
from pathlib import Path

from src.utils.helpers import get_project_root, setup_logging

log = setup_logging("cleaner")

# ── All known sources ────────────────────────────────────────────────
_ALL_SOURCES = ["nhs", "mayo", "medlineplus", "nice"]


# ── Text Cleaning ────────────────────────────────────────────────────
def clean_text(text: str, source: str = "") -> str:
    """Clean scraped text for RAG consumption.

    Removes URLs, emails, review dates, and source-specific boilerplate.

    Args:
        text: Raw scraped text.
        source: Source identifier (e.g. ``'mayo'``) for targeted cleaning.

    Returns:
        Cleaned text string.
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove "Page last reviewed" dates
    text = re.sub(r'Page last reviewed.*?\d{4}', '', text)
    text = re.sub(r'Next review due.*?\d{4}', '', text)

    # Remove "Contents" section markers
    text = re.sub(r'^\s*Contents\s*$', '', text, flags=re.MULTILINE)

    if source.lower() == "mayo":
        # Remove Mayo Clinic specific boilerplate
        text = re.sub(r'(?i)Are you fully protected\?.*?Create my plan', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)Products & Services.*?Home Remedies', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)From Mayo Clinic to your inbox.*?Subscribe!', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)Thank you for subscribing!.*?couple of minutes', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)Request an appointment.*?resubmit the form\.', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)Show references.*', '', text, flags=re.DOTALL)
        text = re.sub(r'(?i)Request an appointment\s+Diagnosis & treatment.*', '', text, flags=re.DOTALL)
        text = text.replace("By Mayo Clinic Staff", "")
        text = text.replace("Print\n", "")

    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Apply regex-based cleaning to raw scraped data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Base directory containing raw source subdirectories.",
    )
    p.add_argument(
        "--output-dir",
        default="data/processed",
        help="Base directory for cleaned output files.",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=_ALL_SOURCES,
        help="Source names to process (subdirectory names under raw-dir).",
    )
    return p.parse_args(argv)


# ── Processing ───────────────────────────────────────────────────────
def process_source(raw_dir: Path, output_dir: Path, source: str) -> int:
    """Clean all JSON files for a single source.

    Args:
        raw_dir: Base raw data directory.
        output_dir: Base output directory.
        source: Source name (e.g. ``'nhs'``).

    Returns:
        Number of files processed.
    """
    source_raw = raw_dir / source
    source_out = output_dir / source
    source_out.mkdir(parents=True, exist_ok=True)

    if not source_raw.is_dir():
        log.warning("Raw directory not found for source '%s': %s", source, source_raw)
        return 0

    count = 0
    for json_file in sorted(source_raw.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        data["content"] = clean_text(data.get("content", ""), source=source)

        out_path = source_out / json_file.name
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

        count += 1
        log.debug("Cleaned: %s (%s)", json_file.name, source)

    return count


# ── Main ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    """Entry point — clean raw scraped files."""
    args = parse_args(argv)
    root = get_project_root()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    total = 0
    for source in args.sources:
        count = process_source(raw_dir, output_dir, source)
        if count:
            log.info("Cleaned %d files from '%s'", count, source)
        total += count

    log.info("Regex cleaning completed. Total files processed: %d", total)


if __name__ == "__main__":
    main()