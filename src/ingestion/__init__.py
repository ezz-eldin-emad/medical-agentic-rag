"""Ingestion package: scrapers, regex cleaning, and LLM cleaning."""

from src.ingestion.cleaner import clean_text
from src.ingestion.loader import run_scraper

__all__ = [
    "clean_text",
    "run_scraper",
]
