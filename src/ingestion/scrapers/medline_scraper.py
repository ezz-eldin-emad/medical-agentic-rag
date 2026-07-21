"""
Module: medline_scraper.py
Purpose: Scrape MedlinePlus health topic pages for medical RAG data.

Usage:
    python -m src.ingestion.scrapers.medline_scraper
    python -m src.ingestion.scrapers.medline_scraper --output-dir data/raw/medlineplus --delay 3
"""

import argparse
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.helpers import get_project_root, setup_logging

log = setup_logging("medline_scraper")

# ── Default URLs ─────────────────────────────────────────────────────
_DEFAULT_URLS = [
    "https://medlineplus.gov/headache.html",
    "https://medlineplus.gov/diabetes.html",
    "https://medlineplus.gov/highbloodpressure.html",
    "https://medlineplus.gov/commoncold.html",
    "https://medlineplus.gov/backpain.html",
]


# ── HTTP Session ─────────────────────────────────────────────────────
def _create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    })
    return session


# ── Scraping ─────────────────────────────────────────────────────────
def scrape_medline_article(session: requests.Session, url: str) -> dict | None:
    """Scrape a MedlinePlus health topic page.

    Args:
        session: Configured requests session.
        url: MedlinePlus page URL.

    Returns:
        Dict with ``title``, ``url``, ``source``, ``content``, or ``None`` on failure.
    """
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        log.error("Failed to fetch %s: %s", url, e)
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    article = soup.find("article", id="main-content") or soup.find("div", class_="main")
    if not article:
        article = soup.find("main") or soup.find("div", id="topic")

    if not article:
        log.warning("Could not find content in %s", url)
        return None

    for element in article.find_all(["script", "style", "nav"]):
        element.decompose()

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown"

    text = article.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return {
        "title": title,
        "url": url,
        "source": "MedlinePlus",
        "content": "\n".join(lines),
    }


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Scrape MedlinePlus health topic pages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default="data/raw/medlineplus",
        help="Output directory for scraped JSON files.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between requests.",
    )
    return p.parse_args(argv)


# ── Main ─────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    """Entry point — scrape MedlinePlus articles."""
    args = parse_args(argv)
    root = get_project_root()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _create_session()

    for i, url in enumerate(_DEFAULT_URLS):
        log.info("Scraping (%d/%d): %s", i + 1, len(_DEFAULT_URLS), url)

        data = scrape_medline_article(session, url)
        if data:
            filename = url.split("/")[-1].replace(".html", "")
            filepath = output_dir / f"{filename}.json"

            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)

            log.info("Saved: %s", filepath.name)

        time.sleep(args.delay)

    log.info("Done! All MedlinePlus articles saved to %s", output_dir)


if __name__ == "__main__":
    main()