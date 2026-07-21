"""
Module: nhs_scraper.py
Purpose: Scrape NHS UK condition pages for medical RAG data.

Usage:
    python -m src.ingestion.scrapers.nhs_scraper
    python -m src.ingestion.scrapers.nhs_scraper --output-dir data/raw/nhs --delay 3
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

log = setup_logging("nhs_scraper")

# ── Default URLs ─────────────────────────────────────────────────────
_DEFAULT_URLS = [
    # Conditions
    "https://www.nhs.uk/conditions/headaches/",
    "https://www.nhs.uk/conditions/diabetes/",
    "https://www.nhs.uk/conditions/high-blood-pressure-hypertension/",
    "https://www.nhs.uk/conditions/common-cold/",
    "https://www.nhs.uk/conditions/back-pain/",
    "https://www.nhs.uk/conditions/migraine/",
    "https://www.nhs.uk/conditions/stomach-ache/",
    "https://www.nhs.uk/conditions/allergies/",
    "https://www.nhs.uk/conditions/insomnia/",
    # Symptoms
    "https://www.nhs.uk/symptoms/fever-in-adults/",
    "https://www.nhs.uk/symptoms/fever-in-children/",
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
def scrape_nhs_article(session: requests.Session, url: str) -> dict | None:
    """Scrape an NHS conditions page and return clean text.

    Args:
        session: Configured requests session.
        url: NHS page URL.

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

    article = soup.find("article") or soup.find("div", class_="nhsuk-main-wrapper")
    if not article:
        log.warning("Could not find article content in %s", url)
        return None

    # Remove unwanted elements
    for element in article.find_all(["script", "style", "nav", "footer", "aside"]):
        element.decompose()

    for element in article.find_all(class_=["nhsuk-review-date", "nhsuk-contents-list", "nhsuk-pagination"]):
        element.decompose()

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown"

    text = article.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    return {
        "title": title,
        "url": url,
        "source": "NHS",
        "content": clean_text,
    }


def scrape_nhs_topic_list(session: requests.Session) -> list[str]:
    """Discover all NHS Health A-Z topic URLs.

    Args:
        session: Configured requests session.

    Returns:
        De-duplicated list of topic URLs.
    """
    url = "https://www.nhs.uk/conditions/"
    response = session.get(url, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/conditions/") and len(href) > 12:
            links.append(f"https://www.nhs.uk{href}")

    return list(set(links))


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Scrape NHS UK condition pages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default="data/raw/nhs",
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
    """Entry point — scrape NHS articles."""
    args = parse_args(argv)
    root = get_project_root()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _create_session()
    urls = _DEFAULT_URLS

    for i, url in enumerate(urls):
        log.info("Scraping (%d/%d): %s", i + 1, len(urls), url)

        data = scrape_nhs_article(session, url)
        if data:
            filename = url.split("/")[-2] if url.endswith("/") else url.split("/")[-1]
            filepath = output_dir / f"{filename}.json"

            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)

            log.info("Saved: %s", filepath.name)

        time.sleep(args.delay)

    log.info("Done! All NHS articles saved to %s", output_dir)


if __name__ == "__main__":
    main()