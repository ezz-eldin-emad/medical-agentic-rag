"""
Module: mayo_scraper.py
Purpose: Scrape Mayo Clinic article pages for medical RAG data.

Usage:
    python -m src.ingestion.scrapers.mayo_scraper
    python -m src.ingestion.scrapers.mayo_scraper --output-dir data/raw/mayo --delay 3
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

log = setup_logging("mayo_scraper")

# ── Default URLs ─────────────────────────────────────────────────────
_DEFAULT_ARTICLES: dict[str, str] = {
    "headache": "https://www.mayoclinic.org/diseases-conditions/headache/symptoms-causes/syc-20372607",
    "diabetes": "https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444",
    "migraine": "https://www.mayoclinic.org/diseases-conditions/migraine-headache/symptoms-causes/syc-20360201",
    "common-cold": "https://www.mayoclinic.org/diseases-conditions/common-cold/symptoms-causes/syc-20351605",
    "back-pain": "https://www.mayoclinic.org/diseases-conditions/back-pain/symptoms-causes/syc-20369906",
}


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
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Sec-Fetch-User": "?1",
    })
    # Warm up: visit homepage to collect cookies before scraping articles.
    try:
        session.get("https://www.mayoclinic.org", timeout=10)
    except Exception:
        pass  # Best-effort; scraping may still work without it.
    return session


# ── Scraping ─────────────────────────────────────────────────────────
def scrape_mayo_article(session: requests.Session, url: str) -> dict | None:
    """Scrape a Mayo Clinic article page.

    Note: Mayo Clinic uses JavaScript for some content, so ``requests``
    might not capture everything. For basic articles it works well.

    Args:
        session: Configured requests session.
        url: Mayo Clinic article URL.

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

    content_div = soup.find("div", class_="content") or soup.find("main")
    if not content_div:
        content_div = soup.find("article") or soup.find("div", role="main")

    if not content_div:
        log.warning("Could not find content in %s", url)
        return None

    # Remove unwanted elements
    for element in content_div.find_all(["script", "style", "nav", "footer", "aside", "header"]):
        element.decompose()

    for element in content_div.find_all(class_=["ad", "advertisement", "promo", "newsletter"]):
        element.decompose()

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown"

    text = content_div.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    return {
        "title": title,
        "url": url,
        "source": "Mayo Clinic",
        "content": clean_text,
    }


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Scrape Mayo Clinic article pages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        default="data/raw/mayo",
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
    """Entry point — scrape Mayo Clinic articles."""
    args = parse_args(argv)
    root = get_project_root()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    session = _create_session()

    for i, (topic, url) in enumerate(_DEFAULT_ARTICLES.items()):
        log.info("Scraping (%d/%d): %s", i + 1, len(_DEFAULT_ARTICLES), url)

        data = scrape_mayo_article(session, url)
        if data:
            filepath = output_dir / f"{topic}.json"

            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)

            log.info("Saved: %s", filepath.name)

        time.sleep(args.delay)

    log.info("Done! All Mayo Clinic articles saved to %s", output_dir)


if __name__ == "__main__":
    main()